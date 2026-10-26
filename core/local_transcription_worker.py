"""MediaIntel — Hybrid Architecture Phase 1: Standalone Local Transcription Worker.

This worker runs EXCLUSIVELY on the user's local machine (Windows laptop) to leverage
residential IP network access for YouTube captions and local Faster-Whisper CPU execution.

Strict Division of Responsibilities:
  LOCAL (This Worker):
    - YouTube caption extraction (manual & auto-generated).
    - Authorized & accessible YouTube audio streaming/acquisition.
    - Faster-Whisper speech-to-text inference on CPU (int8).
    - Secure persistence of completed transcripts directly to Supabase PostgreSQL.

  CLOUD (GitHub Actions & Streamlit Cloud):
    - YouTube channel, video, and comment metadata ingestion.
    - AI sentiment analysis (XLM-RoBERTa aspect-based).
    - Relevance scoring and topic mining.
    - Anomaly detection.
    - RPA reporting and PDF generation.
    - Web dashboard serving.

Safety Guarantees:
  - Requires PostgreSQL (MEDIAINTEL_DB_URL); rejects accidental production SQLite processing.
  - Safe read-only dry-run mode (--dry-run).
  - Single-video targeting (--video-id).
  - Never overwrites valid completed transcripts.
  - Never triggers cloud AI pipelines or dispatches GitHub Actions workflows.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import make_url

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure .env is explicitly loaded before any core modules or database engine initialization
load_dotenv(ROOT_DIR / ".env", override=False)

from core import config, channel_manager, storage_audio
from core.cloud_whisper_worker import (
    extract_youtube_captions,
    save_transcript_record,
    transcribe_audio_pcm,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("local_transcription_worker")


def validate_database_backend(allow_sqlite: bool = False) -> str:
    """Validate that the configured database is PostgreSQL.

    Rejects accidental production runs against SQLite to ensure all local
    transcriptions are safely committed to the central Supabase PostgreSQL database.
    """
    # Ensure .env is reloaded if environment was updated after initial load
    load_dotenv(ROOT_DIR / ".env", override=False)

    raw_url = config.database_url()
    parsed_url = make_url(raw_url)
    dialect = parsed_url.get_backend_name()

    if dialect != "postgresql":
        if allow_sqlite:
            LOGGER.warning(
                "Running against non-PostgreSQL backend (%s) because allow_sqlite is enabled.",
                dialect,
            )
            return dialect

        raise RuntimeError(
            f"Production Local Transcription Worker requires a PostgreSQL database (MEDIAINTEL_DB_URL).\n"
            f"Detected dialect: '{dialect}'. Accidental processing against SQLite is rejected.\n"
            f"Action Required:\n"
            f"  1. Add MEDIAINTEL_DB_URL to your local '.env' file:\n"
            f"     MEDIAINTEL_DB_URL=postgresql://postgres.<project-ref>:[PASSWORD]@aws-0-[region].pooler.supabase.com:6543/postgres?sslmode=require\n"
            f"  2. Rerun the worker. (Or pass --allow-sqlite for offline test runs)."
        )

    return dialect


def get_candidate_videos(
    *,
    limit_per_channel: int = 2,
    video_id: str | None = None,
    channel_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, int, list[str]]:
    """Retrieve candidate videos for transcription.

    Returns:
        (candidates_df, tracked_channels_count, shortfall_warnings)
    """
    tracked_channels_df = config.read_sql(
        "SELECT channel_id, channel_name FROM channels WHERE is_tracked = TRUE ORDER BY channel_name ASC"
    )
    if channel_ids:
        c_set = set(channel_ids)
        tracked_channels_df = tracked_channels_df[tracked_channels_df["channel_id"].isin(c_set)]

    tracked_channels_count = len(tracked_channels_df)

    if video_id and video_id.strip():
        v_clean = video_id.strip()
        v_df = config.read_sql(
            """
            SELECT v.video_id, v.channel_id, v.title, v.published_at, v.duration_sec, COALESCE(v.view_count, 0) AS view_count,
                   c.channel_name, c.is_tracked, t.source AS tsource, t.status AS tstatus,
                   t.transcript_text, t.attempts
            FROM videos v
            JOIN channels c ON c.channel_id = v.channel_id
            LEFT JOIN transcripts t ON t.video_id = v.video_id
            WHERE v.video_id = :vid
            """,
            {"vid": v_clean},
        )
        if v_df.empty:
            LOGGER.warning("Target video_id '%s' not found in database.", v_clean)
            return pd.DataFrame(), tracked_channels_count, []

        first_row = v_df.iloc[0]
        is_tracked_val = first_row.get("is_tracked")
        is_tracked_bool = bool(is_tracked_val) if is_tracked_val is not None else False
        if not is_tracked_bool:
            LOGGER.warning(
                "Target video_id '%s' belongs to channel '%s' which is currently not tracked.",
                v_clean,
                first_row.get("channel_name", "unknown"),
            )

        return v_df, 1, []

    candidates_df = channel_manager.get_transcription_candidate_videos(
        limit_per_channel=limit_per_channel,
        channel_ids=channel_ids,
    )

    shortfalls: list[str] = []
    if not tracked_channels_df.empty:
        c_counts = candidates_df["channel_id"].value_counts().to_dict() if not candidates_df.empty else {}
        for row in tracked_channels_df.itertuples():
            cnt = c_counts.get(row.channel_id, 0)
            if cnt < limit_per_channel:
                shortfalls.append(f"{row.channel_name} ({cnt}/{limit_per_channel} videos)")

    return candidates_df, tracked_channels_count, shortfalls


def identify_cached_transcripts(candidates_df: pd.DataFrame) -> dict[str, tuple[bool, str, str]]:
    """Inspect existing transcripts in database to avoid retranscribing completed videos."""
    if candidates_df.empty:
        return {}

    vids = candidates_df["video_id"].astype(str).tolist()
    if not vids:
        return {}

    # Query existing transcripts for these candidate videos
    tx_df = config.read_sql(
        "SELECT video_id, source, transcript_text, language, model_name, status FROM transcripts"
    )
    cached_map: dict[str, tuple[bool, str, str]] = {}
    if not tx_df.empty:
        for r in tx_df.itertuples():
            vid = str(r.video_id)
            src = str(r.source or "")
            body = str(r.transcript_text or "")
            st = str(getattr(r, "status", "") or "completed")
            is_valid_completed = (
                bool(body.strip())
                and src in (
                    "whisper",
                    "faster_whisper",
                    "captions",
                    "youtube_manual_caption",
                    "youtube_auto_caption",
                )
                and st not in ("failed", "awaiting_authorized_audio")
            )
            cached_map[vid] = (is_valid_completed, src, st)

    return cached_map


def print_run_summary(
    *,
    backend: str,
    dry_run: bool,
    tracked_channels_count: int,
    target_videos_count: int,
    outcomes: dict[str, int],
    total_time: float,
    unsuccessful_videos: list[tuple[str, str, str]],
    pending_count: int = 0,
) -> None:
    """Print an executive, formatted summary of the local transcription run."""
    print()
    print("=" * 60)
    mode_str = "DRY-RUN (READ-ONLY)" if dry_run else f"LIVE PRODUCTION ({backend.upper()})"
    print(f"MEDIAINTEL LOCAL TRANSCRIPTION WORKER — {mode_str}")
    print("=" * 60)
    print(f"Database Backend:            {backend}")
    print(f"Tracked Channels:            {tracked_channels_count}")
    print(f"Target Videos Evaluated:     {target_videos_count}")
    print(f"Skipped (Already Completed): {outcomes.get('reused_existing', 0)}")
    if dry_run:
        print(f"Pending Transcriptions:      {pending_count}")
    else:
        print(f"Manual Captions Extracted:   {outcomes.get('manual_captions_extracted', 0)}")
        print(f"Auto Captions Extracted:     {outcomes.get('auto_captions_extracted', 0)}")
        print(f"Faster-Whisper Completed:    {outcomes.get('faster_whisper_completed', 0)}")
        print(f"Awaiting Authorized Audio:   {outcomes.get('awaiting_authorized_audio', 0)}")
        print(f"Failed:                      {outcomes.get('failed', 0)}")
        print(f"Skipped / Locked:            {outcomes.get('skipped', 0)}")
    print("-" * 60)
    if dry_run:
        total_accounted = outcomes.get("reused_existing", 0) + pending_count
        status_str = "RECONCILED" if total_accounted == target_videos_count else "DISCREPANCY"
        print(f"Outcome Reconciliation:      {status_str} ({total_accounted}/{target_videos_count} accounted)")
    else:
        total_processed = (
            outcomes.get("reused_existing", 0)
            + outcomes.get("manual_captions_extracted", 0)
            + outcomes.get("auto_captions_extracted", 0)
            + outcomes.get("faster_whisper_completed", 0)
            + outcomes.get("awaiting_authorized_audio", 0)
            + outcomes.get("failed", 0)
            + outcomes.get("skipped", 0)
        )
        status_str = "RECONCILED" if total_processed == target_videos_count else "DISCREPANCY"
        print(f"Outcome Reconciliation:      {status_str} ({total_processed}/{target_videos_count})")
    print(f"Total Processing Time:       {total_time:.2f} seconds")
    print("=" * 60)

    if unsuccessful_videos:
        print()
        print("--- Unsuccessful Videos & Action Items ---")
        for vid, title, reason in unsuccessful_videos:
            print(f"- [{vid}] {title[:40]}: {reason}")
        print("-" * 60)
    print()


def run_local_worker(
    *,
    limit: int = 2,
    video_id: str | None = None,
    channel_ids: list[str] | None = None,
    whisper_model: str = "tiny",
    max_seconds: int = 900,
    dry_run: bool = False,
    force_retranscribe: bool = False,
    allow_sqlite: bool = False,
) -> dict[str, Any]:
    """Execute the local transcription worker.

    Safe, production-grade hybrid architecture worker:
      1. Validates PostgreSQL backend (rejects accidental SQLite).
      2. Queries eligible tracked candidate videos (latest 2 per channel or targeted video).
      3. Skips already-completed valid transcripts.
      4. In dry-run mode: prints detailed plan and exits without any DB writes or downloads.
      5. In live mode:
         - Priority 2: Accessible YouTube captions via residential IP.
         - Priority 3: Faster-Whisper if captions absent and audio accessible.
         - Priority 4: Gracefully marks awaiting_authorized_audio if audio download unavailable.
      6. Reconciles outcomes and updates channel run coverage telemetry.
      7. Exits cleanly with zero cloud job dispatch.
    """
    start_time = time.perf_counter()

    # 1. Database validation
    backend = validate_database_backend(allow_sqlite=allow_sqlite)
    LOGGER.info("Connected to database backend: %s", backend)

    # 2. Candidate video selection
    candidates_df, tracked_channels_count, shortfalls = get_candidate_videos(
        limit_per_channel=limit,
        video_id=video_id,
        channel_ids=channel_ids,
    )
    target_videos_count = len(candidates_df)

    if shortfalls:
        LOGGER.info(
            "Channel candidate shortfalls detected: %s",
            ", ".join(shortfalls),
        )

    outcomes = {
        "reused_existing": 0,
        "manual_captions_extracted": 0,
        "auto_captions_extracted": 0,
        "faster_whisper_completed": 0,
        "awaiting_authorized_audio": 0,
        "failed": 0,
        "skipped": 0,
    }
    unsuccessful_videos: list[tuple[str, str, str]] = []

    if candidates_df.empty:
        LOGGER.info("No candidate videos found for transcription.")
        total_time = time.perf_counter() - start_time
        print_run_summary(
            backend=backend,
            dry_run=dry_run,
            tracked_channels_count=tracked_channels_count,
            target_videos_count=0,
            outcomes=outcomes,
            total_time=total_time,
            unsuccessful_videos=unsuccessful_videos,
        )
        return {
            "success": True,
            "dry_run": dry_run,
            "backend": backend,
            "tracked_channels": tracked_channels_count,
            "target_videos": 0,
            **outcomes,
            "total_time_seconds": round(total_time, 2),
        }

    # 3. Check existing completed transcripts
    cached_map = identify_cached_transcripts(candidates_df)
    needed_videos: list[dict[str, Any]] = []

    for row in candidates_df.to_dict("records"):
        vid = str(row["video_id"])
        c_info = cached_map.get(vid)
        if c_info and c_info[0] and not force_retranscribe:
            outcomes["reused_existing"] += 1
            LOGGER.info("Video %s already has valid transcript (%s) - skipped.", vid, c_info[1])
        else:
            needed_videos.append(row)

    LOGGER.info(
        "Candidate evaluation: %d total, %d completed (skipped), %d require transcription.",
        target_videos_count,
        outcomes["reused_existing"],
        len(needed_videos),
    )

    # 4. Safe Dry-Run Handling
    if dry_run:
        LOGGER.info("DRY-RUN: Inspecting candidates without database writes, audio downloads, or Whisper inference.")
        for row in needed_videos:
            vid = str(row["video_id"])
            cname = str(row.get("channel_name", "Unknown"))
            title = str(row.get("title", vid))
            LOGGER.info("[DRY-RUN QUEUED] %s (%s) [Channel: %s]", title[:45], vid, cname)

        total_time = time.perf_counter() - start_time
        print_run_summary(
            backend=backend,
            dry_run=True,
            tracked_channels_count=tracked_channels_count,
            target_videos_count=target_videos_count,
            outcomes=outcomes,
            total_time=total_time,
            unsuccessful_videos=unsuccessful_videos,
            pending_count=len(needed_videos),
        )
        return {
            "success": True,
            "dry_run": True,
            "backend": backend,
            "tracked_channels": tracked_channels_count,
            "target_videos": target_videos_count,
            **outcomes,
            "pending_transcriptions": len(needed_videos),
            "total_time_seconds": round(total_time, 2),
        }

    # 5. Live Processing for Needed Videos
    for row in needed_videos:
        vid = str(row["video_id"])
        cid = str(row.get("channel_id") or "")
        title = str(row.get("title") or vid)
        cname = str(row.get("channel_name") or "Unknown")

        LOGGER.info("--- Processing video: %s ('%s') [%s] ---", vid, title[:40], cname)

        # Step 5A: Priority 2 — Attempt Accessible YouTube Captions First
        cap_res = None
        try:
            LOGGER.info("Attempting YouTube caption extraction for %s...", vid)
            cap_res = extract_youtube_captions(vid)
        except Exception as cap_exc:
            LOGGER.warning("Caption extraction error for %s: %s", vid, cap_exc)

        if cap_res and cap_res.get("text"):
            save_transcript_record(
                video_id=vid,
                source=cap_res["source"],
                status="completed",
                channel_id=cid,
                language=cap_res["language"],
                caption_type=cap_res["caption_type"],
                model_name="youtube-captions",
                transcript_text=cap_res["text"],
                audio_seconds=cap_res["duration_sec"],
                attempts=1,
            )
            if cap_res["source"] == "youtube_manual_caption":
                outcomes["manual_captions_extracted"] += 1
                LOGGER.info("[CAPTIONS MANUAL] %s saved (%s, %d words).", vid, cap_res["language"], cap_res["word_count"])
            else:
                outcomes["auto_captions_extracted"] += 1
                LOGGER.info("[CAPTIONS AUTO] %s saved (%s, %d words).", vid, cap_res["language"], cap_res["word_count"])
            continue

        # Step 5B: Priority 3 — Faster-Whisper Fallback
        LOGGER.info("Captions not accessible for %s. Attempting audio acquisition for Faster-Whisper...", vid)

        # Check storage first, then permitted local streaming
        mapping = storage_audio.get_audio_mapping(vid)
        pcm = None

        if mapping is not None:
            LOGGER.info("Found existing audio mapping in storage for %s.", vid)
            claim = storage_audio.claim_audio_mapping(vid)
            if claim:
                try:
                    pcm = storage_audio.storage_audio_to_pcm(claim["bucket"], claim["object_path"], max_seconds)
                except Exception as pcm_err:
                    LOGGER.warning("Could not load storage audio for %s: %s", vid, pcm_err)
                    storage_audio.fail_audio_claim(vid, claim_token=claim["claim_token"], error=str(pcm_err)[:250])

        video_dur = row.get("duration_sec")
        desc = row.get("description")
        chapters = None
        effective_dur = int(video_dur) if (video_dur and video_dur > 0) else None

        if pcm is None:
            try:
                from core import transcribe_videos
                from core.smart_transcription import SMART_TRANSCRIPTION_THRESHOLD_SEC, plan_transcription_segments

                LOGGER.info("Acquiring audio stream via local residential connection for %s...", vid)

                resolved_dur = None
                try:
                    _, _, resolved_dur, resolved_ch, resolved_desc = transcribe_videos.resolve_stream_info(vid)
                    if chapters is None and resolved_ch:
                        chapters = resolved_ch
                    if not desc and resolved_desc:
                        desc = resolved_desc
                except Exception:
                    pass

                if effective_dur is None and resolved_dur and resolved_dur > 0:
                    effective_dur = int(resolved_dur)

                t_audio_start = time.perf_counter()
                # Check if this is a video >= 15 min requiring smart segment extraction
                if effective_dur is not None and effective_dur >= SMART_TRANSCRIPTION_THRESHOLD_SEC:
                    plan = plan_transcription_segments(
                        video_duration_sec=effective_dur,
                        chapters=chapters,
                        description=desc,
                        max_partial_sec=max_seconds,
                    )
                    LOGGER.info(
                        "Video %s is %ds (>= %ds threshold): extracting %d representative segments totaling %ds via FFmpeg input seeking...",
                        vid, effective_dur, SMART_TRANSCRIPTION_THRESHOLD_SEC, len(plan.segments), plan.total_audio_seconds,
                    )
                    pcm = transcribe_videos.fetch_audio_segments(vid, plan)
                else:
                    fetch_dur = effective_dur if effective_dur else max_seconds
                    pcm = transcribe_videos.fetch_audio(vid, fetch_dur)
                audio_fetch_time = time.perf_counter() - t_audio_start
                LOGGER.info("[PERFORMANCE] Audio acquisition completed in %.2fs.", audio_fetch_time)
            except Exception as dl_exc:
                err_str = str(dl_exc)[:140]
                LOGGER.warning("Audio acquisition unavailable for %s: %s", vid, err_str)
                save_transcript_record(
                    video_id=vid,
                    source="awaiting_authorized_audio",
                    status="awaiting_authorized_audio",
                    channel_id=cid,
                    last_error=f"Audio download unavailable ({err_str}); awaiting authorized audio upload",
                )
                outcomes["awaiting_authorized_audio"] += 1
                unsuccessful_videos.append((vid, title, f"Audio acquisition unavailable: {err_str}"))
                continue

        # Faster-Whisper CPU Inference
        try:
            import tracemalloc
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            t_whisper_start = time.perf_counter()
            target_dur = effective_dur if (effective_dur and effective_dur > 0) else video_dur
            LOGGER.info("Transcribing audio for %s via Faster-Whisper ('%s', duration=%s)...", vid, whisper_model, target_dur)
            w_text, w_lang, w_sec, w_trunc = transcribe_audio_pcm(
                pcm,
                max_seconds=max_seconds,
                model_size=whisper_model,
                video_duration_sec=target_dur,
                chapters=chapters,
                description=desc,
            )
            whisper_infer_time = time.perf_counter() - t_whisper_start
            cur_mem, peak_mem = tracemalloc.get_traced_memory()
            LOGGER.info(
                "[PERFORMANCE] Faster-Whisper inference completed in %.2fs (Peak memory tracked: %.2f MB).",
                whisper_infer_time,
                peak_mem / (1024 * 1024),
            )
            del pcm

            if not w_text or not w_text.strip():
                raise RuntimeError("Faster-Whisper returned empty transcript.")

            save_transcript_record(
                video_id=vid,
                channel_id=cid,
                source="faster_whisper",
                status="completed",
                language=w_lang,
                caption_type="partial_15m" if w_trunc else "full",
                model_name=f"whisper-{whisper_model}",
                transcript_text=w_text,
                audio_seconds=w_sec,
                is_truncated=w_trunc,
                video_duration_sec=target_dur,
            )
            outcomes["faster_whisper_completed"] += 1
            cov_desc = "PARTIAL (<= 15 min)" if w_trunc else "FULL"
            LOGGER.info(
                "[WHISPER SUCCESS] %s transcribed (%s, %s, %d words, %d sec).",
                vid,
                cov_desc,
                w_lang,
                len(w_text.split()),
                w_sec,
            )
        except Exception as w_exc:
            LOGGER.exception("Faster-Whisper transcription failed for %s: %s", vid, w_exc)
            save_transcript_record(
                video_id=vid,
                source="failed",
                status="failed",
                channel_id=cid,
                last_error=f"Faster-Whisper error: {str(w_exc)[:150]}",
            )
            outcomes["failed"] += 1
            unsuccessful_videos.append((vid, title, f"Whisper failed: {str(w_exc)[:80]}"))

    # 6. Refresh channel coverage telemetry in Supabase
    try:
        channel_manager.update_all_channel_run_coverage()
        LOGGER.info("Updated channel run coverage telemetry in database.")
    except Exception as cov_exc:
        LOGGER.warning("Could not refresh channel coverage telemetry: %s", cov_exc)

    total_time = time.perf_counter() - start_time
    print_run_summary(
        backend=backend,
        dry_run=False,
        tracked_channels_count=tracked_channels_count,
        target_videos_count=target_videos_count,
        outcomes=outcomes,
        total_time=total_time,
        unsuccessful_videos=unsuccessful_videos,
    )

    return {
        "success": True,
        "dry_run": False,
        "backend": backend,
        "tracked_channels": tracked_channels_count,
        "target_videos": target_videos_count,
        **outcomes,
        "total_time_seconds": round(total_time, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MediaIntel Hybrid Architecture — Standalone Local Transcription Worker"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform read-only inspection of candidate videos without writes, audio downloads, or Whisper inference.",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Target a specific eligible video ID.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Number of latest eligible videos per tracked channel to evaluate (default: 2).",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="tiny",
        help="Faster-Whisper model size (tiny, base, small; default: tiny).",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=1800,
        help="Maximum audio seconds to transcribe per video (default: 1800s / 30m).",
    )
    parser.add_argument(
        "--force-retranscribe",
        action="store_true",
        help="Bypass cache and force retranscription of eligible videos.",
    )
    parser.add_argument(
        "--allow-sqlite",
        action="store_true",
        help="Allow running against SQLite database for offline automated testing.",
    )

    args = parser.parse_args()

    try:
        res = run_local_worker(
            limit=args.limit,
            video_id=args.video_id,
            whisper_model=args.whisper_model,
            max_seconds=args.max_seconds,
            dry_run=args.dry_run,
            force_retranscribe=args.force_retranscribe,
            allow_sqlite=args.allow_sqlite,
        )
        return 0 if res.get("success") else 1
    except Exception as exc:
        LOGGER.error("Local transcription worker failed: %s", exc)
        print(f"\n[FATAL ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
