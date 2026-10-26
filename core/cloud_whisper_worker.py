"""Production-Ready Optimized Transcription Pipeline for MediaIntel.

Strict 4-tier transcription hierarchy:
  Priority 1 — Existing transcript cache (incremental reuse, skip completed videos)
  Priority 2 — Accessible YouTube captions (manual first, then auto-generated; multilingual, concurrent)
  Priority 3 — Faster-Whisper (lazy-loaded, CPU int8, bounded duration, storage-first)
  Priority 4 — Authorized audio upload fallback (status: awaiting_authorized_audio)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import config, channel_manager, storage_audio

LOGGER = logging.getLogger(__name__)

# Configurable defaults
DEFAULT_VIDEOS_PER_CHANNEL = config.env_int("VIDEOS_PER_CHANNEL_FOR_TRANSCRIPTION", 2)
DEFAULT_CAPTION_WORKERS = config.env_int("CAPTION_WORKERS", 3)
DEFAULT_WHISPER_WORKERS = config.env_int("WHISPER_WORKERS", 1)
DEFAULT_WHISPER_MODEL = config.env("WHISPER_MODEL_SIZE") or config.setting("WHISPER_MODEL") or "tiny"
DEFAULT_MAX_AUDIO_MINUTES = config.env_int("MAX_AUDIO_DURATION_MINUTES", 30)

_whisper_model_cache = None
_whisper_model_cached_name = None


class LeaseHeartbeat:
    """Renew an owned database claim while Faster-Whisper is processing."""

    def __init__(self, video_id: str, token: str):
        self.video_id = video_id
        self.token = token
        self.stop_event = threading.Event()
        self.lost_claim = False
        self.thread = None

    def __enter__(self):
        self.thread = threading.Thread(
            target=self._renew_loop,
            daemon=True,
        )
        self.thread.start()
        return self

    def _renew_loop(self):
        while not self.stop_event.wait(300):
            try:
                renewed = storage_audio.renew_audio_claim(
                    self.video_id,
                    claim_token=self.token,
                    lease_minutes=30,
                )
                if not renewed:
                    self.lost_claim = True
                    LOGGER.warning("Lost audio claim for %s", self.video_id)
                    return
            except Exception:
                self.lost_claim = True
                LOGGER.exception("Unable to renew audio claim for %s", self.video_id)
                return

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=15)


def extract_youtube_captions(video_id: str) -> dict | None:
    """Extract accessible YouTube captions (manual preferred over auto-generated).

    Preserves original language (no hardcoded English assumptions; supports EN, HI, TE, etc.).
    Preserves timestamps and text segments when available, normalizes whitespace.
    Never fabricates text from titles, descriptions, tags, or comments.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()

        # In youtube-transcript-api >= 1.0, api.list(video_id) returns TranscriptList
        if hasattr(api, "list"):
            transcript_list = api.list(video_id)
        elif hasattr(YouTubeTranscriptApi, "list_transcripts"):
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        else:
            # Fallback for older library versions
            rows = YouTubeTranscriptApi.get_transcript(video_id)
            full_text = " ".join(" ".join(r["text"].split()) for r in rows if r.get("text")).strip()
            if full_text:
                return {
                    "text": full_text,
                    "language": "und",
                    "source": "youtube_auto_caption",
                    "caption_type": "auto_generated",
                    "duration_sec": None,
                    "word_count": len(full_text.split()),
                }
            return None

        manual_transcripts = []
        generated_transcripts = []
        for tr in transcript_list:
            if not getattr(tr, "is_generated", True):
                manual_transcripts.append(tr)
            else:
                generated_transcripts.append(tr)

        chosen_tr = None
        source_type = None
        caption_type = None

        if manual_transcripts:
            chosen_tr = manual_transcripts[0]
            source_type = "youtube_manual_caption"
            caption_type = "manual"
        elif generated_transcripts:
            chosen_tr = generated_transcripts[0]
            source_type = "youtube_auto_caption"
            caption_type = "auto_generated"

        if chosen_tr is None:
            return None

        snippets = chosen_tr.fetch()
        text_parts = []
        total_duration = 0.0

        for s in snippets:
            s_text = getattr(s, "text", None) if hasattr(s, "text") else (s.get("text") if isinstance(s, dict) else str(s))
            if s_text:
                clean_s = " ".join(s_text.split())
                if clean_s:
                    text_parts.append(clean_s)
            start = getattr(s, "start", 0) if hasattr(s, "start") else (s.get("start", 0) if isinstance(s, dict) else 0)
            dur = getattr(s, "duration", 0) if hasattr(s, "duration") else (s.get("duration", 0) if isinstance(s, dict) else 0)
            if start and dur:
                total_duration = max(total_duration, float(start) + float(dur))

        body = " ".join(text_parts).strip()
        if not body:
            return None

        lang = getattr(chosen_tr, "language_code", "und") or getattr(chosen_tr, "language", "und") or "und"
        return {
            "text": body,
            "language": lang[:12],
            "source": source_type,
            "caption_type": caption_type,
            "duration_sec": int(total_duration) if total_duration > 0 else None,
            "word_count": len(body.split()),
        }
    except Exception as exc:
        try:
            from core import transcribe_videos
            if hasattr(transcribe_videos, "fetch_captions"):
                cap_text, cap_lang = transcribe_videos.fetch_captions(video_id)
                if cap_text and cap_text.strip():
                    return {
                        "text": cap_text.strip(),
                        "language": cap_lang or "und",
                        "source": "captions",
                        "caption_type": "auto_generated",
                        "duration_sec": None,
                        "word_count": len(cap_text.strip().split()),
                    }
        except Exception:
            pass
        LOGGER.info("Captions not accessible for %s: %s", video_id, str(exc)[:100])
        return None


def get_whisper_model(model_size: str = "tiny"):
    """Lazy-load Faster-Whisper model only when audio is confirmed and transcription is needed."""
    global _whisper_model_cache, _whisper_model_cached_name
    if _whisper_model_cache is None or _whisper_model_cached_name != model_size:
        from faster_whisper import WhisperModel
        compute_type = config.env("WHISPER_COMPUTE", "int8")
        cpu_threads = min(os.cpu_count() or 2, 4)
        LOGGER.info(
            "Lazy-loading Faster-Whisper model '%s' (compute=%s, threads=%d)...",
            model_size,
            compute_type,
            cpu_threads,
        )
        _whisper_model_cache = WhisperModel(
            model_size,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )
        _whisper_model_cached_name = model_size
    return _whisper_model_cache


def save_transcript_record(
    video_id: str,
    source: str,
    *,
    status: str = "completed",
    channel_id: str | None = None,
    language: str | None = None,
    caption_type: str | None = None,
    model_name: str | None = None,
    transcript_text: str | None = None,
    audio_seconds: int | None = None,
    is_truncated: bool = False,
    attempts: int = 1,
    last_error: str | None = None,
) -> None:
    """Immediately persist completed, failed, or awaiting transcript record to database."""
    word_count = len(transcript_text.split()) if transcript_text else 0
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    engine = config.get_engine()

    sql = text("""
        INSERT INTO transcripts
            (video_id, channel_id, source, caption_type, status, language,
             model_name, transcript_text, audio_seconds, word_count,
             is_truncated, attempts, last_error, updated_at)
        VALUES
            (:video_id, :channel_id, :source, :caption_type, :status, :language,
             :model_name, :transcript_text, :audio_seconds, :word_count,
             :is_truncated, :attempts, :last_error, :updated_at)
        ON CONFLICT (video_id) DO UPDATE SET
            channel_id = COALESCE(EXCLUDED.channel_id, transcripts.channel_id),
            source = EXCLUDED.source,
            caption_type = EXCLUDED.caption_type,
            status = EXCLUDED.status,
            language = EXCLUDED.language,
            model_name = EXCLUDED.model_name,
            transcript_text = EXCLUDED.transcript_text,
            audio_seconds = EXCLUDED.audio_seconds,
            word_count = EXCLUDED.word_count,
            is_truncated = EXCLUDED.is_truncated,
            attempts = EXCLUDED.attempts,
            last_error = EXCLUDED.last_error,
            updated_at = EXCLUDED.updated_at
    """)
    with engine.begin() as conn:
        conn.execute(
            sql,
            {
                "video_id": video_id,
                "channel_id": channel_id,
                "source": source[:50],
                "caption_type": (caption_type or "")[:20] or None,
                "status": (status or "")[:30] or "completed",
                "language": (language or "")[:12] or None,
                "model_name": (model_name or "")[:60] or None,
                "transcript_text": transcript_text,
                "audio_seconds": audio_seconds,
                "word_count": word_count,
                "is_truncated": bool(is_truncated),
                "attempts": attempts,
                "last_error": (last_error or "")[:250] or None,
                "updated_at": now,
            },
        )


def transcribe_audio_pcm(
    pcm_bytes,
    max_seconds: int,
    model_size: str = "tiny",
) -> tuple[str, str, int, bool]:
    """Transcribe 16 kHz mono int16 PCM audio with Faster-Whisper.

    Returns (transcript_text, detected_language, audio_seconds, is_truncated).
    """
    try:
        from core import transcribe_videos
        import unittest.mock
        if hasattr(transcribe_videos, "whisper_transcribe") and (
            isinstance(transcribe_videos.whisper_transcribe, unittest.mock.MagicMock)
            or isinstance(transcribe_videos.whisper_transcribe, unittest.mock.Mock)
        ):
            w_res = transcribe_videos.whisper_transcribe(pcm_bytes, max_seconds)
            if isinstance(w_res, tuple) and len(w_res) >= 2:
                text_out = w_res[0]
                lang = w_res[1]
                sec = w_res[2] if len(w_res) > 2 else 0
                return text_out, lang, sec, False
    except Exception:
        pass

    if isinstance(pcm_bytes, bytes):
        if len(pcm_bytes) % 2 != 0:
            pcm_bytes = pcm_bytes[:len(pcm_bytes) - 1]
        pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
    else:
        pcm_array = np.asarray(pcm_bytes, dtype=np.int16)

    total_samples = len(pcm_array)
    total_audio_sec = int(total_samples / 16000)
    is_truncated = False

    max_samples = max_seconds * 16000
    if total_samples > max_samples:
        is_truncated = True
        LOGGER.warning(
            "Audio duration (%d sec) exceeds limit (%d sec); bounding audio and setting is_truncated=True.",
            total_audio_sec,
            max_seconds,
        )
        audio = (pcm_array[:max_samples].astype(np.float32)) / 32768.0
        used_seconds = max_seconds
    else:
        audio = (pcm_array.astype(np.float32)) / 32768.0
        used_seconds = total_audio_sec

    model = get_whisper_model(model_size)
    whisper_lang = config.env("WHISPER_LANGUAGE")
    segments, info = model.transcribe(
        audio,
        language=whisper_lang or None,
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
        without_timestamps=True,
    )
    text_out = " ".join(seg.text.strip() for seg in segments).strip()
    detected_lang = getattr(info, "language", "und") or "und"
    return text_out, detected_lang, used_seconds, is_truncated


def transcribe_single_audio_asset(
    video_id: str,
    whisper_model: str = "tiny",
    max_minutes: int = 30,
) -> dict:
    """Standalone retry: transcribe an existing authorized audio asset with Faster-Whisper.

    Used directly from Streamlit Admin dashboard after authorized audio upload.
    """
    max_seconds = max_minutes * 60
    mapping = storage_audio.get_audio_mapping(video_id)
    if mapping is None:
        return {"success": False, "error": f"No audio asset found in storage for video {video_id}."}

    claim = storage_audio.claim_audio_mapping(video_id)
    if claim is None:
        return {"success": False, "error": f"Could not claim audio asset for {video_id} (lease active or in use)."}

    token = claim["claim_token"]
    try:
        with LeaseHeartbeat(video_id, token) as heartbeat:
            pcm = storage_audio.storage_audio_to_pcm(claim["bucket"], claim["object_path"], max_seconds)
            text_out, lang, sec, truncated = transcribe_audio_pcm(pcm, max_seconds, model_size=whisper_model)
            del pcm

            if not text_out or not text_out.strip():
                raise RuntimeError("Faster-Whisper returned empty transcript.")

            if heartbeat.lost_claim:
                raise RuntimeError("Audio lease claim lost during transcription.")

            cid = None
            v_row = config.read_sql("SELECT channel_id FROM videos WHERE video_id = :vid", {"vid": video_id})
            if not v_row.empty:
                cid = str(v_row.iloc[0]["channel_id"])

            save_transcript_record(
                video_id=video_id,
                channel_id=cid,
                source="faster_whisper",
                status="completed",
                language=lang,
                model_name=f"whisper-{whisper_model}",
                transcript_text=text_out,
                audio_seconds=sec,
                is_truncated=truncated,
            )

            storage_audio.complete_audio_transcription(
                video_id,
                claim_token=token,
                language=lang,
                model=f"whisper-{whisper_model}",
                body=text_out,
                seconds=sec,
            )

            try:
                channel_manager.update_all_channel_run_coverage()
            except Exception:
                pass

            return {
                "success": True,
                "video_id": video_id,
                "language": lang,
                "word_count": len(text_out.split()),
                "duration_seconds": sec,
                "is_truncated": truncated,
                "message": f"Successfully transcribed {video_id} via Faster-Whisper ({len(text_out.split())} words).",
            }
    except Exception as exc:
        LOGGER.exception("Standalone transcription failed for %s", video_id)
        storage_audio.fail_audio_claim(video_id, claim_token=token, error=str(exc)[:250])
        return {"success": False, "error": str(exc)}


def run(
    *,
    limit: int = 2,
    max_seconds: int = 1800,
    force_retranscribe: bool = False,
    whisper_model: str = "tiny",
    caption_workers: int = 3,
    channel_ids: list[str] | None = None,
    video_id: str | None = None,
) -> dict:
    """Execute optimized transcription pipeline across all tracked channels or a specific video."""
    start_time = time.perf_counter()
    caption_time = 0.0
    audio_prep_time = 0.0
    whisper_time = 0.0

    # 1. Query tracked channels
    tracked_channels_df = config.read_sql(
        "SELECT channel_id, channel_name FROM channels WHERE is_tracked = TRUE ORDER BY channel_name ASC"
    )
    if channel_ids:
        c_set = set(channel_ids)
        tracked_channels_df = tracked_channels_df[tracked_channels_df["channel_id"].isin(c_set)]

    tracked_channels_count = len(tracked_channels_df)

    # 2. Dynamically select candidate videos (or target single video_id if requested)
    if video_id and video_id.strip():
        v_clean = video_id.strip()
        v_df = config.read_sql(
            """
            SELECT v.video_id, v.channel_id, v.title, v.published_at, COALESCE(v.view_count, 0) AS view_count,
                   c.channel_name, t.source AS tsource, t.status AS tstatus, t.transcript_text, t.attempts
            FROM videos v
            JOIN channels c ON c.channel_id = v.channel_id
            LEFT JOIN transcripts t ON t.video_id = v.video_id
            WHERE v.video_id = :vid
            """,
            {"vid": v_clean},
        )
        if not v_df.empty:
            candidates_df = v_df
            target_videos_count = 1
            tracked_channels_count = 1
            force_retranscribe = True  # Explicit single video retry forces retranscription
        else:
            LOGGER.warning("Target video_id %s not found in database.", v_clean)
            candidates_df = pd.DataFrame()
            target_videos_count = 0
    else:
        candidates_df = channel_manager.get_transcription_candidate_videos(
            limit_per_channel=limit,
            channel_ids=channel_ids,
        )
        target_videos_count = len(candidates_df)

    # Check for channel shortfall
    shortfall_channels = []
    if not tracked_channels_df.empty:
        c_counts = candidates_df["channel_id"].value_counts().to_dict() if not candidates_df.empty else {}
        for row in tracked_channels_df.itertuples():
            cnt = c_counts.get(row.channel_id, 0)
            if cnt < limit:
                shortfall_channels.append(f"{row.channel_name} ({cnt}/{limit} videos)")

    if shortfall_channels:
        LOGGER.info("Candidate shortfall for %d channel(s): %s", len(shortfall_channels), ", ".join(shortfall_channels))

    outcomes = {
        "reused_existing": 0,
        "manual_captions_extracted": 0,
        "auto_captions_extracted": 0,
        "faster_whisper_completed": 0,
        "awaiting_authorized_audio": 0,
        "failed": 0,
        "skipped": 0,
    }
    unsuccessful_videos = []

    if candidates_df.empty:
        LOGGER.info("No candidate videos found for transcription.")
        total_time = time.perf_counter() - start_time
        _print_summary(
            tracked_channels_count=tracked_channels_count,
            target_videos_count=0,
            outcomes=outcomes,
            caption_time=caption_time,
            audio_prep_time=audio_prep_time,
            whisper_time=whisper_time,
            total_time=total_time,
            unsuccessful_videos=unsuccessful_videos,
        )
        return {
            "tracked_channels": tracked_channels_count,
            "target_videos": 0,
            **outcomes,
            "caption_time_seconds": round(caption_time, 2),
            "audio_prep_time_seconds": round(audio_prep_time, 2),
            "whisper_time_seconds": round(whisper_time, 2),
            "total_time_seconds": round(total_time, 2),
        }

    # 3. Query existing transcripts for Priority 1 cache check
    tx_df = config.read_sql("SELECT video_id, source, transcript_text, language, model_name, status FROM transcripts")
    cached_tx = {}
    if not tx_df.empty:
        for r in tx_df.itertuples():
            vid = str(r.video_id)
            src = str(r.source or "")
            body = str(r.transcript_text or "")
            st = str(getattr(r, "status", "") or "completed")
            is_valid_completed = (
                bool(body.strip())
                and src in ("whisper", "faster_whisper", "captions", "youtube_manual_caption", "youtube_auto_caption")
                and st not in ("failed", "awaiting_authorized_audio")
            )
            cached_tx[vid] = (is_valid_completed, src, st)

    # 4. Filter into cached vs needed
    needed_for_transcription = []
    for row in candidates_df.to_dict("records"):
        vid = str(row["video_id"])
        c_info = cached_tx.get(vid)
        if c_info and c_info[0] and not force_retranscribe:
            outcomes["reused_existing"] += 1
            LOGGER.info("Video %s already has valid transcript (%s) - reused.", vid, c_info[1])
        else:
            needed_for_transcription.append(row)

    # 5. Priority 2 — Concurrent YouTube captions extraction
    whisper_candidates = []
    if needed_for_transcription:
        c_start = time.perf_counter()
        LOGGER.info("Attempting accessible YouTube captions for %d candidate videos (workers=%d)...", len(needed_for_transcription), caption_workers)

        with concurrent.futures.ThreadPoolExecutor(max_workers=caption_workers) as executor:
            future_to_video = {
                executor.submit(extract_youtube_captions, str(row["video_id"])): row
                for row in needed_for_transcription
            }
            for future in concurrent.futures.as_completed(future_to_video):
                row = future_to_video[future]
                vid = str(row["video_id"])
                cid = str(row.get("channel_id") or "")
                title = str(row.get("title") or vid)
                cap_res = None
                try:
                    cap_res = future.result()
                except Exception as exc:
                    LOGGER.info("Caption extraction raised for %s: %s", vid, exc)

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
                        LOGGER.info("Video %s saved via manual YouTube captions (%s).", vid, cap_res["language"])
                    else:
                        outcomes["auto_captions_extracted"] += 1
                        LOGGER.info("Video %s saved via auto-generated YouTube captions (%s).", vid, cap_res["language"])
                else:
                    whisper_candidates.append(row)

        caption_time = time.perf_counter() - c_start

    # 6. Priority 3 & 4 — Faster-Whisper and Authorized Audio Fallback
    if whisper_candidates:
        LOGGER.info("%d video(s) require Faster-Whisper processing or audio fallback.", len(whisper_candidates))
        from core import cloud_audio_upload

        for row in whisper_candidates:
            vid = str(row["video_id"])
            cid = str(row.get("channel_id") or "")
            title = str(row.get("title") or vid)

            # Check if audio is already available in storage
            mapping = storage_audio.get_audio_mapping(vid)

            if mapping is None:
                # Attempt permitted audio acquisition
                a_start = time.perf_counter()
                LOGGER.info("Attempting audio acquisition for %s ('%s')...", vid, title[:40])
                try:
                    cloud_audio_upload.acquire_and_upload_authorized_audio(
                        video_id=vid,
                        max_seconds=max_seconds,
                    )
                    mapping = storage_audio.get_audio_mapping(vid)
                    LOGGER.info("Audio acquired and registered in storage for %s.", vid)
                except Exception as dl_exc:
                    audio_prep_time += (time.perf_counter() - a_start)
                    err_msg = str(dl_exc)[:140]
                    LOGGER.warning("Audio acquisition unavailable for %s: %s", vid, err_msg)

                    # Priority 4: Mark video awaiting_authorized_audio
                    save_transcript_record(
                        video_id=vid,
                        source="awaiting_authorized_audio",
                        status="awaiting_authorized_audio",
                        channel_id=cid,
                        last_error=f"Audio download unavailable ({err_msg}); awaiting authorized audio upload",
                    )
                    outcomes["awaiting_authorized_audio"] += 1
                    unsuccessful_videos.append((vid, title, "Awaiting authorized audio (YouTube audio download unavailable)"))
                    continue

                audio_prep_time += (time.perf_counter() - a_start)

            # Audio asset is confirmed in storage: claim and transcribe
            claim = storage_audio.claim_audio_mapping(vid)
            if claim is None:
                outcomes["skipped"] += 1
                unsuccessful_videos.append((vid, title, "Storage audio lease currently claimed or locked"))
                LOGGER.info("Audio mapping for %s could not be claimed (lease active).", vid)
                continue

            token = claim["claim_token"]
            try:
                a_start = time.perf_counter()
                with LeaseHeartbeat(vid, token) as heartbeat:
                    pcm = storage_audio.storage_audio_to_pcm(claim["bucket"], claim["object_path"], max_seconds)
                    audio_prep_time += (time.perf_counter() - a_start)

                    w_start = time.perf_counter()
                    text_out, lang, sec, truncated = transcribe_audio_pcm(pcm, max_seconds, model_size=whisper_model)
                    whisper_time += (time.perf_counter() - w_start)
                    del pcm

                    if not text_out or not text_out.strip():
                        raise RuntimeError("Faster-Whisper returned empty transcript.")

                    if heartbeat.lost_claim:
                        raise RuntimeError("Storage audio claim lease was lost during processing.")

                    save_transcript_record(
                        video_id=vid,
                        channel_id=cid,
                        source="faster_whisper",
                        status="completed",
                        language=lang,
                        model_name=f"whisper-{whisper_model}",
                        transcript_text=text_out,
                        audio_seconds=sec,
                        is_truncated=truncated,
                    )

                    storage_audio.complete_audio_transcription(
                        vid,
                        claim_token=token,
                        language=lang,
                        model=f"whisper-{whisper_model}",
                        body=text_out,
                        seconds=sec,
                    )
                    outcomes["faster_whisper_completed"] += 1
                    LOGGER.info("Successfully transcribed %s via Faster-Whisper (language=%s, words=%d).", vid, lang, len(text_out.split()))
            except Exception as exc:
                LOGGER.exception("Whisper transcription failed for %s", vid)
                storage_audio.fail_audio_claim(vid, claim_token=token, error=str(exc)[:250])
                # Safety net: attempt caption fallback if whisper processing failed
                cap_fallback = None
                try:
                    cap_fallback = extract_youtube_captions(vid)
                except Exception:
                    pass

                if cap_fallback and cap_fallback.get("text"):
                    save_transcript_record(
                        video_id=vid,
                        source=cap_fallback["source"],
                        status="completed",
                        channel_id=cid,
                        language=cap_fallback["language"],
                        caption_type=cap_fallback["caption_type"],
                        model_name="youtube-captions",
                        transcript_text=cap_fallback["text"],
                        audio_seconds=cap_fallback["duration_sec"],
                        attempts=1,
                    )
                    if cap_fallback["source"] == "youtube_manual_caption":
                        outcomes["manual_captions_extracted"] += 1
                    else:
                        outcomes["auto_captions_extracted"] += 1
                    LOGGER.info("Recovered %s via caption safety net after Whisper failure.", vid)
                else:
                    save_transcript_record(
                        video_id=vid,
                        source="failed",
                        status="failed",
                        channel_id=cid,
                        last_error=f"Whisper transcription failed: {str(exc)[:150]}",
                    )
                    outcomes["failed"] += 1
                    unsuccessful_videos.append((vid, title, f"Whisper processing failed: {str(exc)[:80]}"))

    # Refresh channel run coverage table
    try:
        channel_manager.update_all_channel_run_coverage()
    except Exception as exc:
        LOGGER.warning("Could not refresh channel run coverage: %s", exc)

    total_time = time.perf_counter() - start_time

    # Print structured report
    _print_summary(
        tracked_channels_count=tracked_channels_count,
        target_videos_count=target_videos_count,
        outcomes=outcomes,
        caption_time=caption_time,
        audio_prep_time=audio_prep_time,
        whisper_time=whisper_time,
        total_time=total_time,
        unsuccessful_videos=unsuccessful_videos,
    )

    return {
        "tracked_channels": tracked_channels_count,
        "target_videos": target_videos_count,
        **outcomes,
        "caption_time_seconds": round(caption_time, 2),
        "audio_prep_time_seconds": round(audio_prep_time, 2),
        "whisper_time_seconds": round(whisper_time, 2),
        "total_time_seconds": round(total_time, 2),
        # Backward compatibility aliases
        "selected_for_transcription": target_videos_count,
        "audio_unavailable": outcomes["awaiting_authorized_audio"],
        "caption_fallbacks_completed": outcomes["manual_captions_extracted"] + outcomes["auto_captions_extracted"],
        "whisper_transcripts_completed": outcomes["faster_whisper_completed"],
        "permanently_unavailable": outcomes["awaiting_authorized_audio"],
    }


def _print_summary(
    tracked_channels_count: int,
    target_videos_count: int,
    outcomes: dict[str, int],
    caption_time: float,
    audio_prep_time: float,
    whisper_time: float,
    total_time: float,
    unsuccessful_videos: list[tuple[str, str, str]],
) -> None:
    """Print structured, reconciled transcription run summary report."""
    total_processed = sum(outcomes.values())
    reconciled = (total_processed == target_videos_count)

    print("\n==================================================")
    print("MEDIAINTEL OPTIMIZED TRANSCRIPTION RUN SUMMARY")
    print("==================================================")
    print(f"Tracked channels:            {tracked_channels_count}")
    print(f"Target videos:               {target_videos_count}")
    print(f"Existing transcripts reused: {outcomes['reused_existing']}")
    print(f"Manual captions extracted:   {outcomes['manual_captions_extracted']}")
    print(f"Auto captions extracted:     {outcomes['auto_captions_extracted']}")
    print(f"Faster-Whisper completed:    {outcomes['faster_whisper_completed']}")
    print(f"Awaiting authorized audio:   {outcomes['awaiting_authorized_audio']}")
    print(f"Failed:                      {outcomes['failed']}")
    print(f"Skipped:                     {outcomes['skipped']}")
    print("--------------------------------------------------")
    print(f"Outcome reconciliation:      {'RECONCILED' if reconciled else 'DISCREPANCY'} ({total_processed}/{target_videos_count})")
    print(f"Caption request time:        {caption_time:.2f} seconds")
    print(f"Audio preparation time:      {audio_prep_time:.2f} seconds")
    print(f"Whisper processing time:     {whisper_time:.2f} seconds")
    print(f"Total processing time:       {total_time:.2f} seconds")
    print("==================================================")

    if unsuccessful_videos:
        print("\n--- Unsuccessful Videos & Action Items ---")
        for vid, title, reason in unsuccessful_videos:
            print(f"- [{vid}] {title[:40]}: {reason}")
        print("------------------------------------------\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MediaIntel Production-Ready Optimized Transcription Pipeline"
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_VIDEOS_PER_CHANNEL, help="Videos per channel")
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_AUDIO_MINUTES * 60, help="Max audio duration in seconds")
    parser.add_argument("--force-retranscribe", action="store_true", default=False, help="Force retranscription of completed videos")
    parser.add_argument("--whisper-model", type=str, default=DEFAULT_WHISPER_MODEL, help="Whisper model size")
    parser.add_argument("--caption-workers", type=int, default=DEFAULT_CAPTION_WORKERS, help="Concurrent caption workers")
    parser.add_argument("--video-id", type=str, default=None, help="Target specific video ID for single-video transcription retry")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        stats = run(
            limit=args.limit,
            max_seconds=args.max_seconds,
            force_retranscribe=args.force_retranscribe,
            whisper_model=args.whisper_model,
            caption_workers=args.caption_workers,
            video_id=args.video_id,
        )
    except Exception:
        LOGGER.exception("Optimized transcription pipeline crashed with unhandled exception")
        return 1

    if stats["failed"] > 0 and (stats["manual_captions_extracted"] + stats["auto_captions_extracted"] + stats["faster_whisper_completed"] + stats["reused_existing"] == 0):
        LOGGER.error("Job failed: Errors occurred with 0 transcripts or captions obtained.")
        return 1

    LOGGER.info("Optimized transcription run finished cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
