import argparse
import logging
import os
import sys
import threading
import time

from core import config, channel_manager, storage_audio, transcribe_videos

LOGGER = logging.getLogger(__name__)


class LeaseHeartbeat:
    """Renew an owned database claim while Whisper is processing."""

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
                    LOGGER.warning(
                        "Lost audio claim for %s",
                        self.video_id,
                    )
                    return
            except Exception:
                self.lost_claim = True
                LOGGER.exception(
                    "Unable to renew audio claim for %s",
                    self.video_id,
                )
                return

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=15)


def run(*, limit: int = 2, max_seconds: int = 600) -> dict:
    """Process top candidate videos per channel with audio acquisition, Whisper, and caption fallback."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if max_seconds < 1:
        raise ValueError("max_seconds must be at least 1")

    stats = {
        "selected_for_transcription": 0,
        "audio_downloads_attempted": 0,
        "audio_downloads_successful": 0,
        "whisper_transcripts_completed": 0,
        "caption_fallbacks_completed": 0,
        "permanently_unavailable": 0,
        "retryable_failures": 0,
        "model_storage_failures": 0,
        "skipped": 0,
        # Backward-compatibility keys
        "completed": 0,
        "audio_unavailable": 0,
        "failed": 0,
    }

    # Deterministically select top videos per channel ranked by view_count DESC, published_at DESC, video_id ASC
    candidates_df = channel_manager.get_transcription_candidate_videos(limit_per_channel=limit)
    stats["selected_for_transcription"] = len(candidates_df)

    if candidates_df.empty:
        LOGGER.info("No candidate videos found for transcription.")
        return stats

    # Query existing transcripts to prevent duplicate work
    existing_transcripts = config.read_sql(
        "SELECT video_id, source, transcript_text FROM transcripts"
    )
    tx_by_video = {}
    if not existing_transcripts.empty:
        for r in existing_transcripts.itertuples():
            tx_by_video[str(r.video_id)] = (str(r.source or ""), str(r.transcript_text or ""))

    from core import cloud_audio_upload

    for row in candidates_df.to_dict("records"):
        video_id = str(row["video_id"])
        title = str(row.get("title", video_id))

        # 1. Check for existing completed transcript
        if video_id in tx_by_video:
            src, text_val = tx_by_video[video_id]
            if src in ("whisper", "captions") and text_val.strip():
                stats["skipped"] += 1
                LOGGER.info("Video %s already transcribed (%s) - skipped.", video_id, src)
                continue

        # 2. Check for existing audio mapping in Supabase storage
        mapping = storage_audio.get_audio_mapping(video_id)

        # 3. If no existing storage mapping, attempt lawful audio acquisition via yt-dlp
        if mapping is None:
            stats["audio_downloads_attempted"] += 1
            LOGGER.info("Attempting permitted audio acquisition for %s ('%s')...", video_id, title[:45])
            try:
                cloud_audio_upload.acquire_and_upload_authorized_audio(
                    video_id=video_id,
                    max_seconds=max_seconds,
                )
                stats["audio_downloads_successful"] += 1
                mapping = storage_audio.get_audio_mapping(video_id)
                LOGGER.info("Audio acquired and stored in private Supabase Storage for %s.", video_id)
            except Exception as dl_exc:
                dl_msg = str(dl_exc)
                LOGGER.warning("Audio acquisition unavailable for %s: %s", video_id, dl_msg[:120])

                # Check if this is a systemic storage/configuration failure
                is_systemic_config = (
                    "configuration is missing or invalid" in dl_msg.lower()
                    or "unapproved supabase storage" in dl_msg.lower()
                    or "storage api returned http 40" in dl_msg.lower()
                )
                if is_systemic_config:
                    stats["model_storage_failures"] += 1

                # 4. Fallback to lawful YouTube captions if audio is restricted or unavailable
                caption_acquired = False
                LOGGER.info("Attempting lawful caption fallback for %s...", video_id)
                try:
                    c_text, c_lang = transcribe_videos.fetch_captions(video_id)
                    if c_text and c_text.strip():
                        transcribe_videos.save(
                            video_id,
                            source="captions",
                            language=c_lang,
                            model="youtube-captions",
                            body=c_text.strip(),
                            seconds=None,
                            attempts=1,
                            err=None,
                        )
                        stats["caption_fallbacks_completed"] += 1
                        caption_acquired = True
                        LOGGER.info("Video %s saved with YouTube captions fallback (source=captions).", video_id)
                except Exception as cap_exc:
                    LOGGER.info("Caption fallback unavailable for %s: %s", video_id, str(cap_exc)[:100])

                if not caption_acquired:
                    # Neither audio nor captions could be obtained
                    is_timeout = "timeout" in dl_msg.lower() or "timed out" in dl_msg.lower()
                    reason = f"Audio unavailable: {dl_msg[:100]}; Captions unavailable"
                    transcribe_videos.save(
                        video_id,
                        source="audio_unavailable",
                        model=f"whisper-{transcribe_videos.WHISPER_MODEL}",
                        body=None,
                        seconds=0,
                        attempts=1,
                        err=reason,
                    )
                    if is_timeout:
                        stats["retryable_failures"] += 1
                    else:
                        stats["permanently_unavailable"] += 1
                    stats["audio_unavailable"] += 1
                    LOGGER.info("Video %s marked audio_unavailable.", video_id)
                continue

        # 5. Audio asset is available in Storage: claim and transcribe with Faster-Whisper
        claim = storage_audio.claim_audio_mapping(video_id)
        if claim is None:
            stats["skipped"] += 1
            LOGGER.info("Audio mapping for %s could not be claimed (in use or lease active).", video_id)
            continue

        token = claim["claim_token"]
        whisper_success = False
        try:
            with LeaseHeartbeat(video_id, token) as heartbeat:
                pcm = storage_audio.storage_audio_to_pcm(
                    claim["bucket"],
                    claim["object_path"],
                    max_seconds,
                )

                body, language, seconds = transcribe_videos.whisper_transcribe(
                    pcm,
                    max_seconds,
                )

                del pcm

                if not body or not body.strip():
                    raise RuntimeError("Whisper returned an empty transcript")

                if heartbeat.lost_claim:
                    raise RuntimeError("Audio claim was lost during processing")

                completed = storage_audio.complete_audio_transcription(
                    video_id,
                    claim_token=token,
                    language=language or "unknown",
                    model=f"whisper-{transcribe_videos.WHISPER_MODEL}",
                    body=body,
                    seconds=seconds,
                )

                if completed:
                    stats["whisper_transcripts_completed"] += 1
                    stats["completed"] += 1
                    whisper_success = True
                    LOGGER.info("Successfully transcribed %s via Faster-Whisper (source=whisper).", video_id)
                else:
                    stats["skipped"] += 1
                    LOGGER.warning("Completion rejected for %s", video_id)
        except Exception as exc:
            LOGGER.exception("Whisper transcription failed for %s", video_id)
            stats["model_storage_failures"] += 1
            stats["failed"] += 1
            storage_audio.fail_audio_claim(
                video_id,
                claim_token=token,
                error=str(exc)[:250],
            )
            # Safety net: try caption fallback so video isn't left without text
            try:
                c_text, c_lang = transcribe_videos.fetch_captions(video_id)
                if c_text and c_text.strip():
                    transcribe_videos.save(
                        video_id,
                        source="captions",
                        language=c_lang,
                        model="youtube-captions",
                        body=c_text.strip(),
                        seconds=None,
                        attempts=1,
                        err=f"Whisper failed ({str(exc)[:80]}); fallback to captions",
                    )
                    stats["caption_fallbacks_completed"] += 1
                    LOGGER.info("Video %s saved with caption fallback after Whisper failure.", video_id)
            except Exception:
                pass

    # Refresh channel run coverage table
    try:
        channel_manager.update_all_channel_run_coverage()
    except Exception as exc:
        LOGGER.warning("Could not refresh channel run coverage: %s", exc)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MediaIntel Cloud Whisper Worker with Audio Acquisition and Caption Fallback"
    )
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-seconds", type=int, default=600)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        stats = run(
            limit=args.limit,
            max_seconds=args.max_seconds,
        )
    except Exception:
        LOGGER.exception("Cloud Whisper worker crashed with unhandled exception")
        return 1

    print(f"\n--- Cloud Whisper Worker Summary ---")
    print(f"Candidates selected        : {stats['selected_for_transcription']}")
    print(f"Audio downloads attempted  : {stats['audio_downloads_attempted']}")
    print(f"Audio downloads successful : {stats['audio_downloads_successful']}")
    print(f"Whisper transcripts        : {stats['whisper_transcripts_completed']}")
    print(f"Caption fallbacks          : {stats['caption_fallbacks_completed']}")
    print(f"Permanently unavailable    : {stats['permanently_unavailable']}")
    print(f"Retryable failures         : {stats['retryable_failures']}")
    print(f"Model/storage failures     : {stats['model_storage_failures']}")
    print(f"Skipped (already done)     : {stats['skipped']}")
    print(f"Raw stats dict: {stats}\n")

    # Outcome policy:
    # 1. Fail if systemic configuration or model/storage failures occurred
    if stats["model_storage_failures"] > 0 and (stats["whisper_transcripts_completed"] + stats["caption_fallbacks_completed"] == 0):
        LOGGER.error("Job failed: Systemic storage/configuration/model failures occurred with 0 transcripts or fallbacks.")
        return 1

    # 2. If all selected videos were attempted and 0 transcripts or captions were obtained,
    # and at least one failure was retryable or failed, exit with diagnostic error
    total_text = stats["whisper_transcripts_completed"] + stats["caption_fallbacks_completed"]
    if stats["selected_for_transcription"] > 0 and stats["skipped"] == 0 and total_text == 0:
        if stats["retryable_failures"] > 0:
            LOGGER.warning("All candidate videos failed due to transient/retryable network errors.")
            return 1

    LOGGER.info("Cloud Whisper worker completed successfully with verified outcome.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

