"""Storage-only Cloud Whisper worker for MediaIntel.

Processes audio explicitly mapped to videos in private Supabase Storage.
Does not download audio from YouTube.
"""

import argparse
import logging
import sys
import threading

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
    """Process top candidate videos per channel (ranked by view count) using authorized Storage audio."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if max_seconds < 1:
        raise ValueError("max_seconds must be at least 1")

    stats = {
        "selected_for_transcription": 0,
        "completed": 0,
        "audio_unavailable": 0,
        "skipped": 0,
        "failed": 0,
    }

    # Deterministically select top videos per channel ranked by view_count DESC, published_at DESC, video_id ASC
    candidates_df = channel_manager.get_transcription_candidate_videos(limit_per_channel=limit)
    stats["selected_for_transcription"] = len(candidates_df)

    if candidates_df.empty:
        LOGGER.info("No candidate videos found for transcription.")
        return stats

    # Query existing transcripts
    existing_transcripts = config.read_sql(
        "SELECT video_id, source, transcript_text FROM transcripts"
    )
    tx_by_video = {}
    if not existing_transcripts.empty:
        for r in existing_transcripts.itertuples():
            tx_by_video[str(r.video_id)] = (str(r.source or ""), str(r.transcript_text or ""))

    for row in candidates_df.to_dict("records"):
        video_id = str(row["video_id"])
        title = str(row.get("title", video_id))

        # Check for existing completed transcript
        if video_id in tx_by_video:
            src, text_val = tx_by_video[video_id]
            if src in ("whisper", "captions") and text_val.strip():
                stats["skipped"] += 1
                LOGGER.info("Video %s already transcribed (%s) - skipped.", video_id, src)
                continue

        # Check for authorized audio asset in Supabase storage
        mapping = storage_audio.get_audio_mapping(video_id)
        if mapping is None:
            # Authorized storage audio unavailable - do not bypass restrictions, record audio_unavailable
            transcribe_videos.save(
                video_id,
                source="audio_unavailable",
                model=f"whisper-{transcribe_videos.WHISPER_MODEL}",
                body=None,
                seconds=0,
                attempts=1,
                err="Audio unavailable: YouTube server-side streaming restricted; requires authorized audio asset in Supabase Storage",
            )
            stats["audio_unavailable"] += 1
            LOGGER.info("Video %s ('%s') marked audio_unavailable (no storage asset).", video_id, title[:50])
            continue

        # Authorized audio asset is available: proceed with claim and transcription
        claim = storage_audio.claim_audio_mapping(video_id)
        if claim is None:
            stats["skipped"] += 1
            continue

        token = claim["claim_token"]
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
                    stats["completed"] += 1
                    LOGGER.info("Completed transcription for %s", video_id)
                else:
                    stats["skipped"] += 1
                    LOGGER.warning("Completion rejected for %s", video_id)
        except Exception as exc:
            LOGGER.exception("Transcription failed for %s", video_id)
            failed = storage_audio.fail_audio_claim(
                video_id,
                claim_token=token,
                error=str(exc)[:250],
            )
            if failed:
                stats["failed"] += 1
            else:
                stats["skipped"] += 1

    # Refresh channel run coverage table
    try:
        channel_manager.update_all_channel_run_coverage()
    except Exception as exc:
        LOGGER.warning("Could not refresh channel run coverage: %s", exc)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MediaIntel Storage-only Cloud Whisper worker"
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
        LOGGER.exception("Cloud Whisper worker failed")
        return 1

    print(f"Cloud Whisper worker results: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
