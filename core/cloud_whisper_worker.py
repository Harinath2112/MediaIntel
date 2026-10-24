"""Storage-only Cloud Whisper worker for MediaIntel.

Processes audio explicitly mapped to videos in private Supabase Storage.
Does not download audio from YouTube.
"""

import argparse
import logging
import sys
import threading

from core import storage_audio, transcribe_videos

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


def run(*, limit: int = 1, max_seconds: int = 600) -> dict:
    """Process eligible mapped audio using the existing Whisper pipeline."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if max_seconds < 1:
        raise ValueError("max_seconds must be at least 1")

    stats = {
        "candidates": 0,
        "claimed": 0,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
    }

    candidates = storage_audio.mapped_audio_candidates()
    stats["candidates"] = len(candidates)

    for row in candidates.head(limit).to_dict("records"):
        video_id = str(row["video_id"])
        claim = storage_audio.claim_audio_mapping(video_id)

        if claim is None:
            stats["skipped"] += 1
            continue

        stats["claimed"] += 1
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
    
                # A valid claim is required for the database completion operation.
                # This also refuses to overwrite an existing successful transcript.
                if heartbeat.lost_claim:
                    raise RuntimeError('Audio claim was lost during processing')

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
                    LOGGER.warning(
                        "Completion rejected (claim expired or changed): %s",
                        video_id,
                    )
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
                LOGGER.warning(
                    "Failure update rejected (claim expired or changed): %s",
                    video_id,
                )

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MediaIntel Storage-only Cloud Whisper worker"
    )
    parser.add_argument("--limit", type=int, default=1)
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
