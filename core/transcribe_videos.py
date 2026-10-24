"""Stage 2 - Speech-to-text with OpenAI Whisper.

Uses the original OpenAI Whisper model weights through `faster-whisper` (CTranslate2): identical
transcripts, roughly 4x faster on CPU. It stays fast no matter how many channels are tracked:

  * fair scheduling : every run takes the next WHISPER_VIDEOS_PER_CHANNEL videos of EACH channel (round-robin),
                      so no single channel can use up the whole time budget; later runs continue the backlog
  * time budget     : never longer than WHISPER_BUDGET_MIN minutes per run
  * audio cap       : only the first WHISPER_MAX_AUDIO_MIN minutes of each video are transcribed
  * NO AUDIO FILES  : the first minutes of each video are streamed straight into memory (ffmpeg -> Whisper);
                      nothing is ever saved to disk, only the text is kept
  * parallel I/O    : the audio of the next videos is streamed in background threads while Whisper is busy
  * int8 on CPU     : quantised weights, beam_size=1, voice-activity filter skips silence
  * cached          : a video is transcribed once; reruns only handle new videos
  * safety net      : videos Whisper did not reach get their YouTube captions (seconds, not minutes) so every
                      analytic has text; a later run upgrades them to Whisper

All limits come from SPEED_PROFILE (fast | balanced | accurate) and can be overridden in .env.

Usage:  python core/transcribe_videos.py [--limit N] [--no-captions] [--budget MIN]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import config  # noqa: E402

WHISPER_MODEL = config.setting("WHISPER_MODEL")                    # tiny | base | small | medium ...
WHISPER_COMPUTE = config.env("WHISPER_COMPUTE", "int8")            # int8 (CPU) | float16 (GPU)
WHISPER_LANGUAGE = config.env("WHISPER_LANGUAGE")                  # empty = auto-detect per video
MAX_AUDIO_SEC = config.setting_int("WHISPER_MAX_AUDIO_MIN") * 60
BUDGET_MIN = config.setting_float("WHISPER_BUDGET_MIN")
PER_CHANNEL = max(config.setting_int("WHISPER_VIDEOS_PER_CHANNEL"), 1)
DOWNLOAD_WORKERS = max(config.env_int("DOWNLOAD_WORKERS", 3), 1)
CAPTION_CAP = config.env_int("CAPTION_FALLBACK_MAX", 60)           # captions fetched per run at most
MAX_ATTEMPTS = 3
TEMP_FALLBACK = config.env_bool("WHISPER_ALLOW_TEMP_DOWNLOAD", True)   # last resort if streaming is refused

_model = None


# ---------------------------------------------------------------------------
# audio: streamed into memory - never stored
# ---------------------------------------------------------------------------
def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()      # bundled ffmpeg: nothing to install by hand
    except Exception:  # noqa: BLE001
        return "ffmpeg"                              # fall back to one on PATH


def resolve_stream(video_id: str):
    """Asks yt-dlp for the direct audio stream URL of a video (no download). Returns (url, http_headers)."""
    import yt_dlp
    opts = {"format": "bestaudio[abr<=70]/worstaudio/worst", "quiet": True, "no_warnings": True,
            "noplaylist": True, "skip_download": True, "socket_timeout": 30}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    if info.get("url"):
        return info["url"], info.get("http_headers") or {}
    fmt = (info.get("requested_formats") or info.get("formats") or [])[0]
    return fmt["url"], fmt.get("http_headers") or info.get("http_headers") or {}


def read_pcm(source: str, headers: dict | None, max_seconds: int):
    """Streams the first `max_seconds` of audio from `source` (URL or file) through ffmpeg into memory.

    Returns 16 kHz mono int16 samples. The only thing that ever exists is this in-memory array.
    """
    import numpy as np
    cmd = [_ffmpeg(), "-nostdin", "-loglevel", "error"]
    if headers:
        cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
    cmd += ["-t", str(max_seconds), "-i", source, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, timeout=max_seconds * 3 + 90)
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    if proc.returncode != 0 or pcm.size < 16000:     # need at least one second of audio
        raise RuntimeError("audio stream unavailable: " + proc.stderr.decode("utf-8", "replace").strip()[-160:])
    return pcm


def _temp_download_pcm(video_id: str, max_seconds: int):
    """Last resort when YouTube refuses direct streaming: download into a self-deleting temp folder,
    decode into memory and delete immediately (nothing is kept)."""
    import yt_dlp
    from faster_whisper import decode_audio
    with tempfile.TemporaryDirectory(prefix="mediaintel_") as tmp:
        opts = {"format": "bestaudio[abr<=70]/worstaudio/worst", "outtmpl": os.path.join(tmp, "a.%(ext)s"),
                "quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True, "socket_timeout": 30,
                "ffmpeg_location": _ffmpeg()}
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        files = [f for f in Path(tmp).glob("a.*") if f.suffix not in (".part", ".ytdl")]
        if not files:
            raise RuntimeError("audio download failed")
        pcm = decode_audio(str(files[0]), sampling_rate=16000)[: max_seconds * 16000]
        import numpy as np
        return (pcm * 32767).astype(np.int16)      # the temp folder is wiped when this block ends


def fetch_audio(video_id: str, max_seconds: int):
    """First `max_seconds` of a video's audio as an in-memory int16 array (streamed, never stored)."""
    try:
        url, headers = resolve_stream(video_id)
        return read_pcm(url, headers, max_seconds)
    except Exception as exc:  # noqa: BLE001
        if not TEMP_FALLBACK:
            raise RuntimeError(f"audio stream failed: {str(exc)[:150]}") from exc
        return _temp_download_pcm(video_id, max_seconds)


# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------
def load_whisper():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        device = "auto"
        compute = WHISPER_COMPUTE
        print(f"Loading Whisper '{WHISPER_MODEL}' ({compute}) - first run downloads the model once...")
        _model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute,
                              cpu_threads=os.cpu_count() or 4)
    return _model


def whisper_transcribe(pcm, max_seconds: int):
    """pcm = 16 kHz mono int16 samples in memory. Returns (text, language, seconds_transcribed)."""
    import numpy as np
    audio = (pcm[: max_seconds * 16000].astype(np.float32)) / 32768.0
    model = load_whisper()
    segments, info = model.transcribe(
        audio,
        language=WHISPER_LANGUAGE or None,
        beam_size=1,                       # greedy decoding: much faster, negligible quality loss
        vad_filter=True,                   # skip silence / music
        condition_on_previous_text=False,  # avoids repetition loops and is faster
        without_timestamps=True,
    )
    text_out = " ".join(seg.text.strip() for seg in segments).strip()
    return text_out, info.language, int(len(audio) / 16000)


# ---------------------------------------------------------------------------
# captions safety net
# ---------------------------------------------------------------------------
def fetch_captions(video_id: str):
    """Returns (text, language) from YouTube's own captions; raises if none exist."""
    from youtube_transcript_api import YouTubeTranscriptApi
    api = YouTubeTranscriptApi()
    if hasattr(api, "list"):
        tl = api.list(video_id)
        try:
            tr = tl.find_transcript(["en", "hi", "en-IN"])
        except Exception:
            tr = next(iter(tl))
        parts = [s.text if hasattr(s, "text") else s["text"] for s in tr.fetch()]
        return " ".join(parts).strip(), tr.language_code
    rows = YouTubeTranscriptApi.get_transcript(video_id)  # old (<1.0) library
    return " ".join(r["text"] for r in rows).strip(), "und"


# ---------------------------------------------------------------------------
# database helpers
# ---------------------------------------------------------------------------
TRANSCRIPT_COLS = ["video_id", "source", "language", "model_name", "transcript_text", "audio_seconds",
                   "word_count", "attempts", "last_error", "updated_at"]


def _src(value):
    """pandas turns a missing LEFT JOIN value into NaN/None - normalise to None."""
    return None if value is None or (isinstance(value, float) and value != value) else value


def save(video_id, source, *, language=None, model=None, body=None, seconds=None, attempts=0, err=None):
    with config.get_engine().begin() as conn:
        conn.execute(config.upsert_sql("transcripts", TRANSCRIPT_COLS, ["video_id"]), {
            "video_id": video_id, "source": source, "language": (language or "")[:12] or None,
            "model_name": model, "transcript_text": body, "audio_seconds": seconds,
            "word_count": len(body.split()) if body else 0, "attempts": attempts,
            "last_error": (err or "")[:250] or None, "updated_at": config.utcnow()})


def candidate_videos(limit: int | None = None):
    """Rank the 10 selected active videos per channel by view_count DESC, published_at DESC, video_id ASC,
    and select the top limit (default 2) videos for transcription."""
    df = config.read_sql("""
        SELECT v.video_id, v.title, v.channel_id, v.published_at, COALESCE(v.view_count, 0) AS view_count,
               t.source AS tsource, COALESCE(t.attempts, 0) AS attempts
        FROM videos v
        JOIN channels c ON c.channel_id = v.channel_id
        LEFT JOIN transcripts t ON t.video_id = v.video_id
        WHERE c.is_tracked = TRUE
          AND (t.video_id IS NULL OR (t.source IN ('captions', 'failed', 'audio_unavailable') AND t.attempts < :maxa))""",
                         {"maxa": MAX_ATTEMPTS})
    if df.empty:
        return df
    # Partition by channel and take top 10 most recent videos first (active analysis set)
    df_sorted = df.sort_values(["channel_id", "published_at", "video_id"], ascending=[True, False, True])
    df_active = df_sorted.groupby("channel_id").head(10).copy()

    # Rank the active videos by view count descending with published_at and video_id as deterministic tie-breakers
    df_ranked = df_active.sort_values(["channel_id", "view_count", "published_at", "video_id"],
                                      ascending=[True, False, False, True]).copy()
    df_ranked["rank"] = df_ranked.groupby("channel_id").cumcount()
    return df_ranked if not limit else df_ranked[df_ranked["rank"] < limit]


if config.env_bool("MEDIAINTEL_TEST_FAKES"):   # offline self-test hook
    sys.path.insert(0, str(config.ROOT / "tests"))
    import fakes
    fetch_audio, whisper_transcribe, fetch_captions = fakes.fake_fetch_audio, fakes.fake_whisper, fakes.fake_captions


# ---------------------------------------------------------------------------
# main routine
# ---------------------------------------------------------------------------
def run(limit: int | None = None, use_captions: bool = True, budget_min: float | None = None,
        video_id: str | None = None) -> dict:
    budget = (BUDGET_MIN if budget_min is None else budget_min) * 60
    everything = candidate_videos()
    if video_id is not None:
        everything = everything[everything["video_id"] == video_id].copy()
    stats = {"whisper": 0, "captions": 0, "failed": 0, "queued": 0, "total_todo": len(everything)}
    if everything.empty:
        print("All videos already transcribed - nothing to do.")
        return stats

    # fair schedule: the next PER_CHANNEL videos of every channel, round-robin (rank 0 of all channels first)
    per_channel = limit or PER_CHANNEL
    todo = everything[everything["rank"] < per_channel].sort_values(["rank", "channel_id"])
    print(f"{len(todo)} video(s) need transcription "
          f"(budget {max(round(budget / 60), 1)} min, first {MAX_AUDIO_SEC // 60} min of audio each, "
          f"{per_channel} per channel).")
    started = time.time()
    rows = list(todo.itertuples(index=False))
    handled: set[str] = set()
    failed_saved: set[str] = set()
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        window: deque = deque()          # (row, future) in newest-first order
        nxt = 0
        prefetch = DOWNLOAD_WORKERS * 2

        def refill():
            nonlocal nxt
            while nxt < len(rows) and len(window) < prefetch:
                r = rows[nxt]
                window.append((r, pool.submit(fetch_audio, r.video_id, MAX_AUDIO_SEC)))
                nxt += 1

        refill()
        while window:
            if time.time() - started > budget:
                break                    # out of time: the rest is handled by the safety net / next run
            row, fut = window.popleft()
            refill()
            attempts = int(row.attempts) + 1
            try:
                pcm = fut.result()
                body, lang, secs = whisper_transcribe(pcm, MAX_AUDIO_SEC)
                del pcm                      # the audio only ever lived in memory
                save(row.video_id, "whisper", language=lang, model=f"whisper-{WHISPER_MODEL}",
                     body=body, seconds=secs, attempts=attempts)
                stats["whisper"] += 1
                handled.add(row.video_id)
                print(f"  Whisper  [{stats['whisper']:>3}] {str(row.title)[:60]}")
            except Exception as exc:  # noqa: BLE001
                # keep an existing captions transcript; only record the failure for brand-new videos
                if _src(row.tsource) == "captions":
                    # keep the captions transcript, just count the failed Whisper upgrade
                    config.execute("UPDATE transcripts SET attempts = :a, last_error = :e WHERE video_id = :v",
                                   {"a": attempts, "e": str(exc)[:250], "v": row.video_id})
                else:
                    save(row.video_id, "failed", attempts=attempts, err=str(exc))
                    failed_saved.add(row.video_id)
                print(f"  skipped  {str(row.title)[:50]} ({str(exc)[:70]})")
        for _, fut in window:
            fut.cancel()

    # safety net: videos with no transcript at all that Whisper did not finish (out of time / download failed)
    if use_captions:
        missing = [r for r in everything.itertuples(index=False)
                   if r.video_id not in handled and _src(r.tsource) in (None, "failed")][:CAPTION_CAP]
        if missing:
            print(f"Fetching YouTube captions for {len(missing)} video(s) Whisper could not cover in this run...")

            def grab(r):
                try:
                    body, lang = fetch_captions(r.video_id)
                    return r, body, lang, None
                except Exception as exc:  # noqa: BLE001
                    return r, None, None, str(exc)

            in_a_row = 0
            with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS * 2) as pool:
                for r, body, lang, err in pool.map(grab, missing):
                    in_a_row = 0 if body else in_a_row + 1
                    if in_a_row >= 6 and not body:
                        stats["failed"] += 1
                        continue            # looks blocked by YouTube - stop saving noise, retry next run
                    if body:
                        save(r.video_id, "captions", language=lang, model="youtube-captions", body=body,
                             attempts=int(r.attempts))
                        stats["captions"] += 1
                    else:
                        stats["failed"] += 1
                        if r.video_id not in failed_saved:   # never reached by Whisper: keep it retryable
                            save(r.video_id, "failed", attempts=int(r.attempts), err="no captions available")
    stats["queued"] = max(len(everything) - stats["whisper"] - stats["failed"], 0)
    print(f"\nTranscription done in {time.time() - started:.0f}s: {stats['whisper']} via Whisper, "
          f"{stats['captions']} via captions, {stats['failed']} without transcript"
          + (f", {stats['queued']} more queued for Whisper on later runs." if stats["queued"] else "."))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Whisper videos per channel this run")
    ap.add_argument("--no-captions", action="store_true", help="Whisper only, no caption fallback")
    ap.add_argument("--budget", type=float, default=None, help="time budget in minutes")
    ap.add_argument("--video-id", type=str, default=None, help="Process only this video ID")
    a = ap.parse_args()
    run(a.limit, use_captions=not a.no_captions, budget_min=a.budget, video_id=a.video_id)
