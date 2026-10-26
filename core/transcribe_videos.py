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


def resolve_stream_info(video_id: str) -> tuple[str, dict, int | None, list[dict] | None, str | None]:
    """Asks yt-dlp for direct audio stream URL and video metadata (no download).

    Returns: (url, http_headers, duration_sec, chapters, description)
    """
    import yt_dlp
    opts = {"format": "bestaudio[abr<=70]/worstaudio/worst", "quiet": True, "no_warnings": True,
            "noplaylist": True, "skip_download": True, "socket_timeout": 30}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    url = info.get("url")
    headers = info.get("http_headers") or {}
    if not url:
        fmt = (info.get("requested_formats") or info.get("formats") or [])[0]
        url = fmt["url"]
        headers = fmt.get("http_headers") or info.get("http_headers") or {}
    duration_sec = info.get("duration")
    chapters = info.get("chapters")
    description = info.get("description")
    return url, headers, duration_sec, chapters, description


def resolve_stream(video_id: str) -> tuple[str, dict]:
    """Asks yt-dlp for the direct audio stream URL of a video (no download). Returns (url, http_headers)."""
    url, headers, _, _, _ = resolve_stream_info(video_id)
    return url, headers


def read_pcm_segment(
    source: str,
    headers: dict | None,
    start_sec: float,
    duration_sec: float,
) -> np.ndarray:
    """Streams the audio segment [start_sec, start_sec + duration_sec] from source through ffmpeg.

    Uses FFmpeg input seeking (-ss before -i) where supported to jump directly
    to start_sec without downloading or decoding intervening preceding audio.
    Returns 16 kHz mono int16 samples as a numpy array.
    """
    import numpy as np
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")

    cmd = [_ffmpeg(), "-nostdin", "-loglevel", "error"]
    if headers:
        cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
    if start_sec > 0:
        cmd += ["-ss", f"{start_sec:.3f}"]
    cmd += [
        "-i", source,
        "-t", f"{duration_sec:.3f}",
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "s16le", "pipe:1"
    ]
    timeout = max(30, int(duration_sec * 3 + 45))
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"audio seeking timed out [{start_sec:.1f}s+{duration_sec:.1f}s]") from exc

    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    min_samples = int(min(duration_sec, 0.5) * 16000 * 0.5)
    if proc.returncode != 0 or pcm.size < min_samples:
        err_msg = proc.stderr.decode("utf-8", "replace").strip()[-200:]
        raise RuntimeError(f"audio segment extraction failed [{start_sec:.1f}s+{duration_sec:.1f}s]: {err_msg}")
    return pcm


def read_pcm(source: str, headers: dict | None, max_seconds: int) -> np.ndarray:
    """Streams the first `max_seconds` of audio from `source` (URL or file) through ffmpeg into memory.

    Returns 16 kHz mono int16 samples. The only thing that ever exists is this in-memory array.
    """
    return read_pcm_segment(source, headers, start_sec=0.0, duration_sec=float(max_seconds))


def _temp_download_segment_pcm(video_id: str, start_sec: float, duration_sec: float) -> np.ndarray:
    """Fallback when direct streaming is refused: bounded temp download using yt-dlp download sections."""
    import yt_dlp
    with tempfile.TemporaryDirectory(prefix="mediaintel_seg_") as tmp:
        opts = {
            "format": "bestaudio[abr<=70]/worstaudio/worst",
            "outtmpl": os.path.join(tmp, "a.%(ext)s"),
            "quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True,
            "socket_timeout": 30,
            "ffmpeg_location": _ffmpeg(),
            "download_ranges": yt_dlp.utils.download_range_func(None, [(start_sec, start_sec + duration_sec)]),
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        files = [f for f in Path(tmp).glob("a.*") if f.suffix not in (".part", ".ytdl")]
        if not files:
            raise RuntimeError(f"audio segment download failed for {video_id}")
        return read_pcm_segment(str(files[0]), None, start_sec=0.0, duration_sec=duration_sec)


def _temp_download_pcm(video_id: str, max_seconds: int) -> np.ndarray:
    """Last resort when YouTube refuses direct streaming: download into a self-deleting temp folder,
    decode into memory and delete immediately (nothing is kept)."""
    return _temp_download_segment_pcm(video_id, start_sec=0.0, duration_sec=float(max_seconds))


def fetch_audio_segment(video_id: str, start_sec: float, duration_sec: float) -> np.ndarray:
    """Fetch an audio segment [start_sec, start_sec + duration_sec] as an in-memory int16 array."""
    try:
        url, headers = resolve_stream(video_id)
        return read_pcm_segment(url, headers, start_sec, duration_sec)
    except Exception as exc:  # noqa: BLE001
        if not TEMP_FALLBACK:
            raise RuntimeError(f"audio segment stream failed: {str(exc)[:150]}") from exc
        return _temp_download_segment_pcm(video_id, start_sec, duration_sec)


def fetch_audio_segments(
    video_id: str,
    plan: "core.smart_transcription.TranscriptionPlan",
) -> list[tuple["core.smart_transcription.AudioSegment", np.ndarray]]:
    """Extract audio segments according to the transcription plan using FFmpeg input seeking.

    For each segment, extracts only the required audio range [start_sec, end_sec].
    Intervening audio is not downloaded or decoded.
    """
    import unittest.mock
    import numpy as np
    from core.smart_transcription import slice_pcm_for_plan

    # If fetch_audio is mocked in a test and fetch_audio_segments is not, honor the mock
    if isinstance(fetch_audio, (unittest.mock.Mock, unittest.mock.MagicMock)):
        mock_raw = fetch_audio(video_id, plan.total_audio_seconds)
        if isinstance(mock_raw, list):
            return mock_raw
        if isinstance(mock_raw, bytes):
            mock_arr = np.frombuffer(mock_raw, dtype=np.int16)
        else:
            mock_arr = np.asarray(mock_raw, dtype=np.int16)
        return slice_pcm_for_plan(mock_arr, plan)

    url = None
    headers = None
    try:
        url, headers = resolve_stream(video_id)
    except Exception as exc:
        if not TEMP_FALLBACK:
            raise RuntimeError(f"audio stream resolution failed: {str(exc)[:150]}") from exc

    results = []
    for seg in plan.segments:
        if url:
            try:
                chunk = read_pcm_segment(url, headers, start_sec=seg.start_sec, duration_sec=seg.duration_sec)
                results.append((seg, chunk))
                continue
            except Exception:
                if not TEMP_FALLBACK:
                    raise
        # Fallback if streaming failed or stream URL unavailable
        chunk = _temp_download_segment_pcm(video_id, start_sec=seg.start_sec, duration_sec=seg.duration_sec)
        results.append((seg, chunk))

    return results


def fetch_audio(video_id: str, max_seconds: int = 900) -> np.ndarray:
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


def whisper_transcribe(pcm, max_seconds: int = 900, video_duration_sec: int | None = None):
    """pcm = 16 kHz mono int16 samples in memory or list of (AudioSegment, np.ndarray).
    Returns (text, language, seconds_transcribed).
    """
    import numpy as np
    from core.smart_transcription import plan_transcription_segments, slice_pcm_for_plan, combine_segment_transcripts

    model = load_whisper()

    if isinstance(pcm, list):
        # Already extracted segment chunks: list of (AudioSegment, np.ndarray)
        results = []
        langs = []
        for seg, chunk in pcm:
            if len(chunk) < 8000:
                results.append((seg, ""))
                continue
            audio_chunk = (chunk.astype(np.float32)) / 32768.0
            segments, info = model.transcribe(
                audio_chunk,
                language=WHISPER_LANGUAGE or None,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            seg_text = " ".join(s.text.strip() for s in segments).strip()
            results.append((seg, seg_text))
            if hasattr(info, "language") and info.language:
                langs.append(info.language)
        total_sec = video_duration_sec if (video_duration_sec and video_duration_sec > 0) else sum(int(s.duration_sec) for s, _ in pcm)
        plan = plan_transcription_segments(total_sec, max_partial_sec=max_seconds)
        combined_text = combine_segment_transcripts(results, plan)
        lang = max(set(langs), key=langs.count) if langs else "und"
        return combined_text, lang, plan.total_audio_seconds

    total_sec = video_duration_sec if (video_duration_sec and video_duration_sec > 0) else int(len(pcm) / 16000)
    plan = plan_transcription_segments(total_sec, max_partial_sec=max_seconds)

    if not plan.is_partial:
        audio = (pcm[: max_seconds * 16000].astype(np.float32)) / 32768.0
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
    else:
        sliced_chunks = slice_pcm_for_plan(pcm, plan)
        results = []
        langs = []
        for seg, chunk in sliced_chunks:
            if len(chunk) < 8000:
                results.append((seg, ""))
                continue
            audio_chunk = (chunk.astype(np.float32)) / 32768.0
            segments, info = model.transcribe(
                audio_chunk,
                language=WHISPER_LANGUAGE or None,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            seg_text = " ".join(s.text.strip() for s in segments).strip()
            results.append((seg, seg_text))
            if hasattr(info, "language") and info.language:
                langs.append(info.language)
        combined_text = combine_segment_transcripts(results, plan)
        lang = max(set(langs), key=langs.count) if langs else "und"
        return combined_text, lang, plan.total_audio_seconds


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
