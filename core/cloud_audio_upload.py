"""Safe, server-side audio upload support for MediaIntel."""

from __future__ import annotations

import mimetypes
import re
from pathlib import Path

from core.config import get_engine

BUCKET = "mediaintel-audio"
MAX_AUDIO_BYTES = 25 * 1024 * 1024

ALLOWED_AUDIO_TYPES = {
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
}

VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _has_valid_audio_header(path: Path, ext: str) -> bool:
    """Validate audio header magic bytes to reject disguised or corrupt files."""
    try:
        with path.open("rb") as f:
            header = f.read(32)
        if len(header) < 4:
            return False
        if ext == ".mp3":
            return header.startswith(b"ID3") or (header[0] == 0xFF and (header[1] & 0xE0) == 0xE0)
        if ext == ".wav":
            return header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WAVE"
        if ext == ".ogg":
            return header.startswith(b"OggS")
        if ext == ".m4a":
            return len(header) >= 8 and header[4:8] == b"ftyp"
        if ext == ".webm":
            return header.startswith(b"\x1a\x45\xdf\xa3")
    except Exception:
        return False
    return False


def validate_audio(video_id: str, file_path: str | Path) -> tuple[Path, str]:
    """Validate an existing audio file before any upload or database operation."""
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError("Invalid YouTube video ID.")

    path = Path(file_path).resolve(strict=True)

    if not path.is_file():
        raise ValueError("Audio path must refer to a regular file.")

    ext = path.suffix.lower()
    content_type = ALLOWED_AUDIO_TYPES.get(ext)
    if content_type is None:
        raise ValueError("Unsupported audio file extension.")

    size = path.stat().st_size
    if size == 0 or size > MAX_AUDIO_BYTES:
        raise ValueError("Audio file must be between 1 byte and 25 MB.")

    if not _has_valid_audio_header(path, ext):
        raise ValueError("Audio file content does not match allowed audio format header.")

    return path, content_type

def upload_to_private_storage(video_id: str, file_path: str | Path) -> str:
    """Upload authorized audio without overwriting an existing Storage object."""
    import os
    import uuid
    from urllib.parse import quote

    import requests

    path, content_type = validate_audio(video_id, file_path)

    supabase_url = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    secret_key = (os.getenv("SUPABASE_SECRET_KEY") or "").strip()

    if not supabase_url.startswith("https://") or not secret_key:
        raise RuntimeError("Supabase Storage configuration is missing or invalid.")

    # A unique object path prevents overwriting existing audio.
    object_path = f"videos/{video_id}/{uuid.uuid4().hex}{path.suffix.lower()}"

    url = (
        f"{supabase_url}/storage/v1/object/"
        f"{quote(BUCKET, safe='')}/{quote(object_path, safe='/')}"
    )

    headers = {
        "apikey": secret_key,
        "Authorization": f"Bearer {secret_key}",
        "Content-Type": content_type,
        "x-upsert": "false",
    }

    with path.open("rb") as audio_file:
        response = requests.post(
            url,
            headers=headers,
            data=audio_file,
            timeout=(15, 180),
        )

    if response.status_code not in (200, 201):
        raise RuntimeError(
            f"Private audio upload failed (HTTP {response.status_code})."
        )

    return object_path

def verify_video_upload_eligibility(video_id: str) -> None:
    """Reject uploads for unknown, untracked, or already-processed videos."""
    from sqlalchemy import text

    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError("Invalid YouTube video ID.")

    engine = get_engine()
    if engine.dialect.name != "postgresql":
        raise RuntimeError("Cloud audio uploads require PostgreSQL.")

    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT
                    v.video_id,
                    c.is_tracked,
                    EXISTS (
                        SELECT 1 FROM video_audio_assets a
                        WHERE a.video_id = v.video_id
                    ) AS has_mapping,
                    EXISTS (
                        SELECT 1 FROM transcripts t
                        WHERE t.video_id = v.video_id
                          AND t.source IN ('whisper', 'captions')
                    ) AS has_transcript
                FROM videos v
                JOIN channels c ON c.channel_id = v.channel_id
                WHERE v.video_id = :video_id
            """),
            {"video_id": video_id},
        ).mappings().first()

    if row is None:
        raise ValueError("Video does not exist in the database.")

    if not row["is_tracked"]:
        raise ValueError("Video belongs to an untracked channel.")

    if row["has_mapping"]:
        raise ValueError("Audio mapping already exists for this video.")

    if row["has_transcript"]:
        raise ValueError("Video already has a successful transcript.")


def create_audio_mapping(video_id: str, object_path: str) -> None:
    """Atomically register uploaded audio if the video remains eligible."""
    from sqlalchemy import text

    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError("Invalid YouTube video ID.")

    if not re.fullmatch(
        rf"videos/{re.escape(video_id)}/[0-9a-f]{{32}}\.(m4a|mp3|wav|ogg|webm)",
        object_path,
    ):
        raise ValueError("Invalid audio object path.")

    engine = get_engine()
    if engine.dialect.name != "postgresql":
        raise RuntimeError("Cloud audio mappings require PostgreSQL.")

    with engine.begin() as conn:
        result = conn.execute(
            text("""
                INSERT INTO video_audio_assets
                    (video_id, bucket, object_path, status, attempts)
                SELECT
                    v.video_id, :bucket, :object_path, 'pending', 0
                FROM videos v
                JOIN channels c ON c.channel_id = v.channel_id
                WHERE v.video_id = :video_id
                  AND c.is_tracked = TRUE
                  AND NOT EXISTS (
                      SELECT 1 FROM transcripts t
                      WHERE t.video_id = v.video_id
                        AND t.source IN ('whisper', 'captions')
                  )
                ON CONFLICT DO NOTHING
                RETURNING video_id
            """),
            {
                "video_id": video_id,
                "bucket": BUCKET,
                "object_path": object_path,
            },
        ).first()

        if result is None:
            raise ValueError(
                "Audio mapping was not created: video is ineligible "
                "or a conflicting mapping already exists."
            )


def delete_from_private_storage(object_path: str) -> bool:
    """Delete a specific Storage object from the approved bucket during partial failure cleanup.

    Never deletes unrelated or previously mapped audio.
    """
    import os
    from urllib.parse import quote

    import requests

    if not object_path or not object_path.startswith("videos/") or ".." in object_path:
        raise ValueError("Invalid object path for deletion.")

    supabase_url = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    secret_key = (os.getenv("SUPABASE_SECRET_KEY") or "").strip()

    if not supabase_url.startswith("https://") or not secret_key:
        raise RuntimeError("Supabase Storage configuration is missing or invalid.")

    url = (
        f"{supabase_url}/storage/v1/object/"
        f"{quote(BUCKET, safe='')}/{quote(object_path, safe='/')}"
    )

    headers = {
        "apikey": secret_key,
        "Authorization": f"Bearer {secret_key}",
    }

    try:
        response = requests.delete(url, headers=headers, timeout=(10, 30))
        return response.status_code in (200, 204)
    except Exception:
        return False


def upload_authorized_audio(video_id: str, file_path: str | Path) -> str:
    """Safely upload authorized audio and register atomic database mapping.

    Performs complete end-to-end processing with rollback cleanup if the
    database mapping operation fails.
    """
    verify_video_upload_eligibility(video_id)
    path, _ = validate_audio(video_id, file_path)

    object_path = upload_to_private_storage(video_id, path)

    try:
        create_audio_mapping(video_id, object_path)
    except Exception as exc:
        # Atomic rollback cleanup of only the object uploaded in this operation
        delete_from_private_storage(object_path)
        raise exc

    return object_path


def acquire_and_upload_authorized_audio(
    video_id: str,
    max_seconds: int = 600,
) -> str:
    """Acquire authorized audio in compliance with YouTube terms and upload to private Storage.

    If direct platform audio extraction is restricted or blocked, raises an informative
    exception directing to authorized direct file upload without generating fake data.
    """
    import os
    import tempfile

    verify_video_upload_eligibility(video_id)

    try:
        import yt_dlp
    except ImportError as err:
        raise RuntimeError(
            "yt-dlp is required for direct audio acquisition. Install yt-dlp or "
            "use upload_authorized_audio with a pre-acquired audio file."
        ) from err

    with tempfile.TemporaryDirectory(prefix="mediaintel_acquire_") as tmp_dir:
        out_template = os.path.join(tmp_dir, "audio.%(ext)s")
        ydl_opts = {
            "format": "bestaudio[ext=m4a]/bestaudio/worst",
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "noplaylist": True,
            "socket_timeout": 30,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "m4a",
                }
            ],
        }

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except Exception as exc:
            raise RuntimeError(
                f"Direct YouTube audio acquisition for {video_id} is unavailable "
                f"due to platform access restrictions: {str(exc)[:150]}. "
                "Please use the authorized direct audio upload workflow."
            ) from exc

        audio_candidates = [
            f for f in Path(tmp_dir).iterdir()
            if f.is_file() and f.suffix.lower() in ALLOWED_AUDIO_TYPES
        ]

        if not audio_candidates:
            raise RuntimeError(
                f"No supported audio output produced for video {video_id}."
            )

        acquired_path = audio_candidates[0]
        return upload_authorized_audio(video_id, acquired_path)

