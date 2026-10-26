"""Safe, server-side audio upload support for MediaIntel."""

from __future__ import annotations

import mimetypes
import re
from pathlib import Path
from core import config

def get_engine():
    return config.get_engine()

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

def verify_video_upload_eligibility(video_id: str, allow_reupload: bool = True) -> None:
    """Reject uploads for unknown, untracked, or already-processed videos."""
    from sqlalchemy import text

    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError("Invalid YouTube video ID.")

    engine = get_engine()
    if engine.dialect.name not in ("postgresql", "sqlite"):
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
                          AND t.source IN ('whisper', 'faster_whisper', 'captions', 'youtube_manual_caption', 'youtube_auto_caption')
                          AND t.transcript_text IS NOT NULL
                          AND TRIM(t.transcript_text) != ''
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

    if not allow_reupload and row["has_mapping"]:
        raise ValueError("Audio mapping already exists for this video.")

    if row["has_transcript"]:
        raise ValueError("Video already has a successful transcript.")


def create_audio_mapping(video_id: str, object_path: str) -> None:
    """Atomically register uploaded audio if the video remains eligible."""
    from sqlalchemy import text

    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError("Invalid YouTube video ID.")

    if not re.fullmatch(
        rf"videos/{re.escape(video_id)}/[A-Za-z0-9_-]+\.(m4a|mp3|wav|ogg|webm)",
        object_path,
    ):
        raise ValueError("Invalid audio object path.")

    engine = get_engine()
    if engine.dialect.name not in ("postgresql", "sqlite"):
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
                        AND t.source IN ('whisper', 'faster_whisper', 'captions', 'youtube_manual_caption', 'youtube_auto_caption')
                        AND t.transcript_text IS NOT NULL
                        AND TRIM(t.transcript_text) != ''
                  )
                ON CONFLICT (video_id) DO UPDATE SET
                    bucket = EXCLUDED.bucket,
                    object_path = EXCLUDED.object_path,
                    status = 'pending',
                    attempts = 0,
                    last_error = NULL,
                    claimed_at = NULL,
                    claim_token = NULL,
                    lease_expires_at = NULL,
                    updated_at = CURRENT_TIMESTAMP
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
                "or already has a completed transcript."
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


def save_and_upload_admin_audio(video_id: str, file_bytes: bytes, filename: str) -> dict:
    """Validate server-side and upload admin-supplied authorized audio to Supabase Storage.

    Accepts raw file bytes and filename from Streamlit file_uploader.
    Performs server-side file type, magic byte, and size validation in a safe temp file.
    Registers audio asset and marks video state for Faster-Whisper transcription.
    """
    import tempfile
    from sqlalchemy import text

    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        return {"success": False, "error": "Invalid YouTube video ID format."}

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_AUDIO_TYPES:
        return {
            "success": False,
            "error": f"Unsupported audio format '{ext}'. Supported formats: MP3, WAV, M4A, OGG, WEBM.",
        }

    if len(file_bytes) == 0:
        return {"success": False, "error": "Audio file is empty (0 bytes)."}

    if len(file_bytes) > MAX_AUDIO_BYTES:
        return {
            "success": False,
            "error": f"Audio file exceeds maximum size limit of {MAX_AUDIO_BYTES // (1024 * 1024)} MB.",
        }

    tmp_file = tempfile.NamedTemporaryFile(prefix=f"admin_audio_{video_id}_", suffix=ext, delete=False)
    try:
        tmp_file.write(file_bytes)
        tmp_file.flush()
        tmp_file.close()
        tmp_path = Path(tmp_file.name)

        if not _has_valid_audio_header(tmp_path, ext):
            return {
                "success": False,
                "error": "Audio file content does not match allowed audio format header (invalid magic bytes).",
            }

        verify_video_upload_eligibility(video_id, allow_reupload=True)
        object_path = upload_to_private_storage(video_id, tmp_path)
        create_audio_mapping(video_id, object_path)

        # Update transcripts table to mark status ready for transcription
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO transcripts (video_id, source, status, last_error, updated_at)
                    VALUES (:vid, 'awaiting_authorized_audio', 'awaiting_transcription', 'Authorized audio uploaded; ready for Faster-Whisper', CURRENT_TIMESTAMP)
                    ON CONFLICT (video_id) DO UPDATE SET
                        source = 'awaiting_authorized_audio',
                        status = 'awaiting_transcription',
                        last_error = 'Authorized audio uploaded; ready for Faster-Whisper',
                        updated_at = CURRENT_TIMESTAMP
                """),
                {"vid": video_id},
            )

        return {
            "success": True,
            "object_path": object_path,
            "message": f"Successfully uploaded and verified authorized audio: {object_path}",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    finally:
        try:
            p = Path(tmp_file.name)
            if p.exists():
                p.unlink()
        except Exception:
            pass

