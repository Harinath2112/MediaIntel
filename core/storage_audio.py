
"""Download explicitly mapped audio from private Supabase Storage."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote
from core import config

import requests


MAX_AUDIO_BYTES = 25 * 1024 * 1024
ALLOWED_BUCKET = "mediaintel-audio"


@contextmanager
def temporary_storage_audio(bucket: str, object_path: str):
    """Yield a temporary audio path and remove it afterward."""

    if bucket != ALLOWED_BUCKET:
        raise ValueError("Unapproved Supabase Storage bucket")

    if (
        not object_path
        or object_path.startswith("/")
        or "\\" in object_path
        or any(part in ("", ".", "..") for part in object_path.split("/"))
    ):
        raise ValueError("Invalid Storage object path")

    supabase_url = os.environ["SUPABASE_URL"].rstrip("/")
    secret_key = os.environ["SUPABASE_SECRET_KEY"]

    object_url = (
        f"{supabase_url}/storage/v1/object/authenticated/"
        f"{quote(bucket, safe='')}/{quote(object_path, safe='/')}"
    )

    headers = {
        "apikey": secret_key,
        "Authorization": f"Bearer {secret_key}",
    }

    with tempfile.TemporaryDirectory(prefix="mediaintel_audio_") as tmp:
        audio_path = Path(tmp) / "audio_input"

        with requests.get(
            object_url,
            headers=headers,
            stream=True,
            timeout=(15, 60),
        ) as response:
            response.raise_for_status()

            total = 0
            with audio_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=65536):
                    if not chunk:
                        continue

                    total += len(chunk)

                    if total > MAX_AUDIO_BYTES:
                        raise ValueError("Audio file exceeds 25 MB limit")

                    output.write(chunk)

        if total == 0:
            raise ValueError("Downloaded audio file is empty")

        yield audio_path

def get_audio_mapping(video_id: str) -> tuple[str, str] | None:
    """Return the private Storage location for a mapped video.

    Only pending or previously failed mappings are eligible.
    This function never modifies the database.
    """
    if not video_id or not video_id.strip():
        raise ValueError("A valid video ID is required")

    df = config.read_sql(
        """
        SELECT bucket, object_path
        FROM video_audio_assets
        WHERE video_id = :video_id
          AND status IN ('pending', 'failed')
          AND attempts < :max_attempts
        """,
        {
            "video_id": video_id.strip(),
            "max_attempts": 3,
        },
    )

    if df.empty:
        return None

    row = df.iloc[0]
    return str(row["bucket"]), str(row["object_path"])

def storage_audio_to_pcm(bucket: str, object_path: str, max_seconds: int):
    """Download private Storage audio and decode to 16 kHz mono int16 PCM.

    Returns a NumPy array compatible with whisper_transcribe().
    Does not modify database records or Storage objects.
    """
    import subprocess
    import numpy as np

    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")

    with temporary_storage_audio(bucket, object_path) as audio_path:
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(audio_path),
            "-t", str(max_seconds),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "pipe:1",
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(60, max_seconds + 30),
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Audio decoding failed: "
                f"{result.stderr.decode('utf-8', errors='replace')[:200]}"
            )

        pcm = np.frombuffer(result.stdout, dtype=np.int16).copy()

        if pcm.size == 0:
            raise RuntimeError("Decoded audio is empty")

        return pcm


def mapped_audio_candidates():
    """Find tracked videos with eligible private Storage audio mappings.

    Read-only. Does not modify transcripts or mapping attempts.
    """
    return config.read_sql(
        """
        SELECT
            v.video_id,
            v.title,
            v.channel_id,
            v.published_at,
            a.bucket,
            a.object_path,
            a.status AS audio_status,
            a.attempts AS audio_attempts,
            t.source AS transcript_source,
            COALESCE(t.attempts, 0) AS transcript_attempts
        FROM video_audio_assets a
        JOIN videos v ON v.video_id = a.video_id
        JOIN channels c ON c.channel_id = v.channel_id
        LEFT JOIN transcripts t ON t.video_id = v.video_id
        WHERE c.is_tracked = TRUE
          AND (
            a.status IN ('pending', 'failed')
            OR (
                a.status = 'processing'
                AND a.lease_expires_at < CURRENT_TIMESTAMP
            )
          )
          AND a.attempts < :max_attempts
          AND (
              t.video_id IS NULL
              OR t.source IN ('captions', 'failed')
          )
        ORDER BY v.channel_id, v.published_at DESC
        """,
        {"max_attempts": 3},
    )

def update_audio_mapping_status(
    video_id: str,
    status: str,
    *,
    error: str | None = None,
) -> int:
    """Update a mapped audio asset after a processing attempt.

    Returns the number of updated rows. Never creates a mapping.
    """
    from sqlalchemy import text

    if not video_id or not video_id.strip():
        raise ValueError("A valid video ID is required")

    if status not in ("completed", "failed"):
        raise ValueError("Status must be 'completed' or 'failed'")

    statement = text(
        """
        UPDATE video_audio_assets
        SET status = :status,
            last_error = :error,
            updated_at = CURRENT_TIMESTAMP
        WHERE video_id = :video_id
          AND status IN ('pending', 'failed')
          AND attempts < 3
        """
    )

    with config.get_engine().begin() as connection:
        result = connection.execute(
            statement,
            {
                "video_id": video_id.strip(),
                "status": status,
                "error": (str(error)[:255] if error else None),
            },
        )

        return result.rowcount

def complete_audio_transcription(
    video_id: str,
    *,
    claim_token: str,
    language: str,
    model: str,
    body: str,
    seconds: int,
) -> bool:
    """Atomically save a transcript and complete an owned audio mapping."""
    from sqlalchemy import text
    from core.transcribe_videos import TRANSCRIPT_COLS

    if not video_id or not video_id.strip():
        raise ValueError("A valid video ID is required")

    if not claim_token or not claim_token.strip():
        raise ValueError("A valid claim token is required")

    if not body or not body.strip():
        raise ValueError("Cannot save an empty Whisper transcript")

    engine = config.get_engine()

    if engine.dialect.name != "postgresql":
        raise RuntimeError("Cloud completion requires PostgreSQL")

    video_id = video_id.strip()
    claim_token = claim_token.strip()

    with engine.begin() as connection:
        mapping = connection.execute(
            text("""
                SELECT video_id
                FROM video_audio_assets
                WHERE video_id = :video_id
                  AND status = 'processing'
                  AND claim_token = :claim_token
                  AND lease_expires_at > CURRENT_TIMESTAMP
                FOR UPDATE
            """),
            {
                "video_id": video_id,
                "claim_token": claim_token,
            },
        ).first()

        if mapping is None:
            return False

        existing = connection.execute(
            text("""
                SELECT source
                FROM transcripts
                WHERE video_id = :video_id
                FOR UPDATE
            """),
            {"video_id": video_id},
        ).first()

        if existing is not None and existing.source not in (
            "captions",
            "failed",
        ):
            return False

        connection.execute(
            config.upsert_sql(
                "transcripts",
                TRANSCRIPT_COLS,
                ["video_id"],
            ),
            {
                "video_id": video_id,
                "source": "whisper",
                "language": (language or "")[:12] or None,
                "model_name": model,
                "transcript_text": body.strip(),
                "audio_seconds": seconds,
                "word_count": len(body.split()),
                "attempts": 0,
                "last_error": None,
                "updated_at": config.utcnow(),
            },
        )

        result = connection.execute(
            text("""
                UPDATE video_audio_assets
                SET status = 'completed',
                    claim_token = NULL,
                    claimed_at = NULL,
                    lease_expires_at = NULL,
                    last_error = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE video_id = :video_id
                  AND status = 'processing'
                  AND claim_token = :claim_token
                  AND lease_expires_at > CURRENT_TIMESTAMP
            """),
            {
                "video_id": video_id,
                "claim_token": claim_token,
            },
        )

        if result.rowcount != 1:
            raise RuntimeError("Audio mapping completion failed")

        return True

def claim_audio_mapping(
    video_id: str,
    *,
    lease_minutes: int = 30,
) -> dict | None:
    """Atomically claim an eligible audio mapping for one worker."""
    import uuid
    from sqlalchemy import text

    if not video_id or not video_id.strip():
        raise ValueError("A valid video ID is required")

    if not 1 <= lease_minutes <= 120:
        raise ValueError("Lease must be between 1 and 120 minutes")

    engine = config.get_engine()

    if engine.dialect.name != "postgresql":
        raise RuntimeError("Cloud audio claiming requires PostgreSQL")

    token = uuid.uuid4().hex

    statement = text("""
        WITH eligible AS (
            SELECT a.video_id
            FROM video_audio_assets a
            JOIN videos v ON v.video_id = a.video_id
            JOIN channels c ON c.channel_id = v.channel_id
            LEFT JOIN transcripts t ON t.video_id = a.video_id
            WHERE a.video_id = :video_id
              AND c.is_tracked = TRUE
              AND a.attempts < 3
              AND (
                  a.status IN ('pending', 'failed')
                  OR (
                      a.status = 'processing'
                      AND a.lease_expires_at < CURRENT_TIMESTAMP
                  )
              )
              AND (
                  t.video_id IS NULL
                  OR t.source IN ('captions', 'failed')
              )
            FOR UPDATE OF a SKIP LOCKED
        )
        UPDATE video_audio_assets a
            SET status = 'processing',
            attempts = a.attempts + 1,
            claim_token = :token,
            claimed_at = CURRENT_TIMESTAMP,
            lease_expires_at =
                CURRENT_TIMESTAMP
                + (:lease_minutes * INTERVAL '1 minute'),
            last_error = NULL,
            updated_at = CURRENT_TIMESTAMP
        FROM eligible
        WHERE a.video_id = eligible.video_id
        RETURNING a.video_id, a.bucket, a.object_path,
                  a.claim_token
    """)

    with engine.begin() as connection:
        row = connection.execute(
            statement,
            {
                "video_id": video_id.strip(),
                "token": token,
                "lease_minutes": lease_minutes,
            },
        ).mappings().first()

    return dict(row) if row else None

def fail_audio_claim(
    video_id: str,
    *,
    claim_token: str,
    error: str,
) -> bool:
    """Record a failed audio attempt only if this worker owns the claim."""
    from sqlalchemy import text

    if not video_id or not video_id.strip():
        raise ValueError("A valid video ID is required")

    if not claim_token or not claim_token.strip():
        raise ValueError("A valid claim token is required")

    if not error or not error.strip():
        raise ValueError("A failure reason is required")

    engine = config.get_engine()

    if engine.dialect.name != "postgresql":
        raise RuntimeError("Cloud failure handling requires PostgreSQL")

    statement = text("""
        UPDATE video_audio_assets
        SET status = 'failed',
            last_error = :error,
            claim_token = NULL,
            claimed_at = NULL,
            lease_expires_at = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE video_id = :video_id
          AND status = 'processing'
          AND claim_token = :claim_token
          AND lease_expires_at > CURRENT_TIMESTAMP
    """)

    with engine.begin() as connection:
        result = connection.execute(
            statement,
            {
                "video_id": video_id.strip(),
                "claim_token": claim_token.strip(),
                "error": error.strip()[:255],
            },
        )

        return result.rowcount == 1

def renew_audio_claim(
    video_id: str,
    *,
    claim_token: str,
    lease_minutes: int = 30,
) -> bool:
    """Extend a valid lease only for the worker that owns it."""
    from sqlalchemy import text

    if not video_id or not video_id.strip():
        raise ValueError("A valid video ID is required")

    if not claim_token or not claim_token.strip():
        raise ValueError("A valid claim token is required")

    if not isinstance(lease_minutes, int) or isinstance(lease_minutes, bool):
        raise ValueError("Lease duration must be an integer")

    if not 1 <= lease_minutes <= 120:
        raise ValueError("Lease must be between 1 and 120 minutes")

    engine = config.get_engine()

    if engine.dialect.name != "postgresql":
        raise RuntimeError("Cloud lease renewal requires PostgreSQL")

    statement = text("""
        UPDATE video_audio_assets
        SET lease_expires_at = GREATEST(
                lease_expires_at,
                CURRENT_TIMESTAMP
            ) + (:lease_minutes * INTERVAL '1 minute'),
            updated_at = CURRENT_TIMESTAMP
        WHERE video_id = :video_id
          AND status = 'processing'
          AND claim_token = :claim_token
          AND lease_expires_at > CURRENT_TIMESTAMP
    """)

    with engine.begin() as connection:
        result = connection.execute(
            statement,
            {
                "video_id": video_id.strip(),
                "claim_token": claim_token.strip(),
                "lease_minutes": lease_minutes,
            },
        )

        return result.rowcount == 1