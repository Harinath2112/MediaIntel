"""Isolated cloud Whisper test using private Supabase Storage."""

import os
import tempfile
from pathlib import Path
from urllib.parse import quote

import requests
from faster_whisper import WhisperModel


BUCKET = "mediaintel-audio"
OBJECT_PATH = "whisper-test.m4a"


def main():
    supabase_url = os.environ["SUPABASE_URL"].rstrip("/")
    secret_key = os.environ["SUPABASE_SECRET_KEY"]

    object_url = (
        f"{supabase_url}/storage/v1/object/authenticated/"
        f"{quote(BUCKET, safe='')}/{quote(OBJECT_PATH, safe='/')}"
    )

    headers = {
        "apikey": secret_key,
        "Authorization": f"Bearer {secret_key}",
    }

    print("Downloading test audio from private Supabase Storage...")

    response = requests.get(
        object_url,
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = Path(temp_dir) / "whisper-test.m4a"
        audio_path.write_bytes(response.content)

        print(f"Audio downloaded successfully ({audio_path.stat().st_size} bytes).")
        print("Loading Faster Whisper tiny model on CPU...")

        model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
            cpu_threads=2,
        )

        segments, info = model.transcribe(
            str(audio_path),
            beam_size=1,
            language="en",
        )

        transcript = " ".join(
            segment.text.strip() for segment in segments
        ).strip()

        if not transcript:
            raise RuntimeError("Whisper returned an empty transcript.")

        print("Whisper transcription successful.")
        print(f"Detected language: {info.language}")
        print(f"Transcript: {transcript}")


if __name__ == "__main__":
    main()
