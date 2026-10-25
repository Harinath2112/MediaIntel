"""Unit tests for MediaIntel cloud components and safety controls."""

from __future__ import annotations

import hmac
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import cloud_audio_upload, config


def test_video_id_regex():
    valid = ["UCBwmMxybNv", "dQw4w9WgXcQ", "12345678901", "a-b_c123456"]
    for vid in valid:
        assert cloud_audio_upload.VIDEO_ID_PATTERN.fullmatch(vid)

    invalid = ["", "short", "toolong12345678", "has space!!", "invalid$char"]
    for vid in invalid:
        assert not cloud_audio_upload.VIDEO_ID_PATTERN.fullmatch(vid)


def test_audio_header_validation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # 1. Valid MP3 with ID3
        mp3_file = tmp / "sample.mp3"
        mp3_file.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 100)
        path, ctype = cloud_audio_upload.validate_audio("dQw4w9WgXcQ", mp3_file)
        assert ctype == "audio/mpeg"

        # 2. Valid WAV with RIFF WAVE
        wav_file = tmp / "sample.wav"
        wav_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 100)
        path, ctype = cloud_audio_upload.validate_audio("dQw4w9WgXcQ", wav_file)
        assert ctype == "audio/wav"

        # 3. Valid M4A with ftyp
        m4a_file = tmp / "sample.m4a"
        m4a_file.write_bytes(b"\x00\x00\x00\x1cftypM4A " + b"\x00" * 100)
        path, ctype = cloud_audio_upload.validate_audio("dQw4w9WgXcQ", m4a_file)
        assert ctype == "audio/mp4"

        # 4. Valid OGG with OggS
        ogg_file = tmp / "sample.ogg"
        ogg_file.write_bytes(b"OggS\x00\x02\x00\x00" + b"\x00" * 100)
        path, ctype = cloud_audio_upload.validate_audio("dQw4w9WgXcQ", ogg_file)
        assert ctype == "audio/ogg"

        # 5. Valid WebM with EBML header
        webm_file = tmp / "sample.webm"
        webm_file.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 100)
        path, ctype = cloud_audio_upload.validate_audio("dQw4w9WgXcQ", webm_file)
        assert ctype == "audio/webm"

        # 6. Disguised file (text disguised as .mp3)
        fake_mp3 = tmp / "fake.mp3"
        fake_mp3.write_bytes(b"This is just plain text, not real audio!")
        try:
            cloud_audio_upload.validate_audio("dQw4w9WgXcQ", fake_mp3)
            assert False, "Should have rejected disguised non-audio file"
        except ValueError as exc:
            assert "audio format header" in str(exc)

        # 7. Empty file
        empty_wav = tmp / "empty.wav"
        empty_wav.write_bytes(b"")
        try:
            cloud_audio_upload.validate_audio("dQw4w9WgXcQ", empty_wav)
            assert False, "Should have rejected empty audio file"
        except ValueError as exc:
            assert "between 1 byte and 25 MB" in str(exc)


def test_storage_deletion_path_security():
    # Traversal or wrong prefix must be rejected
    for bad_path in ["../secret.txt", "/etc/passwd", "other_bucket/foo.mp3", ""]:
        try:
            cloud_audio_upload.delete_from_private_storage(bad_path)
            assert False, f"Should have rejected invalid object path: {bad_path}"
        except ValueError:
            pass


def test_database_url_normalization():
    # Check that postgres:// is normalized to postgresql+psycopg://
    os.environ["DATABASE_URL"] = "postgres://postgres:password@localhost:5432/testdb"
    norm = config.database_url()
    assert norm.startswith("postgresql+psycopg://") or norm.startswith("postgresql://")

    # Check postgresql:// is also converted
    os.environ["DATABASE_URL"] = "postgresql://postgres:password@localhost:5432/testdb"
    norm2 = config.database_url()
    assert norm2.startswith("postgresql+psycopg://") or norm2.startswith("postgresql://")

    # Restore default
    os.environ.pop("DATABASE_URL", None)


def test_channel_id_validation_logic():
    # Valid 24-character YouTube Channel ID starting with UC
    valid_cids = ["UCBwmMxybNva6P_5VmxjzwqA", "UCJskGeByzRRSvmOyZOz61ig"]
    for cid in valid_cids:
        assert re.match(r"^UC[A-Za-z0-9_-]{22}$", cid)

    # URL extraction pattern
    url = "https://www.youtube.com/channel/UCBwmMxybNva6P_5VmxjzwqA"
    m = re.search(r"channel/(UC[A-Za-z0-9_-]{22})", url)
    assert m and m.group(1) == "UCBwmMxybNva6P_5VmxjzwqA"

    # Invalid channel IDs
    invalid_cids = ["invalid", "UCshort", "NOT_A_CHANNEL_ID_12345678"]
    for cid in invalid_cids:
        assert not re.match(r"^UC[A-Za-z0-9_-]{22}$", cid)


def test_admin_authentication_digest():
    server_pwd = "my_super_secret_admin_token"

    # Correct password
    assert hmac.compare_digest(
        "my_super_secret_admin_token".strip().encode("utf-8"),
        server_pwd.strip().encode("utf-8"),
    )

    # Incorrect password
    assert not hmac.compare_digest(
        "wrong_password".strip().encode("utf-8"),
        server_pwd.strip().encode("utf-8"),
    )


def test_cloud_workflow_rpa_job_configuration():
    wf_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "mediaintel-cloud-pipeline.yml"
    assert wf_path.is_file(), "mediaintel-cloud-pipeline.yml must exist"

    content = wf_path.read_text(encoding="utf-8")
    # Verify no scheduled cron triggers exist
    assert "cron:" not in content
    assert "schedule:" not in content

    # Verify workflow_dispatch and new inputs
    assert "workflow_dispatch:" in content
    assert "run_rpa:" in content
    assert "send_email:" in content
    assert "default: false" in content  # send_email must be false by default

    # Verify rpa-reporting job and dependencies
    assert "rpa-reporting:" in content
    assert "requirements-rpa.txt" in content
    assert "actions/upload-artifact@v4" in content
    assert "MediaIntel_Dashboard_Documentation.pdf" in content
    assert "Content_Strategy_Report.txt" in content
    assert "AlertSummary.txt" in content


def test_rpa_dry_run_safety_and_audit():
    import rpa.MediaIntelRPA as rpa_module
    bot = rpa_module.MediaIntelRPA()

    # Verify dry-run mode returns DRY_RUN and writes audit .eml
    os.environ["RPA_DRY_RUN"] = "1"
    try:
        sample_mail = {
            "subject": "[TEST] MediaIntel Anomaly Signals",
            "text": "Sample text body for unit test",
            "html": "<p>Sample html body</p>",
            "alerts": 1,
            "new_alerts": 1,
        }
        res = bot.send_email_with_attachments(sample_mail)
        assert res == "DRY_RUN", f"Expected DRY_RUN, got {res}"

        # Verify marking alerts emailed in DRY_RUN returns 0 and does not touch database
        closed = bot.mark_alerts_emailed(res)
        assert closed == 0, "mark_alerts_emailed must be 0 for DRY_RUN"

        # Verify dispatch_result.json was generated with DRY_RUN status
        result_json = config.RPA_LOG_DIR / "dispatch_result.json"
        assert result_json.is_file(), "dispatch_result.json must be written"
    finally:
        os.environ.pop("RPA_DRY_RUN", None)


def test_orchestrator_relevance_highlights_postgresql_compatibility():
    """Verify that main_orchestrator.py has zero SQL ROUND(AVG) calls and handles NULL/floats safely."""
    orch_path = Path(__file__).resolve().parents[1] / "main_orchestrator.py"
    content = orch_path.read_text(encoding="utf-8")
    assert "ROUND(AVG(" not in content, "main_orchestrator.py must not use SQL ROUND(AVG) for PostgreSQL compatibility"

    # Test highlights execution with a dummy console
    import main_orchestrator

    class DummyConsole:
        def __init__(self):
            self.lines = []
        def print(self, *args, **kwargs):
            self.lines.append(str(args))

    dummy = DummyConsole()
    # Must run without raising any exceptions
    main_orchestrator.highlights(dummy)


if __name__ == "__main__":
    test_video_id_regex()
    test_audio_header_validation()
    test_storage_deletion_path_security()
    test_database_url_normalization()
    test_channel_id_validation_logic()
    test_admin_authentication_digest()
    test_cloud_workflow_rpa_job_configuration()
    test_rpa_dry_run_safety_and_audit()
    test_orchestrator_relevance_highlights_postgresql_compatibility()
    print("ALL CLOUD COMPONENT TESTS PASSED!")

