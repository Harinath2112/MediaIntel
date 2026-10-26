"""Automated regression tests for MediaIntel Hybrid Local Transcription Worker.

Covers all required behaviors:
  1. PostgreSQL-only backend validation (rejects SQLite in production mode).
  2. Safe read-only dry-run mode (zero writes, zero audio downloads, zero Whisper).
  3. Deterministic candidate selection (latest 2 videos per tracked channel).
  4. Completed transcript skipping (idempotent cache reuse).
  5. Caption-first processing (extracts captions without calling Whisper).
  6. Faster-Whisper fallback (invoked only when captions are unavailable).
  7. Unavailable audio handling (marks awaiting_authorized_audio without crashing).
  8. Idempotent persistence (no duplicate rows, preserves valid data).
  9. Single-video targeting (--video-id).
 10. Non-existent video handling (safe no-op).
 11. Zero cloud job dispatch (never triggers GitHub Actions).
"""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from core import (
    channel_manager,
    config,
    db_setup,
    local_transcription_worker,
)


class TestLocalTranscriptionWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configure isolated temporary SQLite test database
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="mediaintel_local_worker_test_")
        cls.db_path = Path(cls.temp_dir.name) / "test_local_worker.db"
        cls.engine = create_engine(f"sqlite:///{cls.db_path}")

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()
        cls.temp_dir.cleanup()

    def setUp(self):
        # Patch config engine to use isolated test DB
        self.patcher_engine = patch("core.config.get_engine", return_value=self.engine)
        self.mock_engine = self.patcher_engine.start()

        self.patcher_read_sql = patch("core.config.read_sql", side_effect=self._mock_read_sql)
        self.mock_read_sql = self.patcher_read_sql.start()

        # Rebuild clean database schema
        db_setup.metadata.drop_all(self.engine)
        db_setup.metadata.create_all(self.engine)

    def tearDown(self):
        self.patcher_read_sql.stop()
        self.patcher_engine.stop()

    def _mock_read_sql(self, sql, params=None):
        with self.engine.connect() as conn:
            stmt = text(sql) if isinstance(sql, str) else sql
            return pd.read_sql(stmt, conn, params=params or {})

    # ------------------------------------------------------------------------
    # Test 1: PostgreSQL-only backend validation
    # ------------------------------------------------------------------------
    def test_01_postgresql_validation_rejects_sqlite_in_production(self):
        # Default SQLite URL without allow_sqlite must raise RuntimeError
        with patch("core.config.database_url", return_value="sqlite:///data/mediaintel.db"):
            with self.assertRaises(RuntimeError) as ctx:
                local_transcription_worker.validate_database_backend(allow_sqlite=False)
            self.assertIn("requires a PostgreSQL database", str(ctx.exception))
            self.assertIn("sqlite", str(ctx.exception).lower())

        # When allow_sqlite=True, it should succeed
        with patch("core.config.database_url", return_value="sqlite:///data/mediaintel.db"):
            dialect = local_transcription_worker.validate_database_backend(allow_sqlite=True)
            self.assertEqual(dialect, "sqlite")

        # When URL is PostgreSQL, validation passes normally
        pg_url = "postgresql+psycopg://postgres:secret@aws-0-eu-west-3.pooler.supabase.com:6543/postgres"
        with patch("core.config.database_url", return_value=pg_url):
            dialect = local_transcription_worker.validate_database_backend(allow_sqlite=False)
            self.assertEqual(dialect, "postgresql")

    # ------------------------------------------------------------------------
    # Test 2: Safe read-only dry-run mode
    # ------------------------------------------------------------------------
    def test_02_dry_run_mode_performs_zero_writes(self):
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES ('UC001', 'Test Channel', 1)")
            )
            conn.execute(
                text("""
                    INSERT INTO videos (video_id, channel_id, title, published_at, view_count)
                    VALUES ('VID_DRY_01', 'UC001', 'Dry Run Video', '2026-10-25 12:00:00', 100)
                """)
            )

        with patch("core.local_transcription_worker.extract_youtube_captions") as mock_captions, \
             patch("core.local_transcription_worker.transcribe_audio_pcm") as mock_whisper:

            res = local_transcription_worker.run_local_worker(
                dry_run=True,
                allow_sqlite=True,
                limit=2,
            )

            self.assertTrue(res["success"])
            self.assertTrue(res["dry_run"])
            self.assertEqual(res["target_videos"], 1)
            self.assertEqual(res["pending_transcriptions"], 1)

            # Assert no audio downloads or Whisper inference occurred
            mock_captions.assert_not_called()
            mock_whisper.assert_not_called()

            # Assert database has zero transcripts written
            with self.engine.connect() as conn:
                count = conn.execute(text("SELECT COUNT(*) FROM transcripts")).scalar()
                self.assertEqual(count, 0)

    # ------------------------------------------------------------------------
    # Test 3: Candidate selection (latest 2 videos per tracked channel)
    # ------------------------------------------------------------------------
    def test_03_candidate_selection_latest_two_per_channel(self):
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES ('UC_A', 'Channel A', 1)")
            )
            conn.execute(
                text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES ('UC_B', 'Channel B', 1)")
            )
            conn.execute(
                text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES ('UC_UNTRACKED', 'Untracked', 0)")
            )

            # Channel A: 3 videos (dates: 2026-10-20, 2026-10-22, 2026-10-25)
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title, published_at) VALUES ('VID_A_OLD', 'UC_A', 'Old A', '2026-10-20 00:00:00')"))
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title, published_at) VALUES ('VID_A_MID', 'UC_A', 'Mid A', '2026-10-22 00:00:00')"))
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title, published_at) VALUES ('VID_A_NEW', 'UC_A', 'New A', '2026-10-25 00:00:00')"))

            # Channel B: 2 videos
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title, published_at) VALUES ('VID_B_1', 'UC_B', 'Vid B1', '2026-10-21 00:00:00')"))
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title, published_at) VALUES ('VID_B_2', 'UC_B', 'Vid B2', '2026-10-24 00:00:00')"))

            # Untracked channel video: must not be selected
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title, published_at) VALUES ('VID_IGNORE', 'UC_UNTRACKED', 'Ignored', '2026-10-26 00:00:00')"))

        candidates, tracked_count, shortfalls = local_transcription_worker.get_candidate_videos(limit_per_channel=2)
        self.assertEqual(tracked_count, 2)
        self.assertEqual(len(candidates), 4)

        selected_vids = set(candidates["video_id"].tolist())
        # For Channel A, latest two are VID_A_NEW and VID_A_MID. VID_A_OLD must not be present.
        self.assertIn("VID_A_NEW", selected_vids)
        self.assertIn("VID_A_MID", selected_vids)
        self.assertNotIn("VID_A_OLD", selected_vids)

        # For Channel B, both should be present
        self.assertIn("VID_B_1", selected_vids)
        self.assertIn("VID_B_2", selected_vids)
        self.assertNotIn("VID_IGNORE", selected_vids)

    # ------------------------------------------------------------------------
    # Test 4: Completed transcript skipping (cache reuse)
    # ------------------------------------------------------------------------
    def test_04_completed_transcripts_are_skipped(self):
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES ('UC001', 'Channel 1', 1)")
            )
            conn.execute(
                text("INSERT INTO videos (video_id, channel_id, title, published_at) VALUES ('VID_COMPLETED', 'UC001', 'Done', '2026-10-24 00:00:00')"))
            conn.execute(
                text("""
                    INSERT INTO transcripts (video_id, channel_id, source, status, transcript_text, language)
                    VALUES ('VID_COMPLETED', 'UC001', 'youtube_auto_caption', 'completed', 'Existing completed transcript text', 'en')
                """)
            )

        with patch("core.local_transcription_worker.extract_youtube_captions") as mock_captions:
            res = local_transcription_worker.run_local_worker(
                allow_sqlite=True,
                limit=2,
            )
            self.assertEqual(res["target_videos"], 1)
            self.assertEqual(res["reused_existing"], 1)
            mock_captions.assert_not_called()

            # Transcript text was not overwritten
            with self.engine.connect() as conn:
                body = conn.execute(text("SELECT transcript_text FROM transcripts WHERE video_id = 'VID_COMPLETED'")).scalar()
                self.assertEqual(body, "Existing completed transcript text")

    # ------------------------------------------------------------------------
    # Test 5: Caption-first processing (Priority 2)
    # ------------------------------------------------------------------------
    def test_05_caption_first_processing_saves_without_whisper(self):
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES ('UC001', 'Channel 1', 1)")
            )
            conn.execute(
                text("INSERT INTO videos (video_id, channel_id, title, published_at) VALUES ('VID_CAP', 'UC001', 'Cap Video', '2026-10-25 00:00:00')")
            )

        fake_caption = {
            "text": "This is an authentic caption transcript extracted from YouTube.",
            "language": "te",
            "source": "youtube_manual_caption",
            "caption_type": "manual",
            "duration_sec": 120,
            "word_count": 9,
        }

        with patch("core.local_transcription_worker.extract_youtube_captions", return_value=fake_caption) as mock_caps, \
             patch("core.local_transcription_worker.transcribe_audio_pcm") as mock_whisper:

            res = local_transcription_worker.run_local_worker(allow_sqlite=True, limit=2)
            self.assertEqual(res["manual_captions_extracted"], 1)
            self.assertEqual(res["faster_whisper_completed"], 0)
            mock_caps.assert_called_once_with("VID_CAP")
            mock_whisper.assert_not_called()

            with self.engine.connect() as conn:
                row = conn.execute(text("SELECT source, status, language, caption_type, transcript_text FROM transcripts WHERE video_id = 'VID_CAP'")).fetchone()
                self.assertEqual(row[0], "youtube_manual_caption")
                self.assertEqual(row[1], "completed")
                self.assertEqual(row[2], "te")
                self.assertEqual(row[3], "manual")
                self.assertEqual(row[4], fake_caption["text"])

    # ------------------------------------------------------------------------
    # Test 6: Faster-Whisper fallback (Priority 3)
    # ------------------------------------------------------------------------
    def test_06_faster_whisper_fallback_when_captions_absent(self):
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES ('UC001', 'Channel 1', 1)")
            )
            conn.execute(
                text("INSERT INTO videos (video_id, channel_id, title, published_at) VALUES ('VID_WHISPER', 'UC001', 'Whisper Video', '2026-10-25 00:00:00')")
            )

        fake_pcm = np.zeros(16000 * 5, dtype=np.int16).tobytes()

        with patch("core.local_transcription_worker.extract_youtube_captions", return_value=None), \
             patch("core.transcribe_videos.fetch_audio", return_value=fake_pcm), \
             patch("core.local_transcription_worker.transcribe_audio_pcm", return_value=("Transcribed by Whisper CPU", "hi", 5, False)):

            res = local_transcription_worker.run_local_worker(allow_sqlite=True, limit=2)
            self.assertEqual(res["auto_captions_extracted"], 0)
            self.assertEqual(res["faster_whisper_completed"], 1)

            with self.engine.connect() as conn:
                row = conn.execute(text("SELECT source, status, language, model_name, transcript_text FROM transcripts WHERE video_id = 'VID_WHISPER'")).fetchone()
                self.assertEqual(row[0], "faster_whisper")
                self.assertEqual(row[1], "completed")
                self.assertEqual(row[2], "hi")
                self.assertEqual(row[3], "whisper-tiny")
                self.assertEqual(row[4], "Transcribed by Whisper CPU")

    # ------------------------------------------------------------------------
    # Test 7: Unavailable audio handling (Priority 4)
    # ------------------------------------------------------------------------
    def test_07_audio_download_failure_marks_awaiting_authorized_audio(self):
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES ('UC001', 'Channel 1', 1)")
            )
            conn.execute(
                text("INSERT INTO videos (video_id, channel_id, title, published_at) VALUES ('VID_BLOCKED', 'UC001', 'Blocked Video', '2026-10-25 00:00:00')")
            )

        with patch("core.local_transcription_worker.extract_youtube_captions", return_value=None), \
             patch("core.transcribe_videos.fetch_audio", side_effect=RuntimeError("Private video: audio unavailable")):

            res = local_transcription_worker.run_local_worker(allow_sqlite=True, limit=2)
            self.assertEqual(res["awaiting_authorized_audio"], 1)
            self.assertEqual(res["faster_whisper_completed"], 0)

            with self.engine.connect() as conn:
                row = conn.execute(text("SELECT source, status, last_error FROM transcripts WHERE video_id = 'VID_BLOCKED'")).fetchone()
                self.assertEqual(row[0], "awaiting_authorized_audio")
                self.assertEqual(row[1], "awaiting_authorized_audio")
                self.assertIn("Audio download unavailable", row[2])

    # ------------------------------------------------------------------------
    # Test 8: Idempotent transcript persistence
    # ------------------------------------------------------------------------
    def test_08_idempotent_persistence_no_duplicate_rows(self):
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES ('UC001', 'Channel 1', 1)")
            )
            conn.execute(
                text("INSERT INTO videos (video_id, channel_id, title, published_at) VALUES ('VID_IDEMP', 'UC001', 'Idemp Video', '2026-10-25 00:00:00')")
            )

        fake_caption = {
            "text": "First pass transcript.",
            "language": "en",
            "source": "youtube_auto_caption",
            "caption_type": "auto_generated",
            "duration_sec": 60,
            "word_count": 3,
        }

        with patch("core.local_transcription_worker.extract_youtube_captions", return_value=fake_caption):
            # Pass 1: Transcribes and saves
            res1 = local_transcription_worker.run_local_worker(allow_sqlite=True, limit=2)
            self.assertEqual(res1["auto_captions_extracted"], 1)

            # Pass 2: Re-runs; should detect completed transcript and skip
            res2 = local_transcription_worker.run_local_worker(allow_sqlite=True, limit=2)
            self.assertEqual(res2["reused_existing"], 1)
            self.assertEqual(res2["auto_captions_extracted"], 0)

            with self.engine.connect() as conn:
                count = conn.execute(text("SELECT COUNT(*) FROM transcripts WHERE video_id = 'VID_IDEMP'")).scalar()
                self.assertEqual(count, 1)

    # ------------------------------------------------------------------------
    # Test 9: Single-video targeting (--video-id)
    # ------------------------------------------------------------------------
    def test_09_single_video_targeting(self):
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES ('UC001', 'Channel 1', 1)")
            )
            conn.execute(
                text("INSERT INTO videos (video_id, channel_id, title, published_at) VALUES ('VID_TARGET', 'UC001', 'Target Video', '2026-10-25 00:00:00')")
            )
            conn.execute(
                text("INSERT INTO videos (video_id, channel_id, title, published_at) VALUES ('VID_OTHER', 'UC001', 'Other Video', '2026-10-25 00:00:00')")
            )

        fake_caption = {
            "text": "Target video transcript.",
            "language": "en",
            "source": "youtube_auto_caption",
            "caption_type": "auto_generated",
            "duration_sec": 45,
            "word_count": 3,
        }

        with patch("core.local_transcription_worker.extract_youtube_captions", return_value=fake_caption) as mock_caps:
            res = local_transcription_worker.run_local_worker(
                video_id="VID_TARGET",
                allow_sqlite=True,
            )
            self.assertEqual(res["target_videos"], 1)
            self.assertEqual(res["auto_captions_extracted"], 1)
            mock_caps.assert_called_once_with("VID_TARGET")

            with self.engine.connect() as conn:
                vids = conn.execute(text("SELECT video_id FROM transcripts")).fetchall()
                self.assertEqual(len(vids), 1)
                self.assertEqual(vids[0][0], "VID_TARGET")

    # ------------------------------------------------------------------------
    # Test 10: Non-existent video handling
    # ------------------------------------------------------------------------
    def test_10_non_existent_video_handling(self):
        res = local_transcription_worker.run_local_worker(
            video_id="VID_DOES_NOT_EXIST",
            allow_sqlite=True,
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["target_videos"], 0)

        with self.engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM transcripts")).scalar()
            self.assertEqual(count, 0)

    # ------------------------------------------------------------------------
    # Test 11: Zero cloud job dispatch (never triggers GitHub Actions)
    # ------------------------------------------------------------------------
    def test_11_zero_cloud_job_dispatch(self):
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES ('UC001', 'Channel 1', 1)")
            )
            conn.execute(
                text("INSERT INTO videos (video_id, channel_id, title, published_at) VALUES ('VID_LOCAL', 'UC001', 'Local Only', '2026-10-25 00:00:00')")
            )

        fake_caption = {
            "text": "Local captions only.",
            "language": "en",
            "source": "youtube_auto_caption",
            "caption_type": "auto_generated",
            "duration_sec": 30,
            "word_count": 3,
        }

        with patch("core.local_transcription_worker.extract_youtube_captions", return_value=fake_caption), \
             patch("requests.post") as mock_http_post, \
             patch("core.channel_manager.dispatch_cloud_pipeline") as mock_dispatch:

            res = local_transcription_worker.run_local_worker(allow_sqlite=True, limit=1)
            self.assertTrue(res["success"])
            mock_http_post.assert_not_called()
            mock_dispatch.assert_not_called()

    # ------------------------------------------------------------------------
    # Test 12: Actionable setup guidance and .env loading
    # ------------------------------------------------------------------------
    def test_12_actionable_guidance_when_postgresql_missing(self):
        with patch("core.config.database_url", return_value="sqlite:///data/mediaintel.db"):
            with self.assertRaises(RuntimeError) as ctx:
                local_transcription_worker.validate_database_backend(allow_sqlite=False)
            err = str(ctx.exception)
            self.assertIn("Action Required:", err)
            self.assertIn("MEDIAINTEL_DB_URL=", err)
            self.assertIn("pooler.supabase.com", err)

    # ------------------------------------------------------------------------
    # Test 13: PostgreSQL psycopg driver compatibility
    # ------------------------------------------------------------------------
    def test_13_postgresql_driver_compatibility(self):
        pg_url = "postgresql://postgres.test:mypassword@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"
        with patch.dict("os.environ", {"MEDIAINTEL_DB_URL": pg_url}):
            normalized = config.database_url()
            self.assertTrue(
                normalized.startswith("postgresql+psycopg://") or normalized.startswith("postgresql+psycopg2://"),
                f"Expected postgresql+psycopg or postgresql+psycopg2, got {normalized}"
            )
            # Verify SQLAlchemy make_url recognizes postgresql dialect
            from sqlalchemy.engine import make_url
            url_obj = make_url(normalized)
            self.assertEqual(url_obj.get_backend_name(), "postgresql")


if __name__ == "__main__":
    unittest.main()
