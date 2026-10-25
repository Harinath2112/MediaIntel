"""Unit & Integration Tests for Dashboard Zero Display Fix & Permanent Channel Deletion.

Verifies:
1. Correct video, comment, sentiment, and transcript counts and distinct statuses in coverage ledger.
2. Safe non-destructive deletion preview with exact record counts across all 12 tables.
3. Administrator authentication requirement (rejection of non-admin calls).
4. Exact typed 'DELETE' confirmation requirement.
5. Full dependent-record cleanup across all 12 tables.
6. Atomic transaction rollback on error.
7. Strict isolation: preservation of unrelated channels and shared global topics (scope='all').
8. Concurrency safety: blocking deletion when a cloud workflow is running or audio is locked.
9. Clean re-registration of channel after permanent deletion.
10. Safe Supabase Storage audio cleanup failure reporting and retry mechanism.
11. Idempotent deletion of non-existent or already-deleted channels.
12. Non-destructive tracking disable (disabling tracking never deletes records).
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine, text

from core import channel_manager, config, db_setup, storage_audio


class TestDashboardZeroFixAndChannelDeletion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # In-memory SQLite for deterministic, isolated testing across threads
        from sqlalchemy.pool import StaticPool
        cls.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        db_setup.metadata.create_all(cls.engine)
        cls._orig_get_engine = config.get_engine
        config.get_engine = lambda: cls.engine

    @classmethod
    def tearDownClass(cls):
        config.get_engine = cls._orig_get_engine
        cls.engine.dispose()

    def setUp(self):
        # Clean all tables before each test
        with self.engine.begin() as conn:
            for tbl in [
                "aspect_sentiments", "sentiment_insights", "comments",
                "transcripts", "video_audio_assets", "video_stats_history",
                "video_relevance", "anomaly_alerts", "topic_suggestions",
                "channel_run_coverage", "videos", "channels",
            ]:
                conn.execute(text(f"DELETE FROM {tbl}"))

    def _seed_channel_with_full_records(self, cid: str = "UC_TEST", cname: str = "Test Channel"):
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with self.engine.begin() as conn:
            # 1. Channel
            conn.execute(text("""
                INSERT INTO channels (channel_id, channel_name, is_tracked, created_at)
                VALUES (:cid, :cname, 1, :now)
            """), {"cid": cid, "cname": cname, "now": now})

            # 2. Videos (2 videos)
            v1, v2 = f"vid_1_{cid}", f"vid_2_{cid}"
            for vid, title, views in [(v1, "Video One", 1000), (v2, "Video Two", 2000)]:
                conn.execute(text("""
                    INSERT INTO videos (video_id, channel_id, title, view_count, published_at, fetched_at)
                    VALUES (:vid, :cid, :title, :views, :now, :now)
                """), {"vid": vid, "cid": cid, "title": title, "views": views, "now": now})

            # 3. Comments (2 comments per video = 4 total)
            c1, c2, c3, c4 = f"c1_{cid}", f"c2_{cid}", f"c3_{cid}", f"c4_{cid}"
            for cid_comment, vid, txt in [
                (c1, v1, "Great video!"),
                (c2, v1, "Very informative"),
                (c3, v2, "Awesome explanations"),
                (c4, v2, "Needs more detail"),
            ]:
                conn.execute(text("""
                    INSERT INTO comments (comment_id, video_id, author_name, comment_text, published_at, is_processed)
                    VALUES (:comm_id, :vid, 'User', :txt, :now, 1)
                """), {"comm_id": cid_comment, "vid": vid, "txt": txt, "now": now})

            # 4. Sentiment Insights (4 rows)
            for cid_comment in [c1, c2, c3, c4]:
                conn.execute(text("""
                    INSERT INTO sentiment_insights (comment_id, sentiment_label, sentiment_score, polarity, aspect_category, analyzed_at)
                    VALUES (:comm_id, 'Positive', 0.95, 0.9, 'Content Quality', :now)
                """), {"comm_id": cid_comment, "now": now})

            # 5. Aspect Sentiments (4 rows)
            for idx, cid_comment in enumerate([c1, c2, c3, c4]):
                conn.execute(text("""
                    INSERT INTO aspect_sentiments (comment_id, aspect, sentiment_label, polarity, confidence, evidence)
                    VALUES (:comm_id, :asp, 'Positive', 0.9, 0.88, 'great stuff')
                """), {"comm_id": cid_comment, "asp": f"Teaching_{idx}"})

            # 6. Transcripts (1 for v1)
            conn.execute(text("""
                INSERT INTO transcripts (video_id, source, language, model_name, transcript_text, audio_seconds, word_count, updated_at)
                VALUES (:vid, 'captions', 'en', 'youtube', 'Hello world', 60, 10, :now)
            """), {"vid": v1, "now": now})

            # 7. Video Audio Assets (1 for v2)
            conn.execute(text("""
                INSERT INTO video_audio_assets (video_id, bucket, object_path, status, attempts, created_at, updated_at)
                VALUES (:vid, 'mediaintel-audio', :path, 'pending', 0, :now, :now)
            """), {"vid": v2, "path": f"audio/{cid}/{v2}.mp3", "now": now})

            # 8. Video Stats History (2 rows)
            for vid in [v1, v2]:
                conn.execute(text("""
                    INSERT INTO video_stats_history (video_id, view_count, like_count, comment_count, captured_at)
                    VALUES (:vid, 100, 10, 2, :now)
                """), {"vid": vid, "now": now})

            # 9. Video Relevance (2 rows)
            for vid in [v1, v2]:
                conn.execute(text("""
                    INSERT INTO video_relevance (video_id, relevance_index, comment_alignment, on_topic_share, computed_at)
                    VALUES (:vid, 0.85, 0.8, 0.9, :now)
                """), {"vid": vid, "now": now})

            # 10. Anomaly Alerts (1 row)
            conn.execute(text("""
                INSERT INTO anomaly_alerts (video_id, channel_id, metric, velocity, baseline, ratio, severity, detected_at)
                VALUES (:vid, :cid, 'views', 500, 100, 5.0, 'HIGH', :now)
            """), {"vid": v1, "cid": cid, "now": now})

            # 11. Topic Suggestions (channel-scoped)
            conn.execute(text("""
                INSERT INTO topic_suggestions (scope, rank_no, topic, score, mentions, generated_at)
                VALUES (:cid, 1, 'Python Data Analysis', 0.92, 15, :now)
            """), {"cid": cid, "now": now})

            # 12. Coverage row
            conn.execute(text("""
                INSERT INTO channel_run_coverage (channel_id, channel_name, target_videos, selected_videos, video_status, comment_status, sentiment_status, transcription_status, last_run_status, missing_data_reason, updated_at)
                VALUES (:cid, :cname, 10, 2, 'Partial (2/10)', 'Collected (4)', 'Analyzed (4/4)', 'Audio Pending (1/2)', 'Partial Ingestion', 'Whisper audio pending upload (1/2)', :now)
            """), {"cid": cid, "cname": cname, "now": now})

    def test_01_correct_counts_and_distinct_status_in_coverage_ledger(self):
        """Test 1: Verify correct video & transcript counts and distinct statuses in coverage ledger."""
        self._seed_channel_with_full_records("UC_CB", "Codebasics")
        self._seed_channel_with_full_records("UC_CC", "CrashCourse")

        # Refresh coverage
        channel_manager.update_all_channel_run_coverage(video_limit=2)
        coverage_df = channel_manager.get_channel_run_coverage_df()

        self.assertFalse(coverage_df.empty)
        self.assertEqual(len(coverage_df), 2)

        # Check required columns
        for col in [
            "Channel Name", "Target Videos", "Selected Videos", "Video Status",
            "Target Comments", "Collected Comments", "Comment Status", "Sentiment Status",
            "Transcription Candidates", "Completed Transcripts", "Transcription Status",
            "Last Run Status", "Missing Data Reason"
        ]:
            self.assertIn(col, coverage_df.columns)

        cb_row = coverage_df[coverage_df["Channel Name"] == "Codebasics"].iloc[0]
        self.assertEqual(cb_row["Selected Videos"], 2)
        self.assertEqual(cb_row["Collected Comments"], 4)
        self.assertEqual(cb_row["Completed Transcripts"], 1)
        self.assertEqual(cb_row["Video Status"], "Completed (2/2)")
        self.assertEqual(cb_row["Comment Status"], "Collected (4)")
        self.assertEqual(cb_row["Sentiment Status"], "Analyzed (4/4)")
        self.assertEqual(cb_row["Transcription Status"], "Audio Pending (1/2)")
        # Ingestion status is Complete (NOT Pending Audio!)
        self.assertEqual(cb_row["Last Run Status"], "Ingestion Complete")
        # Audio Pending note is only in transcription status and ledger notes
        self.assertIn("Whisper audio pending upload", cb_row["Missing Data Reason"])

    def test_02_safe_deletion_preview_accuracy(self):
        """Test 2: Non-destructive deletion preview returns exact counts across all 12 tables."""
        self._seed_channel_with_full_records("UC_PREVIEW", "Preview Target")

        preview = channel_manager.preview_channel_deletion("UC_PREVIEW")
        self.assertTrue(preview["exists"])
        self.assertEqual(preview["channel_id"], "UC_PREVIEW")
        self.assertEqual(preview["channel_name"], "Preview Target")
        self.assertTrue(preview["is_tracked"])

        counts = preview["record_counts"]
        self.assertEqual(counts["videos"], 2)
        self.assertEqual(counts["comments"], 4)
        self.assertEqual(counts["sentiment_insights"], 4)
        self.assertEqual(counts["aspect_sentiments"], 4)
        self.assertEqual(counts["transcripts"], 1)
        self.assertEqual(counts["video_audio_assets"], 1)
        self.assertEqual(counts["video_stats_history"], 2)
        self.assertEqual(counts["video_relevance"], 2)
        self.assertEqual(counts["anomaly_alerts"], 1)
        self.assertEqual(counts["topic_suggestions"], 1)
        self.assertEqual(counts["channel_run_coverage"], 1)
        self.assertEqual(counts["channels"], 1)
        self.assertEqual(preview["total_records"], 24)
        self.assertEqual(len(preview["storage_audio_objects"]), 1)
        self.assertIn("audio/UC_PREVIEW/vid_2_UC_PREVIEW.mp3", preview["storage_audio_objects"])

        # Preview is non-destructive
        ch_check = config.read_sql("SELECT COUNT(*) AS cnt FROM channels WHERE channel_id = 'UC_PREVIEW'")
        self.assertEqual(int(ch_check.iloc[0]["cnt"]), 1)

    def test_03_admin_authentication_requirement(self):
        """Test 3: Permanent deletion is blocked if user is not an administrator."""
        self._seed_channel_with_full_records("UC_AUTH", "Auth Target")

        result = channel_manager.delete_channel_permanently(
            "UC_AUTH",
            confirmation="DELETE",
            admin_user=False
        )
        self.assertFalse(result["success"])
        self.assertIn("Administrator privileges are required", result["error"])

        # Verify no data deleted
        ch_check = config.read_sql("SELECT COUNT(*) AS cnt FROM channels WHERE channel_id = 'UC_AUTH'")
        self.assertEqual(int(ch_check.iloc[0]["cnt"]), 1)

    def test_04_typed_deletion_confirmation_requirement(self):
        """Test 4: Permanent deletion requires typing exact uppercase 'DELETE'."""
        self._seed_channel_with_full_records("UC_TYPED", "Typed Target")

        for bad_conf in ["delete", "Delete", "YES", "", "   "]:
            res = channel_manager.delete_channel_permanently("UC_TYPED", confirmation=bad_conf, admin_user=True)
            self.assertFalse(res["success"])
            self.assertIn("type 'DELETE' exactly in uppercase", res["error"])

        # Channel still exists
        ch_check = config.read_sql("SELECT COUNT(*) AS cnt FROM channels WHERE channel_id = 'UC_TYPED'")
        self.assertEqual(int(ch_check.iloc[0]["cnt"]), 1)

    def test_05_full_dependent_record_cleanup(self):
        """Test 5: Successful deletion purges records across all 12 tables."""
        self._seed_channel_with_full_records("UC_PURGE", "Purge Target")

        with patch("core.storage_audio.delete_storage_audio_objects", return_value={"success": True, "deleted_count": 1, "failed_paths": []}):
            res = channel_manager.delete_channel_permanently("UC_PURGE", confirmation="DELETE", admin_user=True)

        self.assertTrue(res["success"])
        self.assertFalse(res.get("already_deleted", False))
        self.assertEqual(res["channel_id"], "UC_PURGE")

        # Verify every table is cleaned
        for tbl, sql in [
            ("channels", "SELECT COUNT(*) AS cnt FROM channels WHERE channel_id = 'UC_PURGE'"),
            ("videos", "SELECT COUNT(*) AS cnt FROM videos WHERE channel_id = 'UC_PURGE'"),
            ("comments", "SELECT COUNT(*) AS cnt FROM comments WHERE video_id LIKE '%UC_PURGE%'"),
            ("sentiment_insights", "SELECT COUNT(*) AS cnt FROM sentiment_insights WHERE comment_id LIKE '%UC_PURGE%'"),
            ("aspect_sentiments", "SELECT COUNT(*) AS cnt FROM aspect_sentiments WHERE comment_id LIKE '%UC_PURGE%'"),
            ("transcripts", "SELECT COUNT(*) AS cnt FROM transcripts WHERE video_id LIKE '%UC_PURGE%'"),
            ("video_audio_assets", "SELECT COUNT(*) AS cnt FROM video_audio_assets WHERE video_id LIKE '%UC_PURGE%'"),
            ("video_stats_history", "SELECT COUNT(*) AS cnt FROM video_stats_history WHERE video_id LIKE '%UC_PURGE%'"),
            ("video_relevance", "SELECT COUNT(*) AS cnt FROM video_relevance WHERE video_id LIKE '%UC_PURGE%'"),
            ("anomaly_alerts", "SELECT COUNT(*) AS cnt FROM anomaly_alerts WHERE channel_id = 'UC_PURGE'"),
            ("topic_suggestions", "SELECT COUNT(*) AS cnt FROM topic_suggestions WHERE scope = 'UC_PURGE'"),
            ("channel_run_coverage", "SELECT COUNT(*) AS cnt FROM channel_run_coverage WHERE channel_id = 'UC_PURGE'"),
        ]:
            df = config.read_sql(sql)
            self.assertEqual(int(df.iloc[0]["cnt"]), 0, f"Table {tbl} still contained records after deletion!")

    def test_06_transaction_rollback_on_error(self):
        """Test 6: Database transaction rolls back completely on any SQL execution error."""
        cid = "UC" + "R" * 22
        self._seed_channel_with_full_records(cid, "Rollback Target")

        # Inject an error into deletion by patching text() to fail on deleting videos
        import core.channel_manager as cm
        orig_text = cm.text

        def failing_text(sql_str):
            if "DELETE FROM videos" in sql_str:
                return orig_text("DELETE FROM non_existent_table_forces_rollback")
            return orig_text(sql_str)

        with patch("core.channel_manager.text", side_effect=failing_text):
            res = cm.delete_channel_permanently(cid, confirmation="DELETE", admin_user=True)

        self.assertFalse(res["success"])
        self.assertIn("Database transaction failed and was completely rolled back", res["error"])

        # Check that transaction rolled back: child records and channel STILL EXIST!
        ch_check = config.read_sql("SELECT COUNT(*) AS cnt FROM channels WHERE channel_id = :cid", {"cid": cid})
        self.assertEqual(int(ch_check.iloc[0]["cnt"]), 1)
        v_check = config.read_sql("SELECT COUNT(*) AS cnt FROM videos WHERE channel_id = :cid", {"cid": cid})
        self.assertEqual(int(v_check.iloc[0]["cnt"]), 2)
        c_check = config.read_sql("SELECT COUNT(*) AS cnt FROM comments WHERE video_id LIKE :pattern", {"pattern": f"%{cid}%"})
        self.assertEqual(int(c_check.iloc[0]["cnt"]), 4)

    def test_07_preservation_of_unrelated_channel_data_and_shared_topics(self):
        """Test 7: Deleting Channel A preserves all of Channel B's data and global shared topics."""
        cid_a = "UC" + "A" * 22
        cid_b = "UC" + "B" * 22
        self._seed_channel_with_full_records(cid_a, "Channel A")
        self._seed_channel_with_full_records(cid_b, "Channel B")

        # Add channel-scoped topics and shared global topic with scope='all'
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with self.engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO topic_suggestions (scope, rank_no, topic, score, mentions, generated_at)
                VALUES ('all', 1, 'Global AI Trends', 0.99, 100, :now)
            """), {"now": now})
            conn.execute(text("""
                INSERT INTO topic_suggestions (scope, rank_no, topic, score, mentions, generated_at)
                VALUES (:cid, 1, 'Channel B Topic', 0.88, 50, :now)
            """), {"cid": cid_b, "now": now})

        # Delete Channel A
        with patch("core.storage_audio.delete_storage_audio_objects", return_value={"success": True, "deleted_count": 1}):
            res = channel_manager.delete_channel_permanently(cid_a, confirmation="DELETE", admin_user=True)
        self.assertTrue(res["success"])

        # Verify Channel A is gone
        ch_a = config.read_sql("SELECT COUNT(*) AS cnt FROM channels WHERE channel_id = :cid", {"cid": cid_a})
        self.assertEqual(int(ch_a.iloc[0]["cnt"]), 0)

        # Verify Channel B is completely intact
        ch_b = config.read_sql("SELECT COUNT(*) AS cnt FROM channels WHERE channel_id = :cid", {"cid": cid_b})
        self.assertEqual(int(ch_b.iloc[0]["cnt"]), 1)
        v_b = config.read_sql("SELECT COUNT(*) AS cnt FROM videos WHERE channel_id = :cid", {"cid": cid_b})
        self.assertEqual(int(v_b.iloc[0]["cnt"]), 2)
        c_b = config.read_sql("SELECT COUNT(*) AS cnt FROM comments WHERE video_id LIKE :pattern", {"pattern": f"%{cid_b}%"})
        self.assertEqual(int(c_b.iloc[0]["cnt"]), 4)

        # Verify Channel B's channel-scoped topics were preserved
        top_b = config.read_sql("SELECT COUNT(*) AS cnt FROM topic_suggestions WHERE scope = :cid", {"cid": cid_b})
        self.assertEqual(int(top_b.iloc[0]["cnt"]), 2)

        # Verify global topic (scope='all') that incorporated Channel A was invalidated
        top_global = config.read_sql("SELECT COUNT(*) AS cnt FROM topic_suggestions WHERE UPPER(scope) = 'ALL'")
        self.assertEqual(int(top_global.iloc[0]["cnt"]), 0)

        # When Channel B (final channel) is deleted, all topics must be wiped
        with patch("core.storage_audio.delete_storage_audio_objects", return_value={"success": True, "deleted_count": 1}):
            res_b = channel_manager.delete_channel_permanently(cid_b, confirmation="DELETE", admin_user=True)
        self.assertTrue(res_b["success"])
        all_topics = config.read_sql("SELECT COUNT(*) AS cnt FROM topic_suggestions")
        self.assertEqual(int(all_topics.iloc[0]["cnt"]), 0)

    def test_08_workflow_concurrency_protection(self):
        """Test 8: Deletion is blocked if a cloud workflow is running or audio is locked."""
        cid = "UC" + "C" * 22
        self._seed_channel_with_full_records(cid, "Concurrency Target")

        with patch("core.channel_manager.check_channel_concurrency", return_value=(True, "Pipeline run #4017 is in_progress")):
            res = channel_manager.delete_channel_permanently(cid, confirmation="DELETE", admin_user=True)

        self.assertFalse(res["success"])
        self.assertIn("concurrency safeguard", res["error"])
        self.assertIn("Pipeline run #4017 is in_progress", res["error"])

        # Data preserved
        ch = config.read_sql("SELECT COUNT(*) AS cnt FROM channels WHERE channel_id = :cid", {"cid": cid})
        self.assertEqual(int(ch.iloc[0]["cnt"]), 1)

    def test_09_reregistration_after_deletion(self):
        """Test 9: Channel can be cleanly re-registered after permanent deletion."""
        valid_cid = "UC" + "1" * 22  # Valid 24-character YouTube Channel ID
        self._seed_channel_with_full_records(valid_cid, "Old Brand Name")

        with patch("core.storage_audio.delete_storage_audio_objects", return_value={"success": True}):
            res = channel_manager.delete_channel_permanently(valid_cid, confirmation="DELETE", admin_user=True)
        self.assertTrue(res["success"])

        # Re-register with new name
        reg_res = channel_manager.register_single_channel(valid_cid, channel_name="New Brand Name", is_tracked=True)
        self.assertTrue(reg_res["success"])

        ch_df = config.read_sql("SELECT channel_name, is_tracked FROM channels WHERE channel_id = :cid", {"cid": valid_cid})
        self.assertEqual(len(ch_df), 1)
        self.assertEqual(ch_df.iloc[0]["channel_name"], "New Brand Name")
        self.assertTrue(bool(ch_df.iloc[0]["is_tracked"]))

    def test_10_storage_cleanup_failure_handling_and_retry(self):
        """Test 10: Relational deletion succeeds and reports storage cleanup failure with retry mechanism."""
        self._seed_channel_with_full_records("UC_STORAGE_FAIL", "Storage Fail Target")

        failed_paths = ["audio/UC_STORAGE_FAIL/vid_2_UC_STORAGE_FAIL.mp3"]
        with patch("core.storage_audio.delete_storage_audio_objects", return_value={
            "success": False,
            "deleted_count": 0,
            "failed_paths": failed_paths,
            "error": "Storage network timeout (HTTP 504)"
        }):
            res = channel_manager.delete_channel_permanently("UC_STORAGE_FAIL", confirmation="DELETE", admin_user=True)

        # Relational deletion committed
        self.assertTrue(res["success"])
        ch_check = config.read_sql("SELECT COUNT(*) AS cnt FROM channels WHERE channel_id = 'UC_STORAGE_FAIL'")
        self.assertEqual(int(ch_check.iloc[0]["cnt"]), 0)

        # Storage cleanup failure reported in payload
        storage_res = res["storage_cleanup"]
        self.assertFalse(storage_res["success"])
        self.assertEqual(storage_res["failed_paths"], failed_paths)
        self.assertIn("Storage network timeout", storage_res["error"])

        # Test retry mechanism
        with patch("core.storage_audio.delete_storage_audio_objects", return_value={
            "success": True,
            "deleted_count": 1,
            "failed_paths": [],
            "error": None
        }):
            retry_res = storage_audio.retry_storage_cleanup("mediaintel-audio", failed_paths)
            self.assertTrue(retry_res["success"])
            self.assertEqual(retry_res["deleted_count"], 1)

    def test_11_idempotent_deletion(self):
        """Test 11: Deleting an already-deleted or non-existent channel is safe and idempotent."""
        res = channel_manager.delete_channel_permanently("UC_NON_EXISTENT", confirmation="DELETE", admin_user=True)
        self.assertTrue(res["success"])
        self.assertTrue(res.get("already_deleted", False))
        self.assertIn("does not exist or has already been deleted", res["message"])

    def test_12_disable_tracking_preserves_all_data(self):
        """Test 12: Disabling tracking NEVER deletes any records."""
        self._seed_channel_with_full_records("UC_DISABLE", "Disable Target")

        # Disable tracking
        ok = channel_manager.set_channel_tracking("UC_DISABLE", False)
        self.assertTrue(ok)

        # Channel is untracked
        ch_df = config.read_sql("SELECT is_tracked FROM channels WHERE channel_id = 'UC_DISABLE'")
        self.assertFalse(bool(ch_df.iloc[0]["is_tracked"]))

        # ALL videos, comments, sentiments, transcripts, etc. still exist!
        v_check = config.read_sql("SELECT COUNT(*) AS cnt FROM videos WHERE channel_id = 'UC_DISABLE'")
        self.assertEqual(int(v_check.iloc[0]["cnt"]), 2)
        c_check = config.read_sql("SELECT COUNT(*) AS cnt FROM comments WHERE video_id LIKE '%UC_DISABLE%'")
        self.assertEqual(int(c_check.iloc[0]["cnt"]), 4)
        s_check = config.read_sql("SELECT COUNT(*) AS cnt FROM sentiment_insights WHERE comment_id LIKE '%UC_DISABLE%'")
        self.assertEqual(int(s_check.iloc[0]["cnt"]), 4)

    def test_13_admin_deletion_preview_attribute_safety(self):
        """Test 13: preview_channel_deletion exists, accepts gh_token, and returns robust preview dict."""
        self._seed_channel_with_full_records("UC_PREV", "Preview Target")

        # 1. Module attribute check
        self.assertTrue(hasattr(channel_manager, "preview_channel_deletion"))
        self.assertTrue(hasattr(channel_manager, "delete_channel_permanently"))
        self.assertTrue(hasattr(channel_manager, "check_channel_concurrency"))

        # 2. Call preview_channel_deletion with gh_token None
        prev_none = channel_manager.preview_channel_deletion("UC_PREV", gh_token=None)
        self.assertTrue(prev_none["exists"])
        self.assertEqual(prev_none["channel_id"], "UC_PREV")
        self.assertEqual(prev_none["channel_name"], "Preview Target")
        self.assertIn("record_counts", prev_none)
        self.assertEqual(prev_none["record_counts"]["videos"], 2)
        self.assertEqual(prev_none["record_counts"]["comments"], 4)
        self.assertGreater(prev_none["total_records"], 0)
        self.assertIn("is_locked", prev_none)

        # 3. Call with gh_token string (kwarg matching dashboard_app.py line 1125)
        prev_tok = channel_manager.preview_channel_deletion("UC_PREV", gh_token="ghp_test_token")
        self.assertTrue(prev_tok["exists"])
        self.assertEqual(prev_tok["record_counts"]["videos"], 2)

        # 4. Safe recovery simulation if module attribute was temporarily missing
        import importlib
        import types
        fake_mod = types.ModuleType("core.channel_manager")
        self.assertFalse(hasattr(fake_mod, "preview_channel_deletion"))
        # Simulating dashboard's dynamic reload protection
        if not hasattr(fake_mod, "preview_channel_deletion"):
            restored = importlib.import_module("core.channel_manager")
            importlib.reload(restored)
            self.assertTrue(hasattr(restored, "preview_channel_deletion"))

    def test_14_admin_dashboard_smoke_test(self):
        """Test 14: Streamlit AppTest verifies admin login path renders without AttributeError or exceptions."""
        import subprocess
        import sys
        root_dir = Path(__file__).resolve().parents[1]
        dashboard_path = root_dir / "dashboard" / "dashboard_app.py"

        probe = (
            "import sys\n"
            "from streamlit.testing.v1 import AppTest\n"
            f"at = AppTest.from_file(r'{dashboard_path}', default_timeout=30)\n"
            "at.session_state['admin_authenticated'] = True\n"
            "at.run()\n"
            "print('EXC', len(at.exception))\n"
            "assert len(at.exception) == 0\n"
            "from core import config\n"
            "config.dispose_engine()\n"
        )
        res = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=root_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(res.returncode, 0, f"Dashboard failed: {res.stderr}")
        self.assertIn("EXC 0", res.stdout)

    def test_15_transcript_count_consistency_and_distinction(self):
        """Test 15: Verify transcript count consistency between coverage ledger, telemetry, and deletion preview.
        
        Reproduces and resolves the discrepancy where a channel with pending/unavailable audio
        has 2 placeholder database rows in transcripts, showing 0/2 in coverage ledger but 2 in deletion preview.
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        # 1. Seed UC_CODEBASICS: 2 videos, but both transcripts are audio_unavailable (0 completed, 2 placeholder rows)
        self._seed_channel_with_full_records("UC_CB_TEST", "Codebasics Replica")
        # Overwrite transcripts for UC_CB_TEST to simulate audio_unavailable
        with self.engine.begin() as conn:
            conn.execute(text("DELETE FROM transcripts WHERE video_id LIKE 'vid_%_UC_CB_TEST'"))
            for v_idx in [1, 2]:
                conn.execute(text("""
                    INSERT INTO transcripts (video_id, source, language, model_name, transcript_text, audio_seconds, word_count, last_error, updated_at)
                    VALUES (:vid, 'audio_unavailable', 'und', 'whisper-tiny', NULL, 0, 0, 'Audio unavailable in Supabase Storage', :now)
                """), {"vid": f"vid_{v_idx}_UC_CB_TEST", "now": now})

        # 2. Seed UC_APNA_TEST: 2 videos, both transcripts completed (whisper)
        self._seed_channel_with_full_records("UC_APNA_TEST", "Apna College Replica")
        with self.engine.begin() as conn:
            conn.execute(text("DELETE FROM transcripts WHERE video_id LIKE 'vid_%_UC_APNA_TEST'"))
            for v_idx in [1, 2]:
                conn.execute(text("""
                    INSERT INTO transcripts (video_id, source, language, model_name, transcript_text, audio_seconds, word_count, updated_at)
                    VALUES (:vid, 'whisper', 'en', 'whisper-tiny', 'Valid completed transcript text', 120, 25, :now)
                """), {"vid": f"vid_{v_idx}_UC_APNA_TEST", "now": now})

        # 3. Update coverage ledger (Active Analysis Set)
        channel_manager.update_all_channel_run_coverage(video_limit=2)
        coverage_df = channel_manager.get_channel_run_coverage_df()

        # Verify Codebasics in Coverage Ledger: 0/2 completed transcripts, status Audio Pending
        cb_cov = coverage_df[coverage_df["Channel Name"] == "Codebasics Replica"].iloc[0]
        self.assertEqual(cb_cov["Transcription Candidates"], 2)
        self.assertEqual(cb_cov["Completed Transcripts"], 0)
        self.assertEqual(cb_cov["Transcription Status"], "Audio Pending (0/2)")

        # Verify Apna in Coverage Ledger: 2/2 completed transcripts, status Completed
        apna_cov = coverage_df[coverage_df["Channel Name"] == "Apna College Replica"].iloc[0]
        self.assertEqual(apna_cov["Transcription Candidates"], 2)
        self.assertEqual(apna_cov["Completed Transcripts"], 2)
        self.assertEqual(apna_cov["Transcription Status"], "Completed (2/2)")

        # 4. Verify All-Time Channel Telemetry
        telemetry_df = channel_manager.get_channel_telemetry_df()
        cb_telem = telemetry_df[telemetry_df["Channel"] == "Codebasics Replica"].iloc[0]
        apna_telem = telemetry_df[telemetry_df["Channel"] == "Apna College Replica"].iloc[0]
        # Telemetry clearly differentiates Whisper, Captions, Total Completed, and Audio Unavailable
        self.assertEqual(cb_telem["Total Completed Transcripts"], 0)
        self.assertEqual(cb_telem["Whisper Transcripts"], 0)
        self.assertEqual(cb_telem["Captions"], 0)
        self.assertEqual(cb_telem["Audio Unavailable"], 2)

        self.assertEqual(apna_telem["Total Completed Transcripts"], 2)
        self.assertEqual(apna_telem["Whisper Transcripts"], 2)
        self.assertEqual(apna_telem["Captions"], 0)
        self.assertEqual(apna_telem["Audio Unavailable"], 0)

        # 5. Verify Permanent Deletion Preview
        cb_prev = channel_manager.preview_channel_deletion("UC_CB_TEST")
        # Physical table rows to be purged must be 2
        self.assertEqual(cb_prev["record_counts"]["transcripts"], 2)
        # Granular completed vs pending breakdown
        self.assertEqual(cb_prev["completed_transcripts"], 0)
        self.assertEqual(cb_prev["pending_audio_records"], 2)
        self.assertEqual(cb_prev["active_set_transcripts"], 0)

        apna_prev = channel_manager.preview_channel_deletion("UC_APNA_TEST")
        self.assertEqual(apna_prev["record_counts"]["transcripts"], 2)
        self.assertEqual(apna_prev["completed_transcripts"], 2)
        self.assertEqual(apna_prev["pending_audio_records"], 0)
        self.assertEqual(apna_prev["active_set_transcripts"], 2)

    def test_16_nlp_pipeline_coverage_refresh(self):
        """Test 16: Verify that coverage ledger synchronizes from Analyzing (0/N) to Analyzed (N/N) after AI analysis."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cid = "UC_NLP_SYNC"
        cname = "NLP Sync Channel"

        with self.engine.begin() as conn:
            conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES (:cid, :cname, TRUE)"),
                         {"cid": cid, "cname": cname})
            for idx in [1, 2]:
                conn.execute(text("""
                    INSERT INTO videos (video_id, channel_id, title, published_at, view_count, like_count, comment_count)
                    VALUES (:vid, :cid, :title, :now, 100, 10, 2)
                """), {"vid": f"vid_{idx}_{cid}", "cid": cid, "title": f"Video {idx}", "now": now})

            for idx in range(1, 5):
                vid = f"vid_1_{cid}" if idx <= 2 else f"vid_2_{cid}"
                conn.execute(text("""
                    INSERT INTO comments (comment_id, video_id, author_name, comment_text, like_count, published_at)
                    VALUES (:cid_comm, :vid, 'Viewer', 'Comment text', 1, :now)
                """), {"cid_comm": f"comm_{idx}_{cid}", "vid": vid, "now": now})

        # Step 1: Coverage right after ingestion (before NLP analysis)
        channel_manager.update_all_channel_run_coverage(video_limit=2)
        cov_df_1 = channel_manager.get_channel_run_coverage_df()
        row_1 = cov_df_1[cov_df_1["Channel Name"] == cname].iloc[0]
        self.assertEqual(row_1["Collected Comments"], 4)
        self.assertEqual(row_1["Sentiment Status"], "Analyzing (0/4)")

        # Step 2: Simulate NLP analysis storing sentiment insights
        with self.engine.begin() as conn:
            for idx in range(1, 5):
                conn.execute(text("""
                    INSERT INTO sentiment_insights (comment_id, sentiment_label, sentiment_score, polarity, aspect_category, analyzed_at)
                    VALUES (:comm_id, 'Positive', 0.95, 0.9, 'Content Quality', :now)
                """), {"comm_id": f"comm_{idx}_{cid}", "now": now})

        # Step 3: Coverage update after NLP analysis completes
        channel_manager.update_all_channel_run_coverage(video_limit=2)
        cov_df_2 = channel_manager.get_channel_run_coverage_df()
        row_2 = cov_df_2[cov_df_2["Channel Name"] == cname].iloc[0]
        self.assertEqual(row_2["Collected Comments"], 4)
        self.assertEqual(row_2["Sentiment Status"], "Analyzed (4/4)")

    def test_17_transcripts_source_and_candidate_consistency(self):
        """Test 17: Verify 2 Whisper + 8 Captions vs Audio Unavailable across telemetry and coverage ledger."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cid_a = "UC_CHAN_A"
        cid_b = "UC_CHAN_B"

        with self.engine.begin() as conn:
            # Channel A: 10 videos, top 2 Whisper, other 8 Captions
            conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES (:cid, 'Channel Alpha', TRUE)"),
                         {"cid": cid_a})
            for i in range(1, 11):
                vid = f"va_{i}"
                views = 1000 - i * 10
                conn.execute(text("""
                    INSERT INTO videos (video_id, channel_id, title, published_at, view_count, like_count, comment_count)
                    VALUES (:vid, :cid, :title, :now, :views, 50, 10)
                """), {"vid": vid, "cid": cid_a, "title": f"Alpha Video {i}", "now": now, "views": views})
                if i <= 2:
                    conn.execute(text("""
                        INSERT INTO transcripts (video_id, source, language, model_name, transcript_text, audio_seconds, word_count)
                        VALUES (:vid, 'whisper', 'en', 'base', 'Whisper spoken audio transcript text', 120, 200)
                    """), {"vid": vid})
                else:
                    conn.execute(text("""
                        INSERT INTO transcripts (video_id, source, language, model_name, transcript_text, audio_seconds, word_count)
                        VALUES (:vid, 'captions', 'en', 'captions', 'Captions transcript text', 120, 200)
                    """), {"vid": vid})

            # Channel B: 10 videos, top 2 audio_unavailable, other 8 no transcripts
            conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES (:cid, 'Channel Beta', TRUE)"),
                         {"cid": cid_b})
            for i in range(1, 11):
                vid = f"vb_{i}"
                views = 500 - i * 10
                conn.execute(text("""
                    INSERT INTO videos (video_id, channel_id, title, published_at, view_count, like_count, comment_count)
                    VALUES (:vid, :cid, :title, :now, :views, 20, 5)
                """), {"vid": vid, "cid": cid_b, "title": f"Beta Video {i}", "now": now, "views": views})
                if i <= 2:
                    conn.execute(text("""
                        INSERT INTO transcripts (video_id, source, language, model_name, transcript_text, audio_seconds, word_count)
                        VALUES (:vid, 'audio_unavailable', 'en', 'none', '[AUDIO_UNAVAILABLE] No audio asset in storage', 0, 0)
                    """), {"vid": vid})

        # 1. Verify All-Time Channel Telemetry
        telem = channel_manager.get_channel_telemetry_df()
        a_tel = telem[telem["Channel"] == "Channel Alpha"].iloc[0]
        b_tel = telem[telem["Channel"] == "Channel Beta"].iloc[0]

        # Channel Alpha: 2 Whisper + 8 Captions = 10 Total Completed Transcripts, 0 Audio Unavailable
        self.assertEqual(a_tel["Videos"], 10)
        self.assertEqual(a_tel["Whisper Transcripts"], 2)
        self.assertEqual(a_tel["Captions"], 8)
        self.assertEqual(a_tel["Total Completed Transcripts"], 10)
        self.assertEqual(a_tel["Audio Unavailable"], 0)

        # Channel Beta: 0 Whisper + 0 Captions = 0 Completed Transcripts, 2 Audio Unavailable
        self.assertEqual(b_tel["Videos"], 10)
        self.assertEqual(b_tel["Whisper Transcripts"], 0)
        self.assertEqual(b_tel["Captions"], 0)
        self.assertEqual(b_tel["Total Completed Transcripts"], 0)
        self.assertEqual(b_tel["Audio Unavailable"], 2)

        # 2. Verify Active Coverage Ledger (evaluates top 2 candidate videos per channel)
        channel_manager.update_all_channel_run_coverage(video_limit=10)
        cov = channel_manager.get_channel_run_coverage_df()
        a_cov = cov[cov["Channel Name"] == "Channel Alpha"].iloc[0]
        b_cov = cov[cov["Channel Name"] == "Channel Beta"].iloc[0]

        self.assertEqual(a_cov["Transcription Candidates"], 2)
        self.assertEqual(a_cov["Completed Transcripts"], 2)
        self.assertEqual(a_cov["Transcription Status"], "Completed (2/2)")

        self.assertEqual(b_cov["Transcription Candidates"], 2)
        self.assertEqual(b_cov["Completed Transcripts"], 0)
        self.assertEqual(b_cov["Transcription Status"], "Audio Pending (0/2)")

        # 3. Verify Permanent Deletion Preview
        a_prev = channel_manager.preview_channel_deletion(cid_a)
        self.assertEqual(a_prev["record_counts"]["transcripts"], 10)
        self.assertEqual(a_prev["completed_transcripts"], 10)
        self.assertEqual(a_prev["active_set_transcripts"], 2)
        self.assertEqual(a_prev["pending_audio_records"], 0)

        b_prev = channel_manager.preview_channel_deletion(cid_b)
        self.assertEqual(b_prev["record_counts"]["transcripts"], 2)
        self.assertEqual(b_prev["completed_transcripts"], 0)
        self.assertEqual(b_prev["active_set_transcripts"], 0)
        self.assertEqual(b_prev["pending_audio_records"], 2)


if __name__ == "__main__":
    unittest.main()


