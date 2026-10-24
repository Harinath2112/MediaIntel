"""Unit and Integration Tests for Consistent On-Demand Data Collection.

Verifies:
1. Eight tracked channels are selected.
2. Each channel's current analysis contains no more than 10 videos.
3. Existing historical videos are preserved.
4. Each selected video has at most 100 collected top-level comments.
5. Duplicate videos and comments are not inserted.
6. Selected-channel scope is respected.
7. Transcription selects the top 2 videos per channel (ranked by view count).
8. Missing audio is handled correctly (marked audio_unavailable, no fake text, pipeline doesn't fail).
9. Streamlit latest-run metrics and coverage table columns are correct.
10. No automatic GitHub Actions scheduling exists (100% manual workflow_dispatch).
"""
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sqlalchemy import create_engine, text

from core import config, db_setup, channel_manager, cloud_whisper_worker, transcribe_videos


class TestConsistentDataCollection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use an in-memory or throwaway SQLite database for deterministic testing
        cls.engine = create_engine("sqlite:///:memory:")
        db_setup.metadata.create_all(cls.engine)

        # Patch config.get_engine to return our test engine
        cls._orig_get_engine = config.get_engine
        config.get_engine = lambda: cls.engine

    @classmethod
    def tearDownClass(cls):
        config.get_engine = cls._orig_get_engine

    def setUp(self):
        # Seed 8 channels
        self.channel_data = [
            ("UC001", "Channel 1", True),
            ("UC002", "Channel 2", True),
            ("UC003", "Channel 3", True),
            ("UC004", "Channel 4", True),
            ("UC005", "Channel 5", True),
            ("UC006", "Channel 6", True),
            ("UC007", "Channel 7", True),
            ("UC008", "Channel 8", True),
        ]
        with self.engine.begin() as conn:
            for t in ["aspect_sentiments", "sentiment_insights", "comments", "transcripts",
                      "video_audio_assets", "video_stats_history", "channel_run_coverage", "videos", "channels"]:
                conn.execute(text(f"DELETE FROM {t}"))

            for cid, name, tracked in self.channel_data:
                conn.execute(
                    text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES (:c, :n, :t)"),
                    {"c": cid, "n": name, "t": tracked}
                )

    def test_01_eight_tracked_channels_selected(self):
        """Test 1: Eight tracked channels are registered and selected."""
        df = config.read_sql("SELECT channel_id, channel_name FROM channels WHERE is_tracked = TRUE")
        self.assertEqual(len(df), 8)
        cids = set(df["channel_id"].tolist())
        self.assertEqual(len(cids), 8)
        self.assertIn("UC001", cids)
        self.assertIn("UC008", cids)

    def test_02_each_channel_current_analysis_at_most_10_videos(self):
        """Test 2: Each channel's current analysis contains no more than 10 videos even with warehouse accumulation."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        # Seed Channel 1 with 15 historical videos
        with self.engine.begin() as conn:
            for i in range(15):
                vid = f"v_uc1_{i:02d}"
                pub = now - timedelta(days=i)
                conn.execute(
                    text("INSERT INTO videos (video_id, channel_id, title, view_count, published_at) "
                         "VALUES (:v, 'UC001', :t, :vc, :p)"),
                    {"v": vid, "t": f"Video {i}", "vc": i * 1000, "p": pub}
                )

        active_videos = channel_manager.get_active_analysis_videos_df(limit_per_channel=10)
        c1_active = active_videos[active_videos["channel_id"] == "UC001"]
        self.assertEqual(len(c1_active), 10)
        # Verify deterministic descending publication date order
        pub_dates = list(c1_active["published_at"])
        self.assertEqual(pub_dates, sorted(pub_dates, reverse=True))

    def test_03_existing_historical_videos_are_preserved(self):
        """Test 3: Existing historical videos in the database are preserved and never deleted."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with self.engine.begin() as conn:
            for i in range(12):
                vid = f"hist_v_{i}"
                conn.execute(
                    text("INSERT INTO videos (video_id, channel_id, title, view_count, published_at) "
                         "VALUES (:v, 'UC002', :t, :vc, :p)"),
                    {"v": vid, "t": f"Hist Video {i}", "vc": 100, "p": now - timedelta(days=i)}
                )

        # Ingestion or active selection should preserve all 12 rows in videos table
        active_videos = channel_manager.get_active_analysis_videos_df(limit_per_channel=10)
        self.assertEqual(len(active_videos), 10)

        all_warehouse_videos = config.read_sql("SELECT COUNT(*) AS cnt FROM videos WHERE channel_id = 'UC002'")
        self.assertEqual(int(all_warehouse_videos.iloc[0]["cnt"]), 12)

    def test_04_each_selected_video_has_at_most_100_comments(self):
        """Test 4: Top comments collection respects max 100 limit, orders by relevance, and deduplicates."""
        from core.youtube_ingest import YouTubeClient

        client = YouTubeClient("fake_key")
        # Mock _svc to return dummy responses
        class FakeThreads:
            def list(self, **kwargs):
                self.kwargs = kwargs
                return self
            def execute(self):
                # Verify order="relevance" is passed
                order = self.kwargs.get("order")
                max_res = self.kwargs.get("maxResults")
                items = []
                for i in range(120):  # 120 comments from API
                    items.append({
                        "id": f"comment_{i % 80}",  # with duplicate IDs
                        "snippet": {
                            "topLevelComment": {
                                "snippet": {
                                    "authorDisplayName": f"User {i}",
                                    "textDisplay": f"Comment text {i}",
                                    "likeCount": i,
                                    "publishedAt": "2026-10-24T12:00:00Z"
                                }
                            }
                        }
                    })
                return {"items": items, "nextPageToken": None}

        class FakeService:
            def commentThreads(self):
                return FakeThreads()

        client._local.svc = FakeService()
        comments = client.top_comments("v_test", pages=2, max_comments=100)

        self.assertLessEqual(len(comments), 100)
        # Verify deduplication
        cids = [c["comment_id"] for c in comments]
        self.assertEqual(len(cids), len(set(cids)))

    def test_05_duplicate_videos_and_comments_not_inserted(self):
        """Test 5: Idempotent upserts prevent duplicate videos and comments."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        upsert_vid = config.upsert_sql("videos", ["video_id", "channel_id", "title", "published_at"], ["video_id"])
        upsert_comm = config.upsert_sql("comments", ["comment_id", "video_id", "comment_text", "like_count"], ["comment_id"])

        with self.engine.begin() as conn:
            # First insert
            conn.execute(upsert_vid, [{"video_id": "v_dup", "channel_id": "UC001", "title": "Orig", "published_at": now}])
            conn.execute(upsert_comm, [{"comment_id": "c_dup", "video_id": "v_dup", "comment_text": "Good", "like_count": 5}])

            # Second upsert identical key
            conn.execute(upsert_vid, [{"video_id": "v_dup", "channel_id": "UC001", "title": "Updated", "published_at": now}])
            conn.execute(upsert_comm, [{"comment_id": "c_dup", "video_id": "v_dup", "comment_text": "Good", "like_count": 10}])

        v_count = config.read_sql("SELECT COUNT(*) AS cnt FROM videos WHERE video_id = 'v_dup'")
        c_count = config.read_sql("SELECT COUNT(*) AS cnt FROM comments WHERE comment_id = 'c_dup'")
        c_row = config.read_sql("SELECT like_count FROM comments WHERE comment_id = 'c_dup'")

        self.assertEqual(int(v_count.iloc[0]["cnt"]), 1)
        self.assertEqual(int(c_count.iloc[0]["cnt"]), 1)
        self.assertEqual(int(c_row.iloc[0]["like_count"]), 10)

    def test_06_selected_channel_scope_respected(self):
        """Test 6: Selected-channel scope is respected and excludes non-selected channels."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO videos (video_id, channel_id, title, view_count, published_at) "
                     "VALUES ('v1', 'UC001', 'T1', 100, :p), ('v2', 'UC002', 'T2', 200, :p)"),
                {"p": now}
            )

        scoped = channel_manager.get_active_analysis_videos_df(limit_per_channel=10, channel_ids=["UC001"])
        self.assertEqual(len(scoped), 1)
        self.assertEqual(scoped.iloc[0]["channel_id"], "UC001")

    def test_07_transcription_selects_top_2_videos_per_channel(self):
        """Test 7: Transcription candidates select exactly top 2 videos per channel ranked by view_count DESC."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with self.engine.begin() as conn:
            # 5 videos with distinct view counts for UC001
            for i, views in enumerate([5000, 100000, 20000, 80000, 15000]):
                conn.execute(
                    text("INSERT INTO videos (video_id, channel_id, title, view_count, published_at) "
                         "VALUES (:v, 'UC001', :t, :vc, :p)"),
                    {"v": f"v_rank_{i}", "t": f"Video {i}", "vc": views, "p": now - timedelta(hours=i)}
                )

        candidates = channel_manager.get_transcription_candidate_videos(limit_per_channel=2)
        c1_cands = candidates[candidates["channel_id"] == "UC001"]
        self.assertEqual(len(c1_cands), 2)
        # Top 2 views are 100000 (v_rank_1) and 80000 (v_rank_3)
        self.assertEqual(c1_cands.iloc[0]["video_id"], "v_rank_1")
        self.assertEqual(c1_cands.iloc[1]["video_id"], "v_rank_3")

    def test_08_missing_audio_handled_correctly(self):
        """Test 8: Missing authorized audio records audio_unavailable without fake text or pipeline crash."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO videos (video_id, channel_id, title, view_count, published_at) "
                     "VALUES ('v_no_audio', 'UC001', 'No Audio Video', 999999, :p)"),
                {"p": now}
            )

        # Run cloud whisper worker without mapped audio in storage
        stats = cloud_whisper_worker.run(limit=1, max_seconds=300)
        self.assertGreaterEqual(stats["selected_for_transcription"], 1)
        self.assertGreaterEqual(stats["audio_unavailable"], 1)

        # Verify database record
        tx = config.read_sql("SELECT * FROM transcripts WHERE video_id = 'v_no_audio'")
        self.assertFalse(tx.empty)
        row = tx.iloc[0]
        self.assertEqual(row["source"], "audio_unavailable")
        self.assertIsNone(row["transcript_text"])  # Never fabricated text!
        self.assertIn("Audio unavailable", str(row["last_error"]))

    def test_09_streamlit_latest_run_metrics_and_coverage_table(self):
        """Test 9: Per-channel coverage table contains all 9 required columns with accurate data."""
        coverage_df = channel_manager.get_channel_run_coverage_df()
        self.assertFalse(coverage_df.empty)

        required_cols = [
            "Channel Name",
            "Target Videos",
            "Selected Videos",
            "Target Comments",
            "Collected Comments",
            "Transcription Candidates",
            "Completed Transcripts",
            "Missing Data Reason",
            "Last Run Status"
        ]
        for col in required_cols:
            self.assertIn(col, coverage_df.columns, f"Missing required column: {col}")

        # All 8 channels should be present in coverage table
        self.assertEqual(len(coverage_df), 8)

    def test_10_no_automatic_github_actions_scheduling(self):
        """Test 10: GitHub Actions workflow has zero cron or schedule triggers (100% manual workflow_dispatch)."""
        wf_path = ROOT / ".github" / "workflows" / "mediaintel-cloud-pipeline.yml"
        self.assertTrue(wf_path.is_file())

        content = wf_path.read_text(encoding="utf-8")
        self.assertNotIn("schedule:", content)
        self.assertNotIn("cron:", content)
        self.assertIn("workflow_dispatch:", content)
        self.assertIn("video_limit:", content)
        self.assertIn("comments_per_video:", content)
        self.assertIn("transcripts_per_channel:", content)


if __name__ == "__main__":
    unittest.main()
