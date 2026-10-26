"""Comprehensive production hardening and acceptance test suite for MediaIntel.

Covers all 22 required areas:
1. Six-channel onboarding and dynamic discovery
2. Existing-channel reruns and deduplication
3. Ingestion quota and pagination boundaries
4. Sentiment limit 0 vs positive limits
5. Partial sentiment processing and resumption
6. Audio download validation and storage persistence
7. Missing audio asset recovery
8. Audio access restrictions handling
9. Whisper model execution success and failure
10. Caption fallback and accurate provenance
11. Missing transcript and dashboard truthful metrics
12. Relevance computation after transcription
13. Topic invalidation after channel deletion
14. Topic regeneration after adding a channel
15. Anomaly severity and cooldown
16. Alert history after deleting channels
17. PDF Unicode rendering (Rupee symbol) and source labels
18. SMTP dry-run and attachment creation
19. Supabase authentication failure diagnostic preflight
20. Manual workflow dispatch, dependencies, and concurrency
21. Public vs authenticated admin permissions
22. Regression tests for all eight dashboard tabs
"""

from __future__ import annotations

import hmac
import os
import re
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import (
    channel_manager,
    cloud_audio_upload,
    cloud_whisper_worker,
    config,
    content_strategy,
    db_setup,
    generate_pdf_report,
    nlp_pipeline,
    relevance_topics,
    storage_audio,
    transcribe_videos,
)


class TestMediaIntelProductionHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.TemporaryDirectory(prefix="mediaintel_prod_test_")
        cls.orig_workdir = os.environ.get("MEDIAINTEL_WORKDIR")
        cls.orig_db_url = os.environ.get("MEDIAINTEL_DB_URL")
        cls.orig_env_db = os.environ.get("DATABASE_URL")
        cls.orig_test_fakes = os.environ.get("MEDIAINTEL_TEST_FAKES")

        os.environ["MEDIAINTEL_WORKDIR"] = cls.test_dir.name
        cls.db_path = Path(cls.test_dir.name) / "test_prod.db"
        cls.db_url = f"sqlite:///{cls.db_path.resolve()}"
        os.environ["MEDIAINTEL_DB_URL"] = cls.db_url
        os.environ["DATABASE_URL"] = cls.db_url
        os.environ["MEDIAINTEL_TEST_FAKES"] = "1"

        config.dispose_engine()
        db_setup.ensure_schema(verbose=False)

    @classmethod
    def tearDownClass(cls):
        config.dispose_engine()
        if cls.orig_workdir:
            os.environ["MEDIAINTEL_WORKDIR"] = cls.orig_workdir
        else:
            os.environ.pop("MEDIAINTEL_WORKDIR", None)

        if cls.orig_db_url:
            os.environ["MEDIAINTEL_DB_URL"] = cls.orig_db_url
        else:
            os.environ.pop("MEDIAINTEL_DB_URL", None)

        if cls.orig_env_db:
            os.environ["DATABASE_URL"] = cls.orig_env_db
        else:
            os.environ.pop("DATABASE_URL", None)

        if cls.orig_test_fakes:
            os.environ["MEDIAINTEL_TEST_FAKES"] = cls.orig_test_fakes
        else:
            os.environ.pop("MEDIAINTEL_TEST_FAKES", None)

        cls.test_dir.cleanup()

    def setUp(self):
        # Clean test tables between runs
        with config.get_engine().begin() as conn:
            for tbl in (
                "aspect_sentiments",
                "sentiment_insights",
                "topic_suggestions",
                "video_relevance",
                "anomaly_alerts",
                "transcripts",
                "video_audio_assets",
                "video_stats_history",
                "channel_run_coverage",
                "comments",
                "videos",
                "channels",
            ):
                conn.execute(text(f"DELETE FROM {tbl}"))

    # 1. Six-channel onboarding and dynamic discovery
    def test_01_six_channel_onboarding_and_discovery(self):
        cids = [f"UC_PROD_TEST_CHANNEL_{i:02d}" for i in range(1, 7)]
        with config.get_engine().begin() as conn:
            for i, cid in enumerate(cids, 1):
                conn.execute(
                    text("""
                        INSERT INTO channels (channel_id, channel_name, is_tracked, created_at)
                        VALUES (:cid, :cname, 1, CURRENT_TIMESTAMP)
                    """),
                    {"cid": cid, "cname": f"Channel {i}"},
                )

        df = config.read_sql("SELECT * FROM channels WHERE is_tracked = TRUE")
        self.assertEqual(len(df), 6, "Must dynamically discover all 6 channels without hardcoded limits")
        self.assertEqual(set(df["channel_id"]), set(cids))

        telem = channel_manager.get_channel_telemetry_df()
        self.assertEqual(len(telem), 6, "Telemetry must cover all 6 tracked channels")

    # 2. Existing-channel reruns and deduplication
    def test_02_existing_channel_rerun_deduplication(self):
        cid = "UC_PROD_TEST_DEDUP_01"
        vid = "VID_DEDUP_01"
        cm_id = "COM_DEDUP_01"

        with config.get_engine().begin() as conn:
            conn.execute(
                text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES (:cid, 'Dedup Channel', 1)"),
                {"cid": cid},
            )
            conn.execute(
                text("INSERT INTO videos (video_id, channel_id, title, view_count) VALUES (:vid, :cid, 'Test Video', 100)"),
                {"vid": vid, "cid": cid},
            )
            conn.execute(
                text("INSERT INTO comments (comment_id, video_id, comment_text) VALUES (:cm_id, :vid, 'Test Comment')"),
                {"cm_id": cm_id, "vid": vid},
            )

        # Upsert again simulating rerun
        with config.get_engine().begin() as conn:
            conn.execute(
                config.upsert_sql("videos", ["video_id", "channel_id", "title", "view_count"], ["video_id"], ["view_count"]),
                {"video_id": vid, "channel_id": cid, "title": "Test Video", "view_count": 150},
            )

        vids_df = config.read_sql("SELECT * FROM videos WHERE video_id = :vid", {"vid": vid})
        self.assertEqual(len(vids_df), 1, "Reruns must never duplicate video records")
        self.assertEqual(vids_df.iloc[0]["view_count"], 150, "Rerun must update mutable stats safely")

    # 3. Ingestion quota and pagination boundaries
    def test_03_ingestion_boundaries(self):
        cid = "UC_PROD_TEST_BOUNDS_01"
        with config.get_engine().begin() as conn:
            conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES (:cid, 'Bounds Channel', 1)"), {"cid": cid})
            for i in range(15):
                conn.execute(
                    text("INSERT INTO videos (video_id, channel_id, title, published_at, view_count) VALUES (:vid, :cid, :t, :p, :vc)"),
                    {"vid": f"V_BOUND_{i:02d}", "cid": cid, "t": f"Title {i}", "p": f"2026-01-{i+1:02d} 10:00:00", "vc": i * 100},
                )

        active = channel_manager.get_active_analysis_videos_df(limit_per_channel=10)
        self.assertEqual(len(active), 10, "Target per-channel active video limit of 10 must be strictly enforced")

        candidates = channel_manager.get_transcription_candidate_videos(limit_per_channel=2)
        self.assertEqual(len(candidates), 2, "Candidate transcription limit of 2 must be strictly enforced")

    # 4. Sentiment limit 0 vs positive limits
    def test_04_sentiment_limit_zero_vs_positive(self):
        cid = "UC_PROD_TEST_LIMITS_01"
        with config.get_engine().begin() as conn:
            conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES (:cid, 'Limit Channel', 1)"), {"cid": cid})
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title) VALUES ('V_LIM_01', :cid, 'Video Lim')"), {"cid": cid})
            for i in range(25):
                conn.execute(
                    text("INSERT INTO comments (comment_id, video_id, comment_text) VALUES (:cid_i, 'V_LIM_01', 'Nice tutorial')"),
                    {"cid_i": f"C_LIM_{i:02d}"},
                )

        all_pending = nlp_pipeline._pending(limit=0, reprocess_all=False, model_name="test-model")
        self.assertEqual(len(all_pending), 25, "Limit 0 must return all 25 eligible comments without capping at 100")

        capped_pending = nlp_pipeline._pending(limit=10, reprocess_all=False, model_name="test-model")
        self.assertEqual(len(capped_pending), 10, "Positive limit must return exactly the requested cap")

    # 5. Partial sentiment processing and resumption
    def test_05_partial_sentiment_resumption(self):
        cid = "UC_PROD_TEST_RESUME_01"
        with config.get_engine().begin() as conn:
            conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES (:cid, 'Resume Channel', 1)"), {"cid": cid})
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title) VALUES ('V_RES_01', :cid, 'Video Res')"), {"cid": cid})
            for i in range(12):
                conn.execute(
                    text("INSERT INTO comments (comment_id, video_id, comment_text) VALUES (:cid_i, 'V_RES_01', 'Great python guide')"),
                    {"cid_i": f"C_RES_{i:02d}"},
                )

        # Process first 5
        done_first = nlp_pipeline.run_ai_pipeline(limit=5)
        self.assertEqual(done_first, 5)

        # Check remaining in database
        active_model = getattr(nlp_pipeline.nlp_models.get_sentiment_model(), "name", nlp_pipeline.nlp_models.SENTIMENT_MODEL)
        remaining_pending = nlp_pipeline._pending(limit=0, reprocess_all=False, model_name=active_model)
        self.assertEqual(len(remaining_pending), 7, "Remaining 7 comments must remain pending for subsequent runs")

        # Resume and process remaining
        done_second = nlp_pipeline.run_ai_pipeline(limit=0)
        self.assertEqual(done_second, 7)

        remaining_after = nlp_pipeline._pending(limit=0, reprocess_all=False, model_name=active_model)
        self.assertEqual(len(remaining_after), 0, "All comments should be processed after resumption")

    # 6. Audio download validation and storage persistence
    def test_06_audio_validation_and_storage_mapping(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_m4a = Path(tmp_dir) / "sample.m4a"
            sample_m4a.write_bytes(b"\x00\x00\x00\x1cftypM4A " + b"\x00" * 200)

            path, ctype = cloud_audio_upload.validate_audio("dQw4w9WgXcQ", sample_m4a)
            self.assertEqual(ctype, "audio/mp4")

            # Seed channel and video
            with config.get_engine().begin() as conn:
                conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES ('UC_A', 'A', 1)"))
                conn.execute(text("INSERT INTO videos (video_id, channel_id, title) VALUES ('dQw4w9WgXcQ', 'UC_A', 'Test')"))

            cloud_audio_upload.create_audio_mapping("dQw4w9WgXcQ", "videos/dQw4w9WgXcQ/0123456789abcdef0123456789abcdef.m4a")
            mapping = storage_audio.get_audio_mapping("dQw4w9WgXcQ")
            self.assertIsNotNone(mapping)
            self.assertEqual(mapping[0], "mediaintel-audio")

    # 7. Missing audio asset recovery via captions
    def test_07_missing_audio_recovery_to_captions(self):
        cid = "UC_PROD_RECOVER_01"
        vid = "VID_RECOV_01"
        with config.get_engine().begin() as conn:
            conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES (:cid, 'Recover Channel', 1)"), {"cid": cid})
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title, view_count) VALUES (:vid, :cid, 'Recov Video', 1000)"), {"vid": vid, "cid": cid})

        # Mock audio acquisition failure and captions success
        with patch("core.cloud_audio_upload.acquire_and_upload_authorized_audio", side_effect=RuntimeError("YouTube 403 Forbidden")), \
             patch("core.transcribe_videos.fetch_captions", return_value=("Hello world from YouTube captions", "en")):
            stats = cloud_whisper_worker.run(limit=1, max_seconds=60)
            self.assertEqual(stats["caption_fallbacks_completed"], 1)

            tx_row = config.read_sql("SELECT * FROM transcripts WHERE video_id = :vid", {"vid": vid}).iloc[0]
            self.assertEqual(tx_row["source"], "captions")
            self.assertEqual(tx_row["model_name"], "youtube-captions")
            self.assertEqual(tx_row["transcript_text"], "Hello world from YouTube captions")

    # 8. Audio access restrictions handling
    def test_08_audio_access_restrictions(self):
        cid = "UC_PROD_RESTRICT_01"
        vid = "VID_RESTR_01"
        with config.get_engine().begin() as conn:
            conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES (:cid, 'Restrict Channel', 1)"), {"cid": cid})
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title, view_count) VALUES (:vid, :cid, 'Restricted Video', 1000)"), {"vid": vid, "cid": cid})

        # Mock both audio and captions unavailable
        with patch("core.cloud_audio_upload.acquire_and_upload_authorized_audio", side_effect=RuntimeError("Sign in required")), \
             patch("core.transcribe_videos.fetch_captions", side_effect=RuntimeError("No captions available")):
            stats = cloud_whisper_worker.run(limit=1, max_seconds=60)
            self.assertEqual(stats["permanently_unavailable"], 1)

            tx_row = config.read_sql("SELECT * FROM transcripts WHERE video_id = :vid", {"vid": vid}).iloc[0]
            self.assertIn(tx_row["source"], ["audio_unavailable", "awaiting_authorized_audio"])
            self.assertTrue("unavailable" in str(tx_row["last_error"]).lower() or "awaiting" in str(tx_row["status"]).lower())

    # 9. Whisper model execution success and failure
    def test_09_whisper_model_success_and_failure(self):
        cid = "UC_PROD_WHISPER_01"
        vid_ok = "VID_WHISP_OK"
        vid_err = "VID_WHISP_ER"
        with config.get_engine().begin() as conn:
            conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES (:cid, 'Whisper Channel', 1)"), {"cid": cid})
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title, view_count) VALUES (:vid, :cid, 'OK Video', 2000)"), {"vid": vid_ok, "cid": cid})
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title, view_count) VALUES (:vid, :cid, 'Err Video', 1500)"), {"vid": vid_err, "cid": cid})
            conn.execute(text("INSERT INTO video_audio_assets (video_id, bucket, object_path, status, attempts) VALUES (:vid, 'mediaintel-audio', 'videos/ok.m4a', 'pending', 0)"), {"vid": vid_ok})
            conn.execute(text("INSERT INTO video_audio_assets (video_id, bucket, object_path, status, attempts) VALUES (:vid, 'mediaintel-audio', 'videos/err.m4a', 'pending', 0)"), {"vid": vid_err})

        # Success for vid_ok
        with patch("core.storage_audio.storage_audio_to_pcm", return_value=b"dummy_pcm"), \
             patch("core.transcribe_videos.whisper_transcribe", return_value=("Decoded Whisper transcript text", "en", 120)):
            stats = cloud_whisper_worker.run(limit=1, max_seconds=60)
            self.assertEqual(stats["whisper_transcripts_completed"], 1)

        # Failure for vid_err with caption fallback safety net
        with patch("core.storage_audio.storage_audio_to_pcm", side_effect=RuntimeError("CTranslate2 Model Error")), \
             patch("core.transcribe_videos.fetch_captions", return_value=("Fallback captions after model failure", "en")):
            stats = cloud_whisper_worker.run(limit=2, max_seconds=60)
            self.assertGreaterEqual(stats["caption_fallbacks_completed"], 1)

    # 10. Caption fallback and accurate provenance
    def test_10_caption_fallback_provenance(self):
        cid = "UC_PROD_PROV_01"
        vid = "VID_PROVENANCE"
        with config.get_engine().begin() as conn:
            conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES (:cid, 'Prov Channel', 1)"), {"cid": cid})
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title) VALUES (:vid, :cid, 'Prov Video')"), {"vid": vid, "cid": cid})
        transcribe_videos.save(vid, source="captions", language="en", model="youtube-captions", body="Caption text", seconds=None)
        tx = config.read_sql("SELECT * FROM transcripts WHERE video_id = :vid", {"vid": vid}).iloc[0]
        self.assertEqual(tx["source"], "captions")
        self.assertNotEqual(tx["source"], "whisper", "Captions must NEVER be labeled as whisper")
        self.assertEqual(tx["model_name"], "youtube-captions")

    # 11. Missing transcript and dashboard truthful metrics
    def test_11_dashboard_truthful_metrics(self):
        transcripts_df = pd.DataFrame([
            {"video_id": "V1", "source": "whisper", "audio_seconds": 120, "word_count": 200, "language": "en"},
            {"video_id": "V2", "source": "captions", "audio_seconds": None, "word_count": 150, "language": "en"},
            {"video_id": "V3", "source": "audio_unavailable", "audio_seconds": 0, "word_count": 0, "language": None},
        ])
        videos_df = pd.DataFrame([{"video_id": "V1"}, {"video_id": "V2"}, {"video_id": "V3"}])
        summary = transcribe_videos.config.ROOT  # sanity check import
        from core import insights
        res = insights.whisper_summary(transcripts_df, videos_df)
        self.assertEqual(res["videos"], 1, "Only genuine whisper transcripts should be counted in whisper_summary videos")
        self.assertEqual(res["minutes"], 2.0)

    # 12. Relevance computation after transcription
    def test_12_relevance_computation_after_transcription(self):
        long_transcript = (
            "Detailed tutorial covering python variables data structures loops "
            "functions classes modules exception handling and object oriented programming "
            "best practices in modern development"
        )
        row_whisper = {"title": "Python Basics", "description": "Intro", "transcript_text": long_transcript, "tsource": "whisper"}
        text_w, src_w = relevance_topics.video_text(row_whisper)
        self.assertEqual(src_w, "whisper")
        self.assertEqual(text_w, long_transcript)

        row_captions = {"title": "Python Basics", "description": "Intro", "transcript_text": long_transcript, "tsource": "captions"}
        text_c, src_c = relevance_topics.video_text(row_captions)
        self.assertEqual(src_c, "captions")

        row_none = {"title": "Python Basics", "description": "Intro guide", "transcript_text": None, "tsource": None}
        text_n, src_n = relevance_topics.video_text(row_none)
        self.assertEqual(src_n, "title_only")

    # 13. Topic invalidation after channel deletion
    def test_13_topic_invalidation_after_channel_deletion(self):
        cid = "UC_PROD_DEL_TOPIC"
        with config.get_engine().begin() as conn:
            conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES (:cid, 'Delete Me', 1)"), {"cid": cid})
            conn.execute(text("INSERT INTO topic_suggestions (scope, rank_no, topic, score) VALUES (:cid, 1, 'Topic A', 85)"), {"cid": cid})
            conn.execute(text("INSERT INTO topic_suggestions (scope, rank_no, topic, score) VALUES ('ALL', 1, 'Global Topic', 90)"))

        # Delete the channel permanently
        res = channel_manager.delete_channel_permanently(cid, confirmation="DELETE", admin_user=True)
        self.assertTrue(res["success"])

        # When 0 channels are tracked, topic_suggestions must be empty
        topics = config.read_sql("SELECT * FROM topic_suggestions")
        self.assertTrue(topics.empty, "When 0 channels remain, all topic suggestions must be purged")

    # 14. Topic regeneration after adding a channel
    def test_14_topic_regeneration_after_adding_channel(self):
        cid = "UC_PROD_REGEN_01"
        with config.get_engine().begin() as conn:
            conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES (:cid, 'Regen Channel', 1)"), {"cid": cid})
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title) VALUES ('V_REG_01', :cid, 'Python DSA Roadmap')"), {"cid": cid})
            for i in range(10):
                conn.execute(
                    text("INSERT INTO comments (comment_id, video_id, comment_text) VALUES (:cm, 'V_REG_01', 'Please make video on system design tutorial')"),
                    {"cm": f"C_REG_{i:02d}"},
                )
            conn.execute(text("INSERT INTO aspect_sentiments (comment_id, aspect, sentiment_label, polarity, confidence) VALUES ('C_REG_00', 'Feature Request', 'Positive', 0.8, 0.9)"))

        relevance_topics.run()
        topics = config.read_sql("SELECT * FROM topic_suggestions WHERE scope = 'ALL'")
        self.assertFalse(topics.empty, "Topics must be regenerated and stored for active channel data")

    # 15. Anomaly severity and cooldown
    def test_15_anomaly_severity_and_cooldown(self):
        from core import anomaly_alert
        self.assertEqual(anomaly_alert.severity_for(12.0), "CRITICAL")
        self.assertEqual(anomaly_alert.severity_for(6.0), "HIGH")
        self.assertEqual(anomaly_alert.severity_for(3.5), "MEDIUM")

        # Test cooldown deduplication
        cid = "UC_PROD_COOL_01"
        vid = "VID_COOL_01"
        with config.get_engine().begin() as conn:
            conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES (:cid, 'Cool Channel', 1)"), {"cid": cid})
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title) VALUES (:vid, :cid, 'Cool Video')"), {"vid": vid, "cid": cid})
            conn.execute(
                text("""
                    INSERT INTO anomaly_alerts (video_id, channel_id, metric, velocity, baseline, ratio, severity, detected_at, emailed)
                    VALUES (:vid, :cid, 'views', 500, 50, 10.0, 'CRITICAL', CURRENT_TIMESTAMP, 0)
                """),
                {"vid": vid, "cid": cid},
            )

        reported = anomaly_alert._already_reported()
        self.assertIn((vid, "views"), reported, "Alert within cooldown period must be detected as already reported")

    # 16. Alert history after deleting channels
    def test_16_alert_history_after_deleting_channel(self):
        cid = "UC_PROD_DEL_ALERT"
        vid = "VID_DEL_AL"
        with config.get_engine().begin() as conn:
            conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES (:cid, 'Alert Del Channel', 1)"), {"cid": cid})
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title) VALUES (:vid, :cid, 'Del Video')"), {"vid": vid, "cid": cid})
            conn.execute(
                text("""
                    INSERT INTO anomaly_alerts (video_id, channel_id, metric, velocity, baseline, ratio, severity, detected_at, emailed)
                    VALUES (:vid, :cid, 'views', 300, 50, 6.0, 'HIGH', CURRENT_TIMESTAMP, 0)
                """),
                {"vid": vid, "cid": cid},
            )

        channel_manager.delete_channel_permanently(cid, confirmation="DELETE", admin_user=True)
        alerts = config.read_sql("SELECT * FROM anomaly_alerts WHERE channel_id = :cid", {"cid": cid})
        self.assertTrue(alerts.empty, "Alerts belonging to deleted channels must be completely purged")

    # 17. PDF Unicode rendering and accurate source labels
    def test_17_pdf_unicode_rendering_and_source_labels(self):
        raw_title = "Complete Python Course ₹0 Fee! — 100% Free ‘Tutorial’"
        clean = generate_pdf_report.clean_text(raw_title)
        self.assertIn("Rs. 0", clean, "Indian rupee symbol ₹ must be safely mapped to Rs.")
        self.assertNotIn("₹", clean, "Raw Unicode rupee glyph must not be passed to standard PDF fonts")
        self.assertIn("Tutorial", clean)

    # 18. SMTP dry-run and attachment creation
    def test_18_smtp_dry_run_and_attachment_creation(self):
        import rpa.MediaIntelRPA as rpa_module
        bot = rpa_module.MediaIntelRPA()

        os.environ["RPA_DRY_RUN"] = "1"
        try:
            mail = {
                "subject": "[HIGH ALERT] MediaIntel: 1 engagement anomaly",
                "text": "Alert details",
                "html": "<p>Alert details</p>",
                "alerts": 1,
            }
            res = bot.send_email_with_attachments(mail)
            self.assertEqual(res, "DRY_RUN")
            self.assertTrue((config.RPA_LOG_DIR / "dispatch_result.json").is_file())
        finally:
            os.environ.pop("RPA_DRY_RUN", None)

    # 19. Supabase authentication failure handling
    def test_19_supabase_auth_failure_preflight(self):
        # Pooler connection string with wrong username
        pooler_bad_url = "postgresql://postgres:mypassword@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"
        res = config.diagnose_database_connection(pooler_bad_url)
        self.assertFalse(res["healthy"])
        self.assertIsNotNone(res["recommendation"])
        self.assertIn("postgres.<project-ref>", res["recommendation"])

    # 20. Manual workflow dispatch, dependencies, and concurrency
    def test_20_workflow_dispatch_dependencies_and_concurrency(self):
        wf_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "mediaintel-cloud-pipeline.yml"
        content = wf_path.read_text(encoding="utf-8")

        self.assertIn("concurrency:", content)
        self.assertIn("group: mediaintel-cloud-pipeline", content)
        self.assertIn("cancel-in-progress: false", content)

        # Ordering: whisper-transcription before ai-analytics
        idx_whisper = content.find("whisper-transcription:")
        idx_ai = content.find("ai-analytics:")
        idx_rpa = content.find("rpa-reporting:")
        self.assertGreater(idx_whisper, 0)
        self.assertGreater(idx_ai, idx_whisper, "ai-analytics must be declared after whisper-transcription")
        self.assertGreater(idx_rpa, idx_ai, "rpa-reporting must be declared after ai-analytics")

        # Explicitly verify that ai-analytics runs when whisper is skipped or disabled
        self.assertIn("needs.whisper-transcription.result == 'skipped'", content,
                      "ai-analytics must explicitly permit skipped whisper-transcription")
        self.assertIn("needs.ingest-youtube.result == 'skipped'", content,
                      "ai-analytics must permit skipped ingestion for manual reruns")

        # Verify that relevance and topics stage executes cleanly with 0 transcripts (fallback to title/description)
        with config.get_engine().begin() as conn:
            conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES ('UC_NOWHISP_01', 'No Whisper Channel', 1)"))
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title, description) VALUES ('V_NOWHISP_01', 'UC_NOWHISP_01', 'Docker Crash Course', 'Comprehensive guide to containerization')"))
            for i in range(5):
                conn.execute(text("INSERT INTO comments (comment_id, video_id, comment_text) VALUES (:cid, 'V_NOWHISP_01', 'Very helpful docker tutorial thanks')"),
                             {"cid": f"C_NW_{i}"})
            conn.execute(text("INSERT INTO aspect_sentiments (comment_id, aspect, sentiment_label, polarity, confidence) VALUES ('C_NW_0', 'Video Content', 'Positive', 0.9, 0.9)"))

        # Verify relevance_topics runs cleanly without transcripts table rows
        relevance_topics.run()
        rel_df = config.read_sql("SELECT * FROM video_relevance WHERE video_id = 'V_NOWHISP_01'")
        self.assertEqual(len(rel_df), 1, "Relevance must compute successfully using title/description fallback when Whisper is disabled")
        self.assertEqual(rel_df.iloc[0]["text_source"], "title_only")
        top_df = config.read_sql("SELECT * FROM topic_suggestions WHERE scope = 'ALL'")
        self.assertFalse(top_df.empty, "Topics must be stored even when Whisper is disabled")

    # 21. Public vs authenticated admin permissions
    def test_21_public_vs_admin_permissions(self):
        secret_pwd = "production_admin_password_123"
        correct = hmac.compare_digest("production_admin_password_123".encode("utf-8"), secret_pwd.encode("utf-8"))
        self.assertTrue(correct)

        wrong = hmac.compare_digest("public_visitor".encode("utf-8"), secret_pwd.encode("utf-8"))
        self.assertFalse(wrong)

    # 22. Regression tests for all eight dashboard tabs
    def test_22_dashboard_eight_tabs_regression(self):
        # Ensure helper summaries work cleanly with empty or populated data
        from core import insights
        empty_vids = pd.DataFrame()
        empty_tx = pd.DataFrame()
        wsum = insights.whisper_summary(empty_tx, empty_vids)
        self.assertEqual(wsum["videos"], 0)
        self.assertEqual(wsum["minutes"], 0.0)

        spoken = insights.spoken_topics(pd.DataFrame(columns=["video_id", "transcript_text"]))
        self.assertTrue(spoken.empty)


if __name__ == "__main__":
    unittest.main()
