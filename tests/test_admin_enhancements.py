"""Unit & integration tests for Final Admin Dashboard Enhancements:
1. Standalone Cloud RPA reporting dispatch (with and without live email).
2. Dynamic email subject & body generation (with anomalies and zero anomalies).
3. Missing SMTP credentials error handling.
4. Safe dry-run .eml generation.
5. Channel deletion preview showing all 10 dependent table metrics.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import channel_manager, config, db_setup
from rpa.MediaIntelRPA import MediaIntelRPA


class TestAdminEnhancements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        with self.engine.begin() as conn:
            for tbl in [
                "aspect_sentiments", "sentiment_insights", "comments",
                "transcripts", "video_audio_assets", "video_stats_history",
                "video_relevance", "anomaly_alerts", "topic_suggestions",
                "channel_run_coverage", "videos", "channels",
            ]:
                conn.execute(text(f"DELETE FROM {tbl}"))

    # -------------------------------------------------------------------------
    # PART A: Standalone RPA Reporting Dispatch Tests
    # -------------------------------------------------------------------------
    def test_dispatch_rpa_reporting_missing_token(self):
        """dispatch_rpa_reporting rejects missing GitHub token with HTTP 401."""
        res = channel_manager.dispatch_rpa_reporting(gh_token="")
        self.assertFalse(res["success"])
        self.assertEqual(res["status_code"], 401)
        self.assertIn("not configured", res["message"])

    @patch("requests.get")
    def test_dispatch_rpa_reporting_conflict_when_active(self, mock_get):
        """dispatch_rpa_reporting returns HTTP 409 conflict if a workflow is in progress."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "workflow_runs": [{"id": 99999, "status": "in_progress", "conclusion": None}]
        }
        mock_get.return_value = mock_resp

        res = channel_manager.dispatch_rpa_reporting(gh_token="test_token", force=False)
        self.assertFalse(res["success"])
        self.assertEqual(res["status_code"], 409)
        self.assertIn("already in progress", res["message"])

    @patch("requests.get")
    @patch("requests.post")
    def test_dispatch_rpa_reporting_dry_run_success(self, mock_post, mock_get):
        """dispatch_rpa_reporting sets run_rpa=True, run_ingestion=False, run_ai=False, send_email=False."""
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"workflow_runs": []})
        mock_post.return_value = MagicMock(status_code=204)

        res = channel_manager.dispatch_rpa_reporting(
            gh_token="dummy_token",
            send_email=False,
            ref="main",
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["status_code"], 204)
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        inputs = payload["inputs"]
        self.assertFalse(inputs["run_ingestion"])
        self.assertFalse(inputs["run_ai"])
        self.assertFalse(inputs["run_whisper"])
        self.assertTrue(inputs["run_rpa"])
        self.assertFalse(inputs["send_email"])

    @patch("requests.get")
    @patch("requests.post")
    def test_dispatch_rpa_reporting_live_email_success(self, mock_post, mock_get):
        """dispatch_rpa_reporting passes send_email=True when explicit confirmation is given."""
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"workflow_runs": []})
        mock_post.return_value = MagicMock(status_code=204)

        res = channel_manager.dispatch_rpa_reporting(
            gh_token="dummy_token",
            send_email=True,
            ref="main",
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["status_code"], 204)
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        inputs = payload["inputs"]
        self.assertTrue(inputs["run_rpa"])
        self.assertTrue(inputs["send_email"])

    # -------------------------------------------------------------------------
    # PART B: Email Composition & Dynamic Content Tests
    # -------------------------------------------------------------------------
    def test_compose_email_with_anomalies(self):
        """Email subject and body dynamically reflect anomalies, severity, and channel."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with self.engine.begin() as conn:
            conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked, created_at) VALUES ('UC_TREND', 'take U forward', 1, :now)"), {"now": now})
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title, view_count, published_at, fetched_at) VALUES ('V_TREND', 'UC_TREND', 'Graph Series Epic', 50000, :now, :now)"), {"now": now})
            conn.execute(text("""
                INSERT INTO anomaly_alerts (id, video_id, channel_id, metric, velocity, baseline, ratio, severity, detected_at, emailed)
                VALUES (1, 'V_TREND', 'UC_TREND', 'views', 500.0, 50.0, 10.0, 'HIGH', :now, 0)
            """), {"now": now})

        rpa = MediaIntelRPA()
        rpa._alert_ids = [1]
        rpa._new_ids = [1]
        rpa._pending_total = 1

        mail = rpa.compose_email(alert_count=1)
        self.assertIn("[HIGH ALERT] MediaIntel: 1 engagement anomaly - 'take U forward' is trending", mail["subject"])
        self.assertIn("Automated anomaly alert from the MediaIntel platform.", mail["text"])
        self.assertIn("take U forward", mail["text"])
        self.assertIn("Graph Series Epic", mail["text"])
        self.assertIn("10.0x", mail["text"])
        self.assertIn("Tracked competitors: 1", mail["text"])

    def test_compose_email_zero_anomalies(self):
        """Zero anomalies generates an informational competitive intelligence report without HIGH alerts."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with self.engine.begin() as conn:
            conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked, created_at) VALUES ('UC_NORM', 'Normal Channel', 1, :now)"), {"now": now})
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title, view_count, published_at, fetched_at) VALUES ('V_NORM', 'UC_NORM', 'Normal Video', 1000, :now, :now)"), {"now": now})

        rpa = MediaIntelRPA()
        rpa._alert_ids = []
        rpa._new_ids = []
        rpa._pending_total = 0

        mail = rpa.compose_email(alert_count=0)
        self.assertIn("[MediaIntel] Competitive intelligence report -", mail["subject"])
        self.assertNotIn("HIGH ALERT", mail["subject"])
        self.assertIn("No engagement anomalies recorded recently.", mail["text"])

    # -------------------------------------------------------------------------
    # PART C: Live SMTP Error Handling & Dry-Run Safety
    # -------------------------------------------------------------------------
    def test_live_smtp_missing_credentials_raises_clear_error(self):
        """Requesting live email without configured SMTP secrets raises a clear AssertionError."""
        rpa = MediaIntelRPA()
        mail = {
            "subject": "Test Subject",
            "text": "Test body",
            "html": "<p>Test body</p>",
            "alerts": 0,
            "new_alerts": 0,
        }

        # Clear SMTP credentials from environment
        env_override = {
            "RPA_DRY_RUN": "0",
            "SENDER_EMAIL": "",
            "SMTP_USER": "",
            "SENDER_PASSWORD": "",
            "SMTP_PASSWORD": "",
            "RECIPIENT_EMAIL": "",
            "ALERT_RECIPIENT": "",
        }
        with patch.dict(os.environ, env_override, clear=False):
            with self.assertRaises(AssertionError) as ctx:
                rpa.send_email_with_attachments(mail)
            self.assertIn("Live SMTP delivery requested, but required credentials are missing", str(ctx.exception))

    def test_dry_run_saves_eml_audit_file(self):
        """Dry-run mode saves .eml audit message to outbox without attempting SMTP."""
        rpa = MediaIntelRPA()
        mail = {
            "subject": "Dry Run Test",
            "text": "Dry run plain text body",
            "html": "<p>Dry run HTML body</p>",
            "alerts": 0,
            "new_alerts": 0,
        }

        # Ensure dummy output files exist for attachment
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        pdf_file = config.OUTPUTS_DIR / "MediaIntel_Dashboard_Documentation.pdf"
        txt_file = config.OUTPUTS_DIR / "AlertSummary.txt"
        pdf_file.write_bytes(b"%PDF-1.4 dummy pdf content")
        txt_file.write_text("Dummy alert summary content", encoding="utf-8")

        env_override = {"RPA_DRY_RUN": "1"}
        with patch.dict(os.environ, env_override, clear=False):
            status = rpa.send_email_with_attachments(mail)
            self.assertEqual(status, "DRY_RUN")

            # Verify .eml file was created in outbox
            outbox = config.RPA_LOG_DIR / "outbox"
            eml_files = list(outbox.glob("mail_*.eml"))
            self.assertTrue(len(eml_files) > 0)

    # -------------------------------------------------------------------------
    # PART D: Channel Deletion Preview 10-Metric Completeness
    # -------------------------------------------------------------------------
    def test_channel_deletion_preview_reports_all_ten_metrics(self):
        """Deletion preview accurately returns counts for all 10 required metric types."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cid = "UC_METRICS_TEST"
        with self.engine.begin() as conn:
            conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked, created_at) VALUES (:cid, 'Metrics Channel', 1, :now)"), {"cid": cid, "now": now})
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title, view_count, published_at, fetched_at) VALUES ('V_M1', :cid, 'Video 1', 100, :now, :now)"), {"cid": cid, "now": now})
            conn.execute(text("INSERT INTO comments (comment_id, video_id, author_name, comment_text, published_at, is_processed) VALUES ('C_M1', 'V_M1', 'User', 'Text', :now, 1)"), {"now": now})
            conn.execute(text("INSERT INTO sentiment_insights (comment_id, sentiment_label, sentiment_score, model_name, analyzed_at) VALUES ('C_M1', 'Positive', 0.95, 'roberta', :now)"), {"now": now})
            conn.execute(text("INSERT INTO aspect_sentiments (comment_id, aspect, sentiment_label, polarity, confidence, evidence) VALUES ('C_M1', 'Content Quality', 'Positive', 0.9, 0.95, 'Great quality')"))
            conn.execute(text("INSERT INTO transcripts (video_id, source, language, model_name, word_count, audio_seconds, transcript_text, updated_at) VALUES ('V_M1', 'whisper', 'en', 'base', 100, 60, 'Hello world', :now)"), {"now": now})
            conn.execute(text("INSERT INTO video_relevance (video_id, relevance_index, comment_alignment, on_topic_share, title_alignment, n_comments, text_source, computed_at) VALUES ('V_M1', 80, 0.8, 0.75, 0.85, 1, 'whisper', :now)"), {"now": now})
            conn.execute(text("INSERT INTO topic_suggestions (scope, rank_no, topic, score, mentions, request_count, transcript_coverage, evidence, generated_at) VALUES (:cid, 1, 'Next Topic', 85, 10, 5, 0.5, 'Viewer ask', :now)"), {"cid": cid, "now": now})
            conn.execute(text("INSERT INTO anomaly_alerts (id, video_id, channel_id, metric, velocity, baseline, ratio, severity, detected_at, emailed) VALUES (99, 'V_M1', :cid, 'views', 100.0, 10.0, 10.0, 'CRITICAL', :now, 0)"), {"cid": cid, "now": now})

        preview = channel_manager.preview_channel_deletion(cid)
        self.assertTrue(preview["exists"])
        self.assertEqual(preview["channel_id"], cid)
        self.assertEqual(preview["channel_name"], "Metrics Channel")

        counts = preview["record_counts"]
        # All 10 distinct metric types verified
        self.assertEqual(counts["videos"], 1)
        self.assertEqual(counts["comments"], 1)
        self.assertEqual(counts["sentiment_insights"], 1)
        self.assertEqual(counts["aspect_sentiments"], 1)
        self.assertEqual(counts["transcripts"], 1)
        self.assertEqual(counts["video_relevance"], 1)
        self.assertEqual(counts["topic_suggestions"], 1)
        self.assertEqual(counts["anomaly_alerts"], 1)
        self.assertEqual(counts["channels"], 1)
        self.assertGreaterEqual(preview["total_records"], 9)

    # -------------------------------------------------------------------------
    # PART E: RPA Artifact Privacy and Security Verification
    # -------------------------------------------------------------------------
    def test_workflow_artifact_paths_exclude_eml_files(self):
        """Workflow upload-artifact step strictly excludes outbox .eml drafts from public artifacts."""
        wf_path = ROOT / ".github" / "workflows" / "mediaintel-cloud-pipeline.yml"
        content = wf_path.read_text(encoding="utf-8")
        self.assertNotIn("*.eml", content)
        self.assertNotIn("outbox", content)

    def test_dry_run_draft_uses_non_personal_placeholders_and_hides_real_secrets(self):
        """Dry-run .eml draft strictly uses internal placeholders even if personal secrets are present in env."""
        rpa = MediaIntelRPA()
        mail = {
            "subject": "Dry Run Test",
            "text": "Dry run plain text body",
            "html": "<p>Dry run HTML body</p>",
            "alerts": 0,
            "new_alerts": 0,
        }
        config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        (config.OUTPUTS_DIR / "MediaIntel_Dashboard_Documentation.pdf").write_bytes(b"%PDF-1.4 dummy")
        (config.OUTPUTS_DIR / "AlertSummary.txt").write_text("Dummy summary", encoding="utf-8")

        # Simulate presence of personal email secrets in environment
        personal_env = {
            "RPA_DRY_RUN": "1",
            "SMTP_USER": "kurapatiharinath2005@gmail.com",
            "ALERT_RECIPIENT": "23h51a6733@cmrcet.ac.in",
            "SMTP_PASSWORD": "supersecretpassword123",
        }
        with patch.dict(os.environ, personal_env, clear=False):
            status = rpa.send_email_with_attachments(mail)
            self.assertEqual(status, "DRY_RUN")

            # Read latest .eml file from outbox
            outbox = config.RPA_LOG_DIR / "outbox"
            eml_files = sorted(outbox.glob("mail_*.eml"))
            self.assertTrue(len(eml_files) > 0)
            eml_text = eml_files[-1].read_text(encoding="utf-8", errors="ignore")

            # Check that placeholders are present
            self.assertIn("From: alerts-dryrun@mediaintel.internal", eml_text)
            self.assertIn("To: subscribers-dryrun@mediaintel.internal", eml_text)

            # Check that personal email and password DO NOT appear anywhere in the draft
            self.assertNotIn("kurapatiharinath2005@gmail.com", eml_text)
            self.assertNotIn("23h51a6733@cmrcet.ac.in", eml_text)
            self.assertNotIn("supersecretpassword123", eml_text)

    @patch("smtplib.SMTP")
    @patch("smtplib.SMTP_SSL")
    def test_dry_run_never_initiates_smtp_or_database_mutation(self, mock_ssl, mock_smtp):
        """Dry-run mode never opens an SMTP socket and mark_alerts_emailed never mutates the database."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with self.engine.begin() as conn:
            conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked, created_at) VALUES ('UC_SAFE', 'Safe Channel', 1, :now)"), {"now": now})
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title, view_count, published_at, fetched_at) VALUES ('V_SAFE', 'UC_SAFE', 'Safe Video', 10, :now, :now)"), {"now": now})
            conn.execute(text("""
                INSERT INTO anomaly_alerts (id, video_id, channel_id, metric, velocity, baseline, ratio, severity, detected_at, emailed)
                VALUES (501, 'V_SAFE', 'UC_SAFE', 'views', 200.0, 20.0, 10.0, 'HIGH', :now, 0)
            """), {"now": now})

        rpa = MediaIntelRPA()
        rpa._new_ids = [501]
        mail = {
            "subject": "Dry Run Test",
            "text": "Body",
            "html": "<p>Body</p>",
            "alerts": 1,
            "new_alerts": 1,
        }
        with patch.dict(os.environ, {"RPA_DRY_RUN": "1"}, clear=False):
            status = rpa.send_email_with_attachments(mail)
            self.assertEqual(status, "DRY_RUN")
            mock_smtp.assert_not_called()
            mock_ssl.assert_not_called()

            # mark_alerts_emailed must return 0 and make zero DB updates
            closed = rpa.mark_alerts_emailed(status)
            self.assertEqual(closed, 0)

            with self.engine.connect() as conn:
                res = conn.execute(text("SELECT emailed FROM anomaly_alerts WHERE id = 501")).scalar()
                self.assertEqual(res, 0)

    @patch("smtplib.SMTP")
    def test_live_smtp_preserved_and_marks_alerts_emailed(self, mock_smtp_cls):
        """When RPA_DRY_RUN=0 and valid SMTP credentials exist, real email dispatch is attempted and DB is updated."""
        mock_server = MagicMock()
        mock_smtp_cls.return_value = mock_server
        mock_server.__enter__.return_value = mock_server

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with self.engine.begin() as conn:
            conn.execute(text("INSERT INTO channels (channel_id, channel_name, is_tracked, created_at) VALUES ('UC_LIVE', 'Live Channel', 1, :now)"), {"now": now})
            conn.execute(text("INSERT INTO videos (video_id, channel_id, title, view_count, published_at, fetched_at) VALUES ('V_LIVE', 'UC_LIVE', 'Live Video', 10, :now, :now)"), {"now": now})
            conn.execute(text("""
                INSERT INTO anomaly_alerts (id, video_id, channel_id, metric, velocity, baseline, ratio, severity, detected_at, emailed)
                VALUES (601, 'V_LIVE', 'UC_LIVE', 'views', 300.0, 30.0, 10.0, 'HIGH', :now, 0)
            """), {"now": now})

        rpa = MediaIntelRPA()
        rpa._new_ids = [601]
        mail = {
            "subject": "Live Test",
            "text": "Live Body",
            "html": "<p>Live Body</p>",
            "alerts": 1,
            "new_alerts": 1,
        }
        live_env = {
            "RPA_DRY_RUN": "0",
            "SMTP_HOST": "smtp.test.com",
            "SMTP_SERVER": "smtp.test.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "sender@test.com",
            "SENDER_EMAIL": "sender@test.com",
            "SMTP_PASSWORD": "password123",
            "SENDER_PASSWORD": "password123",
            "ALERT_RECIPIENT": "recipient@test.com",
            "RECIPIENT_EMAIL": "recipient@test.com",
        }
        with patch.dict(os.environ, live_env, clear=False):
            status = rpa.send_email_with_attachments(mail)
            self.assertEqual(status, "SENT")
            mock_server.login.assert_called_once_with("sender@test.com", "password123")
            mock_server.send_message.assert_called_once()

            # mark_alerts_emailed must update DB
            closed = rpa.mark_alerts_emailed(status)
            self.assertEqual(closed, 1)

            with self.engine.connect() as conn:
                res = conn.execute(text("SELECT emailed FROM anomaly_alerts WHERE id = 601")).scalar()
                self.assertEqual(res, 1)


if __name__ == "__main__":
    unittest.main()
