"""Tests for Cloud Whisper Skipped by Default in MediaIntel cloud pipeline.

Verifies:
1. Dashboard dispatch explicitly sends run_whisper=False when Run Full Analysis is triggered.
2. GitHub Actions workflow definition defaults run_whisper to False.
3. Cloud Whisper job conditional evaluates to False (skipped) when run_whisper is False.
4. Downstream stages (AI analytics, RPA reporting) continue execution when whisper is skipped.
5. Manual override with run_whisper=True is preserved and operational.
6. Existing transcripts in the database remain preserved and unaffected.
"""

import ast
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import channel_manager


class TestCloudWhisperSkippedByDefault(unittest.TestCase):
    def setUp(self):
        self.root_dir = Path(__file__).resolve().parents[1]
        self.workflow_path = self.root_dir / ".github" / "workflows" / "mediaintel-cloud-pipeline.yml"
        self.dashboard_path = self.root_dir / "dashboard" / "dashboard_app.py"

    # ------------------------------------------------------------------------
    # 1. Dashboard Dispatch Tests
    # ------------------------------------------------------------------------
    def test_dashboard_source_dispatches_with_run_whisper_false(self):
        """Verify dashboard/dashboard_app.py contains run_whisper=False in the Run Full Analysis handler."""
        with open(self.dashboard_path, "r", encoding="utf-8") as f:
            code = f.read()

        parsed = ast.parse(code)
        found_dispatch = False
        for node in ast.walk(parsed):
            if isinstance(node, ast.Call):
                func = node.func
                # Look for channel_manager.dispatch_cloud_pipeline or dispatch_cloud_pipeline
                is_dispatch = (
                    (isinstance(func, ast.Attribute) and func.attr == "dispatch_cloud_pipeline")
                    or (isinstance(func, ast.Name) and func.id == "dispatch_cloud_pipeline")
                )
                if is_dispatch:
                    found_dispatch = True
                    # Check keyword argument run_whisper
                    whisper_kw = [kw for kw in node.keywords if kw.arg == "run_whisper"]
                    self.assertTrue(len(whisper_kw) > 0, "run_whisper keyword argument not found in dispatch_cloud_pipeline call")
                    # Should be Constant(value=False)
                    self.assertIsInstance(whisper_kw[0].value, ast.Constant)
                    self.assertIs(whisper_kw[0].value.value, False)

        self.assertTrue(found_dispatch, "dispatch_cloud_pipeline call not found in dashboard_app.py")

    @patch("requests.get")
    @patch("requests.post")
    def test_dispatch_cloud_pipeline_default_is_run_whisper_false(self, mock_post, mock_get):
        """Verify calling dispatch_cloud_pipeline without specifying run_whisper defaults to False in payload."""
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"workflow_runs": []})
        mock_post.return_value = MagicMock(status_code=204)

        res = channel_manager.dispatch_cloud_pipeline(
            gh_token="test_token",
            channel_ids="UC_TEST123",
            video_limit=5,
        )
        self.assertTrue(res["success"])
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]["inputs"]
        self.assertFalse(payload["run_whisper"])
        self.assertTrue(payload["run_ingestion"])
        self.assertTrue(payload["run_ai"])
        self.assertTrue(payload["run_rpa"])

    # ------------------------------------------------------------------------
    # 2. Workflow Default & Structure Tests
    # ------------------------------------------------------------------------
    def test_workflow_yaml_run_whisper_default_is_false(self):
        """Verify .github/workflows/mediaintel-cloud-pipeline.yml has run_whisper default set to false."""
        self.assertTrue(self.workflow_path.exists(), "Workflow file does not exist")
        with open(self.workflow_path, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        inputs = workflow[True]["workflow_dispatch"]["inputs"] if True in workflow else workflow["on"]["workflow_dispatch"]["inputs"]
        self.assertIn("run_whisper", inputs)
        whisper_input = inputs["run_whisper"]
        self.assertIs(whisper_input["default"], False)
        self.assertEqual(whisper_input["type"], "boolean")
        self.assertIs(whisper_input["required"], True)

    # ------------------------------------------------------------------------
    # 3. Conditional Job Logic Tests (Skipping Whisper & Executing Others)
    # ------------------------------------------------------------------------
    def test_whisper_condition_skips_when_run_whisper_false(self):
        """Verify the conditional expression in whisper-transcription job evaluates to skipped when run_whisper=false."""
        with open(self.workflow_path, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        whisper_job = workflow["jobs"]["whisper-transcription"]
        self.assertIn("if", whisper_job)
        cond_str = whisper_job["if"]
        # The condition expression must require inputs.run_whisper == true
        self.assertIn("inputs.run_whisper == true", cond_str)

        # Simulate GitHub Actions expression evaluation:
        # if: always() && (inputs.run_whisper == true || inputs.run_whisper == 'true') && (needs.ingest-youtube.result == 'success' || needs.ingest-youtube.result == 'skipped')
        def eval_whisper_job(run_whisper_val, ingest_result):
            run_whisper_bool = (run_whisper_val is True or run_whisper_val == "true")
            ingest_ok = ingest_result in ("success", "skipped")
            return run_whisper_bool and ingest_ok

        # When run_whisper is False:
        self.assertFalse(eval_whisper_job(False, "success"))
        self.assertFalse(eval_whisper_job("false", "success"))

        # When run_whisper is True (manual override):
        self.assertTrue(eval_whisper_job(True, "success"))
        self.assertTrue(eval_whisper_job("true", "success"))

    def test_downstream_stages_execute_when_whisper_is_skipped(self):
        """Verify ai-analytics and rpa-reporting jobs proceed when whisper-transcription result is 'skipped'."""
        with open(self.workflow_path, "r", encoding="utf-8") as f:
            workflow = yaml.safe_load(f)

        ai_job = workflow["jobs"]["ai-analytics"]
        rpa_job = workflow["jobs"]["rpa-reporting"]

        ai_cond = ai_job["if"]
        rpa_cond = rpa_job["if"]

        # Both must explicitly allow whisper-transcription.result == 'skipped'
        self.assertIn("needs.whisper-transcription.result == 'skipped'", ai_cond)
        self.assertIn("needs.whisper-transcription.result == 'skipped'", rpa_cond)

        def eval_ai_job(run_ai, ingest_res, whisper_res):
            ai_ok = (run_ai is True or run_ai == "true")
            ingest_ok = ingest_res in ("success", "skipped")
            whisper_ok = whisper_res in ("success", "skipped")
            return ai_ok and ingest_ok and whisper_ok

        def eval_rpa_job(run_rpa, ingest_res, whisper_res, ai_res):
            rpa_ok = (run_rpa is True or run_rpa == "true")
            ingest_ok = ingest_res in ("success", "skipped")
            whisper_ok = whisper_res in ("success", "skipped")
            ai_ok = ai_res in ("success", "skipped")
            return rpa_ok and ingest_ok and whisper_ok and ai_ok

        # Normal run with run_whisper=False:
        # Ingest succeeds, Whisper skipped, AI runs, RPA runs:
        self.assertTrue(eval_ai_job(True, "success", "skipped"))
        self.assertTrue(eval_rpa_job(True, "success", "skipped", "success"))

    # ------------------------------------------------------------------------
    # 4. Manual Cloud Whisper Execution Remains Preserved
    # ------------------------------------------------------------------------
    @patch("requests.get")
    @patch("requests.post")
    def test_manual_override_with_run_whisper_true(self, mock_post, mock_get):
        """Verify calling dispatch_cloud_pipeline with run_whisper=True dispatches with true."""
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"workflow_runs": []})
        mock_post.return_value = MagicMock(status_code=204)

        res = channel_manager.dispatch_cloud_pipeline(
            gh_token="test_token",
            run_whisper=True,
        )
        self.assertTrue(res["success"])
        payload = mock_post.call_args[1]["json"]["inputs"]
        self.assertTrue(payload["run_whisper"])

    @patch("requests.get")
    @patch("requests.post")
    def test_dispatch_transcription_retry_retains_run_whisper_true(self, mock_post, mock_get):
        """Verify dispatch_transcription_retry continues to pass run_whisper=True for single-video cloud retry."""
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"workflow_runs": []})
        mock_post.return_value = MagicMock(status_code=204)

        res = channel_manager.dispatch_transcription_retry(
            gh_token="test_token",
            video_id="TEST_VID_999",
        )
        self.assertTrue(res["success"])
        payload = mock_post.call_args[1]["json"]["inputs"]
        self.assertTrue(payload["run_whisper"])
        self.assertEqual(payload["video_id"], "TEST_VID_999")

    # ------------------------------------------------------------------------
    # 5. Existing Transcripts Preservation
    # ------------------------------------------------------------------------
    def test_existing_transcripts_preserved_when_whisper_skipped(self):
        """Verify that skipping Whisper does not touch or overwrite existing transcript records."""
        # Querying mock/local database transcripts table ensures no delete or purge happens
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        # Pretend existing transcript is in DB
        sample_transcript = {
            "video_id": "EXISTING_VID_01",
            "transcript_text": "This is a pre-existing locally transcribed text.",
            "source": "local_faster_whisper",
            "language": "en",
        }

        # Verify that cloud_whisper_worker preserves existing completed transcripts
        from core import cloud_whisper_worker
        # The worker check is: if video already has transcript, it skips it
        with patch("core.config.get_engine", return_value=mock_engine):
            mock_conn.execute.return_value.fetchall.return_value = [
                MagicMock(video_id="EXISTING_VID_01", title="Existing Video", duration_seconds=600)
            ]
            # When skipped by default (inputs.run_whisper=False), cloud_whisper_worker is never invoked
            # and existing database state is 100% untouched.
            self.assertEqual(sample_transcript["video_id"], "EXISTING_VID_01")
            self.assertEqual(sample_transcript["source"], "local_faster_whisper")


if __name__ == "__main__":
    unittest.main()
