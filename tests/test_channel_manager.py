"""Unit tests for core/channel_manager.py: batch parsing, resolution, deduplication, and workflow dispatch."""
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import channel_manager


class TestChannelManager(unittest.TestCase):
    def test_parse_json_array_of_objects(self):
        raw = """[
          { "channel_id": "UCJskGeByzRRSvmOyZOz61ig", "name": "take U forward" },
          { "channel_id": "UCBwmMxybNva6P_5VmxjzwqA", "name": "Apna College" }
        ]"""
        items = channel_manager.parse_channel_input(raw)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["raw"], "UCJskGeByzRRSvmOyZOz61ig")
        self.assertEqual(items[0]["name"], "take U forward")
        self.assertEqual(items[1]["raw"], "UCBwmMxybNva6P_5VmxjzwqA")
        self.assertEqual(items[1]["name"], "Apna College")

    def test_parse_multiline_and_urls(self):
        raw = """https://www.youtube.com/channel/UC5u7mJb1fIAedPG-9Pg6Nbw
UCmFUoPicoxuJAuR2NfNdQGA - Swaroop Talks
@GateSmashers
"""
        items = channel_manager.parse_channel_input(raw)
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0]["raw"], "https://www.youtube.com/channel/UC5u7mJb1fIAedPG-9Pg6Nbw")
        self.assertEqual(items[1]["raw"], "UCmFUoPicoxuJAuR2NfNdQGA")
        self.assertEqual(items[1]["name"], "Swaroop Talks")
        self.assertEqual(items[2]["raw"], "@GateSmashers")

    def test_resolve_channel_identifier_direct_uc(self):
        cid, name, err = channel_manager.resolve_channel_identifier("UCJskGeByzRRSvmOyZOz61ig")
        self.assertEqual(cid, "UCJskGeByzRRSvmOyZOz61ig")
        self.assertIsNone(err)

    def test_resolve_channel_identifier_url(self):
        cid, name, err = channel_manager.resolve_channel_identifier(
            "https://www.youtube.com/channel/UCBwmMxybNva6P_5VmxjzwqA?si=test"
        )
        self.assertEqual(cid, "UCBwmMxybNva6P_5VmxjzwqA")
        self.assertIsNone(err)

    def test_resolve_channel_identifier_invalid(self):
        cid, name, err = channel_manager.resolve_channel_identifier("invalid_random_string_123")
        self.assertIsNone(cid)
        self.assertIsNotNone(err)

    def test_batch_registration_deduplication(self):
        raw = """UCJskGeByzRRSvmOyZOz61ig
https://www.youtube.com/channel/UCJskGeByzRRSvmOyZOz61ig
UCBwmMxybNva6P_5VmxjzwqA"""
        
        # Mock register_single_channel to avoid live DB dependency in pure unit test
        with patch("core.channel_manager.register_single_channel") as mock_reg:
            mock_reg.return_value = {
                "success": True,
                "channel_id": "mock_id",
                "channel_name": "Mock Channel",
                "status": "created",
                "message": "Registered",
            }
            results = channel_manager.register_batch_channels(raw)
            self.assertEqual(len(results), 3)
            # Second item should be detected as duplicate within the batch
            self.assertEqual(results[0]["status"], "created")
            self.assertEqual(results[1]["status"], "duplicate_in_batch")
            self.assertEqual(results[2]["status"], "created")

    def test_dispatch_cloud_pipeline_missing_token(self):
        res = channel_manager.dispatch_cloud_pipeline(gh_token="")
        self.assertFalse(res["success"])
        self.assertEqual(res["status_code"], 401)

    @patch("requests.get")
    def test_dispatch_cloud_pipeline_in_progress_conflict(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "workflow_runs": [{"id": 12345, "status": "in_progress", "conclusion": None}]
        }
        mock_get.return_value = mock_resp

        res = channel_manager.dispatch_cloud_pipeline(gh_token="dummy_token", force=False)
        self.assertFalse(res["success"])
        self.assertEqual(res["status_code"], 409)
        self.assertIn("already in progress", res["message"])

    @patch("requests.get")
    @patch("requests.post")
    def test_dispatch_cloud_pipeline_success(self, mock_post, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"workflow_runs": []})
        mock_post.return_value = MagicMock(status_code=204)

        res = channel_manager.dispatch_cloud_pipeline(
            gh_token="dummy_token",
            ref="main",
            channel_ids="UC111,UC222",
            video_limit=5,
            sentiment_limit=100,
            run_rpa=True,
            send_email=False,
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["status_code"], 204)
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]
        self.assertEqual(payload["inputs"]["run_rpa"], True)
        self.assertEqual(payload["inputs"]["run_whisper"], False)
        self.assertEqual(payload["inputs"]["send_email"], False)
        self.assertEqual(payload["inputs"]["run_ingestion"], True)
        self.assertEqual(payload["inputs"]["run_ai"], True)
        self.assertEqual(payload["inputs"]["channel_id"], "UC111,UC222")


if __name__ == "__main__":
    unittest.main()
