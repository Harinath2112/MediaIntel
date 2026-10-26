"""Automated test suite for MediaIntel Smart 15-Minute Transcription Strategy.

Covers all 12 non-negotiable verification scenarios:
  1. 5-minute video with captions.
  2. 5-minute video without captions.
  3. 14-minute video without captions.
  4. Exactly 15-minute video without captions.
  5. 30-minute video without captions.
  6. 3-hour video without captions.
  7. Long video with captions.
  8. Missing or inaccessible audio.
  9. Segment ordering and total duration <= 900 seconds.
 10. Partial transcript metadata.
 11. Idempotency and preservation of completed transcripts.
 12. Supabase persistence using mocked database writes.
"""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from core import (
    cloud_whisper_worker,
    config,
    db_setup,
    local_transcription_worker,
)
from core.smart_transcription import (
    MAX_PARTIAL_AUDIO_SEC,
    SMART_TRANSCRIPTION_THRESHOLD_SEC,
    AudioSegment,
    TranscriptionPlan,
    combine_segment_transcripts,
    filter_informative_chapters,
    parse_description_timestamps,
    plan_transcription_segments,
    slice_pcm_for_plan,
)


class TestSmartTranscriptionStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Isolated SQLite database for persistence tests
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="mediaintel_smart_tx_test_")
        cls.db_path = Path(cls.temp_dir.name) / "test_smart_tx.db"
        cls.engine = create_engine(f"sqlite:///{cls.db_path}")

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()
        cls.temp_dir.cleanup()

    def setUp(self):
        self.patcher_engine = patch("core.config.get_engine", return_value=self.engine)
        self.mock_engine = self.patcher_engine.start()

        self.patcher_read_sql = patch("core.config.read_sql", side_effect=self._mock_read_sql)
        self.mock_read_sql = self.patcher_read_sql.start()

        # Clean schema rebuild
        db_setup.metadata.drop_all(self.engine)
        db_setup.metadata.create_all(self.engine)

        # Seed sample channel
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO channels (channel_id, channel_name, is_tracked) VALUES ('UC_CHAN_1', 'Test Channel', 1)")
            )

    def tearDown(self):
        self.patcher_read_sql.stop()
        self.patcher_engine.stop()

    def _mock_read_sql(self, sql, params=None):
        with self.engine.connect() as conn:
            stmt = text(sql) if isinstance(sql, str) else sql
            return pd.read_sql(stmt, conn, params=params or {})

    # ------------------------------------------------------------------------
    # 1. 5-minute video with captions
    # ------------------------------------------------------------------------
    def test_01_five_minute_video_with_captions(self):
        """Rule 1: Video < 15 min with captions uses complete captions; Whisper is not invoked."""
        vid = "VID_5M_CAP"
        duration_sec = 300  # 5 minutes
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO videos (video_id, channel_id, title, duration_sec, published_at) "
                     "VALUES (:vid, 'UC_CHAN_1', '5 Min Video', :dur, :pub)"),
                {"vid": vid, "dur": duration_sec, "pub": datetime.now(timezone.utc).replace(tzinfo=None)},
            )

        fake_captions = {
            "text": "This is the complete 5 minute manual caption transcript.",
            "language": "en",
            "source": "youtube_manual_caption",
            "caption_type": "manual",
            "duration_sec": 300,
            "word_count": 9,
        }

        with patch("core.local_transcription_worker.extract_youtube_captions", return_value=fake_captions) as mock_cap, \
             patch("core.local_transcription_worker.transcribe_audio_pcm") as mock_whisper:
            res = local_transcription_worker.run_local_worker(video_id=vid, allow_sqlite=True)

            self.assertEqual(res["manual_captions_extracted"], 1)
            self.assertEqual(res["faster_whisper_completed"], 0)
            mock_cap.assert_called_once_with(vid)
            mock_whisper.assert_not_called()

        # Verify DB row
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM transcripts WHERE video_id = :vid"), {"vid": vid}).mappings().first()
            self.assertIsNotNone(row)
            self.assertEqual(row["source"], "youtube_manual_caption")
            self.assertEqual(row["status"], "completed")
            self.assertFalse(row["is_truncated"])
            self.assertEqual(row["audio_seconds"], 300)

    # ------------------------------------------------------------------------
    # 2. 5-minute video without captions
    # ------------------------------------------------------------------------
    def test_02_five_minute_video_without_captions(self):
        """Rule 1: Video < 15 min without captions transcribes full audio with Whisper; is_truncated=False."""
        vid = "VID_5M_NOCAP"
        duration_sec = 300  # 5 minutes
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO videos (video_id, channel_id, title, duration_sec, published_at) "
                     "VALUES (:vid, 'UC_CHAN_1', '5 Min No Captions', :dur, :pub)"),
                {"vid": vid, "dur": duration_sec, "pub": datetime.now(timezone.utc).replace(tzinfo=None)},
            )

        mock_pcm = np.zeros(duration_sec * 16000, dtype=np.int16)

        with patch("core.local_transcription_worker.extract_youtube_captions", return_value=None), \
             patch("core.transcribe_videos.fetch_audio", return_value=mock_pcm), \
             patch("core.local_transcription_worker.transcribe_audio_pcm",
                   return_value=("Full 5 minute Whisper transcript", "en", 300, False)) as mock_whisper:
            res = local_transcription_worker.run_local_worker(video_id=vid, allow_sqlite=True)

            self.assertEqual(res["faster_whisper_completed"], 1)
            mock_whisper.assert_called_once()
            _, kwargs = mock_whisper.call_args
            self.assertEqual(kwargs.get("video_duration_sec"), 300)

        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM transcripts WHERE video_id = :vid"), {"vid": vid}).mappings().first()
            self.assertIsNotNone(row)
            self.assertEqual(row["source"], "faster_whisper")
            self.assertEqual(row["status"], "completed")
            self.assertFalse(row["is_truncated"])
            self.assertEqual(row["audio_seconds"], 300)

    # ------------------------------------------------------------------------
    # 3. 14-minute video without captions
    # ------------------------------------------------------------------------
    def test_03_fourteen_minute_video_without_captions(self):
        """Rule 1: Video of 14 minutes (840s < 900s) transcribes the entire video without truncation."""
        vid = "VID_14M_NOCAP"
        duration_sec = 840  # 14 minutes
        plan = plan_transcription_segments(duration_sec, video_id=vid)

        self.assertFalse(plan.is_partial)
        self.assertEqual(plan.total_audio_seconds, 840)
        self.assertEqual(len(plan.segments), 1)
        self.assertEqual(plan.segments[0].start_sec, 0.0)
        self.assertEqual(plan.segments[0].end_sec, 840.0)

        # Verify through transcribe_audio_pcm
        fake_pcm = np.zeros(16000 * 2, dtype=np.int16)  # Small stub array
        with patch("core.cloud_whisper_worker.get_whisper_model") as mock_model_getter:
            mock_model = MagicMock()
            seg_mock = MagicMock(text="Complete 14 minute content")
            info_mock = MagicMock(language="hi")
            mock_model.transcribe.return_value = ([seg_mock], info_mock)
            mock_model_getter.return_value = mock_model

            text_out, lang, sec, is_trunc = cloud_whisper_worker.transcribe_audio_pcm(
                fake_pcm,
                video_duration_sec=duration_sec,
            )
            self.assertFalse(is_trunc)
            self.assertEqual(sec, 840)
            self.assertEqual(lang, "hi")
            self.assertIn("Complete 14 minute content", text_out)

    # ------------------------------------------------------------------------
    # 4. Exactly 15-minute video without captions (Check 1: Truncation accuracy)
    # ------------------------------------------------------------------------
    def test_04_exactly_fifteen_minute_video_without_captions(self):
        """Check 1: For a 900s video:
        - If 100% of the audio is transcribed, is_truncated=False.
        - If any meaningful portion is omitted (e.g. intros/outros omitted), is_truncated=True.
        """
        duration_sec = 900  # Exactly 15 minutes

        # Case A: 100% of audio transcribed (full_if_within_budget=True)
        plan_full = plan_transcription_segments(duration_sec, video_id="VID_15M_FULL", full_if_within_budget=True)
        self.assertFalse(plan_full.is_truncated)
        self.assertEqual(plan_full.total_audio_seconds, 900)
        self.assertEqual(len(plan_full.segments), 1)
        self.assertEqual(plan_full.segments[0].start_sec, 0.0)
        self.assertEqual(plan_full.segments[0].end_sec, 900.0)

        # Case B: Meaningful portion omitted (representative sampling with intro/outro omitted)
        plan_partial = plan_transcription_segments(duration_sec, video_id="VID_15M_PARTIAL", full_if_within_budget=False)
        self.assertTrue(plan_partial.is_truncated)
        self.assertLess(plan_partial.total_audio_seconds, 900)
        self.assertEqual(plan_partial.total_audio_seconds, 864)  # 36s omitted (intro/outro)
        self.assertEqual(len(plan_partial.segments), 3)

        # Check chronological ordering and no overlaps
        for i in range(len(plan_partial.segments) - 1):
            self.assertLessEqual(plan_partial.segments[i].end_sec, plan_partial.segments[i + 1].start_sec)
        self.assertLessEqual(plan_partial.segments[-1].end_sec, 900.0)

        # Verify transcribe_audio_pcm marks is_truncated accurately
        fake_pcm = np.zeros(16000 * 2, dtype=np.int16)
        with patch("core.cloud_whisper_worker.get_whisper_model") as mock_model_getter:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = ([MagicMock(text="Segment text")], MagicMock(language="en"))
            mock_model_getter.return_value = mock_model

            # When portion is omitted -> is_truncated=True
            _, _, sec, is_trunc = cloud_whisper_worker.transcribe_audio_pcm(
                fake_pcm,
                video_duration_sec=duration_sec,
            )
            self.assertTrue(is_trunc)
            self.assertLessEqual(sec, 900)

    # ------------------------------------------------------------------------
    # 5. 30-minute video without captions
    # ------------------------------------------------------------------------
    def test_05_thirty_minute_video_without_captions(self):
        """Rule 2: 30-minute video (1800s) selects representative segments across beginning, middle, end <= 900s."""
        duration_sec = 1800  # 30 minutes
        plan = plan_transcription_segments(duration_sec, video_id="VID_30M")

        self.assertTrue(plan.is_partial)
        self.assertLessEqual(plan.total_audio_seconds, 900)
        self.assertEqual(len(plan.segments), 3)

        s1, s2, s3 = plan.segments
        # Beginning: intro omitted
        self.assertGreaterEqual(s1.start_sec, 30.0)
        # Chronological progression
        self.assertLess(s1.end_sec, s2.start_sec)
        self.assertLess(s2.end_sec, s3.start_sec)
        # End: outro omitted
        self.assertLessEqual(s3.end_sec, 1800.0)
        self.assertGreaterEqual(1800.0 - s3.end_sec, 30.0)

        # Total audio budget exactly 900s
        self.assertEqual(plan.total_audio_seconds, 900)

    # ------------------------------------------------------------------------
    # 6. 3-hour video without captions
    # ------------------------------------------------------------------------
    def test_06_three_hour_video_without_captions(self):
        """Rule 2: 3-hour video (10800s) selects representative segments totaling <= 900s in chronological order."""
        duration_sec = 10800  # 3 hours
        plan = plan_transcription_segments(duration_sec, video_id="VID_3H")

        self.assertTrue(plan.is_partial)
        self.assertLessEqual(plan.total_audio_seconds, 900)
        self.assertEqual(len(plan.segments), 3)

        s1, s2, s3 = plan.segments
        self.assertGreaterEqual(s1.start_sec, 30.0)
        # Middle segment is near the center (~5400s)
        self.assertGreater(s2.start_sec, 4000.0)
        self.assertLess(s2.end_sec, 7000.0)
        # End segment is near the end (~10700s)
        self.assertGreater(s3.start_sec, 10000.0)
        self.assertLessEqual(s3.end_sec, 10800.0)
        # Total audio budget <= 900s
        self.assertLessEqual(plan.total_audio_seconds, 900)

    # ------------------------------------------------------------------------
    # 7. Long video with captions
    # ------------------------------------------------------------------------
    def test_07_long_video_with_captions(self):
        """Rule 2 Step 2: Even for long videos (e.g. 1 hour), complete captions are used without running Whisper."""
        vid = "VID_1H_CAP"
        duration_sec = 3600  # 1 hour
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO videos (video_id, channel_id, title, duration_sec, published_at) "
                     "VALUES (:vid, 'UC_CHAN_1', '1 Hour Video with Captions', :dur, :pub)"),
                {"vid": vid, "dur": duration_sec, "pub": datetime.now(timezone.utc).replace(tzinfo=None)},
            )

        fake_captions = {
            "text": "Full 1-hour YouTube auto caption text without any Whisper inference.",
            "language": "en",
            "source": "youtube_auto_caption",
            "caption_type": "auto_generated",
            "duration_sec": 3600,
            "word_count": 10,
        }

        with patch("core.local_transcription_worker.extract_youtube_captions", return_value=fake_captions) as mock_cap, \
             patch("core.local_transcription_worker.transcribe_audio_pcm") as mock_whisper:
            res = local_transcription_worker.run_local_worker(video_id=vid, allow_sqlite=True)

            self.assertEqual(res["auto_captions_extracted"], 1)
            self.assertEqual(res["faster_whisper_completed"], 0)
            mock_cap.assert_called_once_with(vid)
            mock_whisper.assert_not_called()

        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM transcripts WHERE video_id = :vid"), {"vid": vid}).mappings().first()
            self.assertIsNotNone(row)
            self.assertEqual(row["source"], "youtube_auto_caption")
            self.assertFalse(row["is_truncated"])  # Captions are complete

    # ------------------------------------------------------------------------
    # 8. Missing or inaccessible audio
    # ------------------------------------------------------------------------
    def test_08_missing_or_inaccessible_audio(self):
        """Missing or blocked audio is marked as awaiting_authorized_audio without crashing."""
        vid = "VID_BLOCKED_AUDIO"
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO videos (video_id, channel_id, title, duration_sec, published_at) "
                     "VALUES (:vid, 'UC_CHAN_1', 'Blocked Audio Video', 600, :pub)"),
                {"vid": vid, "pub": datetime.now(timezone.utc).replace(tzinfo=None)},
            )

        with patch("core.local_transcription_worker.extract_youtube_captions", return_value=None), \
             patch("core.transcribe_videos.fetch_audio", side_effect=RuntimeError("HTTP 403 Forbidden: Sign in to confirm you are not a bot")):
            res = local_transcription_worker.run_local_worker(video_id=vid, allow_sqlite=True)

            self.assertEqual(res["awaiting_authorized_audio"], 1)
            self.assertEqual(res["faster_whisper_completed"], 0)

        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM transcripts WHERE video_id = :vid"), {"vid": vid}).mappings().first()
            self.assertIsNotNone(row)
            self.assertEqual(row["source"], "awaiting_authorized_audio")
            self.assertEqual(row["status"], "awaiting_authorized_audio")
            self.assertIn("Audio download unavailable", row["last_error"])

    # ------------------------------------------------------------------------
    # 9. Segment ordering and total duration <= 900 seconds
    # ------------------------------------------------------------------------
    def test_09_segment_ordering_and_total_duration_budget(self):
        """Across an extensive range of video durations, segments are strictly ordered and total <= 900s."""
        test_durations = [900, 950, 1000, 1200, 1500, 1800, 2400, 3600, 5400, 7200, 10800, 14400, 21600]

        for dur in test_durations:
            plan = plan_transcription_segments(dur, video_id=f"VID_{dur}S")
            self.assertTrue(plan.is_partial, f"Duration {dur} should be partial")
            self.assertLessEqual(plan.total_audio_seconds, 900, f"Duration {dur} exceeded 900s budget")

            for i in range(len(plan.segments)):
                seg = plan.segments[i]
                self.assertGreater(seg.duration_sec, 0, f"Duration {dur} segment {i} duration <= 0")
                self.assertLessEqual(seg.end_sec, dur, f"Duration {dur} segment {i} exceeded video duration")
                if i < len(plan.segments) - 1:
                    next_seg = plan.segments[i + 1]
                    self.assertLessEqual(seg.end_sec, next_seg.start_sec,
                                         f"Duration {dur} segments {i} and {i+1} overlap")

    # ------------------------------------------------------------------------
    # 10. Partial transcript metadata
    # ------------------------------------------------------------------------
    def test_10_partial_transcript_metadata(self):
        """Partial transcript preserves video ID, channel ID, duration, timestamps, model, and is_truncated."""
        vid = "VID_META_CHECK"
        cid = "UC_CHAN_1"
        duration_sec = 2400  # 40 minutes

        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO videos (video_id, channel_id, title, duration_sec, published_at) "
                     "VALUES (:vid, :cid, 'Metadata Check Video', :dur, :pub)"),
                {"vid": vid, "cid": cid, "dur": duration_sec, "pub": datetime.now(timezone.utc).replace(tzinfo=None)},
            )

        mock_pcm = np.zeros(duration_sec * 16000, dtype=np.int16)
        with patch("core.local_transcription_worker.extract_youtube_captions", return_value=None), \
             patch("core.transcribe_videos.fetch_audio", return_value=mock_pcm), \
             patch("core.cloud_whisper_worker.get_whisper_model") as mock_model_getter:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = ([MagicMock(text="Segment speech content")], MagicMock(language="hi"))
            mock_model_getter.return_value = mock_model

            res = local_transcription_worker.run_local_worker(video_id=vid, allow_sqlite=True)
            self.assertEqual(res["faster_whisper_completed"], 1)

        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM transcripts WHERE video_id = :vid"), {"vid": vid}).mappings().first()
            self.assertIsNotNone(row)
            self.assertEqual(row["video_id"], vid)
            self.assertEqual(row["channel_id"], cid)
            self.assertEqual(row["source"], "faster_whisper")
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["language"], "hi")
            self.assertEqual(row["model_name"], "whisper-tiny")
            self.assertTrue(row["is_truncated"])  # Partial indicator
            self.assertLessEqual(row["audio_seconds"], 900)

            # Check timestamp markers in transcript_text
            text_body = row["transcript_text"]
            self.assertIn("[PARTIAL TRANSCRIPT", text_body)
            self.assertIn("Processed:", text_body)
            self.assertIn("total duration", text_body)
            self.assertIn("[Selected Segments:", text_body)

    # ------------------------------------------------------------------------
    # 11. Idempotency and preservation of completed transcripts
    # ------------------------------------------------------------------------
    def test_11_idempotency_and_preservation_of_completed_transcripts(self):
        """Completed transcripts (such as production candidate records) are NEVER overwritten by partial transcripts."""
        vid = "VID_PRODUCTION_COMPLETED"
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO videos (video_id, channel_id, title, duration_sec, published_at) "
                     "VALUES (:vid, 'UC_CHAN_1', 'Production Video', 3600, :pub)"),
                {"vid": vid, "pub": datetime.now(timezone.utc).replace(tzinfo=None)},
            )
            conn.execute(
                text("INSERT INTO transcripts (video_id, channel_id, source, caption_type, status, language, "
                     "model_name, transcript_text, audio_seconds, word_count, is_truncated) "
                     "VALUES (:vid, 'UC_CHAN_1', 'youtube_auto_caption', 'auto_generated', 'completed', 'en', "
                     "'youtube-captions', 'Production complete transcript preserved untouched', 3600, 6, 0)"),
                {"vid": vid},
            )

        with patch("core.local_transcription_worker.extract_youtube_captions") as mock_cap, \
             patch("core.local_transcription_worker.transcribe_audio_pcm") as mock_whisper:
            res = local_transcription_worker.run_local_worker(video_id=vid, allow_sqlite=True)

            self.assertEqual(res["reused_existing"], 1)
            self.assertEqual(res["faster_whisper_completed"], 0)
            mock_cap.assert_not_called()
            mock_whisper.assert_not_called()

        # Confirm data in DB remains completely unchanged
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM transcripts WHERE video_id = :vid"), {"vid": vid}).mappings().first()
            self.assertEqual(row["transcript_text"], "Production complete transcript preserved untouched")
            self.assertEqual(row["source"], "youtube_auto_caption")
            self.assertFalse(row["is_truncated"])

    # ------------------------------------------------------------------------
    # 12. Supabase persistence using mocked database writes
    # ------------------------------------------------------------------------
    def test_12_supabase_persistence_mocked_writes(self):
        """Validates that save_transcript_record writes correct parameters matching Supabase schema."""
        vid = "VID_SUPABASE_MOCK"
        cid = "UC_CHAN_1"

        with patch("core.config.get_engine") as mock_engine_fn:
            mock_engine = MagicMock()
            mock_conn = MagicMock()
            mock_engine.begin.return_value.__enter__.return_value = mock_conn
            mock_engine_fn.return_value = mock_engine

            cloud_whisper_worker.save_transcript_record(
                video_id=vid,
                channel_id=cid,
                source="faster_whisper",
                status="completed",
                language="hi",
                caption_type="partial_15m",
                model_name="whisper-tiny",
                transcript_text="[PARTIAL TRANSCRIPT] Content...",
                audio_seconds=900,
                is_truncated=True,
                video_duration_sec=2400,
            )

            # Verify execute was called
            self.assertTrue(mock_conn.execute.called)
            sql_arg, params_arg = mock_conn.execute.call_args_list[0][0]
            self.assertEqual(params_arg["video_id"], vid)
            self.assertEqual(params_arg["channel_id"], cid)
            self.assertEqual(params_arg["source"], "faster_whisper")
            self.assertEqual(params_arg["status"], "completed")
            self.assertEqual(params_arg["language"], "hi")
            self.assertEqual(params_arg["audio_seconds"], 900)
            self.assertTrue(params_arg["is_truncated"])

    # ------------------------------------------------------------------------
    # 13. Chapter-based smart selection
    # ------------------------------------------------------------------------
    def test_13_chapter_based_smart_selection(self):
        """Identifies informative chapters and omits intros, outros, and sponsor advertisements."""
        chapters = [
            {"title": "Welcome and Intro", "start_time": 0.0, "end_time": 45.0},
            {"title": "Core Architecture Overview", "start_time": 45.0, "end_time": 500.0},
            {"title": "Our Sponsor: SuperVPN", "start_time": 500.0, "end_time": 580.0},
            {"title": "Implementation Deep Dive", "start_time": 580.0, "end_time": 1200.0},
            {"title": "Key Takeaways & Lessons", "start_time": 1200.0, "end_time": 1750.0},
            {"title": "Outro, Thanks and Subscribe", "start_time": 1750.0, "end_time": 1800.0},
        ]

        plan = plan_transcription_segments(1800, chapters=chapters, video_id="VID_CHAPS")
        self.assertTrue(plan.is_partial)
        self.assertEqual(plan.strategy, "chapters")
        self.assertLessEqual(plan.total_audio_seconds, 900)

        # Confirm excluded chapters are absent
        for seg in plan.segments:
            label_lower = seg.label.lower()
            self.assertNotIn("intro", label_lower)
            self.assertNotIn("sponsor", label_lower)
            self.assertNotIn("outro", label_lower)

    # ------------------------------------------------------------------------
    # 14. Synthetic seeking validation with frequency markers
    # ------------------------------------------------------------------------
    def test_14_synthetic_seeking_validation_frequency_markers(self):
        """Validate actual FFmpeg input seeking behavior using synthetic media with tone markers."""
        import subprocess
        from core.transcribe_videos import _ffmpeg, read_pcm_segment

        with tempfile.TemporaryDirectory(prefix="mediaintel_seeking_test_") as tmp:
            fpath = Path(tmp) / "synthetic_marker.wav"
            # 3 segments: 0..10s = 200 Hz, 10..20s = 800 Hz, 20..30s = 2000 Hz
            cmd = [
                _ffmpeg(), "-y",
                "-f", "lavfi", "-i", "sine=frequency=200:duration=10",
                "-f", "lavfi", "-i", "sine=frequency=800:duration=10",
                "-f", "lavfi", "-i", "sine=frequency=2000:duration=10",
                "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]",
                "-map", "[out]", str(fpath),
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            def dominant_freq(samples: np.ndarray) -> float:
                fft = np.fft.rfft(samples)
                freqs = np.fft.rfftfreq(len(samples), 1 / 16000)
                return float(freqs[np.argmax(np.abs(fft))])

            # Extract 2-second segments from each region via read_pcm_segment
            pcm_beg = read_pcm_segment(str(fpath), None, start_sec=4.0, duration_sec=2.0)
            pcm_mid = read_pcm_segment(str(fpath), None, start_sec=14.0, duration_sec=2.0)
            pcm_end = read_pcm_segment(str(fpath), None, start_sec=24.0, duration_sec=2.0)

            # Bounded memory: each segment must have exactly 2 * 16000 samples
            self.assertEqual(len(pcm_beg), 32000)
            self.assertEqual(len(pcm_mid), 32000)
            self.assertEqual(len(pcm_end), 32000)

            # Frequency verification proves actual timestamp extraction (not reusing first segment)
            self.assertAlmostEqual(dominant_freq(pcm_beg), 200.0, delta=20.0)
            self.assertAlmostEqual(dominant_freq(pcm_mid), 800.0, delta=20.0)
            self.assertAlmostEqual(dominant_freq(pcm_end), 2000.0, delta=20.0)

            # Middle and end segments are distinct from beginning segment
            self.assertFalse(np.array_equal(pcm_beg, pcm_mid))
            self.assertFalse(np.array_equal(pcm_beg, pcm_end))

    # ------------------------------------------------------------------------
    # 15. 30-minute video beginning, middle, and end timestamps
    # ------------------------------------------------------------------------
    def test_15_thirty_minute_video_timestamps_and_no_reuse(self):
        """A 30-minute video uses correct beginning, middle, and end timestamps without reusing the start."""
        vid = "VID_30M_TIMESTAMPS"
        duration_sec = 1800  # 30 minutes
        plan = plan_transcription_segments(duration_sec, video_id=vid)

        self.assertTrue(plan.is_partial)
        self.assertEqual(len(plan.segments), 3)

        s_beg, s_mid, s_end = plan.segments
        # Beginning starts after intro skip
        self.assertGreaterEqual(s_beg.start_sec, 0.0)
        self.assertLess(s_beg.end_sec, s_mid.start_sec)

        # Middle starts well into the video
        self.assertGreater(s_mid.start_sec, s_beg.end_sec)
        self.assertLess(s_mid.end_sec, s_end.start_sec)

        # End starts in the final portion
        self.assertGreater(s_end.start_sec, s_mid.end_sec)
        self.assertLessEqual(s_end.end_sec, duration_sec)

        # Total audio is at most 900 seconds
        self.assertLessEqual(plan.total_audio_seconds, 900)

        # Verify that fetch_audio_segments invokes read_pcm_segment with these exact timestamps
        with patch("core.transcribe_videos.resolve_stream", return_value=("http://example.com/audio", {})) as mock_resolve, \
             patch("core.transcribe_videos.read_pcm_segment") as mock_read_seg:
            dummy_pcm = np.zeros(16000 * 300, dtype=np.int16)
            mock_read_seg.return_value = dummy_pcm

            from core.transcribe_videos import fetch_audio_segments
            chunks = fetch_audio_segments(vid, plan)

            self.assertEqual(len(chunks), 3)
            self.assertEqual(mock_read_seg.call_count, 3)

            calls = mock_read_seg.call_args_list
            # Beginning
            self.assertEqual(calls[0][1]["start_sec"], s_beg.start_sec)
            self.assertEqual(calls[0][1]["duration_sec"], s_beg.duration_sec)
            # Middle
            self.assertEqual(calls[1][1]["start_sec"], s_mid.start_sec)
            self.assertEqual(calls[1][1]["duration_sec"], s_mid.duration_sec)
            # End
            self.assertEqual(calls[2][1]["start_sec"], s_end.start_sec)
            self.assertEqual(calls[2][1]["duration_sec"], s_end.duration_sec)

    # ------------------------------------------------------------------------
    # 16. 3-hour video does not reuse first 15 minutes
    # ------------------------------------------------------------------------
    def test_16_three_hour_video_does_not_reuse_first_15_minutes(self):
        """A 3-hour video does not reuse the first 15 minutes for middle and end segments."""
        vid = "VID_3H_SEEK"
        duration_sec = 10800  # 3 hours = 10,800 seconds
        plan = plan_transcription_segments(duration_sec, video_id=vid)

        self.assertTrue(plan.is_partial)
        self.assertEqual(len(plan.segments), 3)

        s_beg, s_mid, s_end = plan.segments

        # Middle and end segments must start far beyond the first 15 minutes (900s)
        self.assertGreater(s_mid.start_sec, 900.0, "Middle segment must start past 900s")
        self.assertGreater(s_end.start_sec, 5000.0, "End segment must start past 5000s")
        self.assertLessEqual(s_end.end_sec, duration_sec)

        # Combined audio duration is bounded
        self.assertLessEqual(plan.total_audio_seconds, 900)

        # Verify seeking targets
        with patch("core.transcribe_videos.resolve_stream", return_value=("http://example.com/3h_audio", {})), \
             patch("core.transcribe_videos.read_pcm_segment") as mock_read_seg:
            mock_read_seg.return_value = np.zeros(16000 * 300, dtype=np.int16)

            from core.transcribe_videos import fetch_audio_segments
            chunks = fetch_audio_segments(vid, plan)

            self.assertEqual(len(chunks), 3)
            # Check seek arguments passed to read_pcm_segment
            start_times = [c[1]["start_sec"] for c in mock_read_seg.call_args_list]
            self.assertIn(s_beg.start_sec, start_times)
            self.assertIn(s_mid.start_sec, start_times)
            self.assertIn(s_end.start_sec, start_times)

            # Neither middle nor end starts at 0 or within first 15 minutes
            self.assertNotEqual(start_times[1], 0.0)
            self.assertNotEqual(start_times[2], 0.0)
            self.assertGreater(start_times[1], 900.0)
            self.assertGreater(start_times[2], 900.0)

    # ------------------------------------------------------------------------
    # 17. Combined audio duration is at most 900 seconds
    # ------------------------------------------------------------------------
    def test_17_combined_audio_duration_at_most_900_seconds_across_durations(self):
        """For any video duration >= 15 min, combined audio duration is at most 900 seconds."""
        test_durations = [900, 901, 1200, 1800, 2700, 3600, 7200, 10800, 36000]
        for dur in test_durations:
            plan = plan_transcription_segments(dur)
            total = sum(s.duration_sec for s in plan.segments)
            self.assertLessEqual(
                total,
                900.0,
                f"Combined segment duration {total}s exceeds 900s for video duration {dur}s",
            )
            self.assertLessEqual(
                plan.total_audio_seconds,
                900,
                f"Plan total_audio_seconds {plan.total_audio_seconds} exceeds 900 for video duration {dur}s",
            )

    # ------------------------------------------------------------------------
    # 18. Segment text is combined in chronological order
    # ------------------------------------------------------------------------
    def test_18_segment_text_combined_in_chronological_order(self):
        """Segment transcripts are assembled strictly in chronological order with accurate time headers."""
        plan = TranscriptionPlan(
            video_id="VID_CHRONO",
            video_duration_sec=1800,
            is_partial=True,
            segments=[
                AudioSegment(30.0, 330.0, 300.0, "Beginning (intro omitted)"),
                AudioSegment(750.0, 1050.0, 300.0, "Middle / core transition"),
                AudioSegment(1464.0, 1764.0, 300.0, "End / synthesis (outro omitted)"),
            ],
            total_audio_seconds=900,
            strategy="distributed_temporal",
            description="3 representative segments",
        )

        segment_results = [
            (plan.segments[0], "First segment discussion about machine learning foundations."),
            (plan.segments[1], "Second segment deep dive into model architectures and scaling."),
            (plan.segments[2], "Third segment conclusions on future trends and deployment."),
        ]

        combined = combine_segment_transcripts(segment_results, plan)

        # Header check
        self.assertIn("[PARTIAL TRANSCRIPT — Processed: 900s of 1800s total duration]", combined)
        # Chronological sequence check
        pos1 = combined.find("First segment")
        pos2 = combined.find("Second segment")
        pos3 = combined.find("Third segment")
        self.assertTrue(0 < pos1 < pos2 < pos3, "Segments must appear in strict chronological order")

        # Timestamp headers check
        self.assertIn("[00:30 - 05:30] Beginning (intro omitted):", combined)
        self.assertIn("[12:30 - 17:30] Middle / core transition:", combined)
        self.assertIn("[24:24 - 29:24] End / synthesis (outro omitted):", combined)

    # ------------------------------------------------------------------------
    # 19. Exact 15-minute coverage: full vs partial
    # ------------------------------------------------------------------------
    def test_19_exact_15_minute_coverage_full_vs_partial(self):
        """Exact 900-second video: 100% coverage is non-truncated; omitted portions mark is_truncated=True."""
        # Case A: 100% coverage of 900s video
        plan_full = TranscriptionPlan(
            video_id="VID_900_FULL",
            video_duration_sec=900,
            is_partial=False,
            segments=[AudioSegment(0.0, 900.0, 900.0, "Full video audio")],
            total_audio_seconds=900,
            strategy="full_video",
            description="Full video transcript",
        )
        self.assertFalse(plan_full.is_truncated, "100% coverage must have is_truncated=False")

        # Case B: 900s video with intro/outro omitted
        plan_partial = plan_transcription_segments(900, video_id="VID_900_PART")
        self.assertTrue(plan_partial.is_truncated, "Partial coverage omitting intro/outro must have is_truncated=True")

        # Case C: 900s video transcribed from full captions
        fake_captions = {
            "text": "Full caption text covering the complete 900s video duration.",
            "language": "en",
            "source": "youtube_auto_caption",
            "caption_type": "auto_generated",
            "duration_sec": 900,
            "word_count": 9,
        }
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO videos (video_id, channel_id, title, duration_sec, published_at) "
                     "VALUES ('VID_900_CAP', 'UC_CHAN_1', '900s Cap Video', 900, :pub)"),
                {"pub": datetime.now(timezone.utc).replace(tzinfo=None)},
            )
        with patch("core.local_transcription_worker.extract_youtube_captions", return_value=fake_captions):
            res = local_transcription_worker.run_local_worker(video_id="VID_900_CAP", allow_sqlite=True)
            self.assertEqual(res["auto_captions_extracted"], 1)

        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT is_truncated FROM transcripts WHERE video_id = 'VID_900_CAP'")).fetchone()
            self.assertFalse(row[0], "Caption-based transcripts must have is_truncated=False")

    # ------------------------------------------------------------------------
    # 20. Completed transcripts are preserved
    # ------------------------------------------------------------------------
    def test_20_existing_completed_transcripts_are_preserved(self):
        """Worker preserves existing completed transcripts untouched."""
        vid = "VID_PRESERVED_CHECK"
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO videos (video_id, channel_id, title, duration_sec, published_at) "
                     "VALUES (:vid, 'UC_CHAN_1', 'Preserved Video', 1800, :pub)"),
                {"vid": vid, "pub": datetime.now(timezone.utc).replace(tzinfo=None)},
            )
            conn.execute(
                text("INSERT INTO transcripts (video_id, channel_id, source, caption_type, status, language, "
                     "model_name, transcript_text, audio_seconds, word_count, is_truncated) "
                     "VALUES (:vid, 'UC_CHAN_1', 'faster_whisper', 'partial_15m', 'completed', 'en', "
                     "'whisper-tiny', 'Existing complete transcript', 900, 3, 1)"),
                {"vid": vid},
            )

        with patch("core.local_transcription_worker.extract_youtube_captions") as mock_cap, \
             patch("core.local_transcription_worker.transcribe_audio_pcm") as mock_whisper:
            res = local_transcription_worker.run_local_worker(video_id=vid, allow_sqlite=True)

            self.assertEqual(res["reused_existing"], 1)
            mock_cap.assert_not_called()
            mock_whisper.assert_not_called()

        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT transcript_text FROM transcripts WHERE video_id = :vid"), {"vid": vid}).fetchone()
            self.assertEqual(row[0], "Existing complete transcript")

    # ------------------------------------------------------------------------
    # 21. Audio retrieval errors cannot produce falsely completed records
    # ------------------------------------------------------------------------
    def test_21_audio_retrieval_errors_cannot_produce_completed_records(self):
        """Audio retrieval errors safely mark awaiting_authorized_audio and NEVER mark completed."""
        vid = "VID_ERROR_HANDLING"
        with self.engine.begin() as conn:
            conn.execute(
                text("INSERT INTO videos (video_id, channel_id, title, duration_sec, published_at) "
                     "VALUES (:vid, 'UC_CHAN_1', 'Error Video', 1800, :pub)"),
                {"vid": vid, "pub": datetime.now(timezone.utc).replace(tzinfo=None)},
            )

        with patch("core.local_transcription_worker.extract_youtube_captions", return_value=None), \
             patch("core.transcribe_videos.fetch_audio_segments", side_effect=RuntimeError("Audio seeking timed out [750s+300s]")):
            res = local_transcription_worker.run_local_worker(video_id=vid, allow_sqlite=True)

            self.assertEqual(res["awaiting_authorized_audio"], 1)
            self.assertEqual(res["faster_whisper_completed"], 0)

        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT source, status, last_error, transcript_text FROM transcripts WHERE video_id = :vid"),
                {"vid": vid},
            ).mappings().first()

            self.assertIsNotNone(row)
            self.assertEqual(row["source"], "awaiting_authorized_audio")
            self.assertEqual(row["status"], "awaiting_authorized_audio")
            self.assertNotEqual(row["status"], "completed")
            self.assertIn("Audio download unavailable", row["last_error"])
            self.assertIsNone(row["transcript_text"])

    # ------------------------------------------------------------------------
    # 22. read_pcm_segment direct failure handling
    # ------------------------------------------------------------------------
    def test_22_read_pcm_segment_direct_failure_handling(self):
        """read_pcm_segment raises RuntimeError on bad/missing source and ValueError on non-positive duration."""
        from core.transcribe_videos import read_pcm_segment

        # Non-positive duration
        with self.assertRaises(ValueError):
            read_pcm_segment("nonexistent.wav", None, start_sec=0.0, duration_sec=0.0)

        with self.assertRaises(ValueError):
            read_pcm_segment("nonexistent.wav", None, start_sec=0.0, duration_sec=-10.0)

        # Missing file / stream failure
        with self.assertRaises(RuntimeError) as ctx:
            read_pcm_segment("nonexistent_stream_path_12345.wav", None, start_sec=10.0, duration_sec=5.0)
        self.assertIn("audio segment extraction failed", str(ctx.exception))

    # ------------------------------------------------------------------------
    # 23. Downstream insights recommendation source compatibility
    # ------------------------------------------------------------------------
    def test_23_insights_recommendations_support_faster_whisper(self):
        """core.insights.build_recommendations recognizes faster_whisper transcripts for spoken topic differentiation."""
        import pandas as pd
        from core.insights import build_recommendations

        # Test with faster_whisper transcript
        data = {
            "topics": pd.DataFrame(),
            "transcripts": pd.DataFrame([
                {
                    "video_id": "v1",
                    "source": "faster_whisper",
                    "transcript_text": "system design architecture system design database scaling system design",
                },
                {
                    "video_id": "v2",
                    "source": "faster_whisper",
                    "transcript_text": "system design principles system design microservices architecture",
                },
            ]),
            "aspects": pd.DataFrame(),
            "relevance": pd.DataFrame(columns=["relevance_index", "title", "channel_name"]),
            "alerts": pd.DataFrame(),
        }
        recs = build_recommendations(data)
        recs_dict = dict(recs)
        self.assertIn("Differentiate from what competitors already say.", recs_dict)
        self.assertIn("System Design", recs_dict["Differentiate from what competitors already say."])


if __name__ == "__main__":
    unittest.main()
