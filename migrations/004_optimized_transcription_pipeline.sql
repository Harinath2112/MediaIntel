-- Migration 004: Optimized Transcription Pipeline schema enhancements
-- Additive migration: Safe to run multiple times, does not drop or alter existing data.

-- 1. Expand source column to VARCHAR(50) to support new provenance types:
--    'youtube_manual_caption', 'youtube_auto_caption', 'faster_whisper', 'awaiting_authorized_audio'
ALTER TABLE transcripts ALTER COLUMN source TYPE VARCHAR(50);

-- 2. Add channel_id for fast channel-level transcript queries
ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS channel_id VARCHAR(50);

-- 3. Add caption_type to distinguish manual vs auto_generated captions
ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS caption_type VARCHAR(20);

-- 4. Add status to track processing state:
--    'completed', 'awaiting_authorized_audio', 'failed', 'reused_existing'
ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS status VARCHAR(30) DEFAULT 'completed';

-- 5. Add is_truncated to indicate bounded transcription duration
ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS is_truncated BOOLEAN DEFAULT FALSE;
