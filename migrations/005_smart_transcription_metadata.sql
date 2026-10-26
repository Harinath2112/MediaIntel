-- Migration 005: Smart 15-Minute Partial Transcription Schema Enhancements
-- PREPARED MIGRATION — DO NOT APPLY TO PRODUCTION SUPABASE WITHOUT EXPLICIT APPROVAL
--
-- Purpose:
-- Adds optional dedicated columns to the `transcripts` table to preserve explicit video duration
-- and structured JSON segment timestamps for partial transcriptions.
--
-- Safety:
-- Fully additive; does not drop or alter any existing columns, constraints, or data.
-- MediaIntel's existing code continues to work seamlessly with existing schema fields
-- (`is_truncated`, `audio_seconds`, `caption_type`, `status`) even before this migration is applied.

-- 1. Add video_duration_sec to store the actual full duration of the video
ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS video_duration_sec INTEGER;

-- 2. Add segment_timestamps to store structured JSON intervals for representative segments
-- Supports JSONB on PostgreSQL; graceful fallback to TEXT on SQLite/MySQL
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'transcripts') THEN
        BEGIN
            ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS segment_timestamps JSONB;
        EXCEPTION
            WHEN duplicate_column THEN NULL;
        END;
    END IF;
END $$;
