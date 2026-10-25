-- Migration 003: Disentangle channel coverage telemetry into distinct pipeline stages
-- Additive migration: Safe to run multiple times, does not drop or alter existing data.

ALTER TABLE channel_run_coverage ADD COLUMN IF NOT EXISTS video_status VARCHAR(50) DEFAULT 'Completed';
ALTER TABLE channel_run_coverage ADD COLUMN IF NOT EXISTS comment_status VARCHAR(50) DEFAULT 'Completed';
ALTER TABLE channel_run_coverage ADD COLUMN IF NOT EXISTS sentiment_status VARCHAR(50) DEFAULT 'Completed';
ALTER TABLE channel_run_coverage ADD COLUMN IF NOT EXISTS transcription_status VARCHAR(50) DEFAULT 'Audio Pending';
