-- Migration 002: Add channel_run_coverage table for deterministic on-demand collection telemetry
-- Safe, additive migration: never drops or modifies existing tables or columns

CREATE TABLE IF NOT EXISTS channel_run_coverage (
    channel_id VARCHAR(50) PRIMARY KEY REFERENCES channels(channel_id) ON DELETE CASCADE,
    channel_name VARCHAR(100) NOT NULL,
    target_videos INTEGER DEFAULT 10,
    selected_videos INTEGER DEFAULT 0,
    target_comments INTEGER DEFAULT 1000,
    collected_comments INTEGER DEFAULT 0,
    transcription_candidates INTEGER DEFAULT 2,
    completed_transcripts INTEGER DEFAULT 0,
    missing_data_reason VARCHAR(255),
    last_run_status VARCHAR(50) DEFAULT 'success',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_channel_run_coverage_status ON channel_run_coverage(last_run_status);
