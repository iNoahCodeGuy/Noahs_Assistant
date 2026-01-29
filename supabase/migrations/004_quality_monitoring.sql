-- Quality Monitoring Migration
-- Tracks answer_quality_warning frequency, response paths, and flagged responses
-- for data-driven decisions about whether proofreading is needed.
--
-- Run this file in your Supabase SQL Editor to set up the database
-- Dashboard → SQL Editor → New Query → Paste and Run

-- ============================================================================
-- TABLE: quality_events
-- Purpose: Track quality validation events for monitoring and analysis
-- ============================================================================
CREATE TABLE IF NOT EXISTS quality_events (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    conversation_turn INTEGER,
    warning_type TEXT,  -- NULL if no warning, else "answer_relevance_low_0.25" etc
    response_path TEXT NOT NULL,  -- "rag" or "template"
    query_preview TEXT,  -- First 200 chars of query
    answer_preview TEXT,  -- First 500 chars of answer (for flagged only)
    retry_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,  -- role_mode, menu_branch, etc.
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS quality_events_session_idx ON quality_events (session_id);
CREATE INDEX IF NOT EXISTS quality_events_warning_type_idx ON quality_events (warning_type);
CREATE INDEX IF NOT EXISTS quality_events_created_at_idx ON quality_events (created_at DESC);
CREATE INDEX IF NOT EXISTS quality_events_response_path_idx ON quality_events (response_path);

-- Enable Row Level Security (RLS)
ALTER TABLE quality_events ENABLE ROW LEVEL SECURITY;

-- Policy: Allow service role to manage all data
CREATE POLICY "Service role can manage quality_events"
ON quality_events FOR ALL
TO service_role
USING (true);

-- ============================================================================
-- VIEW: quality_summary
-- Purpose: Pre-aggregated view for weekly analysis
-- ============================================================================
CREATE OR REPLACE VIEW quality_summary AS
SELECT
    response_path,
    warning_type,
    COUNT(*) as occurrences,
    AVG(conversation_turn) as avg_turn,
    AVG(retry_count) as avg_retry_count,
    DATE_TRUNC('day', created_at) as day
FROM quality_events
GROUP BY response_path, warning_type, DATE_TRUNC('day', created_at);

-- ============================================================================
-- COMPLETE!
-- ============================================================================
-- Your quality monitoring system is now ready.
--
-- Next steps:
-- 1. Update assistant/analytics/supabase_analytics.py to add logging methods
-- 2. Integrate logging into stage5_quality_validation.py
-- 3. Track retry events in stage6_formatting_nodes.py
-- 4. Use scripts/review_quality_events.py to review flagged responses
-- ============================================================================
