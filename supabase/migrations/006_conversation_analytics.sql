-- Migration 006: Conversation-level analytics tables
-- Tracks: full conversations, turn counts, data capture rates, capture turn numbers

-- ── Session-level metrics ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversation_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT UNIQUE NOT NULL,
    role TEXT,
    visitor_type TEXT DEFAULT 'unknown',
    turn_count INTEGER DEFAULT 0,
    data_captured BOOLEAN DEFAULT FALSE,
    capture_turn INTEGER,           -- which turn data was captured (null if none)
    capture_type TEXT,              -- 'recruiter_lead', 'crush_confession', or null
    referral_source TEXT,
    topics_discussed TEXT[] DEFAULT '{}',          -- accumulated topic list
    projects_asked_about TEXT[] DEFAULT '{}',      -- which projects were discussed
    max_depth_level INTEGER DEFAULT 1,             -- highest depth reached (1=quick, 2=medium, 3=deep-dive)
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conv_sessions_session_id ON conversation_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_conv_sessions_started_at ON conversation_sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_conv_sessions_data_captured ON conversation_sessions(data_captured);

-- ── Per-message log (full transcript) ──────────────────────────────────
CREATE TABLE IF NOT EXISTS conversation_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    role TEXT NOT NULL,              -- 'user' or 'assistant'
    content TEXT NOT NULL,
    message_intent TEXT,             -- intent classification for this turn
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conv_messages_session_id ON conversation_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_conv_messages_session_turn ON conversation_messages(session_id, turn_number);

COMMENT ON TABLE conversation_sessions IS 'Per-conversation metrics: turn counts, capture rates, visitor types';
COMMENT ON TABLE conversation_messages IS 'Full conversation transcripts for review and analysis';
