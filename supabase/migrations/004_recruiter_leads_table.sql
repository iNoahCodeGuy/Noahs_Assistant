-- Migration 004: Create recruiter_leads table for hiring manager data capture
-- Part of the Conversation Strategy Overhaul (earned data capture system)

CREATE TABLE IF NOT EXISTS recruiter_leads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    name TEXT,
    email TEXT,
    phone TEXT,
    company TEXT,
    message TEXT,
    visitor_type TEXT NOT NULL DEFAULT 'hiring_manager',
    buying_signals_count INTEGER DEFAULT 0,
    message_count INTEGER DEFAULT 0,
    capture_trigger TEXT,  -- 'intent_to_connect', 'soft_offer', 'direct_request'
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_recruiter_leads_session_id ON recruiter_leads(session_id);
CREATE INDEX IF NOT EXISTS idx_recruiter_leads_timestamp ON recruiter_leads(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_recruiter_leads_email ON recruiter_leads(email);

COMMENT ON TABLE recruiter_leads IS 'Captures hiring manager contact info from Portfolia earned data capture flow';
