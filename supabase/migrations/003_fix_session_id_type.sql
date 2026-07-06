-- Fix session_id UUID constraint
-- Issue: Frontend generating string IDs like "session_4t1jo080e_1760203305367"
-- Solution: Change session_id from UUID to TEXT

-- messages_with_retrieval (001) depends on this column — Postgres refuses to
-- alter a column's type under a view. Drop and recreate it around the change.
-- (Found applying migrations to a fresh project on 2026-07-05; the original
-- database predated strict ordering.)
DROP VIEW IF EXISTS messages_with_retrieval;

-- Drop existing constraint
ALTER TABLE messages ALTER COLUMN session_id TYPE TEXT;

-- Recreate the view exactly as defined in 001
CREATE OR REPLACE VIEW messages_with_retrieval AS
SELECT
    m.id,
    m.session_id,
    m.role_mode,
    m.query,
    m.answer,
    m.latency_ms,
    m.created_at,
    r.topk_ids,
    r.scores,
    r.grounded
FROM messages m
LEFT JOIN retrieval_logs r ON m.id = r.message_id;

-- Update index to use TEXT
DROP INDEX IF EXISTS messages_session_id_idx;
CREATE INDEX messages_session_id_idx ON messages (session_id);

-- Add comment explaining change
COMMENT ON COLUMN messages.session_id IS 'Session identifier - accepts any string format (UUID or custom format)';
