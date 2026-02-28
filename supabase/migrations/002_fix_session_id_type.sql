-- Fix session_id UUID constraint
-- Issue: Frontend generating string IDs like "session_4t1jo080e_1760203305367"
-- Solution: Change session_id from UUID to TEXT

-- Drop existing constraint
ALTER TABLE messages ALTER COLUMN session_id TYPE TEXT;

-- Update index to use TEXT
DROP INDEX IF EXISTS messages_session_id_idx;
CREATE INDEX messages_session_id_idx ON messages (session_id);

-- Add comment explaining change
COMMENT ON COLUMN messages.session_id IS 'Session identifier - accepts any string format (UUID or custom format)';
