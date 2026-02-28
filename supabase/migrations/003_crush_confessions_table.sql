-- Create crush_confessions table for tracking romantic interest expressions
CREATE TABLE IF NOT EXISTS crush_confessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    anonymous BOOLEAN NOT NULL DEFAULT true,
    name TEXT,
    contact TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create index on session_id for fast lookups
CREATE INDEX IF NOT EXISTS idx_crush_confessions_session_id ON crush_confessions(session_id);

-- Create index on timestamp for chronological queries
CREATE INDEX IF NOT EXISTS idx_crush_confessions_timestamp ON crush_confessions(timestamp DESC);

-- Add comment on table
COMMENT ON TABLE crush_confessions IS 'Tracks crush confessions from Portfolia visitors, supporting both anonymous and revealed identities';

-- Add comments on columns
COMMENT ON COLUMN crush_confessions.id IS 'Unique identifier for the confession';
COMMENT ON COLUMN crush_confessions.session_id IS 'Session ID of the user who made the confession';
COMMENT ON COLUMN crush_confessions.anonymous IS 'True if user chose to stay anonymous, false if they revealed themselves';
COMMENT ON COLUMN crush_confessions.name IS 'User name (null if anonymous)';
COMMENT ON COLUMN crush_confessions.contact IS 'Contact info like email or phone (null if anonymous)';
COMMENT ON COLUMN crush_confessions.timestamp IS 'When the confession was made';
COMMENT ON COLUMN crush_confessions.created_at IS 'Database creation timestamp';
