-- Migration: Add confessions table and update feedback table
-- Purpose: Support confession feature and improve feedback tracking
-- Run in Supabase SQL Editor

-- ============================================================================
-- TABLE: confessions
-- Purpose: Store anonymous or named confession submissions
-- ============================================================================
CREATE TABLE IF NOT EXISTS confessions (
    id BIGSERIAL PRIMARY KEY,
    message TEXT NOT NULL CHECK (char_length(message) >= 10 AND char_length(message) <= 1000),
    is_anonymous BOOLEAN DEFAULT true,
    name TEXT,
    email TEXT,
    phone TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS confessions_created_at_idx 
ON confessions (created_at DESC);

CREATE INDEX IF NOT EXISTS confessions_is_anonymous_idx 
ON confessions (is_anonymous);

-- Enable RLS
ALTER TABLE confessions ENABLE ROW LEVEL SECURITY;

-- Policy: Service role can manage confessions
CREATE POLICY "Service role can manage confessions"
ON confessions FOR ALL
TO service_role
USING (true);

-- Policy: Prevent public read access (privacy)
-- Only service role can read confessions

-- ============================================================================
-- TABLE: feedback (update to add missing columns)
-- ============================================================================
-- Add missing columns if they don't exist
DO $$ 
BEGIN
    -- Add user_name column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'feedback' AND column_name = 'user_name'
    ) THEN
        ALTER TABLE feedback ADD COLUMN user_name TEXT;
    END IF;

    -- Add user_phone column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'feedback' AND column_name = 'user_phone'
    ) THEN
        ALTER TABLE feedback ADD COLUMN user_phone TEXT;
    END IF;

    -- Add user_email column if it doesn't exist (rename from email if needed)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'feedback' AND column_name = 'user_email'
    ) THEN
        -- Check if 'email' column exists and rename it
        IF EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'feedback' AND column_name = 'email'
        ) THEN
            ALTER TABLE feedback RENAME COLUMN email TO user_email;
        ELSE
            ALTER TABLE feedback ADD COLUMN user_email TEXT;
        END IF;
    END IF;
END $$;

-- ============================================================================
-- TABLE: sms_logs (optional, for tracking SMS notifications)
-- Purpose: Track Twilio SMS notifications sent to Noah
-- ============================================================================
CREATE TABLE IF NOT EXISTS sms_logs (
    id BIGSERIAL PRIMARY KEY,
    event TEXT NOT NULL,  -- Event type (e.g., 'contact_request', 'resume_sent', 'confession')
    from_name TEXT,
    from_email TEXT,
    message_preview TEXT,
    is_urgent BOOLEAN DEFAULT false,
    twilio_sid TEXT,  -- Twilio message SID for tracking
    status TEXT DEFAULT 'pending',  -- pending, sent, failed, delivered
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS sms_logs_created_at_idx 
ON sms_logs (created_at DESC);

CREATE INDEX IF NOT EXISTS sms_logs_status_idx 
ON sms_logs (status);

-- Enable RLS
ALTER TABLE sms_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role can manage sms_logs"
ON sms_logs FOR ALL
TO service_role
USING (true);

-- ============================================================================
-- VIEWS: Analytics for new tables
-- ============================================================================

-- View: Recent confessions summary
CREATE OR REPLACE VIEW recent_confessions AS
SELECT 
    id,
    CASE 
        WHEN is_anonymous THEN 'Anonymous'
        ELSE COALESCE(name, 'Unknown')
    END as from_name,
    LEFT(message, 100) || '...' as message_preview,
    created_at
FROM confessions
ORDER BY created_at DESC
LIMIT 50;

-- View: Feedback with contact requests
CREATE OR REPLACE VIEW pending_contacts AS
SELECT 
    f.id,
    f.message_id,
    f.user_name,
    f.user_email,
    f.user_phone,
    f.rating,
    f.comment,
    f.created_at,
    m.query as original_query
FROM feedback f
LEFT JOIN messages m ON f.message_id = m.id
WHERE f.contact_requested = true
  AND f.notification_sent = false
ORDER BY f.created_at DESC;

-- ============================================================================
-- COMPLETE
-- ============================================================================
-- Migration adds:
-- 1. confessions table for crush confession feature
-- 2. Missing columns in feedback table (user_name, user_phone, user_email)
-- 3. sms_logs table for tracking Twilio notifications
-- 4. Helpful views for analytics
--
-- After running this migration:
-- - Test /api/confess endpoint
-- - Test /api/feedback endpoint
-- - Verify RLS policies in Supabase dashboard
-- ============================================================================
