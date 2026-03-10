-- Migration 005: Add referral_source column to recruiter_leads
-- Captures how visitors found Noah's portfolio website

ALTER TABLE recruiter_leads ADD COLUMN IF NOT EXISTS referral_source TEXT;

COMMENT ON COLUMN recruiter_leads.referral_source IS 'How the visitor found the portfolio website';
