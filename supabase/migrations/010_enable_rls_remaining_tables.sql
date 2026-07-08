-- Migration 010: Enable RLS on tables created after 002
-- Fixes Supabase security advisor lint `rls_disabled_in_public` (flagged 2026-07-06).
--
-- Migrations 005/006/008 created their tables without ENABLE ROW LEVEL SECURITY,
-- so PostgREST exposed them to anyone holding the public anon key — full
-- read/insert/update/delete. Verified exposed 2026-07-08: conversation_sessions
-- and conversation_messages were readable as anon (crush_confessions and
-- recruiter_leads were exposed too, but empty at the time).
--
-- Safe for the app: the backend talks to Supabase exclusively with the service
-- role key (assistant/config/supabase_config.py), which bypasses RLS. The
-- service-role policies below match the 001/002 house pattern.

-- Idempotent: ENABLE ROW LEVEL SECURITY is a no-op when already enabled, and
-- each CREATE POLICY is preceded by DROP POLICY IF EXISTS — safe to re-run.

-- ── crush_confessions (from 005) ────────────────────────────────────────────
ALTER TABLE crush_confessions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Service role can manage crush_confessions" ON crush_confessions;
CREATE POLICY "Service role can manage crush_confessions"
ON crush_confessions FOR ALL
TO service_role
USING (true);

-- ── recruiter_leads (from 006) ──────────────────────────────────────────────
ALTER TABLE recruiter_leads ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Service role can manage recruiter_leads" ON recruiter_leads;
CREATE POLICY "Service role can manage recruiter_leads"
ON recruiter_leads FOR ALL
TO service_role
USING (true);

-- ── conversation_sessions / conversation_messages (from 008) ────────────────
ALTER TABLE conversation_sessions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Service role can manage conversation_sessions" ON conversation_sessions;
CREATE POLICY "Service role can manage conversation_sessions"
ON conversation_sessions FOR ALL
TO service_role
USING (true);

ALTER TABLE conversation_messages ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Service role can manage conversation_messages" ON conversation_messages;
CREATE POLICY "Service role can manage conversation_messages"
ON conversation_messages FOR ALL
TO service_role
USING (true);

-- ── Views: respect the caller's RLS ─────────────────────────────────────────
-- Views execute with their owner's privileges by default, and the table owner
-- bypasses RLS — messages_with_retrieval was verified to leak rows of the
-- RLS-protected `messages` table to anon. security_invoker (PG15+) makes each
-- view enforce the querying role's RLS instead. Service-role reads unaffected.
ALTER VIEW messages_with_retrieval SET (security_invoker = true);
ALTER VIEW analytics_by_role SET (security_invoker = true);
ALTER VIEW recent_confessions SET (security_invoker = true);
ALTER VIEW pending_contacts SET (security_invoker = true);
