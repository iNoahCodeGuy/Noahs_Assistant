# Applied migrations — production (project `fyxotgbcjjlsrflxwjdw`, "Portfolia")

The record of what has actually run against prod. Update this whenever a
migration is applied. There is no CLI state tracking — this file is it.

**Project history:** the original project (`tjnlusesinzzlwvlbnnm`, "AI Assistant",
created 2025-10-06) was paused by Supabase around 2026-07-05 with corrupted pause
metadata that blocked dashboard restore. Production was rebuilt the same day on a
fresh project from these migration files; the KB was re-embedded from `data/*.csv`
(parity-verified). Historical 2026 rows in `recruiter_leads` / `crush_confessions` /
conversation analytics live only in the old project — recovery is pending a
Supabase support ticket. See `AUDIT_ROADMAP.md` (incident section).

| Migration | Status | Evidence / notes |
| --- | --- | --- |
| 001_initial_schema | applied 2026-07-05 | via Management API; file amended: HNSW index instead of IVFFLAT (IVFFLAT trains centroids at build time — building on an empty table produced near-random retrieval) |
| 002_add_confessions_and_sms | applied 2026-07-05 | |
| 003_fix_session_id_type | applied 2026-07-05 | file amended: drop/recreate `messages_with_retrieval` around the type change (Postgres refuses ALTER under a dependent view) |
| 004_analytics_helpers | RETIRED 2026-07-05 | never applyable anywhere: referenced `user_query`/`similarity_score` columns and a `tool_invocations` table that never existed, and nothing consumed its functions. File deleted; this row is the tombstone. |
| 005_crush_confessions_table | applied 2026-07-05 | crush flow writes here |
| 006_recruiter_leads_table | applied 2026-07-05 | contact capture writes here |
| 007_add_referral_source | applied 2026-07-05 | |
| 008_conversation_analytics | applied 2026-07-05 | dashboard reads conversation_sessions / conversation_messages |
| 009_fix_match_kb_chunks_type | applied 2026-07-05 | `match_kb_chunks` verified: signature + self-match similarity 1.0 |

Verification after apply (2026-07-05): all tables/views present; KB re-embedded
(234 chunks) and `scripts/verify_kb_parity.py` reports CLEAN; end-to-end pipeline
smoke (retrieval → generation) passes with similarities ~0.7 on topical queries.
