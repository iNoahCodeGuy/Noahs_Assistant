# Changelog

All notable changes to Portfolia are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/); entries are dated.

---

## [Unreleased]

---

## [2026-07-08] — Security: enable RLS on post-002 tables

Supabase's security advisor flagged `rls_disabled_in_public` (CRITICAL) on the
rebuilt production project: every table created after migration 002 shipped
without Row-Level Security.

### Security
- **Anon key had full read/write on four tables** — `crush_confessions`,
  `recruiter_leads`, `conversation_sessions`, `conversation_messages`.
  Verified live before the fix: conversation transcripts were readable with
  the public anon key; the two PII capture tables were exposed but empty
  (historical rows live only in the paused pre-rebuild project). Migration
  `010_enable_rls_remaining_tables.sql` enables RLS with service-role
  policies, matching the 001/002 pattern.
- **Views bypassed base-table RLS**: Postgres views run with their owner's
  privileges by default, and the owner bypasses RLS — `messages_with_retrieval`
  verifiably leaked rows of the RLS-protected `messages` table to anon, and
  `analytics_by_role` leaked its aggregates the same way. Migration 010 sets
  `security_invoker = true` on all four views.
- The backend is unaffected: it talks to Supabase exclusively with the
  service role key (which bypasses RLS) — proven by fresh prod writes to the
  `messages` table, RLS-protected since 001.

### Added
- `scripts/verify_rls.py` — probes every table and view as `anon` (SELECT
  everywhere, INSERT on the four fixed tables) and exits nonzero on any
  exposure. Cited by `supabase/migrations/APPLIED.md` as apply evidence.

---

## [2026-07-07] — Career-asset phase: keep-alive, KB depth, visibility

The audit era is closed; this session optimizes the project as a portfolio
piece. Everything user-visible was re-embedded and verified live.

### Fixed (found by running the live eval suite against the new KB)
- **Degraded-LLM bug**: a bare `RagEngineFactory()` carried `settings=None`,
  so `ChatAnthropic` got a `None` API key, failed validation, and silently
  swapped in the degraded-mode fallback — every edge-case response
  (off-topic and meta questions) was canned text instead of a generated
  answer. The factory now defaults to the settings singleton
- "Where can I see his work?" now retrieves a KB row carrying both the
  GitHub and LinkedIn links (they previously lived only in separate rows)
- `LINKEDIN_URL` constant pointed at a malformed profile slug; the welcome
  message claimed the heatmap uses matplotlib (it is a Streamlit/Plotly
  dashboard)
- The live eval helper now models a post-welcome session like the hermetic
  suite does — before this, stage2 answered every eval question with the
  static role-welcome template. Suite: 9/9
- Gate-flip review procedure corrected: hallucination findings are not in
  LangSmith or Supabase — they exist only as warning lines in Railway
  runtime logs (reset each deploy); the review command lives in the local
  roadmap

### Added
- `GET /health` — liveness plus a kb_chunks count probe (503 when the DB is
  unreachable), with two hermetic tests
- `.github/workflows/keepalive.yml` — Mon+Thu cron + manual dispatch: hits
  /health and asks production a real KB question, asserting a grounded
  answer. Generates the activity that prevents the Supabase free-tier
  inactivity pause (the July 5 root cause) and turns silent retrieval
  degradation into a red workflow run. First dispatch verified green
- `data/incidents_kb.csv` — the July 5 outage-recovery story as 6
  first-person KB rows (the silent failure, three latent bugs, layered
  verification, prevention)
- `docs/INCIDENT_2026-07-05.md` — the same story as an engineering case
  study, linked from the README and docs index
- README "What talking to it looks like" — a real production transcript
- KB self-knowledge for the July hardening: hallucination-gate modes and
  staged rollout, abuse guards, notification throttle, KB provenance/parity,
  CI and the hermetic suite, production failure visibility (6 new rows in
  technical/architecture, 2 existing rows corrected)

### Changed
- `data/analytics_enhanced.csv` rewritten: one 12,798-char emoji/template
  dump → 8 retrieval-granular rows, every claim verified against
  stage7/analytics/dashboard code; fabricated metrics deleted
- `data/lead_response_heatmap_kb.csv` 7 → 14 rows, grounded in the actual
  project source (metric choice, bucketing tradeoff, edge cases, honest
  limitations); `data/enterprise_agentic_kb.csv` +2 rows
- `data/career_kb.csv` +9 rows migrating narratives that lived only in
  `data/noah_career_kb.md` (why-tech, TQL, real estate, coaching
  philosophy, learning approach)
- Frontend ([portfolia-frontend](https://github.com/iNoahCodeGuy/portfolia-frontend)
  `cf832d6`, `7bf1434`): Open Graph + twitter metadata, OG image, favicon
  set, explicit viewport; fail-loud on missing `NEXT_PUBLIC_API_URL` in
  prod; form input aria-labels; stable dashboard list keys; login
  autocomplete

### Removed
- `data/noah_career_kb.md` — never embedded (the migrator reads CSVs only);
  its unique content now lives in career_kb.csv rows
- `.env.bak-20260705` (untracked, un-ignored secrets copy; only diff from
  .env was the old broken EMBEDDING_MODEL value)

### Verified
- Re-embed + parity: 271 chunks across 14 CSVs, `verify_kb_parity.py`
  CLEAN; live probes returned grounded on-voice answers for the outage
  story, the hallucination gate, and heatmap methodology
- 107 hermetic tests green; CI green on both repos; live /health 200 with
  the correct chunk count

---

## [2026-07-05] — Production database rebuilt on a fresh Supabase project

The original Supabase project was paused by the platform with corrupted pause
metadata (dashboard claimed a 2024 pause date and blocked restore; the project
had served traffic the previous day). Production now runs on a fresh project
built from the migration files, with the truth-passed KB re-embedded
(234 chunks, parity CLEAN). Historical 2026 leads/confessions remain on the
old project pending a Supabase support ticket.

### Fixed
- **Migration 003** failed on fresh databases: Postgres refuses to alter a
  column's type under a dependent view — now drops/recreates
  `messages_with_retrieval` around the change
- **Vector index**: 001 built an IVFFLAT index on the empty table, which
  trains degenerate centroids and returns near-random retrieval once data
  loads. Replaced with HNSW (no training step, correct in any build order)
- **Embedding-model mismatch**: a stale `EMBEDDING_MODEL=text-embedding-ada-002`
  in local .env silently broke retrieval once the audit wired that knob up —
  queries embedded with a different model than the KB produce meaningless
  similarities (both are 1536-dim, so nothing errors). The migrator now reads
  the same settings knob as the retriever, so KB and queries cannot diverge
- Migrator now understands all three KB CSV schemas (Question/Answer,
  lowercase q/a with extras, and structured tables like imports_kb/mma_kb);
  `verify_kb_parity.py` imports the same reader so parity cannot drift

### Removed
- **Migration 004 retired**: its analytics helper functions referenced columns
  (`user_query`, `similarity_score`) and a table (`tool_invocations`) that
  never existed anywhere, and nothing called them — it could never have been
  applied. Tombstone recorded in `supabase/migrations/APPLIED.md`, which is
  fully rewritten for the new project (zero unverified rows)

---

## [2026-07-05] — Audit close-out: renames, hardening, KB truth pass, architecture doc

### Added
- `docs/ARCHITECTURE.md` — the design-decision walkthrough: 22-node pipeline
  table, request-lifecycle and capture-FSM diagrams, and the constraint behind
  each architectural choice
- Sentry error reporting, gated on `SENTRY_DSN`: reports both unhandled route
  errors (new global exception handler) and pipeline errors the `/chat`
  endpoint masks behind its graceful fallback
- Notification dispatch throttle — 10/hour global cap on capture SMS/email
  (the `/chat` limiter alone still allowed 20 SMS/min to Noah's phone);
  throttled submissions still reach Supabase. Seven hermetic tests
- KB provenance: the migrator stamps `content_sha256` into chunk metadata and
  gains `--all` for full rebuilds; new `scripts/verify_kb_parity.py` proves
  the live table matches `data/*.csv` and flags orphaned chunks
- Frontend: sliding-window rate limits on the dashboard login (5/min/IP —
  brute-force target) and analytics routes (60/min/IP), with vitest coverage

### Changed
- **Repos renamed**: `Noahs_Assistant` → `portfolia-backend`,
  `portfolia_frontend` → `portfolia-frontend` (GitHub redirects old URLs);
  every reference updated across docs, prompts, test fixtures, and KB
- **KB truth pass**: all 14 KB files audited against the codebase — 99
  findings fixed (langgraph-as-dependency claims, OpenAI credited with
  generation, backend-on-Vercel claims, removed features, fabricated
  metrics, stale schemas and URLs). Not yet re-embedded (see below)
- `docs/EXTERNAL_SERVICES.md` rewritten against reality (it still described
  deleted setup scripts, storage buckets, and sqlite-era variables)
- README/CLAUDE.md testing docs updated: CI runs the full hermetic suite,
  not the legacy 3-file subset

### Known issue
- The Supabase project (`tjnlusesinzzlwvlbnnm`) stopped resolving on
  2026-07-05 — paused or deleted. Production retrieval is degraded (fallback
  answers) and capture writes fail silently until restored. After
  restoration: `migrate_data_to_supabase.py --all --force`, then
  `verify_kb_parity.py --delete-orphans`, then verify
  `supabase/migrations/APPLIED.md` rows 004/008

---

## [2026-07-04] — Structural cleanup completed + production hardening (Phase 5B/6)

### Added
- Per-IP rate limiting (20/min) and a 4k-char message cap on `/chat`, enforced
  before any LLM call; six hermetic guard tests
- `supabase/migrations/APPLIED.md` — the production applied-state record

### Changed
- stage1 intent router split: capture state machines (crush, lead capture,
  notifications, shared marker constants) now live in `assistant/flows/capture/`;
  `classify_intent` is a short dispatcher (1,981 → 1,113 lines). Verified with
  byte-level AST comparison and a 294-case differential battery
- stage6 formatting split by responsibility: voice enforcement, link throttling,
  and followup generation are separate modules (2,633 → 1,203 lines)
- Migrations renumbered uniquely (two 002s/003s resolved); one-off variants archived
- langsmith pinned explicitly (tracing is a feature, not a transitive accident)

### Fixed
- Capture-trigger contact form was unrecoverable from history in stateless mode
  (marker case mismatch — detection now case-insensitive)
- Frontend: chat requests time out instead of hanging; login button can't stick
  on "Checking…"; dashboard fetches surface auth expiry properly; analytics
  routes fail loud on missing env; labeled inputs + aria-live for screen readers

---

## [2026-07-04] — Structural cleanup (Phase 5B, part 1)

### Added
- Structured `form` field in the `/chat` response (`"crush" | "contact" | null`) —
  the frontend renders forms from the pipeline's state machines instead of
  pattern-matching answer text

### Changed
- Orchestrator slimmed from 879 to ~300 lines: discovery hooks/pacing extracted to
  `util_discovery_hooks.py`
- Twilio/Resend read their config from the central settings object (one place env
  vars are read)
- CORS: strict origin allow-list when `FRONTEND_URL` is set (was wildcard +
  credentials, a spec-invalid pair), logged permissive fallback otherwise

### Removed
- langgraph dependency: the Studio StateGraph was built at module import
  (constructing network clients on every cold start, for visualization nobody
  used); `chat_history` is now typed as the plain message list it is

---

## [2026-07-04] — Code-level credibility (Phase 5, part 1)

### Added
- Real hallucination check: percentages, dollar amounts, and URLs in generated
  answers are verified against retrieved sources; staged rollout via
  `HALLUCINATION_GATE` (log → enforce), 13 new tests

### Fixed
- False "Resume dispatched" SMS on resume requests (dead resume-email actions
  removed; resume requests now route to the reach-out offer)
- Generation could append a factually wrong tech-stack line to answers
  ("Ensure test expectation" hack removed)
- Architecture snippet shown to visitors was a fabricated 18-node LangGraph
  sketch — now a faithful view of the real 22-node pipeline
- Code index scanned a nonexistent `src/` directory (silently indexing nothing)
- `EMBEDDING_MODEL` setting was unread with the wrong default (ada-002)

### Changed
- All dependency-fallback shims fail loud (ImportError) instead of silently
  substituting fake embeddings, empty loaders, or an echo LLM
- Pipeline docstring now lists the nodes that actually run; invented latency
  figures removed

### Removed
- ~1,300 lines of admitted-dead code: role/visitor-type machinery, menu
  validators, `ChatOpenAI` alias, legacy `OPENAI_*` settings, empty
  `settings.py`, unused monitor module, eight one-off scripts, frontend
  mock-response module

---

## [2026-07-04] — Test suite resurrection (Phase 4)

### Added
- Hermetic tests for the flagship claims: intent routing (classifier faked, incl.
  outage fallback), grounding-validation thresholds, verbatim-copy detection,
  conversation phases, pgvector threshold plumbing
- Rewritten end-to-end pipeline tests against the current partial-dict contract
- Frontend: ESLint (next/core-web-vitals), 19 vitest tests for form detection,
  and its first CI workflow (lint, typecheck, test, build)
- pyproject pytest/coverage config; pinned `requirements-dev.txt`; working
  pre-commit setup

### Changed
- Live-API evals are opt-in (`pytest -m live`) — collection no longer requires keys
- Backend CI runs the full suite with coverage (79 tests; was a 3-file allowlist)

### Removed
- Test modules and scripts that exercised removed architecture (RoleRouter,
  Vercel handlers, visitor-type detection, resume-dispatch actions)

---

## [2026-07-04] — One deployment story (Phase 3)

### Removed
- Dead Vercel/Next.js deployment target: the in-repo Next.js app (a stale copy of the
  deployed frontend), Vercel Python serverless handlers, `vercel.json`, `runtime.txt`,
  `.railwayignore`, and their orphaned scripts/tests
- Dead Streamlit tree (`assistant/main.py`, `assistant/ui/`, `assistant/integration/`)
  and the test files that imported it

### Changed
- Dockerfile no longer copies `.env` into the image; ships `data/` (read by the RAG
  factory at startup); `.dockerignore` added
- Python version declared once: 3.12 (`.python-version`, `pyproject.toml`)
- `api/README.md` rewritten against the actual `POST /chat` contract

---

## [2026-07-04] — Repo professionalization (Phases 1–2)

### Added
- Passing CI workflow (`tests.yml`) running the hermetic test subset on every push
- MIT LICENSE (this repo and portfolia_frontend)
- README and `.env.example` for portfolia_frontend
- `docs/README.md` index

### Changed
- Dashboard auth hardened: fail-closed middleware, HMAC-signed expiring session cookie
  (replaces plaintext-password cookie), constant-time comparisons
- README rewritten against the current system (Claude Sonnet 4.5, pgvector, FastAPI on
  Railway, Next.js frontend on Vercel) with a Mermaid architecture diagram
- `.env.example` regenerated from the variables the code actually reads
  (fixes `SUPABASE_SERVICE_KEY` → `SUPABASE_SERVICE_ROLE_KEY`)
- CLAUDE.md slimmed to lean agent instructions; persona playbook moved to a local file
- docs/ reduced to four maintained references (glossary, external services, LangSmith,
  observability) — 30+ stale session/planning documents removed

### Fixed
- Crush-confession DB write read the wrong env var name (`SUPABASE_SERVICE_KEY`),
  causing silent insert failures when the legacy alias wasn't set
- Broken CI workflows that invoked nonexistent files (red on all 309 runs)
- Tracked build artifacts, backup files, and FAISS-era stubs removed from git

### Removed
- 10 stale remote branches (preserved as `archive/*` tags)
- `.github/copilot-instructions.md` (documented a nonexistent `src/` layout)

---

## [2026-03] — Claude migration & universal pipeline (consolidated)

### Changed
- Generation moved from OpenAI GPT-4o to Anthropic Claude Sonnet 4.5; intent
  classification to Claude Haiku
- Role-based conversation branching replaced with a single universal pipeline
  (4 welcome buttons, no per-visitor routing)
- Retrieval consolidated on Supabase pgvector (`match_kb_chunks` RPC, 0.50/0.30
  thresholds); FAISS removed
- Frontend split into its own repo (portfolia_frontend) deployed on Vercel;
  backend deployed as FastAPI on Railway
- Knowledge base rewritten and re-embedded (voice, accuracy, K-Means project added)

### Added
- Crush-confession flow as a single-form state machine with Twilio SMS notification
- Universal contact-capture flow writing to `recruiter_leads`
- Markdown rendering in the frontend (react-markdown + remark-gfm)

---

*Entries before 2026-03 described the earlier Streamlit/GPT-3.5 iteration of the
project and were removed during the July 2026 documentation cleanup; they remain
available in git history.*
