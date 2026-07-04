# Changelog

All notable changes to Portfolia are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/); entries are dated.

---

## [Unreleased]

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
