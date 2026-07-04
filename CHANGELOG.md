# Changelog

All notable changes to Portfolia are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/); entries are dated.

---

## [Unreleased]

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
