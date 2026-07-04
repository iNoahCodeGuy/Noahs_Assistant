# Docs

Reference documentation for the Portfolia backend. Start with the
[main README](../README.md) for architecture, quickstart, and project layout.

| Doc | What it covers |
| --- | --- |
| [GLOSSARY.md](GLOSSARY.md) | Terms used across the codebase — RAG, embeddings, pgvector, grounding, testing vocabulary |
| [EXTERNAL_SERVICES.md](EXTERNAL_SERVICES.md) | Setting up the third-party services: Supabase, Twilio, Resend, API keys |
| [LANGSMITH.md](LANGSMITH.md) | LangSmith account setup and trace configuration |
| [OBSERVABILITY.md](OBSERVABILITY.md) | What gets traced and measured: retrieval metrics, generation metrics, evaluation sampling |

Database schema lives in [supabase/migrations/](../supabase/migrations/).
Knowledge-base content lives in [data/](../data/) — remember that CSV edits require
re-embedding (`python3 scripts/migrate_data_to_supabase.py`).

Historical planning and session documents were removed in July 2026 — they described
earlier, since-replaced iterations of the system and are preserved in git history if
ever needed.
