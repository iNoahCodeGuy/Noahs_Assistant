# Scripts

Operational utilities for the Portfolia backend. One canonical script per job —
one-off setup wizards and superseded migrators were removed in July 2026 (git
history has them if ever needed).

## `migrate_data_to_supabase.py` — the KB embed/migrate entrypoint

Re-embeds the knowledge-base CSVs in [data/](../data/) into the Supabase
`kb_chunks` table (OpenAI text-embedding-3-small, 1536 dims, batched, idempotent).
**Run this after any `data/*.csv` edit** — local changes don't affect retrieval
until re-embedded.

```bash
python3 scripts/migrate_data_to_supabase.py           # skips existing rows
python3 scripts/migrate_data_to_supabase.py --force   # delete + re-import
```

## `test_pgvector_search.py` — retrieval diagnostic

Live end-to-end check of the `match_kb_chunks` RPC: embeds a test query, runs
the similarity search, prints results. Useful after migrations or threshold
changes. Requires real API keys in `.env`.

## `test_live_api.sh` — deployed-API smoke test

Curl-based checks against a running backend (local or Railway).

## `verify_migration.sh` — migration sanity checks

Helper for verifying database migrations were applied (see
[supabase/migrations/](../supabase/migrations/)).
