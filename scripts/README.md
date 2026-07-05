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
python3 scripts/migrate_data_to_supabase.py                       # default: data/career_kb.csv, skips existing
python3 scripts/migrate_data_to_supabase.py --csv data/technical_kb.csv --force   # one file, delete + re-import
python3 scripts/migrate_data_to_supabase.py --all --force         # full rebuild of every data/*.csv
```

Each chunk's `metadata` records provenance: `source` (CSV filename),
`content_sha256`, and `migrated_at` — which is what `verify_kb_parity.py`
checks against.

## `verify_kb_parity.py` — prove the live table matches the CSVs

Retrieval serves whatever is in Supabase, not what is in git. This compares
row counts and content hashes per CSV against the live `kb_chunks` table and
flags orphaned doc_ids (chunks from older embedding runs that no current CSV
produces — they keep getting retrieved until deleted).

```bash
python3 scripts/verify_kb_parity.py                   # report; exit 0 = clean, 1 = drift
python3 scripts/verify_kb_parity.py --delete-orphans  # also remove orphaned chunks
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
