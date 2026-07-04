# Contributing

This repo powers **Portfolia**, Noah's AI portfolio assistant
(live at [noahdelacalzada.com](https://noahdelacalzada.com/)). Conventions below keep
changes safe and reviewable.

## Layout

```
assistant/            core package (pipeline, RAG engine, retrieval, services, config)
  flows/              22-node pipeline — conversation_flow.py is the entry point
  flows/node_logic/   node implementations, stage0–stage7
api/                  FastAPI app (api/main.py)
data/                 knowledge-base CSVs (re-embed after editing!)
scripts/              KB migration & utilities
supabase/migrations/  database schema
tests/                pytest suite
docs/                 reference docs
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in the four required keys
```

## Running

```bash
python3 chat_with_portfolia.py                 # terminal client
uvicorn api.main:app --reload --port 8000     # API server
```

## Testing

```bash
pytest tests/test_documentation_alignment.py tests/test_memory.py tests/test_roles.py
```

This hermetic subset runs in CI on every push and must stay green. If you change
pipeline behavior, run the manual smoke queries listed in [CLAUDE.md](CLAUDE.md) too.

## Rules of the road

- **Knowledge base:** editing `data/*.csv` does nothing until you re-embed:
  `python3 scripts/migrate_data_to_supabase.py`
- **Migrations:** schema changes go in `supabase/migrations/` as a new file — never
  edit an applied migration
- **Branches:** `feature/…`, `fix/…`, `docs/…`; commit messages in the imperative
  ("add X", not "added X")
- **No dead code:** delete replaced code in the same commit that ships its replacement —
  git history is the archive
- **Secrets:** never commit `.env`; new config goes in `.env.example` with a placeholder
