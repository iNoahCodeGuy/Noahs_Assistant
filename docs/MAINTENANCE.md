# Maintenance Guide

## Codebase Indexing

### Selective Indexing (Recommended During Development)

During active development, we use selective indexing to index only core architecture files. This avoids frequent re-indexing while still enabling self-referential queries.

**Core Files** (20 files):
- Essential architecture: `rag_engine.py`, `conversation_flow.py`, `retrieval_nodes.py`, `query_classification.py`, etc.
- Supporting architecture: `rag_factory.py`, `code_index.py`, `memory.py`, `guardrails.py`, etc.

**Re-index core files**:
```bash
python scripts/index_codebase.py --core-only
```

**Expected**: ~139 chunks, ~$0.002 cost, ~40 seconds

### Custom File List

Index specific files:
```bash
python scripts/index_codebase.py --files assistant/core/rag_engine.py assistant/flows/conversation_flow.py
```

### Full Indexing (When Codebase Stabilizes)

When the codebase is more stable, run full indexing:
```bash
python scripts/index_codebase.py
```

**Expected**: ~500-800 chunks, ~$5-10 cost, ~10-20 minutes

### When to Re-index

- **After major architecture changes** to core files
- **After adding new core components** (new RAG engines, major pipeline changes)
- **Before important demos** to ensure latest code is searchable
- **Periodically** (monthly) to keep codebase chunks up-to-date

### Documentation Indexing

Documentation is more stable and rarely needs re-indexing:
```bash
python scripts/index_documentation.py
```

**Expected**: ~1,296 chunks, ~$0.004 cost, ~4 minutes

Re-index documentation when:
- Major documentation updates
- New architecture documents added
- Documentation structure changes significantly

## Verification

Verify chunks in database:
```bash
python scripts/verify_codebase_chunks.py
```

This will show:
- Codebase chunk count and unique files
- Documentation chunk count
- Sample file paths

## Troubleshooting

### No chunks retrieved for self-referential queries

1. Verify chunks exist: `python scripts/verify_codebase_chunks.py`
2. Check `doc_id` values match exactly ('codebase', 'documentation')
3. Verify embeddings were generated (check `embedding` column not NULL)
4. Re-index if needed: `python scripts/index_codebase.py --core-only`

### Retrieval returns empty results

1. Check similarity threshold (default 0.60 in `pgvector_retriever.py`)
2. Verify query embeddings are generated successfully
3. Check logs for retrieval errors
4. Verify `doc_id` filtering is working in retrieval logic

### Chunks out of date

1. Re-index core files: `python scripts/index_codebase.py --core-only`
2. For specific files: `python scripts/index_codebase.py --files path/to/file.py`
3. For full refresh: `python scripts/index_codebase.py`
