# Legacy Vector Stores Archive

This directory contains deprecated FAISS vector indexes.

## Archived November 16, 2025

### FAISS Indexes (Replaced)
- `faiss_career/` - Local career knowledge base index
- `code_index/` - Local code snippets index

## Current Vector Storage

Production uses **Supabase pgvector**:
- Centralized vector storage
- IVFFLAT indexes for fast search
- Managed in `assistant/retrieval/pgvector_retriever.py`
- No local vector stores needed
