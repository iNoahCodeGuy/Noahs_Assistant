# Scripts Directory

This folder contains utility scripts for data migration, testing, and maintenance of Noah's AI Assistant.

## 📝 Available Scripts

### `migrate_data_to_supabase.py`
**Purpose**: Migrate career knowledge base from CSV to Supabase with pgvector embeddings.

**Features**:
- Reads `data/career_kb.csv` (21 Q&A pairs)
- Generates embeddings using OpenAI `text-embedding-3-small` (1536 dimensions)
- Batch processing (100 texts per API call for efficiency)
- Idempotent inserts (prevents duplicates)
- Exponential backoff retry logic for API failures
- Progress tracking with cost estimates
- Structured logging for observability

**Usage**:
```bash
# Standard migration (skips if data already exists)
python scripts/migrate_data_to_supabase.py

# Force re-import (deletes existing data first)
python scripts/migrate_data_to_supabase.py --force

# Use custom CSV path
python scripts/migrate_data_to_supabase.py --csv path/to/custom.csv
```

**Requirements**:
- `OPENAI_API_KEY` environment variable
- `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` environment variables
- `data/career_kb.csv` file present

**Expected Output**:
```
🚀 Starting data migration to Supabase...
📄 Reading data/career_kb.csv...
   Found 21 rows

🔢 Creating chunks...
   Created 21 chunks

🧠 Generating embeddings...
   Batch 1/1: Processing 21 texts...
   Progress: 21/21 (100.0%)
   ✅ Generated 21 embeddings
   💰 Estimated cost: $0.0003

💾 Inserting chunks to Supabase...
   Batch 1/1: Inserting 21 chunks...
   Progress: 21/21 (100.0%)
   ✅ Inserted 21 chunks

✨ Migration complete!
   📊 Summary:
      CSV rows read: 21
      Chunks created: 21
      Embeddings generated: 21
      Chunks inserted: 21
      API calls: 1
      Failed operations: 0
      Total cost: $0.0003
      Duration: 3.2s
```

**Cost Estimate**:
- ~$0.0003 for 21 chunks (text-embedding-3-small: $0.00002 per 1K tokens)
- Negligible cost for testing and development

---

### `test_pgvector_search.py`
**Purpose**: Verify that pgvector similarity search is working correctly.

**Features**:
- Tests multiple queries covering different topics
- Uses same embedding model as production (text-embedding-3-small)
- Calls Supabase `search_kb_chunks()` RPC function
- Displays similarity scores and retrieved content

**Usage**:
```bash
python scripts/test_pgvector_search.py
```

**Expected Output**:
```
============================================================
pgvector Search Test
============================================================

🔍 Testing query: 'What programming languages does Noah know?'

🧠 Generating query embedding...
   ✅ Generated 1536-dimensional vector

🔎 Searching Supabase for top 2 results...
   ✅ Found 2 results

📄 Result 1:
   Section: What technical skills do you have?...
   Content: Q: What technical skills do you have?
A: Programming & Development: Python...
   Similarity: 0.8234

📄 Result 2:
   Section: How strong is Noah's Python?...
   Content: Q: How strong is Noah's Python?
A: Noah's Python skills are at an intermediate...
   Similarity: 0.7891

------------------------------------------------------------
```

**What to Look For**:
- ✅ Similarity scores > 0.7 indicate good matches
- ✅ Retrieved content should be relevant to query
- ❌ If no results or low scores, check embeddings in database

---

## 🔧 Troubleshooting

### "OPENAI_API_KEY not found"
```bash
# Add to your .env file
OPENAI_API_KEY=sk-...
```

### "Supabase configuration invalid"
```bash
# Add to your .env file
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...
```

### "CSV file not found"
- Ensure `data/career_kb.csv` exists
- Or specify custom path with `--csv` flag

### "No results found" in test script
1. Run migration first: `python scripts/migrate_data_to_supabase.py`
2. Check Supabase SQL Editor: `SELECT COUNT(*) FROM kb_chunks;`
3. Verify `search_kb_chunks()` function exists (created by migration SQL)

### Rate limit errors (429)
- Script automatically retries with exponential backoff
- If persistent, check OpenAI API usage limits
- Free tier: 3 RPM (requests per minute)

---

## 📊 Performance Notes

**Migration Performance**:
- 21 chunks: ~3-5 seconds
- 100 chunks: ~10-15 seconds
- 1000 chunks: ~60-90 seconds

**Batch Sizes**:
- Embedding generation: 100 texts per call (configurable)
- Database inserts: 50 chunks per batch (safe for pgvector)

**Cost Optimization**:
- Batching reduces API calls by 100x
- text-embedding-3-small is 5x cheaper than ada-002
- Idempotency prevents accidental duplicate charges

---

## 🚀 Next Steps

After running these scripts successfully:

1. **Update RAG engine** (`src/core/rag_engine.py`)
   - Replace FAISS with pgvector queries
   - Use `search_kb_chunks()` function

2. **Build Next.js frontend**
   - Create `/api/chat` route
   - Implement chat interface

3. **Add evaluation metrics**
   - Track retrieval quality
   - Measure response latency
   - Log to LangSmith

---

## 📚 Additional Resources

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Supabase pgvector Documentation](https://supabase.com/docs/guides/database/extensions/pgvector)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)

---

**Last Updated**: October 5, 2025  
**Maintainer**: Noah De La Calzada (@iNoahCodeGuy)
