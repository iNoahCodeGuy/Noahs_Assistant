# Phase 1: Data Migration Setup Guide

This guide walks you through running the data migration to populate Supabase with pgvector embeddings.

## ✅ Prerequisites Checklist

Before running the migration, ensure you have:

- [ ] **Supabase project created** at [supabase.com](https://supabase.com)
- [ ] **Database schema deployed** (ran `supabase/migrations/001_initial_schema.sql` in SQL Editor)
- [ ] **Environment variables set** in `.env` file
- [ ] **OpenAI API key** with available credits
- [ ] **Python virtual environment** activated

## 🔧 Step 1: Configure Environment

Create or update your `.env` file:

```bash
# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbG...your-service-role-key
SUPABASE_ANON_KEY=eyJhbG...your-anon-key  # Optional

# OpenAI Configuration
OPENAI_API_KEY=sk-...your-openai-key

# External Services (Optional for Phase 1)
RESEND_API_KEY=re_...
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM=+1...
```

### How to Get Supabase Credentials:

1. Go to your Supabase project dashboard
2. Click **Settings** (⚙️) in the left sidebar
3. Click **API** section
4. Copy:
   - **Project URL** → `SUPABASE_URL`
   - **service_role key** → `SUPABASE_SERVICE_ROLE_KEY` (click "Reveal" button)

⚠️ **Never commit your `.env` file to Git!**

## 🗄️ Step 2: Deploy Database Schema

1. Open your Supabase project
2. Go to **SQL Editor** in the left sidebar
3. Click **New Query**
4. Copy and paste the entire contents of `supabase/migrations/001_initial_schema.sql`
5. Click **Run** (or press Ctrl+Enter)

You should see:
```
Success. No rows returned
```

### Verify Schema:

Run this query in SQL Editor:
```sql
-- Check tables exist
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('kb_chunks', 'messages', 'retrieval_logs', 'links', 'feedback');

-- Check pgvector extension
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Check search function exists
SELECT routine_name
FROM information_schema.routines
WHERE routine_name = 'search_kb_chunks';
```

Expected output:
```
table_name
-----------
kb_chunks
messages
retrieval_logs
links
feedback

extname | extversion
--------|-----------
vector  | 0.5.0

routine_name
--------------
search_kb_chunks
```

## 🚀 Step 3: Run Data Migration

### Activate Virtual Environment

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Mac/Linux
source .venv/bin/activate
```

### Run Migration Script

```bash
python scripts/migrate_data_to_supabase.py
```

### Expected Output:

```
🚀 Starting data migration to Supabase...
   Embedding model: text-embedding-3-small
   Dimensions: 1536

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

### If Data Already Exists:

```bash
# Force re-import (deletes existing data first)
python scripts/migrate_data_to_supabase.py --force
```

## ✅ Step 4: Verify Migration

### Option A: Run Test Script

```bash
python scripts/test_pgvector_search.py
```

Expected output shows similarity scores > 0.7 for relevant results.

### Option B: Manual SQL Query

In Supabase SQL Editor:

```sql
-- Check row count
SELECT COUNT(*) FROM kb_chunks;
-- Expected: 21

-- View sample chunks
SELECT id, doc_id, section, LEFT(content, 100) as content_preview
FROM kb_chunks
LIMIT 5;

-- Test similarity search
SELECT
    section,
    LEFT(content, 100) as content_preview,
    1 - (embedding <=> (SELECT embedding FROM kb_chunks LIMIT 1)) as similarity
FROM kb_chunks
ORDER BY embedding <=> (SELECT embedding FROM kb_chunks LIMIT 1)
LIMIT 3;
```

## 🐛 Troubleshooting

### Error: "OPENAI_API_KEY not found"

**Solution**: Add API key to `.env` file and restart terminal

```bash
# .env
OPENAI_API_KEY=sk-proj-...your-key
```

### Error: "Supabase configuration invalid"

**Solution**: Check URL and service role key format

```bash
# URL should be: https://PROJECT_ID.supabase.co
# Key should start with: eyJ...
```

### Error: "relation 'kb_chunks' does not exist"

**Solution**: Run the SQL migration first (Step 2)

### Error: "function search_kb_chunks does not exist"

**Solution**: The SQL migration wasn't fully executed. Re-run Step 2.

### Error: Rate limit (429)

**Solution**:
- Script automatically retries with backoff
- OpenAI free tier: 3 requests/minute
- Wait a few minutes or upgrade to paid tier

### Warning: "Found X existing chunks"

**Solution**:
- Data already migrated (safe to ignore)
- Use `--force` flag to re-import if needed

## 💰 Cost Breakdown

**For 21 chunks** (current career_kb.csv):
- OpenAI embedding cost: ~$0.0003
- Supabase storage: Free (well within free tier limits)
- **Total**: < $0.001 per migration

**Scaling estimates**:
- 100 chunks: ~$0.0014
- 1,000 chunks: ~$0.014
- 10,000 chunks: ~$0.14

text-embedding-3-small pricing: $0.00002 per 1K tokens

## 📊 What Happens Next?

After successful migration:

1. ✅ **21 Q&A pairs** stored in `kb_chunks` table
2. ✅ **1536-dimensional embeddings** indexed with IVFFLAT
3. ✅ **Similarity search** ready via `search_kb_chunks()` function
4. ✅ **RAG pipeline** can now retrieve relevant context

Next steps:
- Update `rag_engine.py` to use pgvector (replacing FAISS)
- Build `/api/chat` Next.js endpoint
- Test end-to-end RAG workflow

## 🎯 Success Criteria

Migration is successful when:
- [x] Script completes without errors
- [x] `kb_chunks` table has 21 rows
- [x] Test script returns relevant results
- [x] Similarity scores > 0.7 for matching queries
- [x] Cost < $0.001

---

**Questions?** Check `scripts/README.md` or open an issue on GitHub.

**Last Updated**: October 5, 2025
