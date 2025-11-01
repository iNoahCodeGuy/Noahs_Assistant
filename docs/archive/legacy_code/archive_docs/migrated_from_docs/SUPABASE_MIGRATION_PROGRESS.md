# Supabase + Vercel Migration Progress

## Overview
Migrating Noah's AI Assistant from GCP (Cloud SQL, Pub/Sub, Secret Manager) to **Supabase + Vercel** for:
- Lower costs (~$25-50/month vs ~$100-200/month)
- Simpler architecture (no Pub/Sub, Redis, or BigQuery)
- Unified backend (Postgres + Storage + Auth in one service)
- Better developer experience (Supabase Studio UI, real-time subscriptions, auto-generated APIs)

---

## ✅ Completed (Steps 1-4)

### 1. Architecture Analysis ✅
- Reviewed current GCP implementation (Cloud SQL + Pub/Sub + Secret Manager)
- Designed Supabase schema with pgvector for embeddings
- Planned hybrid deployment: Streamlit for chat + Next.js API routes for external services

### 2. Supabase Configuration ✅
**File:** `src/config/supabase_config.py` (195 lines)

Key features:
- `SupabaseConfig` dataclass with URL, service role key, bucket names
- `SupabaseSettings` class with environment variable management
- `get_supabase_client()` for lazy client initialization
- Backward compatibility: maintains `api_key`, `validate_api_key()`, `career_kb_path`

Environment variables:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_ANON_KEY=your-anon-key
OPENAI_API_KEY=sk-...
RESEND_API_KEY=re_...
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM=+1234567890
```

### 3. Database Schema ✅
**File:** `supabase/migrations/001_initial_schema.sql` (364 lines)

Tables created:
- **kb_chunks**: Knowledge base with `embedding vector(1536)`, doc_id, section, content, metadata
- **messages**: Chat logs with session_id, role_mode, query, answer, latency_ms, token counts
- **retrieval_logs**: RAG tracking with topk_ids[], scores[], grounded flag
- **links**: External resources (MMA fight, LinkedIn, GitHub)
- **feedback**: User ratings (1-5), comments, contact requests

Indexes:
- IVFFLAT on `kb_chunks.embedding` for fast similarity search (lists=100)
- Composite indexes on session_id, role_mode, created_at

Security:
- Row Level Security (RLS) enabled on all tables
- `service_role` policies for backend ALL operations
- `anon` policies for SELECT on kb_chunks, links (future public API)

Functions:
- `search_kb_chunks(query_embedding, match_threshold=0.7, match_count=3)` - pgvector similarity search
- `update_updated_at_column()` - Auto-update timestamp trigger

Views:
- `messages_with_retrieval` - Joins messages + retrieval_logs
- `analytics_by_role` - Aggregates messages by role_mode

### 4. Supabase Analytics ✅
**File:** `src/analytics/supabase_analytics.py` (330 lines)

Replaces `cloud_analytics.py` with simpler implementation:
- ❌ No Pub/Sub → ✅ Direct database writes
- ❌ No Secret Manager → ✅ Environment variables
- ❌ No connection pooling → ✅ Supabase handles it

Key classes:
- `UserInteractionData` - Dataclass for message logging
- `RetrievalLogData` - Dataclass for RAG event logging
- `SupabaseAnalytics` - Main analytics class

Methods:
- `log_interaction(UserInteractionData)` → Returns message_id
- `log_retrieval(RetrievalLogData)` → Logs RAG chunks
- `log_feedback(message_id, rating, comment, email)` → Stores user feedback
- `get_user_behavior_insights(days=30)` → Analytics aggregation
- `health_check()` → System status

### 5. Updated Files ✅
- ✅ `requirements.txt` - Replaced GCP packages with Supabase/Resend/Twilio
- ✅ `src/main.py` - Updated imports from cloud_analytics → supabase_analytics
- ✅ `src/agents/role_router.py` - Updated cloud_settings → supabase_settings
- ✅ `src/core/rag_engine.py` - Updated cloud_settings → supabase_settings
- ✅ `src/analytics/__init__.py` - Updated exports

---

## 🔄 In Progress (Step 5)

### 5. RAG Engine pgvector Migration
**File:** `src/core/rag_engine.py` (needs update)

Current state:
- Uses local FAISS vector stores
- Loads career_kb.csv on startup

Target state:
- Replace FAISS with Supabase pgvector
- Use `search_kb_chunks()` SQL function
- Load embeddings from `kb_chunks` table

Implementation plan:
1. Add `search_similar_chunks(query_embedding, threshold=0.7, top_k=3)` method
2. Update `retrieve()` to call Supabase instead of FAISS
3. Create data migration script: `scripts/migrate_career_kb_to_supabase.py`
4. Populate `kb_chunks` table from `data/career_kb.csv`

---

## 📝 Pending (Steps 6-9)

### 6. Next.js API Routes
**Files to create:**
- `pages/api/chat.ts` - Main RAG endpoint (embed → search → generate)
- `pages/api/email.ts` - Resume delivery via Resend
- `pages/api/feedback.ts` - Store feedback + trigger Twilio SMS
- `pages/api/health.ts` - System health check

Example chat endpoint:
```typescript
// pages/api/chat.ts
import { createClient } from '@supabase/supabase-js'
import { OpenAI } from 'openai'

export default async function handler(req, res) {
  const { query, session_id, role_mode } = req.body

  // 1. Generate embedding
  const embedding = await openai.embeddings.create(...)

  // 2. Search KB with pgvector
  const { data: chunks } = await supabase.rpc('search_kb_chunks', {
    query_embedding: embedding,
    match_threshold: 0.7,
    match_count: 3
  })

  // 3. Generate response with context
  const response = await openai.chat.completions.create(...)

  // 4. Log to database
  await supabase.from('messages').insert(...)

  return res.json({ answer: response })
}
```

### 7. External Services Integration
**Files to create:**
- `src/services/resend_service.py` - Email delivery
- `src/services/twilio_service.py` - SMS notifications

Email service (Resend):
```python
from resend import Resend
from config.supabase_config import supabase_settings

resend = Resend(supabase_settings.resend_api_key)

def send_resume_email(to_email: str, resume_url: str):
    """Send Noah's resume via email."""
    resend.emails.send({
        "from": "noah@yourdomain.com",
        "to": to_email,
        "subject": "Noah's Resume",
        "html": f"<p>Here's my resume: <a href='{resume_url}'>Download</a></p>"
    })
```

SMS service (Twilio):
```python
from twilio.rest import Client
from config.supabase_config import supabase_settings

client = Client(
    supabase_settings.twilio_account_sid,
    supabase_settings.twilio_auth_token
)

def send_contact_notification(email: str, message: str):
    """Send SMS when contact is requested."""
    client.messages.create(
        to="+1234567890",  # Noah's phone
        from_=supabase_settings.twilio_from,
        body=f"New contact request from {email}: {message}"
    )
```

### 8. Vercel Deployment Configuration
**Files to create:**
- `vercel.json` - Deployment configuration
- `package.json` - Next.js dependencies
- `tsconfig.json` - TypeScript configuration
- `next.config.js` - Next.js settings

Example vercel.json:
```json
{
  "version": 2,
  "builds": [
    {
      "src": "pages/api/**/*.ts",
      "use": "@vercel/node"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "/pages/api/$1"
    }
  ],
  "env": {
    "SUPABASE_URL": "@supabase-url",
    "SUPABASE_SERVICE_ROLE_KEY": "@supabase-service-role-key",
    "OPENAI_API_KEY": "@openai-api-key"
  }
}
```

Note: Streamlit app will be deployed separately (Streamlit Cloud or Vercel with custom config)

### 9. Documentation Updates
**Files to update:**
- `README.md` - Add Supabase + Vercel setup instructions
- `docs/ARCHITECTURE.md` - Update architecture diagram
- `API_KEY_SETUP.md` - Add Resend and Twilio keys

---

## Cost Comparison

### GCP (Previous)
- Cloud SQL: $50-100/month (db-f1-micro + storage)
- Cloud Run: $0-20/month (depends on usage)
- Pub/Sub: $5-10/month (message ingestion)
- Secret Manager: $0.30/month (6 secrets)
- **Total: ~$100-200/month**

### Supabase + Vercel (New)
- Supabase Pro: $25/month (500GB bandwidth, 8GB database, 100GB storage)
- Vercel Hobby: $0/month (or Pro $20/month for team)
- OpenAI API: Variable (based on usage, ~$10-30/month)
- Resend: $0/month (3,000 emails/month free)
- Twilio: $0/month (pay-as-you-go, ~$0.0075/SMS)
- **Total: ~$25-50/month**

Savings: **$50-150/month (50-75% reduction)**

---

## Testing Strategy

### Unit Tests (Update existing)
- `tests/test_supabase_analytics.py` - Mock Supabase client
- `tests/test_rag_engine.py` - Mock pgvector search
- `tests/test_role_router.py` - Verify supabase_settings usage

### Integration Tests (New)
- `tests/test_supabase_integration.py` - Real Supabase connection
- `tests/test_pgvector_search.py` - Vector similarity search
- `tests/test_api_routes.py` - Next.js API endpoint tests

### Test fixtures
```python
@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client for offline testing."""
    client = Mock()
    client.table.return_value.insert.return_value.execute.return_value = Mock(data=[{'id': 1}])
    return client
```

---

## Next Steps (Immediate)

1. **Create data migration script** (`scripts/migrate_career_kb_to_supabase.py`)
   - Read `data/career_kb.csv`
   - Generate embeddings for each row
   - Insert into `kb_chunks` table with pgvector

2. **Update RagEngine for pgvector**
   - Replace FAISS with Supabase search_kb_chunks()
   - Test retrieval accuracy
   - Verify similarity threshold (0.7) is appropriate

3. **Create Next.js API routes**
   - Implement `/api/chat` endpoint
   - Test with Postman/curl
   - Deploy to Vercel preview

4. **Set up environment variables**
   - Create `.env.local` for local development
   - Add secrets to Vercel dashboard
   - Test configuration loading

---

## Migration Checklist

- [x] Create Supabase project
- [x] Run SQL migration (001_initial_schema.sql)
- [x] Update Python configuration (supabase_config.py)
- [x] Replace analytics system (supabase_analytics.py)
- [x] Update imports in main.py, role_router.py, rag_engine.py
- [x] Update requirements.txt
- [ ] Migrate career KB data to kb_chunks
- [ ] Update RAG engine for pgvector
- [ ] Create Next.js API routes
- [ ] Set up Resend email service
- [ ] Set up Twilio SMS service
- [ ] Configure Vercel deployment
- [ ] Update documentation
- [ ] Run integration tests
- [ ] Deploy to production

---

## Resources

- **Supabase Docs**: https://supabase.com/docs
- **pgvector Guide**: https://supabase.com/docs/guides/ai/vector-columns
- **Resend API**: https://resend.com/docs
- **Twilio SMS**: https://www.twilio.com/docs/sms
- **Vercel Next.js**: https://vercel.com/docs/frameworks/nextjs

---

*Last updated: 2025-10-05*
