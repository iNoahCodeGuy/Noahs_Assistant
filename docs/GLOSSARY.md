# 📖 Technical Glossary

**Purpose**: Definitions for technical terms used throughout Noah's AI Assistant codebase.

---

## 🔍 Vector Search & Embeddings

### **pgvector**
PostgreSQL extension that adds vector similarity search capabilities. Allows storing and querying embeddings directly in Postgres without a separate vector database.

**Why we use it**: Simplifies architecture by consolidating database and vector search into one system (Supabase Postgres).

---

### **Embedding**
A numerical representation of text as a vector (list of numbers). OpenAI's `text-embedding-3-small` model converts text into 1536-dimensional vectors.

**Example**:
```python
"Python developer" → [0.23, -0.45, 0.67, ..., 0.12]  # 1536 numbers
```

**Why 1536 dimensions**: OpenAI's model choice, balances accuracy vs size.

---

### **Cosine Similarity**
Measures how similar two vectors are by calculating the angle between them.

**Range**: 
- `1.0` = Identical (0° angle)
- `0.7` = Moderately similar (45° angle) ← Our threshold
- `0.0` = Completely unrelated (90° angle)

**Why cosine over Euclidean**: Works better for high-dimensional embeddings, ignores magnitude differences.

---

### **IVFFLAT Index**
Inverted File with Flat Compression. An indexing algorithm that makes vector searches faster by pre-clustering vectors.

**Trade-off**:
- **Speed**: 10x faster searches on large datasets
- **Accuracy**: 95-98% recall (might miss 2-5% of exact matches)

**When to use**: Datasets with 10,000+ vectors. Below that, exact search is fine.

---

### **Top-K Retrieval**
Retrieve the K most similar vectors to a query.

**Example**: `top_k=5` returns the 5 most relevant knowledge chunks.

**Why not return all matches**: More context ≠ better. LLMs perform worse with too much noise. 5-10 chunks is the sweet spot.

---

## 🤖 RAG (Retrieval Augmented Generation)

### **Retrieval**
Finding relevant knowledge from a database based on semantic similarity.

**Example**:
```python
Query: "What's your Python experience?"
Retrieved: ["Built APIs with FastAPI...", "10 years Python development..."]
```

---

### **Augmentation**
Adding retrieved context to the LLM prompt.

**Before Augmentation** (bad):
```
User: What's your Python experience?
LLM: [hallucinates random answer]
```

**After Augmentation** (good):
```
Context: "10 years Python development, FastAPI expert..."
User: What's your Python experience?
LLM: Based on my experience, I have 10 years...
```

---

### **Generation**
LLM creating the final response using the augmented prompt.

---

### **Grounded Response**
A response that cites specific sources from retrieved context. Reduces hallucinations.

**Ungrounded** (bad): "I have Python experience."  
**Grounded** (good): "I have 10 years of Python development experience (source: career_kb, chunk_15)."

---

### **Hybrid RAG**
Supporting multiple retrieval backends (pgvector for production, FAISS for tests).

**Why hybrid**: 
- Production: pgvector (centralized, observable)
- Tests: FAISS (no external dependencies, faster CI/CD)

---

## 🗄️ Supabase Terms

### **RLS (Row-Level Security)**
PostgreSQL feature that controls which rows users can access.

**Example**:
```sql
-- Only authenticated users can read their own messages
CREATE POLICY "Users read own messages" ON messages
FOR SELECT USING (auth.uid() = user_id);
```

**Why we use it**: Security without application code. Database enforces permissions.

---

### **Service Role**
Supabase's admin-level access key that bypasses RLS policies.

**⚠️ Security**: Never expose in client code! Only use server-side (like in Python scripts).

---

### **Anon Key**
Supabase's public access key for client-side code. Respects RLS policies.

**Use case**: Next.js frontend will use anon key to ensure users only see their own data.

---

### **RPC (Remote Procedure Call)**
Calling a PostgreSQL function from your application.

**Example**:
```python
# Call custom similarity_search function
supabase.rpc('similarity_search', {
    'query_embedding': [0.23, -0.45, ...],
    'match_count': 5
})
```

**Why use RPC**: Complex SQL logic stays in database, not scattered in application code.

---

### **Realtime**
Supabase's feature for subscribing to database changes.

**Example**: Get notified when a new message is inserted.

**Not used yet**: Will be useful for Phase 3 (live chat updates).

---

## 🧠 OpenAI API Terms

### **text-embedding-3-small**
OpenAI's embedding model that converts text → vectors.

**Cost**: $0.00002 per 1,000 tokens (~750 words)  
**Dimensions**: 1536  
**Speed**: ~50ms per embedding

**Why this model**: 5x cheaper than `text-embedding-ada-002`, same quality.

---

### **Token**
A piece of text (~4 characters on average).

**Example**: "Hello world" = 2 tokens

**Why it matters**: OpenAI charges per token, not per word.

---

### **Context Window**
Maximum amount of text an LLM can process at once.

**GPT-4o-mini**: 128,000 tokens (~96,000 words)  
**Our typical usage**: 2,000 tokens (query + 5 retrieved chunks)

---

### **Temperature**
Controls randomness in LLM responses.

**Range**: 0.0 (deterministic) to 2.0 (creative)  
**Our setting**: 0.7 (balanced)

**Why 0.7**: Accurate but not robotic. 0.0 is too repetitive, 1.5+ is too random.

---

## 🌐 Deployment Terms

### **Vercel**
Serverless hosting platform for Next.js applications.

**Why Vercel**: 
- Free tier for small projects
- Automatic scaling
- Built-in CI/CD
- Optimized for Next.js

---

### **Serverless Function**
Code that runs on-demand without managing servers.

**Example**: `/api/chat` endpoint processes requests then shuts down.

**Benefits**:
- Pay per request (not per hour)
- Auto-scales to zero when unused
- No server maintenance

---

### **Cold Start**
Time it takes to spin up a serverless function from scratch.

**FAISS cold start**: 2-3 seconds (loading vector files)  
**pgvector cold start**: 0.2 seconds (stateless SQL query)

**Why pgvector wins**: No files to load, just query Supabase.

---

## 🧪 Testing Terms

### **Fixture** (pytest)
Reusable test setup code.

**Example**:
```python
@pytest.fixture
def mock_rag_engine():
    return RagEngine(use_pgvector=False)  # Use FAISS for tests
```

**Why fixtures**: Don't repeat setup in every test.

---

### **Mock**
Fake object that simulates real behavior.

**Example**: Mock OpenAI API to avoid real API calls in tests.

---

### **Integration Test**
Tests multiple components working together.

**Example**: Test RoleRouter → RagEngine → ResponseGenerator flow.

---

### **Unit Test**
Tests a single function in isolation.

**Example**: Test `embed()` function alone.

---

## 📊 Analytics Terms

### **Latency**
Time from request to response.

**Target**: <2 seconds for good UX  
**Current**: ~500ms (pgvector) vs ~3s (FAISS)

---

### **Throughput**
Requests per second the system can handle.

**Current**: ~10 req/sec (limited by OpenAI API rate limits)

---

### **Observability**
Ability to understand system behavior from logs/metrics.

**Our tools**:
- Supabase Analytics (message logs, retrieval logs)
- LangSmith (optional, for detailed LLM traces)

---

## 🔧 Architecture Patterns

### **Singleton**
Design pattern ensuring only one instance of a class exists.

**Example**: `get_supabase_client()` returns cached client.

**Why**: Reusing connections is faster (100ms → 10ms on subsequent calls).

---

### **Connection Pooling**
Reusing database connections instead of creating new ones.

**Benefit**: 10x faster queries (no TCP handshake overhead).

**Handled by**: Supabase's PostgREST layer automatically.

---

### **Idempotent**
Operation that produces the same result if run multiple times.

**Example**: Migration script checks if data exists before inserting.

**Why**: Safe to re-run without creating duplicates.

---

### **Lazy Loading**
Creating objects only when needed.

**Example**: `get_supabase_client()` creates client on first call, not at import.

**Why**: Faster startup, avoids unnecessary work.

---

## 🎯 Domain-Specific Terms

### **Role Mode**
User's context: Hiring Manager (technical), Software Developer, etc.

**Why it matters**: Determines retrieval strategy and response style.

---

### **Query Type**
Classification of user's question: technical, career, mma, fun, general.

**How classified**: Keyword matching (e.g., "code" → technical, "experience" → career).

---

### **Dual-Audience Response**
Response format for technical hiring managers with two sections:
1. Engineer Detail (code examples, file:line citations)
2. Plain-English Summary (business-friendly explanation)

**Why**: Technical managers need both technical depth and executive summary.

---

### **Code Chunk**
Snippet of source code with metadata (file path, line numbers, language).

**Size**: 50-200 lines per chunk (sweet spot for context without overwhelming LLM).

---

### **Knowledge Base (KB)**
Collection of text chunks stored for retrieval.

**Types**:
- Career KB: Resume/experience Q&A pairs
- Code Index: Source code snippets

---

## 🛠️ Development Tools

### **Streamlit**
Python framework for building data apps.

**Why we use it**: Quick prototyping, built-in chat interface.

**Next step**: Migrate to Next.js for production (better UX, SEO).

---

### **Next.js**
React framework for building web applications.

**Why**: 
- Better SEO (server-side rendering)
- Faster load times
- Professional UI/UX
- Vercel integration

---

### **LangSmith**
Observability platform for LLM applications.

**Features**:
- Trace every LLM call
- Debug prompt issues
- Compare model versions
- Monitor costs

**Status**: Optional, not yet integrated.

---

## 💰 Cost Terms

### **Supabase Pro**
$25/month tier with:
- 8GB database
- 50GB bandwidth
- pgvector included
- No cold starts

---

### **OpenAI API Costs**
- **Embeddings**: $0.00002 per 1K tokens (~$5-10/month for 100K queries)
- **GPT-4o-mini**: $0.00015 per 1K input tokens (~$20-30/month)

**Total estimated**: $30-35/month (vs $100-200/month on GCP).

---

## 📚 Further Reading

- [pgvector GitHub](https://github.com/pgvector/pgvector) - Vector extension docs
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) - How embeddings work
- [Supabase Docs](https://supabase.com/docs) - Comprehensive guides
- [RAG Paper](https://arxiv.org/abs/2005.11401) - Original research

---

**Last Updated**: January 2025  
**Maintainer**: Noah De La Calzada
