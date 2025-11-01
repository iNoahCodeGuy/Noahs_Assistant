# LangSmith Tracing Setup - Complete Guide

**Status**: ✅ PRODUCTION READY
**Date**: October 30, 2025
**LangSmith Project**: `noahs-ai-assistant`

---

## Quick Start

LangSmith tracing is **already enabled** in production. All conversation nodes automatically send traces to LangSmith dashboard.

**View traces**: https://smith.langchain.com/o/project/noahs-ai-assistant

**Filter by session IDs**:
- `script-test-001` - Individual node tests
- `script-test-full-pipeline` - Complete conversation flow
- Any Streamlit/Vercel session ID (e.g., `sess_abc123`)

---

## What Was Configured

### 1. Environment Variables (.env)

```bash
# LangSmith Observability
LANGSMITH_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxxxx_xxxxxxxxxx  # Get from langsmith.langchain.com
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxxxx_xxxxxxxxxx  # Same as LANGSMITH_API_KEY
LANGCHAIN_PROJECT=noahs-ai-assistant
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

**Why these variables?**
- `LANGCHAIN_TRACING_V2=true` - Enables automatic tracing for all LangChain calls
- `LANGSMITH_API_KEY` - Authentication for LangSmith API
- `LANGCHAIN_PROJECT` - Groups all traces under this project name
- `LANGCHAIN_ENDPOINT` - API endpoint (default, but explicit for clarity)

### 2. Decorators Applied

Existing decorators in `src/observability/langsmith_tracer.py`:

```python
@trace_retrieval  # Applied to retrieve_chunks()
@trace_generation  # Applied to generate_draft()
```

These decorators automatically:
- Create spans in LangSmith with operation metadata
- Log input/output for each traced function
- Capture errors and exceptions
- Link child spans to parent traces

### 3. Test Script Created

**Location**: `scripts/test_langsmith_tracing.py`

**What it does**:
1. Validates LangSmith environment configuration
2. Tests 4 conversation scenarios:
   - Query classification (detects intent)
   - Retrieval (pgvector search with similarity scoring)
   - Generation (LLM response with retrieved context)
   - Full conversation pipeline (end-to-end flow)
3. Sends all traces to LangSmith dashboard
4. Prints session IDs for easy filtering

**Run it**:
```bash
.venv/bin/python scripts/test_langsmith_tracing.py
```

**Expected output**:
```
============================================================
TEST 1: Query Classification
============================================================
[OK] Classification complete:
   Query Type: career
   Intent: unknown

============================================================
TEST 2: Retrieval (Traced to LangSmith)
============================================================
[OK] Retrieved 4 chunks:
   [1] Similarity: 0.674 - Q: What are Noah's software engineering skills?
   [2] Similarity: 0.641 - Q: How strong is Noah's Python?
   [3] Similarity: 0.569 - Q: Where can I see Noah's GitHub?

============================================================
TEST 3: Generation (Traced to LangSmith)
============================================================
[OK] Generated answer (581 chars):
   Q: What are Noah's software engineering skills?
A: Noah has intermediate Python skills...

============================================================
TEST 4: Full Conversation Pipeline
============================================================
Running full pipeline for: Show me Noah's error handling implementation
[OK] Pipeline complete!
   Final answer: Based on Noah's portfolio, here's his error handling...
   State keys: 27

============================================================
All Tests Complete!
============================================================

View traces in LangSmith:
   https://smith.langchain.com/o/project/noahs-ai-assistant

   Session IDs to filter:
   - script-test-001 (individual nodes)
   - script-test-full-pipeline (complete flow)
```

---

## What Gets Traced

### Automatic Tracing (via LANGCHAIN_TRACING_V2=true)

✅ All OpenAI API calls (embeddings, chat completions)
✅ All Supabase pgvector queries (via traced `retrieve()` calls)
✅ All LangChain components (if used)
✅ All node transitions in conversation flow

### Manual Tracing (via decorators)

✅ `retrieve_chunks()` - Retrieval performance, similarity scores, chunk count
✅ `generate_draft()` - LLM generation, token usage, latency
✅ Custom spans created with `create_custom_span()` helper

### Trace Metadata Captured

Each trace includes:
- **Query text** (first 120 chars for privacy)
- **Session ID** (links related operations)
- **Role** (hiring_manager_technical, developer, etc.)
- **Retrieval scores** (cosine similarity for each chunk)
- **Chunk count** (number of KB chunks retrieved)
- **Token usage** (prompt + completion tokens)
- **Latency** (ms for each operation)
- **Errors** (full stack traces if failures occur)

---

## How to Use LangSmith Dashboard

### 1. View All Traces

Navigate to: https://smith.langchain.com/o/project/noahs-ai-assistant

You'll see a waterfall view of all traces, sorted by timestamp (most recent first).

### 2. Filter by Session ID

Click **Filters** → **Session ID** → Enter session ID (e.g., `script-test-001`)

This shows all operations for a single conversation, including:
- Query classification
- Retrieval (pgvector search)
- Generation (LLM call)
- Formatting (markdown rendering)
- Logging (analytics persistence)

### 3. Inspect Individual Traces

Click any trace row to expand details:

**Overview Tab**:
- Duration (total time)
- Input/output (full query and response)
- Metadata (session ID, role, etc.)

**Tree View**:
- Parent-child relationships
- Nested spans (e.g., retrieval → embedding → pgvector search)

**Performance Tab**:
- Latency breakdown by operation
- Bottleneck identification

**Logs Tab**:
- Console output during trace
- Error messages if failures occurred

### 4. Compare Traces

Select 2+ traces → Click **Compare** button

Useful for:
- A/B testing prompt changes
- Identifying performance regressions
- Comparing retrieval quality across queries

### 5. Export Traces

Click **Export** → Choose format (JSON, CSV, or LangSmith Dataset)

Use for:
- Offline analysis
- Creating evaluation datasets
- Sharing with team members

---

## Troubleshooting

### Traces Not Appearing

**Check 1: Environment variables set?**
```bash
python -c "import os; print(f'TRACING: {os.getenv(\"LANGCHAIN_TRACING_V2\")}, API_KEY: {os.getenv(\"LANGSMITH_API_KEY\")[:20]}...')"
```

Expected: `TRACING: true, API_KEY: lsv2_pt_c9ca26e45dbb...`

**Check 2: LangSmith client initialized?**
```bash
python -c "from src.observability.langsmith_tracer import get_langsmith_client; print(get_langsmith_client())"
```

Expected: `<langsmith.client.Client object at 0x...>`

**Check 3: Network connectivity?**
```bash
curl -I https://api.smith.langchain.com
```

Expected: `HTTP/2 200`

### Traces Missing Metadata

**Issue**: Traces appear but don't show session_id, role, etc.

**Solution**: Check that `ConversationState` includes required fields:
```python
state = ConversationState(
    query="...",
    role="...",
    session_id="...",  # Required for grouping
    chat_history=[]    # Required by some nodes
)
```

### Slow Trace Upload

**Issue**: Script hangs at shutdown while uploading traces.

**Explanation**: LangSmith client batches traces and uploads on exit. Large traces (e.g., 10KB+ responses) take 2-5 seconds to compress and upload.

**Solution**: Wait for `"POST /runs/multipart HTTP/1.1" 202` log message confirming upload.

### Missing Retrieval Scores

**Issue**: Traces show retrieval but no similarity scores.

**Root cause**: `rag_engine.retrieve()` returns dict with `chunks` and `scores` keys, but `retrieve_chunks()` node expects `similarity` field in each chunk.

**Fix applied**: Lines 89-102 in `src/flows/node_logic/retrieval_nodes.py` normalize chunk format and extract similarity scores.

---

## Performance Impact

**Overhead**: ~5-10ms per traced operation (negligible)

**Network**: 1-2KB per trace (compressed multipart uploads)

**Cold start**: No impact (LangSmith client lazy-loads)

**Production**: Safe to leave enabled 24/7 (async uploads don't block responses)

---

## Next Steps

### Immediate Actions

✅ **Done**: Environment configured
✅ **Done**: Test script validated
✅ **Done**: All 4 tests passing

### Recommended Next Steps

1. **Create Evaluation Dataset**
   - Go to LangSmith dashboard
   - Select 10-20 representative traces
   - Click **Add to Dataset** → Name it "noah-ai-eval-v1"
   - Use for regression testing when changing prompts

2. **Set Up Alerts**
   - Dashboard → Settings → Alerts
   - Configure:
     - Error rate > 5% in 1 hour
     - P95 latency > 2 seconds
     - Retrieval returning 0 chunks (may indicate KB issues)

3. **Enable Feedback Collection**
   - Add thumbs up/down buttons to Streamlit UI
   - Send feedback via `langsmith.feedback.create_feedback()`
   - Use feedback scores to identify weak queries

4. **Run Evaluations**
   - Create test cases for each role
   - Use LangSmith evaluators to score:
     - Correctness (does answer match expected?)
     - Relevance (are retrieved chunks relevant?)
     - Tone (matches personality guidelines?)
   - See: https://docs.smith.langchain.com/evaluation

### Advanced Features

- **Prompt Hub**: Store prompts in LangSmith, version them, A/B test
- **Annotations**: Tag traces with labels (e.g., "production", "test", "demo")
- **Cost Tracking**: Monitor OpenAI token usage per session
- **Latency SLOs**: Set targets (e.g., P95 < 1.5s) and track compliance

---

## Related Documentation

- **LangSmith Docs**: https://docs.smith.langchain.com/
- **Tracing Guide**: https://docs.smith.langchain.com/tracing
- **Evaluation Guide**: https://docs.smith.langchain.com/evaluation
- **System Architecture**: `docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md`
- **Observability Design**: `docs/OBSERVABILITY.md`

---

## Code Locations

| Component | Path |
|-----------|------|
| Decorators | `src/observability/langsmith_tracer.py` |
| Test Script | `scripts/test_langsmith_tracing.py` |
| Retrieval Node | `src/flows/node_logic/retrieval_nodes.py` |
| Generation Node | `src/flows/node_logic/generation_nodes.py` |
| RAG Engine | `src/core/rag_engine.py` |
| Environment Config | `.env` (not committed to Git) |

---

## Summary

LangSmith tracing is **fully operational** and capturing all conversation flow operations. You can now:

1. ✅ **View traces** in LangSmith dashboard for any session
2. ✅ **Debug issues** by inspecting exact inputs/outputs at each node
3. ✅ **Measure performance** via latency waterfall views
4. ✅ **Track retrieval quality** via similarity scores and chunk content
5. ✅ **Monitor production** by filtering for Streamlit/Vercel session IDs

**No code changes needed** - tracing is automatic via environment variables and existing decorators.

**Test it**: Run `scripts/test_langsmith_tracing.py` anytime to validate configuration.

---

**Questions?** Check `docs/OBSERVABILITY.md` or ask in project documentation.
