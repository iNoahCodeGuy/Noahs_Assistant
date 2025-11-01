# LangSmith - Quick Start Card

## 🎯 View Your Traces

**Dashboard**: https://smith.langchain.com/o/project/noahs-ai-assistant

**Filter by Session**:
- `script-test-001` - Individual node tests
- `script-test-full-pipeline` - Complete conversation
- Any Streamlit session (check browser console for ID)

---

## ✅ Test LangSmith Setup

```bash
# Run test script (validates everything works)
.venv/bin/python scripts/test_langsmith_tracing.py

# Should output:
# ✅ TEST 1: Query Classification - [OK]
# ✅ TEST 2: Retrieval (4 chunks) - [OK]
# ✅ TEST 3: Generation (581 chars) - [OK]
# ✅ TEST 4: Full Pipeline - [OK]
```

---

## 🔍 What Gets Traced

**Automatically**:
- All OpenAI API calls (embeddings, chat)
- All Supabase pgvector queries
- Node transitions in conversation flow

**Metadata Captured**:
- Query text, role, session ID
- Retrieval scores (cosine similarity)
- Token usage, latency
- Errors with stack traces

---

## 🛠️ Troubleshooting

### Traces not appearing?

**Check environment**:
```bash
python -c "import os; print(f'TRACING: {os.getenv(\"LANGCHAIN_TRACING_V2\")}')"
# Expected: TRACING: true
```

**Check client**:
```bash
python -c "from src.observability.langsmith_tracer import get_langsmith_client; print(get_langsmith_client())"
# Expected: <langsmith.client.Client object...>
```

### Missing session IDs?

Ensure `ConversationState` includes:
```python
state = ConversationState(
    query="...",
    role="...",
    session_id="...",  # Required!
    chat_history=[]    # Required by some nodes
)
```

---

## 📚 Full Documentation

See: `docs/LANGSMITH_TRACING_SETUP.md` for complete guide.

---

## 🚀 Next Steps

1. **Create Evaluation Dataset** - Select 10-20 good traces → "Add to Dataset"
2. **Set Up Alerts** - Dashboard → Settings → Alerts (error rate, latency)
3. **Enable Feedback** - Add 👍👎 buttons, send to LangSmith
4. **Run Evaluations** - Test prompts against dataset, score quality

---

**Questions?** Check `docs/LANGSMITH_TRACING_SETUP.md` or LangSmith docs: https://docs.smith.langchain.com/
