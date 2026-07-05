# CLAUDE.md

## Project: Portfolia AI Assistant
Noah's AI-powered portfolio assistant built with RAG architecture. Live at
https://noahdelacalzada.com (frontend repo: iNoahCodeGuy/portfolia-frontend).

## Tech Stack
- **Pipeline:** 22-node functional pipeline (LangGraph-style; plain function loop at runtime)
- **Backend:** FastAPI (Python 3.12) on Railway — single endpoint `POST /chat`
- **Database:** Supabase + pgvector (vector storage, semantic retrieval)
- **Generation:** Anthropic Claude Sonnet 4.5 · **Classification:** Claude Haiku
- **Embeddings:** OpenAI text-embedding-3-small (1536 dims)
- **Observability:** LangSmith

## Project Structure
- `assistant/flows/conversation_flow.py` — pipeline entry point (`run_conversation_flow`)
- `assistant/flows/node_logic/stage*.py` — node implementations (stage0–stage7)
- `assistant/flows/capture/` — capture state machines (crush flow, lead capture,
  notifications) + the marker constants both producer and detector must share
- `assistant/core/rag_engine.py` — RAG orchestration
- `assistant/retrieval/pgvector_retriever.py` — vector search against Supabase
- `api/main.py` — FastAPI app

## Commands
- Terminal client: `python3 chat_with_portfolia.py` (must use `run_conversation_flow()`, not `generate_response()` directly)
- API server: `uvicorn api.main:app --reload --port 8000`
- Tests (hermetic, same as CI): `pytest` — live evals are opt-in via `pytest -m live`
- Re-embed KB after editing `data/*.csv`: `python3 scripts/migrate_data_to_supabase.py`

## Key Architecture Decisions
- Intent classification happens BEFORE RAG retrieval (not every message needs vector search)
- Special flows (crush confession, greetings, small talk) bypass RAG entirely
- Supabase RPC function `match_kb_chunks` handles vector similarity search
- Two similarity thresholds: 0.5 (strict) and 0.3 (fallback/broader)
- Multi-step flows (contact capture, crush confession) are state machines recovered from
  chat-history markers each turn — stateless, serverless-compatible; the pipeline (not
  the LLM) triggers side effects (Supabase writes, Twilio SMS, Resend email)
- Nodes return partial dicts; the pipeline merges them via `state.update(result)`
- LangSmith tracing is enabled for all LLM calls

## Tone & Personality
Portfolia is dry, confident, direct, and conversational — never resume-speak or
Wikipedia voice. Lead with the most interesting fact; end with a hook, never a menu.
The full persona spec, scripted conversation examples, and conversation-flow design live
in `CLAUDE.local.md` (gitignored — ask Noah for it if missing).

## Testing
When testing changes, try these queries to verify quality:
1. "What is Noah's professional background?" — should be conversational, not a dry list
2. "What are some projects by Noah?" — should return specific projects with personality
3. "I would like to confess a crush" — should route to crush flow, NOT hit RAG
4. "asdfghjkl" — should gracefully handle gibberish with a redirect

## Common Pitfalls
- Don't send non-knowledge queries through RAG — they return 0 chunks and the fallback is bad
- Local `data/*.csv` edits do NOT affect retrieval until re-embedded to Supabase
- hpack/urllib3/langsmith loggers are noisy — keep them at WARNING in production
- The system prompt is the #1 lever for response quality — if responses sound robotic,
  the system prompt needs work, not the retrieval
- `initialize_conversation_state` MUST clear volatile per-turn fields
  (`pipeline_halt`, `answer`, `skip_rag`, `message_intent`, etc.)
- The pipeline mutates `chat_history` in place — deep-copy first if you need the original
