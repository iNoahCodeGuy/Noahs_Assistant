# GitHub Copilot Instructions for Portfolia (Noah's AI Assistant)

## Quick Context Links (Always Reference)

Before implementing features or answering questions, open these master docs:

1. 📘 **Project Overview** → `docs/context/PROJECT_REFERENCE_OVERVIEW.md` (purpose, roles, stack)
2. 🧩 **System Architecture** → `docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md` (control flow, RAG pipeline, data layer)
3. 🧮 **Data & Schema Reference** → `docs/context/DATA_COLLECTION_AND_SCHEMA_REFERENCE.md` (tables, queries, presentation rules)
4. 💬 **Conversation Personality** → `docs/context/CONVERSATION_PERSONALITY.md` (warmth, enthusiasm, engagement)
5. 🔍 **LangSmith Tracing** → `docs/LANGSMITH_TRACING_SETUP.md` (observability, debugging, performance monitoring)

These define:
- What the assistant is and why it exists
- Role-specific behaviors and conversation modes (narrative vs data)
- When to show code, tables, or long explanations
- Data contracts and grounding rules
- Personality: warmth, excitement, invitation culture

## System Architecture Overview

This is a **role-based RAG (Retrieval-Augmented Generation) application** serving as an interactive résumé assistant. The system uses:

- **Hybrid deployment**: Streamlit UI (`src/main.py`) for local dev + Vercel serverless (`api/`) for production
- **LangGraph-style orchestration**: Modular node-based conversation flow with organized node logic:
  - `conversation_nodes.py` - Central export hub (re-exports all nodes from `node_logic/`)
  - `node_logic/` package - Focused modules (<200 lines each):
    - `session_management.py` - State initialization
    - `role_routing.py` - Role classification
    - `query_classification.py` - Intent detection
    - `entity_extraction.py` - Company/role/contact extraction
    - `clarification.py` - Vague query handling
    - `query_composition.py` - Retrieval-ready queries
    - `presentation_control.py` - Depth/display formatting
    - `retrieval_nodes.py` - Retrieval pipeline (pgvector search, re-ranking, grounding validation)
    - `generation_nodes.py` - Generation pipeline (LLM call, hallucination check)
    - `formatting_nodes.py` - Formatting pipeline (structured layout, toggles, enrichments)
    - `logging_nodes.py` - Logging pipeline (analytics persistence, followups, memory)
    - `core_nodes.py` - Backward-compatible aliases (re-exports from split modules)
    - `action_planning.py` - Role-based action decisions
    - `action_execution.py` - Side effects (email, SMS, storage)
    - `code_validation.py` - Sanitization and validation
    - `greetings.py` - Role-specific welcome messages
    - `resume_distribution.py` - Hiring signal detection
    - `analytics_renderer.py` - Analytics display
    - `performance_metrics.py` - Performance tracking
  - `content_blocks.py` - Reusable enterprise messaging blocks
  - `data_reporting.py` - Analytics display with markdown tables
- **Supabase pgvector**: Centralized vector storage replacing any local FAISS (see `src/retrieval/pgvector_retriever.py`)
- **Role-driven retrieval**: Each of 5 roles (technical/nontechnical hiring managers, developers, casual visitors, confessions) triggers different knowledge sources and formatting

### Critical Flow: User Query → Response

```
User input → classify_query → retrieve_chunks (pgvector) → generate_answer →
plan_actions → apply_role_context → execute_actions → log_and_notify
```

All conversation state lives in `ConversationState` dataclass (immutable updates via nodes). Never modify `state.chat_history` or `state.answer` directly—use `state.set_answer()` and `state.stash()` helpers.

## Project-Specific Conventions

### Import Patterns

**Always use the compatibility layer for LangChain imports**:
```python
from src.core.langchain_compat import (
    OpenAIEmbeddings, CSVLoader, ChatOpenAI, Document
)
```
Never import directly from `langchain.*` or `langchain_openai.*`—the compat layer handles version migrations and fallbacks.

### Configuration & Environment

- **Settings**: Use `supabase_settings` (singleton) from `src/config/supabase_config.py` for all config
- **Environment detection**: Check `supabase_settings.is_vercel` or `is_production` for deployment-specific logic
- **Required env vars**: `OPENAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` (see `.env.example` if creating)

### Service Pattern

External integrations use factory functions returning singletons:
```python
from src.services.resend_service import get_resend_service
from src.services.twilio_service import get_twilio_service
from src.services.storage_service import get_storage_service

resend = get_resend_service()  # Handles None gracefully if keys missing
```

Never instantiate `ResendService()` or `TwilioService()` directly—factories handle degraded mode and config validation.

## Role-Specific Logic

**5 distinct roles** with different retrieval strategies (see `src/agents/roles.py` and `ROLE_FEATURES.md`):

1. **Hiring Manager (nontechnical)**: Career KB only, offers resume after 2 turns
2. **Hiring Manager (technical)**: Career KB + code snippets, dual-audience formatting
3. **Software Developer**: Code index prioritized, includes architecture diagrams on "show me" queries
4. **Just looking around**: Lightweight, adds fun facts, MMA fight links on keyword match
5. **Looking to confess crush**: Bypasses RAG flow entirely, stores in `data/confessions.csv`

### Query Type Classification

In `RoleRouter._classify_query()` (line ~38), specific regex patterns detect:
- `\bmma\b|\bfight\b|\bufc\b` → MMA query (returns YouTube link)
- `"show me"|"display"|"diagram"` → Raw content display (no LLM rewrite)
- Technical/career keywords → Route to appropriate KB

**When adding new query types**: Update both `_classify_query()` and corresponding handler in `conversation_nodes.py`.

## RAG Engine Architecture

**Production uses pgvector exclusively** (no local vector stores):

```python
# ✅ Correct: Centralized pgvector retrieval
results = rag_engine.retrieve(query, top_k=4)
chunks = results.get("chunks", [])

# ❌ Wrong: FAISS is deprecated
career_kb = FAISS.load_local(...)  # DO NOT USE
```

Key methods:
- `retrieve(query, top_k=4)` → Dict with `chunks`, `matches`, `scores` keys
- `retrieve_with_code(query, role)` → Adds `code_snippets` field for technical roles
- `generate_response(query, chat_history)` → Role-aware LLM generation

**Observability**: All retrieval/generation calls auto-traced to LangSmith if `LANGSMITH_API_KEY` set (see decorators in `src/observability/`).

## Testing & Validation

Run tests with **pytest** (no unittest discovery):
```bash
pytest tests/test_code_display_edge_cases.py -v
pytest tests/ -k "role" --maxfail=1
```

**Testing pattern**: Mock Supabase client via `patch('supabase.create_client')`, provide fixture response:
```python
@patch('supabase.create_client')
def test_retrieval(mock_supabase):
    mock_client = MagicMock()
    mock_client.rpc.return_value.execute.return_value.data = [...]
    mock_supabase.return_value = mock_client
```

See `tests/test_code_display_edge_cases.py` for edge case patterns (empty queries, malformed input, XSS attempts).

## Data Migration & Setup

**One-time setup** after cloning:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables (create .env)
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...

# 3. Run database migrations in Supabase SQL Editor
# Go to Supabase Dashboard → SQL Editor → New Query
# Copy/paste and run each migration file in order:
- supabase/migrations/001_initial_schema.sql
- supabase/migrations/002_add_confessions_and_sms.sql

# 4. Run data migration (idempotent, safe to re-run)
python scripts/migrate_data_to_supabase.py

# 5. Start local Streamlit
streamlit run src/main.py
```

**Schema lives in** `supabase/migrations/` (apply via Supabase dashboard SQL editor).

**Common migration issues**:
- `404 Not Found` on API calls → Table doesn't exist, run migrations
- `FUNCTION_INVOCATION_FAILED` → Check Vercel logs for specific table name
- Missing columns → Run migration 002 to add `confessions`, `sms_logs` tables

## Vercel Deployment

API routes in `api/*.py` follow **Vercel Python runtime** pattern:
```python
class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Parse JSON body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        data = json.loads(body)

        # Process with LangGraph flow
        state = ConversationState(role=data['role'], query=data['query'])
        result = run_conversation_flow(state, rag_engine, session_id=...)

        # Return JSON
        self._send_json(200, {'answer': result.answer})
```

Routes configured in `vercel.json` (no Next.js required, pure Python functions).

## Common Workflows

### Adding a New Conversation Node
1. Define node function in `src/flows/node_logic/<module>.py` (choose appropriate module or create new one):
   ```python
   def my_node(state: ConversationState) -> ConversationState:
       # Mutate state immutably
       state.stash("my_key", value)
       return state
   ```
2. Export from `src/flows/node_logic/__init__.py`:
   ```python
   from src.flows.node_logic.<module> import my_node
   __all__ = [..., "my_node"]
   ```
3. Re-export from `src/flows/conversation_nodes.py`:
   ```python
   from src.flows.node_logic.<module> import my_node
   __all__ = [..., "my_node"]
   ```
4. Insert into pipeline in `conversation_flow.py`:
   ```python
   pipeline = (classify_query, retrieve_chunks, my_node, generate_answer, ...)
   ```
5. Add tests in `tests/test_conversation_flow.py`

**Module organization guidelines:**
- **Retrieval concerns** → `retrieval_nodes.py` (pgvector search, re-ranking, grounding)
- **Generation concerns** → `generation_nodes.py` (LLM calls, hallucination checks)
- **Formatting concerns** → `formatting_nodes.py` (layout, toggles, enrichments)
- **Logging concerns** → `logging_nodes.py` (analytics, followups, memory)
- **New concern** → Create new module following <200 line principle
- **Backward compatibility** → Add aliases in `core_nodes.py` if needed for tests

### Adding Enterprise Content Blocks
1. Add new function to `src/flows/content_blocks.py`:
   ```python
   def my_content_block() -> str:
       return "- Bullet point content\n- More content"
   ```
2. Reference in `conversation_nodes.py` via `content_blocks.my_content_block()`
3. Trigger via action in `plan_actions()` node

### Adding New Action Types
1. Add action handler to `src/flows/node_logic/action_execution.py`:
   ```python
   def execute_my_action(self, state: ConversationState, action: Dict[str, Any]) -> None:
       # Perform side effect
       pass
   ```
2. Wire in `ActionExecutor.execute()` method
3. Plan action in `src/flows/node_logic/action_planning.py`'s `plan_actions()` node

### Updating Knowledge Base
```bash
# Edit CSV directly
vim data/career_kb.csv

# Re-run migration (deletes old data)
python scripts/migrate_data_to_supabase.py --force
```

Embeddings regenerated on each migration (idempotent by content hash).

### Debugging Retrieval Issues
```bash
# Check pgvector connection
python -c "from src.retrieval.pgvector_retriever import get_retriever; print(get_retriever())"

# Test specific query
python -c "from src.core.rag_engine import RagEngine; print(RagEngine().retrieve('Python experience'))"

# View LangSmith traces
open https://smith.langchain.com/  # Requires LANGSMITH_API_KEY
```

## Anti-Patterns to Avoid

1. **Don't bypass ConversationState**: Directly calling `rag_engine.generate_response()` skips analytics and action planning
2. **Don't use local file storage**: Resume URLs, images → Supabase Storage (`storage_service.py`)
3. **Don't hardcode OpenAI models**: Use `supabase_settings.openai_model` for A/B testing
4. **Don't add untraced LLM calls**: Wrap with `@trace_generation` decorator from `src/observability/`
5. **Don't assume sync execution**: Vercel functions have 10s timeout—use batch operations for bulk data

## Troubleshooting Vercel Deployment

### API Key Issues

**Problem**: `httpcore.LocalProtocolError: Illegal header value` or `APIConnectionError: Connection error`

**Root cause**: Environment variables contain trailing newlines or whitespace.

**Fix**:
```bash
# In Vercel dashboard → Settings → Environment Variables
# Ensure NO newlines after API keys

# Test locally first
python -c "import os; key=os.getenv('OPENAI_API_KEY'); print(f'Length: {len(key)}, Has newline: {repr(key[-5:])}')"

# Should show: Length: 164, Has newline: '...abc'
# NOT: Length: 165, Has newline: '...abc\n'
```

**Prevention**: Add stripping logic in `supabase_config.py`:
```python
self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
self.supabase_config.service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
```

### Empty Embedding Warnings

**Problem**: `WARNING: Empty embedding, returning no results`

**Root cause**: OpenAI client fails before retry logic completes (network timeout or malformed auth header).

**Debug**:
```bash
# Check Vercel function logs for full traceback
vercel logs --follow

# Test OpenAI connection in isolation
python -c "from openai import OpenAI; client = OpenAI(); print(client.embeddings.create(input='test', model='text-embedding-3-small'))"
```

### Analytics Logging Failures

**Problem**: `Failed to log interaction: Invalid non-printable ASCII character in URL`

**Root cause**: Supabase URL or analytics payload contains unescaped characters.

**Fix**: Validate environment variables in `supabase_settings.validate_supabase()`:
```python
if '\n' in self.supabase_config.url or '\n' in self.supabase_config.service_role_key:
    raise ValueError("Supabase credentials contain newlines - check environment variables")
```

### Cold Start Performance

**Issue**: First request takes 5-6 seconds (seen in logs: 5511ms duration).

**Optimization strategies**:
1. **Lazy load heavy imports**: Import LangChain only when needed in functions
2. **Connection pooling**: Reuse Supabase client across invocations (use module-level singleton)
3. **Reduce bundle size**: Remove unused dependencies from `requirements.txt`
4. **Provisioned concurrency**: Enable in Vercel Pro for pre-warmed instances

### Service Initialization Failures

**Problem**: `/api/email`, `/api/feedback`, `/api/confess` return `FUNCTION_INVOCATION_FAILED`

**Common causes**:
1. **Missing optional API keys**: Resend, Twilio services fail to initialize
2. **Import errors**: Service dependencies not in `requirements.txt`
3. **Supabase client issues**: Table schema mismatches

**Debug**:
```bash
# Check Vercel logs for specific error
vercel logs --follow

# Test service initialization locally
python -c "from src.services.resend_service import get_resend_service; print(get_resend_service())"
python -c "from src.services.twilio_service import get_twilio_service; print(get_twilio_service())"
```

**Fix**: Ensure services handle missing credentials gracefully (degraded mode):
```python
def get_resend_service():
    """Factory returns None if API key missing, doesn't crash."""
    if not os.getenv("RESEND_API_KEY"):
        logger.warning("Resend API key not set, email disabled")
        return None
    return ResendService()
```

Add `try/except` in API handlers:
```python
resend = get_resend_service()
if resend:
    resend.send_email(...)
else:
    return {"success": False, "error": "Email service unavailable"}
```

## Key Files Reference

| Path | Purpose |
|------|---------|
| `src/main.py` | Streamlit entry point, role selection UI |
| `api/chat.py` | Vercel serverless endpoint for chat |
| `src/flows/conversation_flow.py` | LangGraph pipeline orchestrator |
| `src/flows/conversation_nodes.py` | Central export hub for all nodes (re-exports from node_logic/) |
| `src/flows/node_logic/` | Package containing all node implementations (19 modules, <200 lines each) |
| `src/flows/node_logic/retrieval_nodes.py` | Retrieval pipeline: pgvector search, re-ranking, grounding (273 lines) |
| `src/flows/node_logic/generation_nodes.py` | Generation pipeline: LLM call, hallucination check (304 lines) |
| `src/flows/node_logic/formatting_nodes.py` | Formatting pipeline: structured layout, toggles, enrichments (435 lines) |
| `src/flows/node_logic/logging_nodes.py` | Logging pipeline: analytics persistence, followups, memory (193 lines) |
| `src/flows/node_logic/core_nodes.py` | Backward-compatible aliases (re-exports from split modules) (65 lines) |
| `src/flows/content_blocks.py` | Reusable enterprise messaging blocks (127 lines) |
| `src/flows/data_reporting.py` | Analytics display with markdown tables (172 lines) |
| `src/core/rag_engine.py` | RAG logic, pgvector retrieval, LLM generation |
| `src/agents/role_router.py` | Role-based query routing (legacy, migrating to nodes) |
| `src/config/supabase_config.py` | All environment config, settings singleton |
| `src/retrieval/pgvector_retriever.py` | Supabase vector search wrapper |
| `scripts/migrate_data_to_supabase.py` | One-time data migration script |

## Additional Resources

- **Master docs (authoritative)**: `docs/context/` - PROJECT_REFERENCE_OVERVIEW, SYSTEM_ARCHITECTURE_SUMMARY, DATA_COLLECTION_AND_SCHEMA_REFERENCE
- **Role behavior specs**: `ROLE_FEATURES.md` and `ROLE_FUNCTIONALITY_CHECKLIST.md`
- **Supabase schema**: `supabase/migrations/*.sql`
- **API contracts**: `api/README.md`
- **Supplementary guides**: `docs/GLOSSARY.md`, `docs/EXTERNAL_SERVICES.md`, `docs/OBSERVABILITY.md`, `docs/LANGSMITH_TRACING_SETUP.md`
- **Legacy docs**: `docs/archive/legacy/` - archived for historical reference only
