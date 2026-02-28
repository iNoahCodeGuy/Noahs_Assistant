# Conversation Pipeline - Module Structure

## 📖 For Junior Developers

The conversation pipeline used to be a single 624-line file (`conversation_nodes.py`). We've broken it into **5 focused modules** to make the code easier to understand and maintain.

## 🗂️ Module Breakdown

### 1. `conversation_nodes.py` (47 lines)
**Purpose**: Central orchestrator - imports and re-exports all pipeline functions

**What it does**: Acts as a convenience wrapper so existing code doesn't break. Just imports functions from the specialized modules below.

**When to edit**: Rarely. Only when adding entirely new pipeline stages.

---

### 2. `query_classification.py` (113 lines)
**Purpose**: Detect user intent from their query

**Key function**: `classify_query(state) -> state`

**What it does**: 
- Detects if the query is technical, career-focused, data display, MMA, or fun
- Sets `query_type` and `data_display_requested` in state
- First node in the pipeline

**Example queries it handles**:
- "Show me your Python code" → technical
- "What did you do at IBM?" → career
- "Display analytics data" → data
- "Tell me about your UFC fight" → mma

**When to edit**: When adding new query types or detection patterns

---

### 3. `code_validation.py` (111 lines)
**Purpose**: Defensive utilities to validate and sanitize content

**Key functions**: 
- `is_valid_code_snippet(code) -> bool`: Checks if retrieved code is real Python (not metadata)
- `sanitize_generated_answer(answer) -> str`: Strips SQL artifacts that leak from retrieval

**What it does**: 
- Prevents malformed code snippets from showing to users
- Cleans up LLM output before displaying
- Fixes the "display data bug" (SQL tokens appearing in responses)

**When to edit**: When you discover new patterns of malformed output that need filtering

---

### 4. `action_planning.py` (152 lines)
**Purpose**: Build a "shopping list" of actions based on query type and role

**Key function**: `plan_actions(state) -> state`

**What it does**: 
- Looks at query type, role, turn count, and conversation history
- Builds a list of actions like "include_code_snippets", "offer_resume_prompt", "send_linkedin"
- Different roles get different action plans (developers get code, hiring managers get enterprise content)

**Example action plans**:
- Developer + technical query → ["include_code_snippets", "explain_architecture"]
- Hiring manager + career query → ["offer_resume_prompt", "include_purpose_overview"]
- Casual visitor + "just looking" → ["share_fun_facts", "share_mma_link"]

**When to edit**: When adding new actions or changing role-specific behavior

---

### 5. `core_nodes.py` (392 lines)
**Purpose**: The main pipeline stages (retrieve → generate → enrich → log)

**Key functions**:
- `retrieve_chunks(state, rag_engine) -> state`: Fetch relevant knowledge base content
- `generate_answer(state, rag_engine) -> state`: Create LLM response with retrieved context
- `apply_role_context(state, rag_engine) -> state`: Add role-specific content blocks (code, data, links)
- `log_and_notify(state, session_id, latency_ms) -> state`: Save analytics to database

**What it does**: 
- `retrieve_chunks`: Calls the RAG engine to get top 4 relevant chunks from vector DB
- `generate_answer`: Feeds chunks to LLM, gets conversational response, sanitizes output
- `apply_role_context`: Reads pending_actions list and appends content blocks (code, data tables, resume links)
- `log_and_notify`: Saves conversation to Supabase for analytics

**When to edit**: 
- `retrieve_chunks`: When changing retrieval strategy or top_k value
- `generate_answer`: When tweaking LLM prompts or adding special cases
- `apply_role_context`: When adding new content blocks or changing enrichment logic
- `log_and_notify`: When adding new analytics fields

---

## 🔄 Full Pipeline Flow

```
User query 
    ↓
1. classify_query (query_classification.py)
    → Detects intent, sets query_type
    ↓
2. retrieve_chunks (core_nodes.py)
    → Fetches relevant KB content
    ↓
3. generate_answer (core_nodes.py)
    → LLM creates response, sanitizes output
    ↓
4. plan_actions (action_planning.py)
    → Builds action shopping list based on role
    ↓
5. apply_role_context (core_nodes.py)
    → Adds code, data tables, links per actions
    ↓
6. execute_actions (action_execution.py - not shown here)
    → Performs side effects (email, SMS, storage)
    ↓
7. log_and_notify (core_nodes.py)
    → Saves to analytics DB
    ↓
Final answer returned to user
```

## 📝 How to Add a New Feature

### Example: Add "show me a diagram" support

1. **Update query_classification.py**:
   ```python
   if "diagram" in query_lower or "architecture" in query_lower:
       state.stash("query_type", "diagram")
   ```

2. **Update action_planning.py**:
   ```python
   if query_type == "diagram":
       actions.append({"type": "include_architecture_diagram"})
   ```

3. **Update core_nodes.py** (in `apply_role_context`):
   ```python
   if "include_architecture_diagram" in actions:
       diagram_url = "https://example.com/architecture.png"
       components.append(f"\n\n![Architecture]({diagram_url})")
   ```

4. **Test it**:
   ```bash
   pytest tests/test_diagram_display.py -v
   ```

## 🎯 Design Principles

- **Single Responsibility**: Each module does one thing well
- **Immutable State**: State updates via `.stash()`, `.set_answer()`, not direct mutation
- **Defensive Coding**: Validate inputs, handle errors gracefully
- **Junior-Friendly**: Docstrings explain "what and why", not just "how"

## 📚 Further Reading

- **Full architecture**: `docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md`
- **Data flow**: `docs/context/DATA_COLLECTION_AND_SCHEMA_REFERENCE.md`
- **Role behavior**: `ROLE_FEATURES.md` and `ROLE_FUNCTIONALITY_CHECKLIST.md`
