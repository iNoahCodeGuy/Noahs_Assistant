# Conversation Pipeline - Module Structure

## 📖 For Junior Developers

The pipeline used to live in a single 600+ line module. It now spans a set of small, purpose-built files so each responsibility is easy to own, test, and extend.

## 🗂️ Module Breakdown

### 1. `conversation_nodes.py` (orchestrator)
**Purpose**: Central import/export hub so the rest of the codebase can pull nodes from one place.

**What it does**: Re-exports every node while keeping backwards-compatible aliases (e.g., `classify_query` still points at `classify_intent`). Only touch this when adding a brand-new node.

---

### 2. `session_management.py`
**Purpose**: Normalize incoming state and load any stored memory.

**Key function**: `initialize_conversation_state(state) -> ConversationState`

**What it does**: Ensures mandatory keys exist, attaches analytics metadata, and hydrates chat history prior to the rest of the pipeline.

---

### 3. `greetings.py`
**Purpose**: Handle first-turn greetings without incurring retrieval/generation cost.

**Key functions**:
- `should_show_greeting(query, chat_history)`
- `handle_greeting(state, rag_engine)` (exported through `conversation_nodes`)

**What it does**: Detects greeting phrases, injects the role-specific welcome message, and short-circuits the pipeline when appropriate.

---

### 4. `role_routing.py`
**Purpose**: Apply role-specific defaults before intent detection.

**Key function**: `classify_role_mode(state) -> state`

**What it does**: Confirms the active persona, seeds teaching defaults, and keeps persona logic separate from pure query intent.

---

### 5. `query_classification.py`
**Purpose**: Detect user intent and determine if code/data should be surfaced.

**Key function**: `classify_intent(state) -> state` (alias `classify_query` kept for legacy callers)

**What it does**: Flags teaching moments, code/data affordances, MMA easter eggs, and query risk categories. Runs after role mode is established.

---

### 6. `resume_distribution.py`
**Purpose**: Passive hiring signal tracking and explicit resume handling.

**Key functions**: `detect_hiring_signals`, `handle_resume_request`, `should_add_availability_mention`, `should_gather_job_details`

**What it does**: Monitors conversation for hiring interest, sets follow-up prompts, and wires in email/SMS triggers for resume delivery.

---

### 7. `entity_extraction.py`
**Purpose**: Capture structured entities (company, role, timeline, contact hints) for later prompts and analytics.

**Key function**: `extract_entities(state) -> state`

---

### 8. `clarification.py`
**Purpose**: Decide whether to ask for clarification before spending tokens.

**Key functions**: `assess_clarification_need`, `ask_clarifying_question`

---

### 9. `query_composition.py`
**Purpose**: Build the retrieval-ready prompt that the RAG engine uses.

**Key function**: `compose_query(state) -> state`

---

### 10. `core_nodes.py`
**Purpose**: Retrieval, generation, formatting, follow-ups, and analytics hooks.

**Key functions**:
- `retrieve_chunks(state, rag_engine)`
- `re_rank_and_dedup(state)`
- `validate_grounding(state)` / `handle_grounding_gap(state)`
- `generate_draft(state, rag_engine)`
- `hallucination_check(state)`
- `format_answer(state, rag_engine)` *(exports legacy alias `apply_role_context`)*
- `suggest_followups(state)`
- `update_memory(state)`
- `log_and_notify(state, session_id, latency_ms)`

**What it does**: Handles pgvector retrieval, drafts the answer, applies role-aware formatting (including follow-up prompts) inside `format_answer`, and finally persists analytics via `log_and_notify`.

---

### 11. `action_planning.py`
**Purpose**: Build the “shopping list” of actions the assistant should take.

**Key function**: `plan_actions(state) -> state`

**What it does**: Looks at role, query type, retrieved content, and conversation context to decide on resumes, LinkedIn links, code snippets, data tables, etc.

---

### 12. `action_execution.py`
**Purpose**: Execute side effects such as email, SMS, and storage lookups.

**Key function**: `execute_actions(state) -> state`

**What it does**: Calls the appropriate service singletons (Resend, Twilio, Supabase) and handles degraded-mode fallbacks.

---

### 13. `code_validation.py`
**Purpose**: Defensive utilities to validate and sanitize content before display.

**Key functions**: `is_valid_code_snippet`, `sanitize_generated_answer`

---

These modules are intentionally small (<200 lines) and follow the “one responsibility per file” rule so junior contributors can reason about them quickly.

## 🔄 Full Pipeline Flow

```
User query
    ↓
0. initialize_conversation_state (session_management.py)
    → Normalize state, hydrate memory, attach analytics metadata
    ↓
1. handle_greeting (greetings.py)
    → Short-circuit warm intro for first-turn "hi" messages
    ↓
2. classify_role_mode (role_routing.py)
    → Confirm persona and load role defaults
    ↓
3. classify_intent (query_classification.py)
    → Detect teaching moments, code/data affordances, easter eggs
    ↓
4. detect_hiring_signals / handle_resume_request (resume_distribution.py)
    → Track passive interest, respond to explicit resume asks
    ↓
5. extract_entities (entity_extraction.py)
    → Capture company, role, timeline, contact hints
    ↓
6. assess_clarification_need → ask_clarifying_question (clarification.py)
    → Guardrail vague prompts before retrieval spend
    ↓
7. compose_query (query_composition.py)
    → Build retrieval-ready query with persona + entity context
    ↓
8. retrieve_chunks → re_rank_and_dedup (core_nodes.py)
    → Call pgvector, diversify results
    ↓
9. validate_grounding → handle_grounding_gap (core_nodes.py)
    → Halt if similarity too low, ask for more detail
    ↓
10. generate_draft → hallucination_check (core_nodes.py)
     → Produce draft answer, attach lightweight citations
    ↓
11. plan_actions (action_planning.py)
     → Decide on resumes, code blocks, analytics, follow-ups
    ↓
12. format_answer (core_nodes.py)
    → Apply role-specific framing, inject content blocks, and seed follow-up prompts
    ↓
13. execute_actions (action_execution.py)
    → Fire side effects (email, SMS, storage) with graceful fallbacks
    ↓
14. suggest_followups → update_memory (core_nodes.py)
    → Backstop follow-up prompts and store soft signals for later turns
    ↓
15. log_and_notify (core_nodes.py)
    → Persist analytics to Supabase + LangSmith tracing
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

3. **Update core_nodes.py** (inside `format_answer`):
   ```python
   if any(a.get("type") == "include_architecture_diagram" for a in state.get("pending_actions", [])):
       diagram_url = "https://example.com/architecture.png"
       components.append(f"\n\n![Architecture Overview]({diagram_url})")
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
