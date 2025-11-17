# Node Complete Reference

> Consolidated guide to LangGraph node architecture and state management.

## Quick References

- **State Logic Flow**: See `HOW_TO_FOLLOW_NODE_STATE_LOGIC.md`
- **State Fields**: See `NODE_STATE_QUICK_REFERENCE.md`
- **Migration Guide**: See `NODE_MIGRATION_GUIDE.md`
- **QA Migration**: See `QA_LANGGRAPH_MIGRATION.md`

---

## Node Pipeline Overview

18-node consolidated pipeline (see `assistant/flows/conversation_flow.py`):

### Stage 0: Initialization
- `initialize_conversation_state` - Normalize state, load memory

### Stage 1: Greeting & Role
- `handle_greeting` - Warm intro without RAG cost
- `classify_role_mode` - Role detection + routing

### Stage 2: Query Understanding
- `classify_intent` - Engineering vs business focus
- `extract_entities` - Company, role, timeline, contact hints

### Stage 3: Query Refinement
- `assess_clarification_need` - Detect vague queries
- `compose_query` - Build retrieval-ready prompt

### Stage 4: Retrieval
- `retrieve_chunks` - pgvector search + MMR
- `validate_grounding` - Stop hallucinations early

### Stage 5: Generation
- `generate_draft` - Role-aware LLM generation
- `hallucination_check` - Citations + safety

### Stage 6: Formatting
- `plan_actions` - Action decisions + hiring signals
- `format_answer` - Structure + followups

### Stage 7: Logging
- `log_and_notify` - Supabase analytics + LangSmith

---

## State Management

### Core State Fields

```python
ConversationState = TypedDict({
    # Identity
    "query": str,
    "role": str,
    "role_mode": str,
    "session_id": str,
    
    # Pipeline
    "is_greeting": bool,
    "query_type": str,
    "composed_query": str,
    
    # Retrieval
    "retrieved_chunks": List[Dict],
    "retrieval_scores": List[float],
    "grounding_status": str,
    
    # Generation
    "draft_answer": str,
    "answer": str,
    "hallucination_safe": bool,
    
    # Actions
    "planned_actions": List[Dict],
    "executed_actions": List[Dict],
    
    # Memory
    "session_memory": Dict,
    "chat_history": List[Dict],
})
```

### State Transitions

See `docs/HOW_TO_FOLLOW_NODE_STATE_LOGIC.md` for detailed flow diagrams.

---

## Migration Guides

### LangGraph Migration

See `docs/QA_LANGGRAPH_MIGRATION.md`:
- Pre-LangGraph → LangGraph conversion patterns
- Node boundary decisions
- State management patterns

### Node Refactoring

See `docs/NODE_MIGRATION_GUIDE.md`:
- Splitting large nodes
- Consolidating redundant nodes
- Performance optimization patterns

---

## Implementation Details

**Node Logic**: `assistant/flows/node_logic/` (19 modules)
**Orchestration**: `assistant/flows/conversation_flow.py`
**State Definition**: `assistant/state/conversation_state.py`

Each module < 200 lines for maintainability.
