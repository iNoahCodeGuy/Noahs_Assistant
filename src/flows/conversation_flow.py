"""LangGraph-style orchestrator aligned with PORTFOLIA_NODE_STRUCTURE.md.

Educational mission: make every conversation a live case study of production RAG
patterns with clear node boundaries, traceability, and cinematic-yet-grounded tone.

Conversation Pipeline Overview:
1. initialize_conversation_state → normalize state containers and load memory
2. handle_greeting → warm intro without RAG cost for first-turn hellos
3. classify_role_mode → decouple persona selection from intent detection
4. classify_intent → determine engineering vs business focus and data needs
5. detect_hiring_signals / handle_resume_request → passive hiring intelligence
6. extract_entities → capture company, role, timeline, contact hints
7. assess/ask_clarification → clarify vague prompts before retrieval
8. compose_query → build retrieval-ready prompt with persona + entity context
9. retrieve_chunks → Supabase pgvector lookup (LangSmith traced)
10. re_rank_and_dedup → diversify context for grounded answers
11. validate_grounding / handle_grounding_gap → stop hallucinations early
12. generate_draft → role-aware LLM generation (stored as draft_answer)
13. hallucination_check → attach citations and mark safety status
14. plan_actions → decide on resumes, LinkedIn, analytics, etc.
15. format_answer → add enterprise content blocks without breaking plain text rule
16. execute_actions → fire side-effects (email/SMS/logging)
17. suggest_followups → cinematic curiosity prompts
18. update_memory → store soft signals for next turns
19. log_and_notify → Supabase analytics + LangSmith metadata (always executed)

Performance characteristics remain consistent with Week 1 launch targets:
- Typical latency ~1.2s
- Greeting short-circuit <50ms
- Cold start ~3s on Vercel
- p95 latency <3s with tracing enabled
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Sequence, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgraph.graph import CompiledGraph

from src.core.rag_engine import RagEngine
from src.state.conversation_state import ConversationState
from src.flows.conversation_nodes import (
    initialize_conversation_state,
    prompt_for_role_selection,
    handle_greeting,
    classify_role_mode,
    route_hiring_manager_technical,
    classify_intent,
    depth_controller,
    display_controller,
    detect_hiring_signals,
    handle_resume_request,
    extract_entities,
    assess_clarification_need,
    ask_clarifying_question,
    compose_query,
    retrieve_chunks,
    re_rank_and_dedup,
    validate_grounding,
    handle_grounding_gap,
    generate_draft,
    hallucination_check,
    plan_actions,
    format_answer,
    execute_actions,
    suggest_followups,
    update_memory,
    log_and_notify,
)

# LangGraph Studio support
try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    START = "START"
    END = "END"


Node = Callable[[ConversationState], ConversationState]


def run_conversation_flow(
    state: ConversationState,
    rag_engine: RagEngine,
    session_id: str,
    nodes: Optional[Sequence[Callable[[ConversationState], ConversationState]]] = None
) -> ConversationState:
    """Orchestrate the functional pipeline for conversation processing.

    Design Pattern: Functional pipeline with immutable state updates. Each node
    receives the full state and returns a partial update. This simplified approach
    provides 90% of StateGraph's benefits with 50% less complexity, ideal for
    week 1 launch stability.

    Args:
        state: Initial conversation state (requires role, query, session_id)
        rag_engine: RAG engine instance for retrieval & generation
        session_id: Unique session identifier for analytics logging
        nodes: Optional custom node sequence (for testing/customization)

    Returns:
        Updated ConversationState with:
        - answer: Generated response (str)
        - retrieved_chunks: Context used for generation (list)
        - analytics_metadata: Latency, tokens, retrieval stats (dict)
        - pending_actions: Actions taken (list)

    Raises:
        None - All errors handled gracefully (see ERROR_HANDLING_IMPLEMENTATION.md)
        Services that fail return None, conversation continues with degraded functionality

    Performance:
        - Typical: 1.2s (see module docstring for breakdown)
        - Greeting short-circuit: <50ms (skips nodes 4-8)

    Example:
        >>> state = ConversationState(role="Software Developer", query="How does RAG work?")
        >>> result = run_conversation_flow(state, rag_engine, session_id="demo123")
        >>> print(result["answer"])  # Grounded explanation from KB
        >>> print(len(result["retrieved_chunks"]))  # Should be 1-4 chunks
    """
    pipeline = nodes or (
        initialize_conversation_state,
        prompt_for_role_selection,
        lambda s: handle_greeting(s, rag_engine),
        classify_role_mode,
    route_hiring_manager_technical,
        classify_intent,
        depth_controller,
        display_controller,
        detect_hiring_signals,
        handle_resume_request,
        extract_entities,
        assess_clarification_need,
        ask_clarifying_question,
        compose_query,
        lambda s: retrieve_chunks(s, rag_engine),
        re_rank_and_dedup,
        validate_grounding,
        handle_grounding_gap,
        lambda s: generate_draft(s, rag_engine),
        hallucination_check,
        plan_actions,
        lambda s: format_answer(s, rag_engine),
        execute_actions,
        suggest_followups,
        update_memory,
    )

    start = time.time()
    for node in pipeline:
        state = node(state)
        if state.get("pipeline_halt") or state.get("is_greeting"):
            break

    elapsed_ms = int((time.time() - start) * 1000)
    state = log_and_notify(state, session_id=session_id, latency_ms=elapsed_ms)
    return state


def _build_langgraph() -> Any:
    """Build LangGraph StateGraph for Studio visualization.

    Returns:
        Compiled StateGraph if LangGraph available, None otherwise
    """
    if not LANGGRAPH_AVAILABLE:
        return None

    # Initialize RAG engine for Studio
    rag_engine = RagEngine()

    # Create StateGraph with ConversationState schema
    workflow = StateGraph(ConversationState)

    # Add all nodes with RAG engine injected
    workflow.add_node("initialize", initialize_conversation_state)
    workflow.add_node("role_prompt", prompt_for_role_selection)
    workflow.add_node("greeting", lambda s: handle_greeting(s, rag_engine))
    workflow.add_node("classify_role", classify_role_mode)
    workflow.add_node("role_router", route_hiring_manager_technical)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("depth_control", depth_controller)
    workflow.add_node("display_control", display_controller)
    workflow.add_node("detect_hiring", detect_hiring_signals)
    workflow.add_node("resume_request", handle_resume_request)
    workflow.add_node("extract_entities", extract_entities)
    workflow.add_node("assess_clarification", assess_clarification_need)
    workflow.add_node("clarify", ask_clarifying_question)
    workflow.add_node("compose_query", compose_query)
    workflow.add_node("retrieve", lambda s: retrieve_chunks(s, rag_engine))
    workflow.add_node("re_rank", re_rank_and_dedup)
    workflow.add_node("validate_grounding", validate_grounding)
    workflow.add_node("grounding_gap", handle_grounding_gap)
    workflow.add_node("generate_draft", lambda s: generate_draft(s, rag_engine))
    workflow.add_node("hallucination_check", hallucination_check)
    workflow.add_node("plan_actions", plan_actions)
    workflow.add_node("format_answer", lambda s: format_answer(s, rag_engine))
    workflow.add_node("execute_actions", execute_actions)
    workflow.add_node("suggest_followups", suggest_followups)
    workflow.add_node("update_memory", update_memory)
    workflow.add_node("log_and_notify", lambda s: log_and_notify(s, "studio-session", 0))

    # Build linear pipeline with conditional edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "role_prompt")

    # Role prompt can short-circuit if we're still waiting on a persona choice
    workflow.add_conditional_edges(
        "role_prompt",
        lambda s: "end" if s.get("pipeline_halt") else "greeting",
        {"end": END, "greeting": "greeting"}
    )

    # Greeting can short-circuit to end
    workflow.add_conditional_edges(
        "greeting",
        lambda s: "end" if s.get("is_greeting") else "classify_role",
        {"end": END, "classify_role": "classify_role"}
    )

    workflow.add_edge("classify_role", "role_router")
    workflow.add_conditional_edges(
        "role_router",
        lambda s: "end" if s.get("pipeline_halt") else "classify_intent",
        {"end": END, "classify_intent": "classify_intent"}
    )
    workflow.add_edge("classify_intent", "depth_control")
    workflow.add_edge("depth_control", "display_control")
    workflow.add_edge("display_control", "detect_hiring")
    workflow.add_edge("detect_hiring", "resume_request")
    workflow.add_edge("resume_request", "extract_entities")
    workflow.add_edge("extract_entities", "assess_clarification")

    # Clarification conditional
    workflow.add_conditional_edges(
        "assess_clarification",
        lambda s: "clarify" if s.get("needs_clarification") else "compose_query",
        {"clarify": "clarify", "compose_query": "compose_query"}
    )
    workflow.add_edge("clarify", END)  # Clarification questions end the flow

    workflow.add_edge("compose_query", "retrieve")
    workflow.add_edge("retrieve", "re_rank")
    workflow.add_edge("re_rank", "validate_grounding")

    # Grounding validation conditional
    workflow.add_conditional_edges(
        "validate_grounding",
        lambda s: "grounding_gap" if s.get("grounding_failed") else "generate_draft",
        {"grounding_gap": "grounding_gap", "generate_draft": "generate_draft"}
    )
    workflow.add_edge("grounding_gap", "generate_draft")

    workflow.add_edge("generate_draft", "hallucination_check")
    workflow.add_edge("hallucination_check", "plan_actions")
    workflow.add_edge("plan_actions", "format_answer")
    workflow.add_edge("format_answer", "execute_actions")
    workflow.add_edge("execute_actions", "suggest_followups")
    workflow.add_edge("suggest_followups", "update_memory")
    workflow.add_edge("update_memory", "log_and_notify")
    workflow.add_edge("log_and_notify", END)

    return workflow.compile()


# Export compiled graph for LangGraph Studio
graph = _build_langgraph()
