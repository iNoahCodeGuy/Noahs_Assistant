"""LangGraph-style orchestrator with 22-node pipeline including quality validation and phase tracking.

Educational mission: make every conversation a live case study of production RAG
patterns with clear node boundaries, traceability, and cinematic-yet-grounded tone.

Enhanced Pipeline (18→22 nodes with quality assurance + phase tracking):
1. initialize_conversation_state → normalize state containers and load memory
2. handle_greeting → warm intro without RAG cost for first-turn hellos
3. classify_role_mode → role detection + technical HM onboarding routing
4. classify_intent → determine engineering vs business focus and data needs
5. detect_conversation_phase → determine phase: discovery/exploration/synthesis/extended (NEW)
6. presentation_controller → depth level (1-3) + display toggles in one pass
7. extract_entities → capture company, role, timeline, contact hints
8. assess/ask_clarification → clarify vague prompts before retrieval
9. compose_query → build retrieval-ready prompt with persona + entity context
10. retrieve_chunks → pgvector search + MMR diversification
11. validate_retrieval_relevance → quality gate: check chunks match intent
12. validate_grounding / handle_grounding_gap → stop hallucinations early
13. generate_draft → role-aware LLM generation (stored as draft_answer)
14. validate_answer_quality → quality gate: relevance + novelty checks
15. validate_conversation_guidance → detect guidance needs (stuck, progression)
16. hallucination_check → attach citations and mark safety status
17. plan_actions → decide on actions + hiring signal detection
18. format_answer → structure answer with enrichments + followup generation
19. execute_actions → fire side-effects (email/SMS/storage)
20. update_memory → store soft signals + affinity tracking + memory pruning
21. log_and_notify → Supabase analytics + LangSmith metadata (always executed)

Quality Assurance Features:
- Answer relevance check: Ensures answer addresses query key terms
- Novelty check: Prevents repetitive answers (compares with last 4 responses)
- Conversation guidance: Detects stuck patterns, progression opportunities
- Retrieval relevance: Validates chunks match query intent
- Memory pruning: Bounded memory for indefinite conversations (100+ turns)
- Enhanced retry logic: Quality warnings trigger regeneration with specific instructions
- Guided followups: Context-aware suggestions based on conversation flow

Conversation Phase Tracking (NEW):
- discovery: Turns 1-3, user exploring/getting oriented
- exploration: Turns 4-8, user diving into specific areas
- synthesis: Turns 8+ with 4+ topics, ready for connections
- extended: Turns 15+, long-running conversation
- Phase influences guidance suggestions and followup generation

Scalability for Indefinite Conversations:
- All quality checks use sliding windows (last 4-10 items)
- Memory pruning: topics (10), entities (20), files (10), chat backup (6)
- O(1) or O(recent_history) complexity, not O(full_history)
- Supports 100+ turn conversations without memory bloat

Merged nodes (Part 1 & 2):
- route_hiring_manager_technical → classify_role_mode
- depth_controller + display_controller → presentation_controller
- re_rank_and_dedup → retrieve_chunks
- detect_hiring_signals + handle_resume_request → plan_actions
- suggest_followups → format_answer
- update_enterprise_affinity + update_technical_affinity → update_memory

Performance characteristics:
- Typical latency ~1.3s (slight increase due to quality checks)
- Quality validation adds <5ms per check (in-memory operations)
- Greeting short-circuit <50ms
- Cold start ~3s on Vercel
- p95 latency <3.2s with tracing enabled
- Recursion depth: 22 nodes (quality gates are lightweight)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Callable, Optional, Sequence, TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from langgraph.graph import CompiledGraph

from assistant.core.rag_engine import RagEngine
from assistant.state.conversation_state import ConversationState
from assistant.flows.conversation_nodes import (
    initialize_conversation_state,
    prompt_for_role_selection,
    handle_greeting,
    classify_message_intent,  # NEW: Intent router before RAG
    handle_non_knowledge_intent,  # NEW: Handler for non-RAG intents
    classify_role_mode,
    detect_repeated_query,  # Detect user asking same question twice
    classify_intent,
    presentation_controller,  # Merged depth_controller + display_controller
    extract_entities,
    assess_clarification_need,
    ask_clarifying_question,
    compose_query,
    retrieve_chunks,  # Now includes re_rank_and_dedup logic
    validate_grounding,
    handle_grounding_gap,
    generate_draft,
    hallucination_check,
    plan_actions,  # Now includes hiring detection logic
    format_answer,  # Now includes followup generation
    execute_actions,
    update_memory,  # Now includes affinity tracking
    log_and_notify,
)

# Import quality validation nodes
from assistant.flows.node_logic.stage5_quality_validation import (
    validate_answer_quality,
    validate_conversation_guidance,
    detect_conversation_phase,  # NEW: Phase detection for conversation progression
)
from assistant.flows.node_logic.stage4_retrieval_nodes import (
    validate_retrieval_relevance,
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


# ── Discovery-question post-hook ──────────────────────────────────────────
# Appends a discovery question (or curiosity hook) to early-phase responses
# that don't already end with one.  Works for both hardcoded pipeline_halt
# responses and LLM-generated answers.
_SKIP_INTENTS = frozenset({
    "crush_confession", "greeting", "contact_info_submission",
})

_CAPTURE_QUESTIONS = (
    "What brings you here?",
)

_KNOWLEDGE_HOOKS = (
    "The architecture behind this conversation is the best demo "
    "of his engineering — ask me how I work.",
    "The attrition model hit 94.75% on imbalanced data. "
    "The last 12 points are where the real work happened.",
    "Most portfolio sites are static pages. This one answers back.",
    "The retrieval system behind this conversation uses the same math "
    "that powers the attrition model.",
    "Noah's biology degree is the least obvious and most important part "
    "of the technical foundation.",
    "The MMA coaching story connects to the professional background "
    "in a way most people don't expect.",
    "The Response Time Analysis app is worth asking about if you care "
    "about how he thinks through statistical problems.",
)


# Patterns that identify menu-style questions (multi-option "or" questions).
# Reused by _maybe_append_discovery_question to detect trailing menu endings
# that _strip_menu_endings may have missed.
_MENU_QUESTION_RE = re.compile(
    r'(?:Would you like|Want|Shall I|Should I|Interested in|Curious about|Wanna|'
    r'Would you rather|What would you like|Which (?:one|topic|area))'
    r'.*?\bor\b.*?\?',
    re.IGNORECASE,
)


def _is_menu_ending(answer: str) -> bool:
    """Check if the answer's trailing question is a menu-style multi-option prompt.

    Extracts the last sentence (from last '. '/'! '/'\\n' boundary to end)
    and checks whether it matches a menu pattern (trigger word + 'or' + '?').
    Single discovery questions like "What brings you here?" return False.
    """
    # Find the last sentence boundary
    last_boundary = 0
    for m in re.finditer(r'(?:[.!?]\s+|\n)', answer):
        last_boundary = m.end()

    last_sentence = answer[last_boundary:].strip()
    if not last_sentence:
        return False

    is_menu = bool(_MENU_QUESTION_RE.search(last_sentence))
    logger.info(
        "Discovery hook _is_menu_ending check | last_sentence='%s' | is_menu=%s",
        last_sentence[:100], is_menu,
    )
    return is_menu


def _strip_trailing_menu(answer: str) -> str:
    """Remove the trailing menu question from the answer text.

    Returns the answer with the last menu-style sentence removed.
    """
    last_boundary = 0
    for m in re.finditer(r'(?:[.!?]\s+|\n)', answer):
        last_boundary = m.end()

    if last_boundary > 0:
        return answer[:last_boundary].rstrip()
    # Entire answer is the menu question — return empty so caller keeps original
    return ""


def _maybe_append_discovery_question(state: dict) -> dict:
    """Append a capture question AND knowledge hook to substantive answers that lack them.

    Every substantive response (not greetings, not crush flow) should end with:
    1. A capture/discovery question — draws out who they are or why they're here
    2. A knowledge hook — a statement that invites curiosity about an uncovered topic
    """
    logger.info(
        "ENTER _maybe_append_discovery_question: injected=%s answer_len=%d intent=%s msg=%d",
        state.get("_discovery_injected"),
        len(state.get("answer", "")),
        state.get("message_intent"),
        state.get("message_count", 0),
    )
    if state.get("_discovery_injected"):
        return state

    answer = (state.get("answer") or "").strip()
    if not answer:
        return state

    # If answer ends with "?", check whether it's a menu ending or a real question.
    # Menu endings ("Want to see X or Y?") get replaced with our two-part ending.
    # Real questions are left alone only if a knowledge hook is also present.
    if answer.endswith("?"):
        if _is_menu_ending(answer):
            stripped = _strip_trailing_menu(answer)
            if stripped:
                answer = stripped
                logger.info(
                    "Discovery hook: replaced menu ending, answer now ends: '...%s'",
                    answer[-80:],
                )
            else:
                # Entire answer was the menu question — leave it alone
                return state

    # Don't touch special flows
    if state.get("message_intent") in _SKIP_INTENTS:
        return state
    if state.get("hm_capture_step"):
        return state
    if state.get("is_greeting"):
        return state

    msg_count = state.get("message_count", 0)
    if msg_count < 1:
        return state

    # Check what's already present at the END of the answer (last 200 chars).
    # Only look at the tail to avoid false-positiving on phrases that appear
    # incidentally in the prose body (e.g., "worth a look" mid-paragraph).
    answer_tail = answer[-200:].lower()
    has_capture = any(
        frag in answer_tail for frag in [
            "what brings you", "what's your angle", "hiring, building",
            "hiring, curiosity", "hiring, exploring",
            "want to share what you",
            "want noah to reach out", "noah can follow up",
            "say the word", "just let me know",
        ]
    )
    has_hook = any(
        frag in answer_tail for frag in [
            "worth a look", "same math that powers",
            "statistical foundation", "architecture behind this",
            "attrition model", "if you want to see",
            "ask me how i work",
        ]
    )

    # If both already present, nothing to do
    logger.info(
        "DIAG _maybe_append: ends_with_q=%s has_capture=%s has_hook=%s tail=%r",
        answer.endswith("?"), has_capture, has_hook, answer_tail,
    )
    if has_capture and has_hook:
        state["_discovery_injected"] = True
        return state

    # Single capture question for all phases
    capture = _CAPTURE_QUESTIONS[0]  # "What brings you here?"

    # Pick a knowledge hook (cycle through them based on msg_count)
    hook = _KNOWLEDGE_HOOKS[msg_count % len(_KNOWLEDGE_HOOKS)]

    # Append whichever parts are missing
    suffix_parts = []
    if not has_capture:
        suffix_parts.append(capture)
    if not has_hook:
        suffix_parts.append(hook)

    if suffix_parts:
        state["answer"] = answer + "\n\n" + "\n".join(suffix_parts)
        state["_discovery_injected"] = True
        logger.info(
            "Discovery hook appended (msg_count=%d): capture=%s hook=%s",
            msg_count, not has_capture, not has_hook,
        )

    return state


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
        # ═══════════════════════════════════════════════════════════════════════════
        # STAGE 0: INITIALIZATION (Setup)
        # Purpose: Ensure all 46 state fields exist with safe defaults
        # State Modified: All containers initialized (lists→[], dicts→{}, flags→False)
        # Performance: <5ms (no I/O)
        # ═══════════════════════════════════════════════════════════════════════════
        initialize_conversation_state,

        # ═══════════════════════════════════════════════════════════════════════════
        # STAGE 1: GREETING (First Contact)
        # Purpose: Portfolia messages first, detect simple greetings
        # State Modified: answer, is_greeting, pipeline_halt, session_memory.persona_hints
        # Short-Circuit: Exits if is_greeting=True (user said "hi/hello")
        # Performance: <50ms (no LLM call)
        # ═══════════════════════════════════════════════════════════════════════════
        prompt_for_role_selection,
        lambda s: handle_greeting(s, rag_engine),

        # ═══════════════════════════════════════════════════════════════════════════
        # STAGE 1.5: INTENT ROUTING (NEW)
        # Purpose: Classify message intent before RAG (knowledge vs non-knowledge)
        # State Modified: message_intent, skip_rag, awaiting_crush_choice
        # Short-Circuit: Routes to non-RAG handler if skip_rag=True
        # Performance: ~150ms (fast LLM classification call)
        # ═══════════════════════════════════════════════════════════════════════════
        classify_message_intent,
        lambda s: handle_non_knowledge_intent(s, rag_engine),

        # ═══════════════════════════════════════════════════════════════════════════
        # STAGE 2: CLASSIFICATION (Understanding Intent)
        # Purpose: Infer role, detect intent, extract entities
        # State Modified: role, query_type, query_intent, entities, topic_focus
        # Performance: ~100ms (regex + keyword matching, no LLM)
        # ═══════════════════════════════════════════════════════════════════════════
        classify_role_mode,  # Now includes HM technical routing
        classify_intent,
        detect_conversation_phase,  # Phase tracking (discovery/exploration/synthesis/extended)
        extract_entities,

        # ═══════════════════════════════════════════════════════════════════════════
        # STAGE 3: QUERY PREPARATION (Optimize for Retrieval)
        # Purpose: Clarify vague queries, compose retrieval prompt, set presentation
        # State Modified: composed_query, depth_level, display_toggles, clarification_needed
        # Short-Circuit: Exits if clarification_needed=True (user query too vague)
        # Performance: ~50ms (no LLM, just string manipulation)
        # ═══════════════════════════════════════════════════════════════════════════
        assess_clarification_need,
        ask_clarifying_question,
        presentation_controller,  # Merged depth + display
        compose_query,

        # ═══════════════════════════════════════════════════════════════════════════
        # STAGE 4: RETRIEVAL (Find Relevant Knowledge)
        # Purpose: pgvector search, validate grounding sufficiency
        # State Modified: retrieved_chunks, retrieval_scores, grounding_status
        # Short-Circuit: Exits if grounding_status="insufficient" (low similarity scores)
        # Performance: ~300ms (pgvector RPC + MMR dedup)
        # ═══════════════════════════════════════════════════════════════════════════
        lambda s: retrieve_chunks(s, rag_engine),  # Now includes MMR dedup
        validate_grounding,
        handle_grounding_gap,

        # ═══════════════════════════════════════════════════════════════════════════
        # STAGE 5: GENERATION (Create Answer)
        # Purpose: LLM generation with hallucination checks
        # State Modified: draft_answer, answer, hallucination_safe, citations
        # Performance: ~600ms (OpenAI API call)
        # ═══════════════════════════════════════════════════════════════════════════
        lambda s: generate_draft(s, rag_engine),
        hallucination_check,

        # ═══════════════════════════════════════════════════════════════════════════
        # STAGE 6: ENRICHMENT (Enhance Answer)
        # Purpose: Plan actions, format with followups/toggles
        # State Modified: planned_actions, followup_prompts, hiring_signals
        # Performance: ~200ms (followup generation via pgvector + string formatting)
        # ═══════════════════════════════════════════════════════════════════════════
        plan_actions,  # Now includes hiring detection
        lambda s: format_answer(s, rag_engine),  # Now includes followup generation

        # ═══════════════════════════════════════════════════════════════════════════
        # STAGE 7: FINALIZATION (Side Effects & Logging)
        # Purpose: Execute actions, update memory, log analytics
        # State Modified: executed_actions, session_memory, analytics_metadata
        # Always Executed: log_and_notify runs even on short-circuit
        # Performance: ~50ms (Supabase inserts)
        # ═══════════════════════════════════════════════════════════════════════════
        execute_actions,
        update_memory,  # Now includes affinity tracking
    )

    start = time.time()
    for node in pipeline:
        result = node(state)
        # Some nodes return partial update dicts (designed for LangGraph StateGraph merging).
        # The functional pipeline needs to merge these into the full state.
        if result is not state and isinstance(result, dict):
            state.update(result)
        else:
            state = result
        # Short-circuit conditions: greeting, pipeline halt, or skip_rag (non-knowledge intent)
        if state.get("pipeline_halt") or state.get("is_greeting"):
            # #region agent log
            with open('/Users/noahdelacalzada/NoahsAIAssistant/NoahsAIAssistant-/.cursor/debug.log', 'a') as f:
                import json
                f.write(json.dumps({
                    "location": "conversation_flow.py:215",
                    "message": "Pipeline break detected",
                    "data": {
                        "pipeline_halt": state.get("pipeline_halt"),
                        "is_greeting": state.get("is_greeting"),
                        "skip_rag": state.get("skip_rag"),
                        "message_intent": state.get("message_intent"),
                        "has_answer": bool(state.get("answer")),
                        "chat_history_len": len(state.get("chat_history", []))
                    },
                    "timestamp": int(time.time() * 1000),
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A"
                }) + "\n")
            # #endregion
            break

    # ── Strip leaked source citations ──────────────────────────────────
    # hallucination_check appends "Sources: 1. entry_X; ..." to draft_answer.
    # format_answer normally strips it, but early-return paths (welcome messages,
    # action requests) or LLM-generated citations can leak through.
    answer = state.get("answer") or ""
    if answer:
        answer = re.sub(r'\n*Sources:\s*[\d].*$', '', answer, flags=re.DOTALL).rstrip()
        state["answer"] = answer

    # ── Post-loop discovery-question hook ─────────────────────────────
    # Runs after BOTH pipeline_halt (hardcoded) and full-pipeline paths
    # so discovery questions fire regardless of which path produced the answer.
    _maybe_append_discovery_question(state)

    # Sync chat_history if the discovery hook modified the answer.
    # For normal (non-halt) pipeline runs, update_memory already appended
    # the answer to chat_history BEFORE this hook ran, so the history has
    # the old version. Update the last assistant entry to match.
    if state.get("_discovery_injected") and not state.get("pipeline_halt"):
        chat_history = state.get("chat_history", [])
        new_answer = state.get("answer", "")
        if chat_history and new_answer:
            for i in range(len(chat_history) - 1, -1, -1):
                msg = chat_history[i]
                if isinstance(msg, dict) and (msg.get("role") or msg.get("type", "")) in ("assistant", "ai"):
                    msg["content"] = new_answer
                    break

    # Append user query and assistant answer to chat_history for conversation continuity
    # Skip only for actual greetings (initial greeting before role selection)
    # Role welcome messages and menu selections are part of conversation and should be preserved
    # #region agent log
    with open('/Users/noahdelacalzada/NoahsAIAssistant/NoahsAIAssistant-/.cursor/debug.log', 'a') as f:
        import json
        f.write(json.dumps({
            "location": "conversation_flow.py:221",
            "message": "Before chat_history append check",
            "data": {
                "has_answer": bool(state.get("answer")),
                "is_greeting": state.get("is_greeting"),
                "pipeline_halt": state.get("pipeline_halt"),
                "chat_history_len": len(state.get("chat_history", [])),
                "condition_passes": bool(state.get("answer") and not state.get("is_greeting"))
            },
            "timestamp": int(time.time() * 1000),
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "B"
        }) + "\n")
    # #endregion

    # For normal pipeline runs, chat_history append is handled by update_memory()
    # in stage7_logging_nodes.py. But when pipeline_halt is True (crush flow, etc.),
    # the pipeline breaks early and update_memory never runs — so we append here.
    if state.get("pipeline_halt") and state.get("answer") and not state.get("is_greeting"):
        chat_history = state.get("chat_history", [])
        if state.get("query"):
            chat_history.append({"role": "user", "content": state["query"]})
        chat_history.append({"role": "assistant", "content": state["answer"]})
        state["chat_history"] = chat_history

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

    # Add all nodes with RAG engine injected (22-node pipeline with quality validation)
    workflow.add_node("initialize", initialize_conversation_state)
    workflow.add_node("role_prompt", prompt_for_role_selection)
    workflow.add_node("greeting", lambda s: handle_greeting(s, rag_engine))
    workflow.add_node("classify_role", classify_role_mode)  # Now includes HM routing
    workflow.add_node("detect_repeated", detect_repeated_query)  # Detect user asking same question
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("detect_phase", detect_conversation_phase)  # NEW: Conversation phase detection
    workflow.add_node("presentation", presentation_controller)  # Merged depth + display
    workflow.add_node("extract_entities", extract_entities)
    workflow.add_node("assess_clarification", assess_clarification_need)
    workflow.add_node("clarify", ask_clarifying_question)
    workflow.add_node("compose_query", compose_query)
    workflow.add_node("retrieve", lambda s: retrieve_chunks(s, rag_engine))  # Now includes MMR dedup
    workflow.add_node("validate_retrieval", validate_retrieval_relevance)  # NEW: Quality gate for retrieval
    workflow.add_node("validate_grounding", validate_grounding)
    workflow.add_node("grounding_gap", handle_grounding_gap)
    workflow.add_node("generate_draft", lambda s: generate_draft(s, rag_engine))
    workflow.add_node("validate_quality", validate_answer_quality)  # NEW: Quality gate for answer
    workflow.add_node("validate_guidance", validate_conversation_guidance)  # NEW: Conversation guidance
    workflow.add_node("hallucination_check", hallucination_check)
    workflow.add_node("plan_actions", plan_actions)  # Now includes hiring detection
    workflow.add_node("format_answer", lambda s: format_answer(s, rag_engine))  # Now includes followups
    workflow.add_node("execute_actions", execute_actions)
    workflow.add_node("update_memory", update_memory)  # Now includes affinity tracking
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

    # Consolidated edges (8 nodes removed from original 26-node pipeline)
    workflow.add_conditional_edges(
        "classify_role",
        lambda s: "end" if s.get("pipeline_halt") else "detect_repeated",
        {"end": END, "detect_repeated": "detect_repeated"}
    )
    workflow.add_edge("detect_repeated", "classify_intent")  # Repeated query detection before intent
    workflow.add_edge("classify_intent", "detect_phase")  # Phase detection after intent
    workflow.add_edge("detect_phase", "presentation")
    workflow.add_edge("presentation", "extract_entities")
    workflow.add_edge("extract_entities", "assess_clarification")

    # Clarification conditional
    workflow.add_conditional_edges(
        "assess_clarification",
        lambda s: "clarify" if s.get("needs_clarification") else "compose_query",
        {"clarify": "clarify", "compose_query": "compose_query"}
    )
    workflow.add_edge("clarify", END)  # Clarification questions end the flow

    workflow.add_edge("compose_query", "retrieve")
    workflow.add_edge("retrieve", "validate_retrieval")  # NEW: Quality gate after retrieval
    workflow.add_edge("validate_retrieval", "validate_grounding")  # MMR dedup now in retrieve

    # Grounding validation conditional
    workflow.add_conditional_edges(
        "validate_grounding",
        lambda s: "grounding_gap" if s.get("grounding_failed") else "generate_draft",
        {"grounding_gap": "grounding_gap", "generate_draft": "generate_draft"}
    )
    workflow.add_edge("grounding_gap", "generate_draft")

    workflow.add_edge("generate_draft", "validate_quality")  # NEW: Quality gate after generation
    workflow.add_edge("validate_quality", "validate_guidance")  # NEW: Conversation guidance check
    workflow.add_edge("validate_guidance", "hallucination_check")
    workflow.add_edge("hallucination_check", "plan_actions")  # Hiring detection now in plan_actions
    workflow.add_edge("plan_actions", "format_answer")  # Followup generation now in format_answer
    workflow.add_edge("format_answer", "execute_actions")
    workflow.add_edge("execute_actions", "update_memory")  # Affinity tracking now in update_memory
    workflow.add_edge("update_memory", "log_and_notify")
    workflow.add_edge("log_and_notify", END)

    # Compile the workflow (recursion_limit reduced from 50→30 due to consolidation)
    try:
        return workflow.compile(recursion_limit=30)
    except TypeError:
        # Fallback for older LangGraph versions without recursion_limit parameter
        return workflow.compile()


# Export compiled graph for LangGraph Studio
graph = _build_langgraph()
