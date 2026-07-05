"""Functional orchestrator for the 22-node conversation pipeline.

The runtime pipeline is a plain function loop (`state = node(state)` with
partial-dict merging), LangGraph-style in structure but with no graph engine
in the hot path. This list mirrors the `pipeline` tuple in
run_conversation_flow — if you change one, change both:

Stage 0 — initialization
 1. initialize_conversation_state → defaults + clear volatile per-turn fields

Stage 1 — first contact
 2. prompt_for_role_selection → first-turn menu prompt
 3. handle_greeting → deterministic hello handling, no LLM cost

Stage 1.5 — intent routing (classify before you retrieve)
 4. classify_message_intent → Claude Haiku intent + capture/crush FSM steps
 5. handle_non_knowledge_intent → answers greeting/small-talk/off-topic/crush

Stage 2 — understanding
 6. classify_role_mode → welcome-message routing (universal pipeline after)
 7. classify_intent → query type + topic focus (regex/keywords, no LLM)
 8. detect_conversation_phase → discovery/exploration/synthesis/extended
 9. extract_entities → company, role, timeline, contact hints

Stage 3 — query preparation
10. assess_clarification_need → flag vague queries
11. ask_clarifying_question → short-circuit with a question when too vague
12. presentation_controller → depth level + display toggles
13. compose_query → retrieval-ready prompt

Stage 4 — retrieval
14. retrieve_chunks → pgvector search (0.50 strict / 0.30 fallback) + MMR
15. validate_grounding → similarity threshold gate before generation
16. handle_grounding_gap → self-knowledge injection or graceful degradation

Stage 5 — generation
17. generate_draft → Claude Sonnet 4.5 generation (+ verbatim-copy detection)
18. hallucination_check → deterministic claim verification vs sources
    (HALLUCINATION_GATE=log|enforce|off)

Stage 6 — enrichment
19. plan_actions → queue side effects + hiring-signal detection
20. format_answer → voice enforcement, followups, link throttling

Stage 7 — finalization
21. execute_actions → fire queued side effects (Supabase/Twilio/Resend)
22. update_memory → session memory + affinity tracking + pruning

Post-loop (always runs, even on short-circuit):
    log_and_notify → Supabase analytics + latency metadata

Memory is bounded for long conversations: quality checks use sliding
windows, and pruning caps topics (10), entities (20), and chat backup (6).

Latency is dominated by the LLM calls (Haiku classification, then Sonnet
generation for knowledge queries); everything else is in-memory or a single
pgvector RPC. Real per-stage timings are visible in LangSmith traces.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Callable, Optional, Sequence

from assistant.flows.node_logic.util_discovery_hooks import (
    _maybe_append_discovery_question,
)

logger = logging.getLogger(__name__)

from assistant.core.rag_engine import RagEngine
from assistant.state.conversation_state import ConversationState
from assistant.flows.conversation_nodes import (
    initialize_conversation_state,
    prompt_for_role_selection,
    handle_greeting,
    classify_message_intent,  # NEW: Intent router before RAG
    handle_non_knowledge_intent,  # NEW: Handler for non-RAG intents
    classify_role_mode,
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

from assistant.flows.node_logic.stage5_quality_validation import (
    detect_conversation_phase,
)


Node = Callable[[ConversationState], ConversationState]



def run_conversation_flow(
    state: ConversationState,
    rag_engine: RagEngine,
    session_id: str,
    nodes: Optional[Sequence[Callable[[ConversationState], ConversationState]]] = None
) -> ConversationState:
    """Orchestrate the functional pipeline for conversation processing.

    Design Pattern: functional pipeline. Each node receives the full state
    and returns a partial update that the loop merges via state.update().

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
        None - errors are handled gracefully; services that fail return None
        and the conversation continues with degraded functionality.

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
        classify_role_mode,  # Welcome message routing only
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
        # Performance: ~600ms (Anthropic API call)
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
            break

    # ── Strip leaked source citations ──────────────────────────────────
    # Guard against the LLM emitting its own "Sources: 1. ..." trailer
    # (prompt-driven; nothing in the pipeline appends citations anymore).
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

    # For normal pipeline runs, chat_history append is handled by update_memory()
    # in stage7_logging_nodes.py. But when pipeline_halt is True (crush flow, etc.),
    # the pipeline breaks early and update_memory never runs — so we append here.
    if state.get("pipeline_halt") and state.get("answer") and not state.get("is_greeting"):
        chat_history = state.get("chat_history", [])
        if state.get("query"):
            chat_history.append({"role": "user", "content": state["query"]})
        chat_history.append({"role": "assistant", "content": state["answer"]})
        state["chat_history"] = chat_history

    # ── Universal em-dash removal (final gate) ─────────────────────────
    # Runs LAST, after all hooks and modifications, so em-dashes never
    # reach the user regardless of which branch produced the answer.
    from assistant.flows.node_logic.stage6_formatting_nodes import _strip_em_dashes
    final_answer = state.get("answer") or ""
    if final_answer:
        cleaned = _strip_em_dashes(final_answer)
        if cleaned != final_answer:
            state["answer"] = cleaned
            # Also update chat_history if the answer was already appended
            chat_history = state.get("chat_history", [])
            for i in range(len(chat_history) - 1, -1, -1):
                msg = chat_history[i]
                if isinstance(msg, dict) and (msg.get("role") or msg.get("type", "")) in ("assistant", "ai"):
                    if msg.get("content") == final_answer:
                        msg["content"] = cleaned
                    break

    elapsed_ms = int((time.time() - start) * 1000)
    state = log_and_notify(state, session_id=session_id, latency_ms=elapsed_ms)
    return state
