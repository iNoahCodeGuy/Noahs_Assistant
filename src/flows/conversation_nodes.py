"""Conversation nodes orchestrator - imports modular functions.

This module acts as the central import point for all conversation pipeline nodes.
All node logic has been extracted into focused modules for maintainability:

- query_classification.py: Intent detection and routing
- core_nodes.py: Retrieval, generation, and logging
- action_planning.py: Role-based action generation
- action_execution.py: Side effects (email, SMS, storage)
- code_validation.py: Sanitization and validation utilities
- greetings.py: Role-specific welcome messages

Each module is <200 lines and handles a single responsibility.
This file simply re-exports the functions so callers can import from one place.

See docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md for the full conversation flow diagram.
See docs/CONVERSATION_PIPELINE_MODULES.md for implementation details.
"""

from __future__ import annotations

# Import all conversation nodes from the node_logic package
from src.flows.node_logic.session_management import initialize_conversation_state
from src.flows.node_logic.role_routing import classify_role_mode
from src.flows.node_logic.query_classification import classify_intent, classify_query
from src.flows.node_logic.entity_extraction import extract_entities
from src.flows.node_logic.clarification import assess_clarification_need, ask_clarifying_question
from src.flows.node_logic.query_composition import compose_query
from src.flows.node_logic.presentation_control import depth_controller, display_controller
from src.flows.node_logic.core_nodes import (
    retrieve_chunks,
    re_rank_and_dedup,
    validate_grounding,
    handle_grounding_gap,
    generate_draft,
    generate_answer,
    hallucination_check,
    format_answer,
    apply_role_context,
    log_and_notify,
    suggest_followups,
    update_memory,
)
from src.flows.node_logic.action_planning import plan_actions
from src.flows.node_logic.action_execution import execute_actions
from src.flows.node_logic.code_validation import (
    is_valid_code_snippet,
    sanitize_generated_answer
)
from src.flows.node_logic.greetings import get_role_greeting, should_show_greeting, is_first_turn
from src.flows.node_logic.resume_distribution import (
    detect_hiring_signals,
    handle_resume_request,
    should_add_availability_mention,
    extract_email_from_query,
    extract_name_from_query,
    should_gather_job_details,
    get_job_details_prompt,
    extract_job_details_from_query
)


def handle_greeting(state, rag_engine):
    """Check if this is a first-turn greeting and respond appropriately.

    If the user's first query is a simple greeting (hi/hello/hey), we respond
    with a warm, role-specific introduction per CONVERSATION_PERSONALITY.md.

    Design Principles:
    - Defensibility (#6): Uses .get() for optional chat_history field
    - Performance: Short-circuits RAG pipeline for greetings (no LLM call)
    - Clarity: Single responsibility (greeting detection only)

    Args:
        state: ConversationState with query and role
        rag_engine: RAG engine (not used for greetings, but part of node signature)

    Returns:
        Updated state with greeting as answer, or unchanged if not a greeting

    Performance:
        - Greeting response: <50ms (no LLM calls)
        - Non-greeting: Pass through to next node (~0ms overhead)
    """
    # Defensive access: chat_history and query may be optional in test scenarios
    query = state.get("query", "")
    if query and should_show_greeting(query, state.get("chat_history", [])):
        greeting = get_role_greeting(state.get("role", ""))
        state["answer"] = greeting
        state["is_greeting"] = True
    return state


# Export all nodes for use in conversation_flow.py
__all__ = [
    "initialize_conversation_state",
    "classify_role_mode",
    "classify_intent",
    "classify_query",  # backward-compatible alias
    "depth_controller",
    "display_controller",
    "extract_entities",
    "assess_clarification_need",
    "ask_clarifying_question",
    "compose_query",
    "retrieve_chunks",
    "re_rank_and_dedup",
    "validate_grounding",
    "handle_grounding_gap",
    "generate_draft",
    "generate_answer",  # backward-compatible alias
    "hallucination_check",
    "format_answer",
    "apply_role_context",  # backward-compatible alias
    "plan_actions",
    "execute_actions",
    "log_and_notify",
    "suggest_followups",
    "update_memory",
    "handle_greeting",
    "get_role_greeting",
    "is_first_turn",
    "should_show_greeting",
    "is_valid_code_snippet",
    "sanitize_generated_answer",
    # Resume distribution nodes (NEW - Intelligent Resume Distribution System)
    "detect_hiring_signals",
    "handle_resume_request",
    "should_add_availability_mention",
    "extract_email_from_query",
    "extract_name_from_query",
    "should_gather_job_details",
    "get_job_details_prompt",
    "extract_job_details_from_query",
]
