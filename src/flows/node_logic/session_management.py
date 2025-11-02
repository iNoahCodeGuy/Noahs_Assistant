"""Session initialization utilities for the LangGraph pipeline.

This module centralizes default state preparation before any other
conversation nodes execute. The helper keeps the pipeline defensive by
ensuring required collections exist and perf metrics reset every turn.
"""

from __future__ import annotations

from src.state.conversation_state import ConversationState
from src.observability.langsmith_tracer import create_custom_span


def initialize_conversation_state(state: ConversationState) -> ConversationState:
    """Populate the ConversationState with safe defaults.

    The frontend guarantees ``query``, ``role``, ``session_id`` and
    ``chat_history``, but Studio testing may provide empty state.
    Downstream nodes expect structured containers to exist,
    so this initializer normalizes the state and provides empty defaults.
    """
    with create_custom_span(
        name="initialize_state",
        inputs={"session_id": state.get("session_id"), "role": state.get("role")}
    ):
        # Core fields (required by all nodes)
        state.setdefault("query", "")
        state.setdefault("role", "")
        state.setdefault("session_id", "")
        state.setdefault("chat_history", [])

        # Structured containers
        state.setdefault("analytics_metadata", {})
        state.setdefault("pending_actions", [])
        state.setdefault("planned_actions", [])
        state.setdefault("executed_actions", [])
        state.setdefault("retrieved_chunks", [])
        state.setdefault("retrieval_scores", [])
        state.setdefault("code_snippets", [])
        state.setdefault("hiring_signals", [])
        state.setdefault("session_memory", {})
        state.setdefault("entities", {})
        state.setdefault("job_details", {})
        state.setdefault("followup_prompts", [])
        state.setdefault("topic_focus", "general")
        state.setdefault("grounding_status", "unknown")
        state.setdefault("hallucination_safe", True)
        state.setdefault("clarification_needed", False)
        state.setdefault("clarifying_question", "")

    return state
