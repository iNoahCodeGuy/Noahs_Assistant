"""Session management helpers for the LangGraph pipeline.

This module centralizes default state preparation and onboarding steps
before any other conversation nodes execute. The helpers keep the
pipeline defensive by ensuring required collections exist, perf metrics
reset every turn, and the assistant can guide users through persona
selection when the UI does not provide it up front.
"""

from __future__ import annotations

from textwrap import dedent

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


_INITIAL_GREETING = dedent(
    """\
    Hello! I'm Noah's AI assistant, Portfolia.

    I'm here to help you understand Noah's work, skills, and experience—or just to chat!
    What brings you here today?
    """
)


def prompt_for_role_selection(state: ConversationState) -> ConversationState:
    """Show initial greeting if no role is set yet; infer role from first response.

    This node runs before greeting handling. On the very first turn (no role set),
    Portfolia introduces herself and asks what brings the user here. The response
    is analyzed in classify_role_mode to infer the appropriate persona.
    """
    persona_hints = state.setdefault("session_memory", {}).setdefault("persona_hints", {})

    # If role already set, move on
    if state.get("role"):
        persona_hints["initial_greeting_shown"] = True
        return state

    # First turn: show initial greeting and wait for user response
    if not persona_hints.get("initial_greeting_shown"):
        persona_hints["initial_greeting_shown"] = True
        state["answer"] = _INITIAL_GREETING
        state["pipeline_halt"] = True
        state["is_greeting"] = True
        return state

    # After first response, let role classification infer the persona
    return state
