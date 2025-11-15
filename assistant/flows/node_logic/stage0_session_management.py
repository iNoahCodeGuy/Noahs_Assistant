"""Session management helpers for the LangGraph pipeline.

This module handles the initial conversation setup in two stages:

1. initialize_conversation_state (Node 1): Sets up default state structure
   - Ensures all required fields exist with safe defaults
    Before we dive in, which of these feels closest to you?
    1️⃣ Hiring Manager (Nontechnical)
    2️⃣ Hiring Manager (Technical)
    3️⃣ Software Developer
    4️⃣ Just Looking Around
    5️⃣ Looking to Confess Crush 💌

    I can go deep on architecture and code, talk through business value and ROI, share career insights about Noah, or just hang out and be a fun demo. What would you like to explore first?

Design: Portfolia messages first, no explicit role selector menu.
Role is inferred from natural conversation in classify_role_mode (Node 3).
"""

from __future__ import annotations

from textwrap import dedent

from assistant.state.conversation_state import ConversationState
from assistant.observability.langsmith_tracer import create_custom_span


# ============================================================================
# Stage 1: State Initialization
# ============================================================================

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
        state.setdefault("relate_to_enterprise", False)
        state.setdefault("grounding_status", "unknown")
        state.setdefault("hallucination_safe", True)
        state.setdefault("clarification_needed", False)
        state.setdefault("clarifying_question", "")

    return state


# ============================================================================
# Stage 2: First Message & Role Selection
# ============================================================================

_INITIAL_GREETING = dedent(
    """\
    👋 Hey! I'm Portfolia — Noah's AI Assistant,

    Before we dive in, what best describes you?
    1️⃣ Hiring Manager (Nontechnical)
    2️⃣ Hiring Manager (Technical)
    3️⃣ Software Developer
    4️⃣ Just Looking Around
    5️⃣ Looking to Confess Crush 💌
    """
)


def prompt_for_role_selection(state: ConversationState) -> ConversationState:
    """Show initial greeting if no role is set yet; infer role from first response.

    Flow:
        1. If role exists → Skip (already classified)
        2. First turn, no greeting shown → Show _INITIAL_GREETING, halt pipeline
        3. User responds → Clear halt flag, continue to classify_role_mode

    This ensures Portfolia always messages first with a conversational greeting,
    then the user's natural response is analyzed for role inference.
    """
    persona_hints = state.setdefault("session_memory", {}).setdefault("persona_hints", {})

    # Case 1: Role already set (skip greeting)
    if state.get("role"):
        persona_hints["initial_greeting_shown"] = True
        return state

    # Case 2: First turn (show greeting, halt pipeline)
    if not persona_hints.get("initial_greeting_shown"):
        persona_hints["initial_greeting_shown"] = True
        state["answer"] = _INITIAL_GREETING
        state["pipeline_halt"] = True  # Wait for user response
        state["is_greeting"] = True
        return state

    # Case 3: User responded (clear halt, continue to role classification)
    state.pop("pipeline_halt", None)
    if state.get("is_greeting"):
        state["is_greeting"] = False
    return state
