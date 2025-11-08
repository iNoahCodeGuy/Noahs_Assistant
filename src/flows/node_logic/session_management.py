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
        state.setdefault("relate_to_enterprise", False)
        state.setdefault("grounding_status", "unknown")
        state.setdefault("hallucination_safe", True)
        state.setdefault("clarification_needed", False)
        state.setdefault("clarifying_question", "")

    return state


_INITIAL_GREETING = dedent(
    """\
    👋 Hey! I'm Portfolia — Noah's AI Assistant, and I'm genuinely excited you're here.

    I'm a full-stack generative AI application built to help people understand how production AI systems actually work. Think of me as both a working demo and a teaching tool — every conversation shows you real RAG architecture, vector search, LLM orchestration, and enterprise-grade patterns in action.

    I can walk you through:
    • The engineering side (architecture, code, data pipelines)
    • The business value (ROI, team efficiency, enterprise adoption)
    • Career insights about Noah and full-stack AI development
    • Or just have a conversation and see where it goes!

    Before we dive in, what best describes you?
    1️⃣ Hiring Manager (Nontechnical)
    2️⃣ Hiring Manager (Technical)
    3️⃣ Software Developer
    4️⃣ Just Looking Around
    5️⃣ Looking to Confess Crush 💌

    What sounds most interesting to you?
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
    # Clear any greeting guards so the pipeline can continue normally
    state.pop("pipeline_halt", None)
    if state.get("is_greeting"):
        state["is_greeting"] = False
    return state
