"""Clarification nodes keep the dialogue grounded before retrieval."""

from __future__ import annotations

from typing import List

from assistant.state.conversation_state import ConversationState
from assistant.observability.langsmith_tracer import create_custom_span


def assess_clarification_need(state: ConversationState) -> ConversationState:
    """Decide whether the assistant should pause to ask for more context."""
    with create_custom_span(
        name="assess_clarification",
        inputs={
            "ambiguous": state.get("ambiguous_query", False),
            "detail_preference_needed": state.get("detail_preference_needed", False)
        }
    ):
        # PRIORITY: Menu selections are explicit - never need clarification
        if state.get("query_type") == "menu_selection" or state.get("menu_choice"):
            state["clarification_needed"] = False
            state["clarification_type"] = None
            return state

        # Check both topic ambiguity and detail preference need
        topic_ambiguous = bool(state.get("ambiguous_query"))
        detail_needed = bool(state.get("detail_preference_needed"))

        # Determine clarification type
        if topic_ambiguous and detail_needed:
            state["clarification_type"] = "both"
            state["clarification_needed"] = True
        elif topic_ambiguous:
            state["clarification_type"] = "topic"
            state["clarification_needed"] = True
        elif detail_needed:
            state["clarification_type"] = "detail"
            state["clarification_needed"] = True
        else:
            state["clarification_type"] = None
            state["clarification_needed"] = False

    return state


def ask_clarifying_question(state: ConversationState) -> ConversationState:
    """If clarification is required, craft the targeted follow-up question."""
    if not state.get("clarification_needed"):
        return state

    clarification_type = state.get("clarification_type", "topic")
    query = state.get("query", "this topic")
    topic_focus = state.get("topic_focus", "this topic")

    # Build clarifying question based on type
    if clarification_type == "detail":
        # Detail preference question only
        clarifier = (
            f"I'd love to explain {topic_focus}! Would you prefer:\n"
            f"1) Brief overview (2-3 sentences, high-level)\n"
            f"2) Moderate detail (paragraph with examples)\n"
            f"3) Comprehensive walkthrough (full explanation with code/data)"
        )
    elif clarification_type == "both":
        # Both topic and detail needed - prioritize topic first, mention detail
        options: List[str] = state.get("ambiguity_options", []) or []
        context = state.get("ambiguity_context", "Portfolia's architecture")

        options_text = ", ".join(options[:-1])
        if options:
            if len(options) > 1:
                options_text = f"{options_text}, or {options[-1]}" if options_text else options[-1]
            else:
                options_text = options[0]

        clarifier = (
            f"I am excited to go deeper on \"{query}\". I can focus on {options_text} "
            f"using my own system as the walkthrough so you get a real example. "
            f"Which part of {context} would you like first?\n\n"
            f"Also, would you prefer a brief overview, moderate detail, or comprehensive walkthrough?"
        )
    else:
        # Topic ambiguity only (default behavior)
        options: List[str] = state.get("ambiguity_options", []) or []
        context = state.get("ambiguity_context", "Portfolia's architecture")

        options_text = ", ".join(options[:-1])
        if options:
            if len(options) > 1:
                options_text = f"{options_text}, or {options[-1]}" if options_text else options[-1]
            else:
                options_text = options[0]

        clarifier = (
            f"I am excited to go deeper on \"{query}\". I can focus on {options_text} "
            f"using my own system as the walkthrough so you get a real example. "
            f"Which part of {context} would you like first?"
        )

    with create_custom_span(
        name="ask_clarifier",
        inputs={"question": clarifier, "type": clarification_type}
    ):
        state["answer"] = clarifier
        state["clarifying_question"] = clarifier
        state["pipeline_halt"] = True

    return state
