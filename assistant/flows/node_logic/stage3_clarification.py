"""Clarification nodes keep the dialogue grounded before retrieval."""

from __future__ import annotations

from typing import List

from assistant.state.conversation_state import ConversationState
from assistant.observability.langsmith_tracer import create_custom_span


def assess_clarification_need(state: ConversationState) -> ConversationState:
    """Decide whether the assistant should pause to ask for more context."""
    with create_custom_span(
        name="assess_clarification",
        inputs={"ambiguous": state.get("ambiguous_query", False)}
    ):
        # PRIORITY: Menu selections are explicit - never need clarification
        if state.get("query_type") == "menu_selection" or state.get("menu_choice"):
            state["clarification_needed"] = False
            return state

        needs_clarification = bool(state.get("ambiguous_query"))
        state["clarification_needed"] = needs_clarification

    return state


def ask_clarifying_question(state: ConversationState) -> ConversationState:
    """If clarification is required, craft the targeted follow-up question."""
    if not state.get("clarification_needed"):
        return state

    options: List[str] = state.get("ambiguity_options", []) or []
    context = state.get("ambiguity_context", "Portfolia's architecture")
    query = state.get("query", "this topic")

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
        inputs={"question": clarifier}
    ):
        state["answer"] = clarifier
        state["clarifying_question"] = clarifier
        state["pipeline_halt"] = True

    return state
