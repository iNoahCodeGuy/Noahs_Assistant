"""Compose retrieval-ready queries that respect role and entity context."""

from __future__ import annotations

from src.state.conversation_state import ConversationState
from src.observability.langsmith_tracer import create_custom_span


def compose_query(state: ConversationState) -> ConversationState:
    """Blend the user question with role and entity hints for retrieval."""
    with create_custom_span(
        name="compose_query",
        inputs={"query": state.get("query", "")[:120], "role_mode": state.get("role_mode")}
    ):
        base_query = state.get("expanded_query") or state.get("query", "")
        role_hint = state.get("role_mode", "")
        entity_hint = state.get("entities", {})

        entity_fragments = []
        if company := entity_hint.get("company"):
            entity_fragments.append(f"company={company}")
        if position := entity_hint.get("position"):
            entity_fragments.append(f"position={position}")
        if timeline := entity_hint.get("timeline"):
            entity_fragments.append(f"timeline={timeline}")

        composed = base_query
        if role_hint:
            composed = f"[{role_hint}] {composed}"
        if entity_fragments:
            fragments = " | ".join(entity_fragments)
            composed = f"{composed} :: {fragments}"

        state["composed_query"] = composed.strip()

    return state
