"""Entity extraction nodes for the conversation pipeline."""

from __future__ import annotations

from typing import Dict, Any

from src.state.conversation_state import ConversationState
from src.flows.node_logic.resume_distribution import (
    extract_email_from_query,
    extract_name_from_query,
    extract_job_details_from_query,
)
from src.observability.langsmith_tracer import create_custom_span


CONTACT_KEYWORDS = {
    "call": "phone",
    "reach": "follow_up",
    "email": "email",
    "contact": "follow_up",
}


def extract_entities(state: ConversationState) -> ConversationState:
    """Pull lightweight entities (company, role, timeline, contact info) from the query."""
    with create_custom_span(
        name="extract_entities",
        inputs={"query": state.get("query", "")[:120]}
    ):
        entities: Dict[str, Any] = state.get("entities", {}).copy()

        # Update job details using existing resume distribution helpers
        extract_job_details_from_query(state)
        job_details = state.get("job_details", {})
        if job_details:
            entities.update(job_details)

        email = extract_email_from_query(state["query"])
        if email:
            entities["email"] = email

        name = extract_name_from_query(state["query"])
        if name:
            entities["name"] = name

        lowered = state["query"].lower()
        for keyword, value in CONTACT_KEYWORDS.items():
            if keyword in lowered:
                entities.setdefault("contact_preference", value)

        state["entities"] = entities

    return state
