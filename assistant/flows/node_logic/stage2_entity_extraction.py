"""Entity extraction nodes for the conversation pipeline."""

from __future__ import annotations

import logging
from typing import Dict, Any

from assistant.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)
from assistant.flows.node_logic.util_resume_distribution import (
    extract_email_from_query,
    extract_name_from_query,
    extract_job_details_from_query,
)
from assistant.observability.langsmith_tracer import create_custom_span


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

        # Extract menu selection for analytics visibility
        menu_choice = state.get("menu_choice")
        query_type = state.get("query_type")
        role_mode = state.get("role_mode", "")

        # DEBUG: Log incoming state
        logger.info(f"🔍 Entity extraction START: menu_choice={menu_choice}, query_type={query_type}, role_mode={role_mode}")

        if menu_choice and query_type == "menu_selection":
            entities["menu_selection"] = menu_choice
            # Add context about what the menu option means
            if role_mode == "hiring_manager_technical":
                menu_context_map = {
                    "1": "full_tech_stack",
                    "2": "orchestration_layer",
                    "3": "enterprise_adaptation",
                    "4": "technical_background"
                }
                entities["menu_context"] = menu_context_map.get(menu_choice, "unknown")
            logger.info(f"✅ Extracted menu entity: selection={menu_choice}, context={entities.get('menu_context')}")
        else:
            logger.warning(f"⚠️ Menu extraction SKIPPED: menu_choice={menu_choice}, query_type={query_type}")

        # Update job details using existing resume distribution helpers
        extract_job_details_from_query(state)
        job_details = state.get("job_details", {})
        if job_details:
            entities.update(job_details)

        query = state.get("query", "")
        if query:
            email = extract_email_from_query(query)
            if email:
                entities["email"] = email

            name = extract_name_from_query(query)
            if name:
                entities["name"] = name

            lowered = query.lower()
        else:
            lowered = ""
        for keyword, value in CONTACT_KEYWORDS.items():
            if keyword in lowered:
                entities.setdefault("contact_preference", value)

        state["entities"] = entities

        # DEBUG: Log outgoing state
        logger.info(f"🔍 Entity extraction END: entities={entities}")

    return state
