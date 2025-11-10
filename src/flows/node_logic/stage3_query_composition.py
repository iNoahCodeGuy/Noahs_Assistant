"""Compose retrieval-ready queries that respect role and entity context."""

from __future__ import annotations

from src.state.conversation_state import ConversationState
from src.observability.langsmith_tracer import create_custom_span


def _expand_menu_selection(menu_choice: str, role_mode: str) -> str:
    """Expand menu number to full retrieval query based on role.

    Maps menu selections to rich queries that retrieve relevant content.
    Different roles have different menu options, so we check role_mode first.

    Args:
        menu_choice: The menu number (1, 2, 3, or 4)
        role_mode: The user's role (hiring_manager_technical, etc.)

    Returns:
        Expanded query optimized for vector search retrieval
    """

    # Technical Hiring Manager menu (from stage2_role_routing.py welcome message)
    if role_mode == "hiring_manager_technical":
        menu_map = {
            "1": "full technology stack architecture frontend backend data pipeline observability LangGraph Supabase pgvector deployment infrastructure",
            "2": "LangGraph orchestration layer nodes states safeguards conversation flow pipeline stage routing error handling",
            "3": "enterprise adaptation patterns large-scale deployment customization scalability reliability production best practices",
            "4": "Noah technical background certifications GitHub projects engineering foundation credentials proof skills"
        }
        return menu_map.get(menu_choice, menu_choice)

    # Nontechnical Hiring Manager menu
    elif role_mode == "hiring_manager_nontechnical":
        menu_map = {
            "1": "career progression achievements key accomplishments milestones growth",
            "2": "business impact ROI value delivered results outcomes",
            "3": "leadership teamwork collaboration communication soft skills",
            "4": "resume download contact information availability"
        }
        return menu_map.get(menu_choice, menu_choice)

    # Software Developer menu
    elif role_mode == "software_developer":
        menu_map = {
            "1": "code architecture implementation technical decisions patterns",
            "2": "system design infrastructure deployment production engineering",
            "3": "debugging troubleshooting problem-solving technical challenges",
            "4": "open source contributions GitHub projects technical portfolio"
        }
        return menu_map.get(menu_choice, menu_choice)

    # Explorer menu
    elif role_mode == "explorer":
        menu_map = {
            "1": "career stories professional journey highlights interesting projects",
            "2": "technical projects how they work architecture overview",
            "3": "behind the scenes how assistant built RAG pipeline",
            "4": "fun facts personal interests hobbies MMA fights"
        }
        return menu_map.get(menu_choice, menu_choice)

    # Fallback: return original if role not recognized
    return menu_choice


def compose_query(state: ConversationState) -> ConversationState:
    """Blend the user question with role and entity hints for retrieval."""
    with create_custom_span(
        name="compose_query",
        inputs={"query": state.get("query", "")[:120], "role_mode": state.get("role_mode")}
    ):
        # PRIORITY: Handle menu selections first
        if state.get("query_type") == "menu_selection":
            menu_choice = state.get("menu_choice", state.get("query", ""))
            role_mode = state.get("role_mode", "")

            # Expand menu selection to full retrieval query
            expanded_query = _expand_menu_selection(menu_choice, role_mode)

            # Add role context for better retrieval
            if role_mode:
                composed = f"[{role_mode}] {expanded_query}"
            else:
                composed = expanded_query

            state["composed_query"] = composed.strip()
            state["menu_expanded"] = True  # Flag for debugging
            return state

        # Regular query composition for non-menu queries
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
