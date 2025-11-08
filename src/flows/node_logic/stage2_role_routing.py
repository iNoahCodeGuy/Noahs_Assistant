"""Role classification and routing logic for the conversation pipeline.

Handles role detection, normalization, and technical hiring manager onboarding.
Merged route_hiring_manager_technical logic for single-pass routing.
"""

from __future__ import annotations

from typing import Dict

from src.state.conversation_state import ConversationState
from src.observability.langsmith_tracer import create_custom_span

_ROLE_ALIASES = {
    "hiring manager (technical)": "hiring_manager_technical",
    "hiring manager (nontechnical)": "hiring_manager_nontechnical",
    "hiring manager (non-technical)": "hiring_manager_nontechnical",
    "software developer": "software_developer",
    "just looking around": "explorer",
    "looking to confess crush": "confession",
}

_ROLE_DISPLAY = {
    "hiring_manager_technical": "Hiring Manager (technical)",
    "hiring_manager_nontechnical": "Hiring Manager (nontechnical)",
    "software_developer": "Software Developer",
    "explorer": "Just looking around",
    "confession": "Looking to confess crush",
}

_ROLE_SELECTION_MAP = {
    "1": "hiring_manager_nontechnical",
    "1️⃣": "hiring_manager_nontechnical",
    "2": "hiring_manager_technical",
    "2️⃣": "hiring_manager_technical",
    "3": "software_developer",
    "3️⃣": "software_developer",
    "4": "explorer",
    "4️⃣": "explorer",
    "5": "confession",
    "5️⃣": "confession",
}


def classify_role_mode(state: ConversationState) -> ConversationState:
    """Infer the user's persona from their query and conversation context.

    If no role is set yet, this node analyzes the user's message to determine
    which persona best fits their intent (hiring manager, developer, casual, etc.).
    """
    with create_custom_span(
        name="classify_role_mode",
        inputs={"role": state.get("role", "unknown"), "query": state.get("query", "")}
    ):
        # If role already set, just normalize it
        if state.get("role"):
            raw_role = state.get("role", "").strip().lower()
            normalized = _ROLE_ALIASES.get(raw_role, raw_role.replace(" ", "_"))
            state["role_mode"] = normalized
            state["role_confidence"] = 1.0
            state["role"] = _ROLE_DISPLAY.get(normalized, state.get("role", "Just looking around"))

            persona_hints: Dict[str, str] = state["session_memory"].setdefault("persona_hints", {})
            persona_hints.setdefault("role_mode", normalized)
            return state

        # Infer role from query content
        query_raw = state.get("query", "")
        query = query_raw.lower().strip()
        inferred_role = None
        confidence = 0.7

        # Number-based selections from initial prompt (1-5)
        selection_key = query if query in _ROLE_SELECTION_MAP else query.replace(" ", "")
        if selection_key in _ROLE_SELECTION_MAP:
            normalized = _ROLE_SELECTION_MAP[selection_key]
            inferred_role = _ROLE_DISPLAY.get(normalized, "Just looking around")
            confidence = 1.0

        # First check if user explicitly stated their role (exact match)
        if not inferred_role and query in _ROLE_ALIASES:
            inferred_role = _ROLE_DISPLAY[_ROLE_ALIASES[query]]
            confidence = 1.0
        # Also check for close matches (e.g., "I'm a hiring manager (technical)")
        elif not inferred_role and any(role_name in query for role_name in _ROLE_ALIASES.keys()):
            for role_name, role_key in _ROLE_ALIASES.items():
                if role_name in query:
                    inferred_role = _ROLE_DISPLAY[role_key]
                    confidence = 0.95
                    break
        # Check for hiring/recruiting signals
        elif not inferred_role and any(keyword in query for keyword in ["hire", "hiring", "recruit", "position", "job opening", "candidate"]):
            if any(tech_keyword in query for tech_keyword in ["technical", "tech", "engineering", "code", "developer"]):
                inferred_role = "Hiring Manager (technical)"
            else:
                inferred_role = "Hiring Manager (nontechnical)"
            confidence = 0.9

        # Check for developer/engineer signals
        elif not inferred_role and any(keyword in query for keyword in ["code", "developer", "engineer", "programming", "technical", "architecture", "api", "database"]):
            inferred_role = "Software Developer"
            confidence = 0.85

        # Check for confession signals
        elif not inferred_role and any(keyword in query for keyword in ["confess", "crush", "secret", "anonymous"]):
            inferred_role = "Looking to confess crush"
            confidence = 0.95

        # Default to casual explorer
        if not inferred_role:
            inferred_role = "Just looking around"
            confidence = 0.6

        # Set the inferred role
        state["role"] = inferred_role
        raw_role = inferred_role.strip().lower()
        normalized = _ROLE_ALIASES.get(raw_role, raw_role.replace(" ", "_"))

        state["role_mode"] = normalized
        state["role_confidence"] = confidence
        state["role"] = _ROLE_DISPLAY.get(normalized, inferred_role)

        # Attach persona hints for downstream nodes (analytics + memory)
        persona_hints: Dict[str, str] = state["session_memory"].setdefault("persona_hints", {})
        persona_hints.setdefault("role_mode", normalized)

        # Merged: Handle technical HM routing (from route_hiring_manager_technical)
        # If technical HM role detected, check for menu handling or onboarding
        if normalized == "hiring_manager_technical":
            from src.flows.node_logic.util_role_specific import (
                handle_hm_technical_menu_selection,
                onboard_hiring_manager_technical,
            )

            if state.get("awaiting_hm_tech_menu"):
                return handle_hm_technical_menu_selection(state)

            if not persona_hints.get("hm_technical_onboarded"):
                return onboard_hiring_manager_technical(state)

    return state
