"""Role classification node for the conversation pipeline."""

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
        query = state.get("query", "").lower()
        inferred_role = None
        confidence = 0.7

        # Check for hiring/recruiting signals
        if any(keyword in query for keyword in ["hire", "hiring", "recruit", "position", "job opening", "candidate"]):
            if any(tech_keyword in query for tech_keyword in ["technical", "tech", "engineering", "code", "developer"]):
                inferred_role = "Hiring Manager (technical)"
            else:
                inferred_role = "Hiring Manager (nontechnical)"
            confidence = 0.9

        # Check for developer/engineer signals
        elif any(keyword in query for keyword in ["code", "developer", "engineer", "programming", "technical", "architecture", "api", "database"]):
            inferred_role = "Software Developer"
            confidence = 0.85

        # Check for confession signals
        elif any(keyword in query for keyword in ["confess", "crush", "secret", "anonymous"]):
            inferred_role = "Looking to confess crush"
            confidence = 0.95

        # Default to casual explorer
        else:
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

    return state
