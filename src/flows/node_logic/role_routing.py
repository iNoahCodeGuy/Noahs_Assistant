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


def classify_role_mode(state: ConversationState) -> ConversationState:
    """Normalize the selected persona into an internal role mode."""
    with create_custom_span(
        name="classify_role_mode",
        inputs={"role": state.get("role", "unknown")}
    ):
        raw_role = (state.get("role") or "Just looking around").strip().lower()
        normalized = _ROLE_ALIASES.get(raw_role, raw_role.replace(" ", "_"))

        state["role_mode"] = normalized
        state["role_confidence"] = 1.0

        # Attach persona hints for downstream nodes (analytics + memory)
        persona_hints: Dict[str, str] = state["session_memory"].setdefault("persona_hints", {})
        persona_hints.setdefault("role_mode", normalized)

    return state
