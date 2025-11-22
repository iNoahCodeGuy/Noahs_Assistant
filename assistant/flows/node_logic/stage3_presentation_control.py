"""Presentation control for Portfolia's teach-first experience.

Unified presentation controller that determines depth level (1-3) and display toggles
(code, data, diagrams) in a single pass, reducing pipeline complexity.

Merged depth_controller + display_controller logic for streamlined presentation decisions.

Exports:
    presentation_controller(state) -> ConversationState
        Chooses depth level (1-3) and display toggles based on role, intent, turn count,
        and teaching signals. Single-pass presentation strategy.

    depth_controller(state) -> ConversationState [DEPRECATED - alias for presentation_controller]
        Backward compatibility alias. New code should use presentation_controller.

    display_controller(state) -> ConversationState [DEPRECATED - no-op]
        Backward compatibility no-op. Logic merged into presentation_controller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from assistant.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)


ENGINEERING_INTENTS = {"technical", "engineering"}
BUSINESS_INTENTS = {"business_value", "career", "analytics", "data"}


@dataclass(frozen=True)
class DepthRule:
    name: str
    level: int
    reason: str


def _resolve_role_mode(state: ConversationState) -> str:
    return state.get("role_mode") or state.get("role", "explorer").lower()


def presentation_controller(state: ConversationState) -> ConversationState:
    """Unified presentation controller: depth level + display toggles in one pass.

    Merges depth_controller and display_controller logic for streamlined decision-making.

    Depth selection (1-3):
    - Level 1: Opening overview, default for casual queries
    - Level 2: Guided detail for technical roles, multi-turn conversations
    - Level 3: Deep dive for explicit teaching requests or engineering drilldowns

    Display toggles (code, data, diagram):
    - Code: Engineering queries, "how" questions, depth ≥2
    - Data: Business/reliability queries with metrics keywords
    - Diagram: Depth ≥2, non-greeting contexts

    Args:
        state: ConversationState with role, intent, query, conversation_turn

    Returns:
        Updated state with depth_level, detail_strategy, layout_variant, followup_variant,
        display_toggles, display_reasons
    """
    role_mode = _resolve_role_mode(state)
    intent = state.get("query_intent") or state.get("query_type") or "general"
    conversation_turn = state.get("conversation_turn", 0)
    lowered_query = state.get("query", "").lower()

    # ========== DEPTH SELECTION ==========
    rules: Tuple[DepthRule, ...] = (
        DepthRule("default", 1, "Opening overview"),
        DepthRule("technical_role", 2, "Technical persona expects guided detail"),
        DepthRule("teaching_moment", 3, "User explicitly asked for a deep explanation"),
        DepthRule("multi_turn", 2, "Conversation has progressed beyond the opener"),
        DepthRule("business_depth", 2, "Business questions need context + outcomes"),
    )

    depth = 1
    reason = "default"

    for rule in rules:
        if rule.name == "technical_role" and role_mode in {
            "software developer",
            "hiring manager (technical)",
            "hiring_manager_technical",
        }:
            depth = max(depth, rule.level)
            reason = rule.reason
        elif rule.name == "teaching_moment" and state.get("teaching_moment"):
            depth = max(depth, rule.level)
            reason = rule.reason
        elif rule.name == "multi_turn" and conversation_turn >= 2:
            depth = max(depth, rule.level)
            reason = rule.reason
        elif rule.name == "business_depth" and intent in BUSINESS_INTENTS:
            depth = max(depth, rule.level)
            reason = rule.reason

    if intent in ENGINEERING_INTENTS and state.get("needs_longer_response"):
        depth = 3
        reason = "Engineering deep dive requested"

    state["depth_level"] = min(depth, 3)
    state["detail_strategy"] = reason

    # Validate depth progression
    session_memory = state.get("session_memory", {})
    persona_hints = session_memory.setdefault("persona_hints", {})
    previous_depth = persona_hints.get("previous_depth_level", 1)

    # Store previous depth for next turn
    persona_hints["previous_depth_level"] = state["depth_level"]

    # Validate depth increases with turn count (or maintains)
    if conversation_turn >= 2 and state["depth_level"] < 2:
        logger.warning(
            f"Depth not progressing: turn={conversation_turn}, depth={state['depth_level']}, "
            f"previous={previous_depth}. Multi-turn conversations should have depth >= 2."
        )
    elif conversation_turn >= 3 and state["depth_level"] < previous_depth and previous_depth > 1:
        logger.warning(
            f"Depth decreased: turn={conversation_turn}, depth={state['depth_level']}, "
            f"previous={previous_depth}. Depth should maintain or increase with conversation progression."
        )

    # ========== LAYOUT VARIANT SELECTION ==========
    if intent in ENGINEERING_INTENTS:
        state["layout_variant"] = "engineering"
        state["followup_variant"] = "engineering"
    elif intent in BUSINESS_INTENTS:
        state["layout_variant"] = "business"
        state["followup_variant"] = "business"
    else:
        state["layout_variant"] = "mixed"
        state["followup_variant"] = "mixed"

    # ========== DISPLAY TOGGLES ==========
    toggles: Dict[str, bool] = {"code": False, "data": False, "diagram": False}
    reasons: Dict[str, str] = {}

    code_triggers = ("how ", "how do", "how does", "code", "sql", "langgraph")
    if depth >= 2 and (
        any(trigger in lowered_query for trigger in code_triggers)
        or intent in ENGINEERING_INTENTS
    ):
        toggles["code"] = True
        reasons["code"] = "Engineering-oriented question benefits from code context"

    data_triggers = ("latency", "cost", "reliability")
    if any(trigger in lowered_query for trigger in data_triggers) or intent == "business_value":
        toggles["data"] = True
        reasons["data"] = "Business or reliability question warrants metrics"

    if depth >= 2 and not state.get("is_greeting"):
        toggles["diagram"] = True
        reasons["diagram"] = "Depth ≥2 unlocks architecture diagrams"

    state["display_toggles"] = toggles
    state["display_reasons"] = reasons

    return state


def depth_controller(state: ConversationState) -> ConversationState:
    """DEPRECATED: Backward compatibility alias for presentation_controller.

    Legacy function preserved for existing imports. New code should use
    presentation_controller() directly.
    """
    return presentation_controller(state)


def display_controller(state: ConversationState) -> ConversationState:
    """DEPRECATED: No-op for backward compatibility.

    Logic merged into presentation_controller. This function does nothing
    since presentation_controller now handles both depth and display in one pass.
    Kept for import compatibility only.
    """
    return state


__all__ = [
    "presentation_controller",
    "depth_controller",  # Deprecated alias
    "display_controller",  # Deprecated no-op
]
