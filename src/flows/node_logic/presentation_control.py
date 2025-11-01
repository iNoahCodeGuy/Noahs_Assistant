"""Presentation control nodes for Portfolia's teach-first experience.

These nodes sit between intent classification and retrieval to decide how much
structure and which supporting artifacts should be presented. They do not
invoke the LLM. Instead, they annotate the ConversationState with presentation
metadata used later by format_answer.

Exports:
    depth_controller(state) -> ConversationState
        Chooses depth level (1-3) based on role, intent, turn count, and
        teaching signals.

    display_controller(state) -> ConversationState
        Decides whether to surface code, data/metrics, or diagrams based on
        heuristics plus the selected depth level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from src.state.conversation_state import ConversationState


ENGINEERING_INTENTS = {"technical", "engineering"}
BUSINESS_INTENTS = {"business_value", "career", "analytics", "data"}


@dataclass(frozen=True)
class DepthRule:
    name: str
    level: int
    reason: str


def _resolve_role_mode(state: ConversationState) -> str:
    return state.get("role_mode") or state.get("role", "explorer").lower()


def depth_controller(state: ConversationState) -> ConversationState:
    """Select a presentation depth level (1-3) aligned with teaching-first UX."""
    role_mode = _resolve_role_mode(state)
    intent = state.get("query_intent") or state.get("query_type") or "general"
    conversation_turn = state.get("conversation_turn", 0)

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

    if intent in ENGINEERING_INTENTS:
        state["layout_variant"] = "engineering"
        state["followup_variant"] = "engineering"
    elif intent in BUSINESS_INTENTS:
        state["layout_variant"] = "business"
        state["followup_variant"] = "business"
    else:
        state["layout_variant"] = "mixed"
        state["followup_variant"] = "mixed"

    return state


def display_controller(state: ConversationState) -> ConversationState:
    """Determine which supporting artifacts should be offered."""
    depth = state.get("depth_level", 1)
    intent = state.get("query_intent") or state.get("query_type") or "general"
    lowered_query = state.get("query", "").lower()

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


__all__ = ["depth_controller", "display_controller"]
