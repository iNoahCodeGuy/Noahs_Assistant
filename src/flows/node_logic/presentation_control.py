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


def update_enterprise_affinity(state: ConversationState) -> ConversationState:
    """Adjust enterprise framing based on query focus.

    Uses a relevance score (0-4) that increases with enterprise keywords
    and decreases with pure technical queries. Score ≥2 enables enterprise tie-ins.

    This allows technical hiring managers to start with enterprise context,
    then naturally shift to pure technical discussion as they drill deeper,
    and snap back to enterprise framing if they ask business-oriented questions.

    Args:
        state: ConversationState with query and session_memory

    Returns:
        Updated state with relate_to_enterprise flag set based on score

    Score mechanics:
        - Enterprise keywords (+2, cap at 4): "governance", "scale", "enterprise",
          "rollout", "value", "roi", "team", "compliance", "production"
        - Technical keywords (-1, floor at 0): "code", "implementation", "trace",
          "architecture", "pipeline", "model", "algorithm", "function"
        - Threshold: score ≥2 → relate_to_enterprise=True

    Example progression:
        Turn 1: "How does this scale to enterprise?" → score=2 (+2) → True
        Turn 2: "Show me the retrieval code" → score=1 (-1) → False
        Turn 3: "What's the governance model?" → score=3 (+2) → True
    """
    query_lower = state.get("query", "").lower()

    # Retrieve current score from session memory
    persona_hints = state.setdefault("session_memory", {}).setdefault("persona_hints", {})
    current_score = persona_hints.get("enterprise_relevance_score", 0)

    # Enterprise-focused keywords (increase affinity)
    enterprise_keywords = {
        "governance", "scale", "enterprise", "rollout", "value", "roi",
        "team", "compliance", "production", "deployment", "multi-tenant",
        "audit", "business", "customer", "operational", "reliability"
    }

    # Technical deep-dive keywords (decrease affinity)
    technical_keywords = {
        "code", "implementation", "trace", "architecture", "pipeline",
        "model", "algorithm", "function", "debug", "error", "bug",
        "test", "how does", "how do", "explain the", "walk me through"
    }

    # Calculate score adjustment
    has_enterprise = any(kw in query_lower for kw in enterprise_keywords)
    has_technical = any(kw in query_lower for kw in technical_keywords)

    if has_enterprise:
        current_score = min(current_score + 2, 4)
    elif has_technical:
        current_score = max(current_score - 1, 0)

    # Store updated score
    persona_hints["enterprise_relevance_score"] = current_score

    # Set flag based on threshold
    state["relate_to_enterprise"] = current_score >= 2

    # Log decision for analytics (useful for debugging affinity drift)
    state.setdefault("analytics_metadata", {})["enterprise_affinity_score"] = current_score

    return state


def update_technical_affinity(state: ConversationState) -> ConversationState:
    """Adjust technical depth preference based on query focus.

    Uses a relevance score (0-4) that increases with technical keywords
    and decreases with business/enterprise-only queries. Score ≥2 enables
    technical deep-dives with code, architecture diagrams, and implementation details.

    This allows business-oriented users to start light, then drill into
    technical specifics as they become curious, and return to high-level
    discussion when asking about outcomes or ROI.

    Args:
        state: ConversationState with query and session_memory

    Returns:
        Updated state with show_technical_depth flag set based on score

    Score mechanics:
        - Technical keywords (+2, cap at 4): "code", "implementation", "how does",
          "architecture", "algorithm", "trace", "debug", "api", "function", "error"
        - Business keywords (-1, floor at 0): "roi", "value", "outcome", "business",
          "stakeholder", "budget", "governance", "compliance", "rollout"
        - Threshold: score ≥2 → show_technical_depth=True

    Example progression:
        Turn 1: "Tell me about your career" → score=0 → False
        Turn 2: "How does the RAG pipeline work?" → score=2 (+2) → True
        Turn 3: "Show me the retrieval code" → score=4 (+2, capped) → True
        Turn 4: "What's the business value?" → score=3 (-1) → True (still above threshold)
        Turn 5: "How much does this cost?" → score=2 (-1) → True
        Turn 6: "Tell me about ROI" → score=1 (-1) → False
    """
    query_lower = state.get("query", "").lower()

    # Retrieve current score from session memory
    persona_hints = state.setdefault("session_memory", {}).setdefault("persona_hints", {})
    current_score = persona_hints.get("technical_relevance_score", 0)

    # Technical deep-dive keywords (increase affinity)
    technical_keywords = {
        "code", "implementation", "how does", "how do", "architecture",
        "algorithm", "trace", "debug", "api", "function", "error", "bug",
        "test", "pipeline", "model", "retrieval", "embedding", "vector",
        "sql", "query", "langgraph", "node", "workflow", "state", "async",
        "performance", "latency", "optimization", "refactor", "class", "method"
    }

    # Business/high-level keywords (decrease affinity)
    business_keywords = {
        "roi", "value", "outcome", "business", "stakeholder", "budget",
        "governance", "compliance", "rollout", "why", "what benefit",
        "customer", "user", "team", "scale", "enterprise", "production",
        "career", "experience", "background", "tell me about", "overview"
    }

    # Calculate score adjustment
    has_technical = any(kw in query_lower for kw in technical_keywords)
    has_business = any(kw in query_lower for kw in business_keywords)

    if has_technical:
        current_score = min(current_score + 2, 4)
    elif has_business:
        current_score = max(current_score - 1, 0)

    # Store updated score
    persona_hints["technical_relevance_score"] = current_score

    # Set flag based on threshold
    state["show_technical_depth"] = current_score >= 2

    # Log decision for analytics (useful for debugging affinity drift)
    state.setdefault("analytics_metadata", {})["technical_affinity_score"] = current_score

    return state


__all__ = ["depth_controller", "display_controller", "update_enterprise_affinity", "update_technical_affinity"]
