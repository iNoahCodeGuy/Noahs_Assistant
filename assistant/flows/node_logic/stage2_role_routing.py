"""Role classification and routing logic for the conversation pipeline.

Handles role detection, normalization, and technical hiring manager onboarding.
Merged route_hiring_manager_technical logic for single-pass routing.
"""

from __future__ import annotations

import logging
import time
from textwrap import dedent
from typing import Any, Dict

from assistant.state.conversation_state import ConversationState
from assistant.observability.langsmith_tracer import create_custom_span
from assistant.config.settings import get_debug_log_path

logger = logging.getLogger(__name__)

# ============================================================================
# Role-Specific Welcome Messages
# ============================================================================

def _get_role_welcome_message(role_mode: str) -> str:
    """Return role-specific welcome message explaining available knowledge base."""

    messages = {
        "professional_background": dedent("""\
            Great choice! Let me tell you about Noah's professional background.

            Noah has built a strong foundation in sales across multiple industries:
            • **Tesla Sales Advisor** (Current) - Consistently exceeds targets in the competitive EV market
            • **Real Estate** - Developed client relationships and negotiation skills
            • **Gym Sales** - Mastered consultative selling and membership conversions
            • **Logistics** - Built operational efficiency and customer service excellence

            Noah combines this sales expertise with a passion for technology, making him uniquely positioned to bridge business and technical conversations.

            Would you like me to send you Noah's resume? I can email it to you and Noah will receive a notification that you're interested!
        """),

        "technical_background": dedent("""\
            Perfect! I'd love to show you Noah's technical side.

            Would you like me to:
            1️⃣ Go over Noah's certifications?
            2️⃣ Show what kind of programs Noah has built?
        """),

        "explorer": dedent("""\
            Awesome! Just looking around? I've got some fun stuff for you!

            What would you like to see?
            1️⃣ A video of Noah eating 10 hotdogs 🌭
            2️⃣ A video of Noah in a cage fight 🥊
        """),

        "confession": dedent("""\
            💌 Aww, this is sweet! I'm here to help deliver your message to Noah.

            Would you like to confess:
            1️⃣ Anonymously (your identity stays secret)
            2️⃣ With your identity (so Noah can reach out)
        """),

        # Keep legacy roles for backwards compatibility
        "hiring_manager_nontechnical": dedent("""\
            Perfect! I'll focus on business value and career insights.

            I have access to Noah's complete professional background including:
            • Career progression and key achievements
            • Business impact and leadership examples
            • Team collaboration and soft skills
            • Industry experience and domain knowledge

            Where would you like to start?
        """),

        "hiring_manager_technical": dedent("""\
            Since you selected 'Technical Hiring Manager', I can focus on 5 key areas:

            1️⃣ The orchestration layer — how my nodes, states, and safeguards work together
            2️⃣ My full tech stack — architecture overview (frontend → backend → observability)
            3️⃣ Enterprise adaptation — how assistants like me are customized for large-scale deployments
            4️⃣ Data pipeline management — embeddings, vector storage, chunking, and analytics
            5️⃣ See Noah's technical background — certifications, GitHub projects, engineering foundation
        """),

        "software_developer": dedent("""\
            Great! Let's dive into the technical details.

            I have access to:
            • Code samples and architecture patterns
            • Implementation details of Noah's projects
            • Technical decision-making and trade-offs
            • System design and infrastructure choices
            • This very codebase (I'm built with LangGraph, RAG, pgvector, and Supabase!)

            Want to see code, discuss architecture, or ask about specific technical implementations?
        """),
    }

    return messages.get(role_mode, "")

# ============================================================================
# Role Mapping & Normalization
# ============================================================================

_ROLE_ALIASES = {
    # New 4-option menu aliases
    "looking to learn about noah's professional background": "professional_background",
    "professional background": "professional_background",
    "looking to learn about his technical background": "technical_background",
    "technical background": "technical_background",
    "just looking around": "explorer",
    "looking to confess crush": "confession",
    # Legacy aliases for backwards compatibility
    "hiring manager (technical)": "hiring_manager_technical",
    "hiring manager (nontechnical)": "hiring_manager_nontechnical",
    "hiring manager (non-technical)": "hiring_manager_nontechnical",
    "software developer": "software_developer",
}

_ROLE_DISPLAY = {
    # New roles
    "professional_background": "Professional Background",
    "technical_background": "Technical Background",
    "explorer": "Just looking around",
    "confession": "Looking to confess crush",
    # Legacy roles
    "hiring_manager_technical": "Hiring Manager (technical)",
    "hiring_manager_nontechnical": "Hiring Manager (nontechnical)",
    "software_developer": "Software Developer",
}

_ROLE_SELECTION_MAP = {
    # New 4-option menu (matches _INITIAL_GREETING)
    "1": "professional_background",
    "1️⃣": "professional_background",
    "2": "technical_background",
    "2️⃣": "technical_background",
    "3": "explorer",
    "3️⃣": "explorer",
    "4": "confession",
    "4️⃣": "confession",
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

            # Clear pipeline_halt if it exists (from previous welcome message)
            # Menu selections after role selection should proceed normally
            # #region agent log
            with open(get_debug_log_path(), 'a') as f:
                import json
                f.write(json.dumps({
                    "location": "stage2_role_routing.py:146",
                    "message": "Before pipeline_halt clear check",
                    "data": {
                        "pipeline_halt": state.get("pipeline_halt"),
                        "role_welcome_shown": persona_hints.get("role_welcome_shown"),
                        "condition_passes": bool(state.get("pipeline_halt") and persona_hints.get("role_welcome_shown"))
                    },
                    "timestamp": int(time.time() * 1000),
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "C"
                }) + "\n")
            # #endregion

            if state.get("pipeline_halt") and persona_hints.get("role_welcome_shown"):
                state.pop("pipeline_halt", None)
                # #region agent log
                with open(get_debug_log_path(), 'a') as f:
                    import json
                    f.write(json.dumps({
                        "location": "stage2_role_routing.py:160",
                        "message": "Pipeline_halt cleared",
                        "data": {
                            "role": normalized,
                            "role_welcome_shown": persona_hints.get("role_welcome_shown"),
                            "pipeline_halt_after": state.get("pipeline_halt")
                        },
                        "timestamp": int(time.time() * 1000),
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": "C"
                    }) + "\n")
                # #endregion
                logger.debug(f"Cleared pipeline_halt for menu selection after role welcome: role={normalized}, role_welcome_shown={persona_hints.get('role_welcome_shown')}")

            # #region agent log
            with open(get_debug_log_path(), 'a') as f:
                import json
                f.write(json.dumps({
                    "location": "stage2_role_routing.py:186",
                    "message": "Returning from classify_role_mode (role already set)",
                    "data": {
                        "pipeline_halt": state.get("pipeline_halt"),
                        "role": normalized,
                        "role_welcome_shown": persona_hints.get("role_welcome_shown"),
                        "has_answer": bool(state.get("answer"))
                    },
                    "timestamp": int(time.time() * 1000),
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "E"
                }) + "\n")
            # #endregion

            # Check if we're awaiting a sub-menu selection for an already established role
            if state.get("awaiting_sub_menu") and state.get("current_menu_branch") == normalized:
                from assistant.flows.node_logic.util_menu_handlers import (
                    handle_sub_menu_selection,
                )
                # Handle the sub-menu selection and return the updated state immediately
                return handle_sub_menu_selection(state)

            # Return partial update dict (not full state) to avoid preserving old answer
            # Only include top-level fields that were modified
            # Note: session_memory is modified in place, so LangGraph will merge it automatically
            # We don't need to include it in the partial update to avoid overwriting nested dicts
            partial_update: Dict[str, Any] = {
                "role_mode": normalized,
                "role_confidence": 1.0,
                "role": state["role"],
                # CRITICAL: Clear old answer when role is already set and we're continuing pipeline
                # This prevents preserving the welcome message from Turn 2 when Turn 3 runs
                "answer": None,
                "draft_answer": None
            }
            # If pipeline_halt was cleared (popped), explicitly set it to None in update
            # This ensures LangGraph clears it from state
            if not state.get("pipeline_halt"):
                partial_update["pipeline_halt"] = None

            return partial_update

        # Check for navigation keywords (menu, back, etc.)
        from assistant.flows.node_logic.util_menu_handlers import _check_navigation_keywords
        nav_action = _check_navigation_keywords(state.get("query", ""))
        if nav_action == "menu":
            # Reset to main menu
            state["awaiting_sub_menu"] = False
            state.pop("current_menu_branch", None)
            state.pop("sub_menu_type", None)
            from assistant.flows.node_logic.stage0_session_management import _INITIAL_GREETING
            state["answer"] = _INITIAL_GREETING
            state["pipeline_halt"] = True
            return state

        # Infer role from query content
        query_raw = state.get("query", "")
        query = query_raw.lower().strip()
        inferred_role = None
        confidence = 0.7

        # Number-based selections from initial prompt (1-4)
        selection_key = query if query in _ROLE_SELECTION_MAP else query.replace(" ", "")
        if selection_key in _ROLE_SELECTION_MAP:
            normalized = _ROLE_SELECTION_MAP[selection_key]
            inferred_role = _ROLE_DISPLAY.get(normalized, "Just looking around")
            confidence = 1.0

        # First check if user explicitly stated their role (exact match)
        if not inferred_role and query in _ROLE_ALIASES:
            inferred_role = _ROLE_DISPLAY[_ROLE_ALIASES[query]]
            confidence = 1.0
        # Also check for close matches
        elif not inferred_role and any(role_name in query for role_name in _ROLE_ALIASES.keys()):
            for role_name, role_key in _ROLE_ALIASES.items():
                if role_name in query:
                    inferred_role = _ROLE_DISPLAY[role_key]
                    confidence = 0.95
                    break

        # Check for professional/sales/career signals
        elif not inferred_role and any(keyword in query for keyword in ["professional", "sales", "career", "background", "experience", "resume", "cv"]):
            inferred_role = "Professional Background"
            confidence = 0.9

        # Check for technical/developer/code signals
        elif not inferred_role and any(keyword in query for keyword in ["technical", "code", "developer", "engineer", "programming", "architecture", "certifications", "projects", "github"]):
            inferred_role = "Technical Background"
            confidence = 0.9

        # Check for confession signals
        elif not inferred_role and any(keyword in query for keyword in ["confess", "crush", "secret", "anonymous", "love"]):
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

        # Show role-specific welcome message on first role detection
        if not persona_hints.get("role_welcome_shown"):
            welcome_msg = _get_role_welcome_message(normalized)
            if welcome_msg:
                state["answer"] = welcome_msg
                state["pipeline_halt"] = True  # Wait for user's first real query
                persona_hints["role_welcome_shown"] = True

                # Set sub-menu awaiting flag for roles with sub-menus
                if normalized in ("technical_background", "explorer", "confession"):
                    state["awaiting_sub_menu"] = True
                    state["current_menu_branch"] = normalized
                    state["sub_menu_type"] = f"{normalized}_choice"

                # #region agent log
                with open(get_debug_log_path(), 'a') as f:
                    import json
                    f.write(json.dumps({
                        "location": "stage2_role_routing.py:252",
                        "message": "Setting role welcome message",
                        "data": {
                            "role": normalized,
                            "role_welcome_shown": persona_hints.get("role_welcome_shown"),
                            "pipeline_halt": state.get("pipeline_halt"),
                            "awaiting_sub_menu": state.get("awaiting_sub_menu"),
                            "answer_length": len(welcome_msg)
                        },
                        "timestamp": int(time.time() * 1000),
                        "sessionId": "debug-session",
                        "runId": "run2",
                        "hypothesisId": "F"
                    }) + "\n")
                # #endregion
                return state

        # Handle branch-specific sub-menu routing
        # Each branch may have its own sub-menu that needs handling
        if normalized in ("technical_background", "explorer", "confession", "professional_background"):
            from assistant.flows.node_logic.util_menu_handlers import (
                handle_sub_menu_selection,
            )

            # Check if we're waiting for a sub-menu selection
            if state.get("awaiting_sub_menu"):
                return handle_sub_menu_selection(state)

        # Legacy: Handle technical HM routing (from route_hiring_manager_technical)
        if normalized == "hiring_manager_technical":
            from assistant.flows.node_logic.util_role_specific import (
                handle_hm_technical_menu_selection,
                onboard_hiring_manager_technical,
            )

            if state.get("awaiting_hm_tech_menu"):
                return handle_hm_technical_menu_selection(state)

            if not persona_hints.get("hm_technical_onboarded"):
                return onboard_hiring_manager_technical(state)

    return state


# ============================================================================
# Repeated Query Detection
# ============================================================================

def detect_repeated_query(state: ConversationState) -> ConversationState:
    """Detect if user asked the same question in recent turns.

    This helps Portfolia avoid giving identical responses when users
    repeat themselves, instead offering to explore different angles
    or clarify what they're looking for.

    Args:
        state: ConversationState with query and chat_history

    Returns:
        Updated state with is_repeated_query flag if detected
    """
    with create_custom_span(
        name="detect_repeated_query",
        inputs={"query": state.get("query", "")[:100]}
    ):
        query = state.get("query", "").lower().strip()
        chat_history = state.get("chat_history", [])

        if not query or len(query) < 5:
            return state

        # Get last 4 messages (2 exchanges) to find recent user queries
        recent_user_queries = []
        for msg in chat_history[-4:]:
            if isinstance(msg, dict):
                msg_type = msg.get("type") or msg.get("role", "")
                content = msg.get("content", "").lower().strip()
            elif hasattr(msg, "type"):
                msg_type = msg.type
                content = getattr(msg, "content", "").lower().strip()
            else:
                continue

            if msg_type in ["human", "user"]:
                recent_user_queries.append(content)

        # Check for exact or near-exact match (within last 2 user messages)
        if recent_user_queries:
            for prev_query in recent_user_queries:
                # Exact match
                if query == prev_query:
                    state["is_repeated_query"] = True
                    state["repeated_query_count"] = 2
                    logger.info(f"Detected repeated query (exact match): '{query[:50]}...'")
                    return state

                # Near-exact match (90% word overlap)
                query_words = set(query.split())
                prev_words = set(prev_query.split())
                if query_words and prev_words:
                    overlap = len(query_words & prev_words) / max(len(query_words), len(prev_words))
                    if overlap > 0.9:
                        state["is_repeated_query"] = True
                        state["repeated_query_count"] = 2
                        logger.info(f"Detected repeated query (90% overlap): '{query[:50]}...'")
                        return state

        return state
