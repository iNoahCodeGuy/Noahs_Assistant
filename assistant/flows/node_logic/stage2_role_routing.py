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

logger = logging.getLogger(__name__)

# ============================================================================
# Role-Specific Welcome Messages
# ============================================================================

def _get_role_welcome_message(role_mode: str) -> str:
    """Return role-specific welcome message explaining available knowledge base."""

    messages = {
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
            Since you selected 'Technical Hiring Manager', I can focus on the areas most relevant to you:

            • My architecture and full-stack design — LangGraph orchestration, RAG pipeline, Supabase vector storage, and observability through LangSmith
            • How Noah applied software engineering, AI engineering, and data pipeline best practices to make me production-ready
            • The business and enterprise value of agentic systems like me — how organizations deploy assistants to improve reliability, scalability, and customer satisfaction

            You can choose where to start:
            1️⃣ My full tech stack — architecture overview (frontend → backend → data pipeline → observability)
            2️⃣ The orchestration layer — how my nodes, states, and safeguards work together as a conversation progresses
            3️⃣ Enterprise adaptation — how assistants like me are customized for large-scale deployments
            4️⃣ See Noah's technical background — certifications, GitHub projects, and proof of his engineering foundation
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

        "explorer": dedent("""\
            Awesome! I'm here to show you whatever interests you.

            I can share:
            • Career stories and professional highlights
            • Technical projects and how they work
            • Behind-the-scenes of how I'm built
            • Fun facts and personal interests (including MMA!)

            What sounds interesting to you?
        """),

        "confession": dedent("""\
            💌 Aww, this is sweet! I'm here to help.

            Your confession will be completely anonymous and sent directly to Noah. I won't judge or analyze it — just pass it along safely.

            When you're ready, go ahead and share what's on your mind!
        """),
    }

    return messages.get(role_mode, "")

# ============================================================================
# Role Mapping & Normalization
# ============================================================================

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

            # Clear pipeline_halt if it exists (from previous welcome message)
            # Menu selections after role selection should proceed normally
            # #region agent log
            with open('/Users/noahdelacalzada/NoahsAIAssistant/NoahsAIAssistant-/.cursor/debug.log', 'a') as f:
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
                with open('/Users/noahdelacalzada/NoahsAIAssistant/NoahsAIAssistant-/.cursor/debug.log', 'a') as f:
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
            with open('/Users/noahdelacalzada/NoahsAIAssistant/NoahsAIAssistant-/.cursor/debug.log', 'a') as f:
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

        # Show role-specific welcome message on first role detection
        if not persona_hints.get("role_welcome_shown"):
            welcome_msg = _get_role_welcome_message(normalized)
            if welcome_msg:
                state["answer"] = welcome_msg
                state["pipeline_halt"] = True  # Wait for user's first real query
                persona_hints["role_welcome_shown"] = True
                # #region agent log
                with open('/Users/noahdelacalzada/NoahsAIAssistant/NoahsAIAssistant-/.cursor/debug.log', 'a') as f:
                    import json
                    f.write(json.dumps({
                        "location": "stage2_role_routing.py:252",
                        "message": "Setting role welcome message",
                        "data": {
                            "role": normalized,
                            "role_welcome_shown": persona_hints.get("role_welcome_shown"),
                            "pipeline_halt": state.get("pipeline_halt"),
                            "answer_length": len(welcome_msg)
                        },
                        "timestamp": int(time.time() * 1000),
                        "sessionId": "debug-session",
                        "runId": "run2",
                        "hypothesisId": "F"
                    }) + "\n")
                # #endregion
                return state

        # Merged: Handle technical HM routing (from route_hiring_manager_technical)
        # If technical HM role detected, check for menu handling or onboarding
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
