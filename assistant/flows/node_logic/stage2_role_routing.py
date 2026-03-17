"""Welcome message routing for the conversation pipeline.

Handles the initial welcome flow: user clicks one of 4 buttons (or types a
number/alias), gets a tailored welcome message, then enters a single universal
conversation pipeline. No role-based branching after the welcome.
"""

from __future__ import annotations

import logging
from textwrap import dedent
from typing import Any, Dict

from assistant.state.conversation_state import ConversationState
from assistant.observability.langsmith_tracer import create_custom_span

logger = logging.getLogger(__name__)

# ============================================================================
# Welcome Messages (one per button option)
# ============================================================================

def _get_welcome_message(role_mode: str) -> str:
    """Return welcome message for the selected entry path."""

    messages = {
        "explorer": dedent("""\
            Noah is a software developer who builds machine learning models and generative AI applications. The flagship project is what you're talking to -- a RAG-powered conversational assistant with semantic search and intent classification. He also built predictive models (logistic regression, Naive Bayes), unsupervised clustering studies, and data visualization tools.

            Professionally: Inside Sales Advisor at Tesla, top 10% performer. Before that, logistics operations at TQL and real estate transactions at Signature Real Estate Group. He's also been coaching BJJ and MMA at Xtreme Couture since 2021. The technical foundation comes from a Biology degree at UNLV -- biostatistics and experimental design, not computer science.

            What brings you here?"""),

        "software_developer": dedent("""\
            Seven projects. Each one started with a problem worth solving.

            **Portfolia AI Assistant** -- Most portfolio sites are static pages you skim and forget. This is a 21-node RAG pipeline with pgvector semantic search, Claude Sonnet 4.5 for generation, and intent classification before retrieval so greetings don't waste a vector search. You're using it.

            **Employee Attrition -- Logistic Regression** -- HR teams can't predict who's leaving until it's too late. Feature engineering on imbalanced data, cross-validation, ROC analysis. 94.75% accuracy where naive approaches plateau around 83%.

            **Employee Attrition -- Naive Bayes** -- Same dataset, different question. Logistic regression maximizes overall accuracy but misses leavers. Naive Bayes trades that for 10% higher recall on the class that actually costs money -- 58% vs 48%. Five class imbalance variants tested.

            **Customer Segmentation -- Decision Trees** -- A telecom company's customer labels don't explain behavior. Supervised classification found education and tenure drive 81% of segmentation. Region, gender, and age contribute nothing. Interpretable rules over raw accuracy.

            **Customer Segmentation -- K-Means Clustering** -- Same dataset, opposite approach. Forget the labels -- what structure actually exists? Two algorithms independently found four natural segments driven by life-stage and income. The existing labels mapped to none of them.

            **Response Time Analysis** -- No quick way to test whether response time differences are real or noise. A Streamlit app with statistical hypothesis testing and time-series visualization.

            **Lead Response Heatmap** -- Sales teams can't see when leads go unanswered. A reusable Python dashboard using pandas and matplotlib to visualize coverage gaps.

            Pick one and I'll walk you through the architecture decisions."""),

        "enterprise_ai": dedent("""\
            You're talking to one right now. I'm a 21-node agentic pipeline — intent classification, semantic retrieval, grounding validation, hallucination checking, and deterministic tool execution. Every pattern in this system maps directly to enterprise AI applications.

            Intent-first routing is how customer support agents handle millions of tickets without embedding every "hello" into a vector database. Stateless state machines are how booking flows and onboarding sequences run in serverless environments without server-side sessions. Deterministic tool execution — where the pipeline decides when to fire an API call, not the LLM — is how you keep an agent from hallucinating a function call when real money is on the line.

            I write to databases, send SMS via Twilio, and fire transactional email via Resend. None of those are demos. They run unsupervised in production.

            Pick any layer — retrieval, generation, agent coordination, quality gates — and I'll show you how the pattern transfers to enterprise scale."""),

        "casual": dedent("""\
            No agenda required. I know about Noah's projects, his career background, his technical stack, and there's an MMA coaching story that's better than you'd expect. Ask whatever you want."""),

        # Confession welcome is handled by handle_crush_confession in stage1
        "confession": "",
    }

    return messages.get(role_mode, "")


# ============================================================================
# Role Mapping (button click / text → internal key)
# ============================================================================

_ROLE_ALIASES = {
    # Button text (lowercase)
    "learn more about noah": "explorer",
    "see what noah has built": "software_developer",
    "how i relate to enterprise ai": "enterprise_ai",
    "confess a crush": "confession",
    # Casual aliases (no button — triggered by free-text input)
    "just looking around": "casual",
    "just browsing": "casual",
    "just looking": "casual",
    # Legacy aliases (kept for backwards compat with any stored roles)
    "hiring manager (technical)": "explorer",
    "hiring manager (nontechnical)": "explorer",
    "hiring manager (non-technical)": "explorer",
    "software developer": "software_developer",
    "looking to confess crush": "confession",
}

_ROLE_DISPLAY = {
    "software_developer": "See what Noah has built",
    "explorer": "Learn more about Noah",
    "enterprise_ai": "How I relate to Enterprise AI",
    "casual": "Just looking around",
    "confession": "Confess a crush",
}

_ROLE_SELECTION_MAP = {
    "1": "explorer",
    "1️⃣": "explorer",
    "2": "software_developer",
    "2️⃣": "software_developer",
    "3": "enterprise_ai",
    "3️⃣": "enterprise_ai",
    "4": "confession",
    "4️⃣": "confession",
}


def classify_role_mode(state: ConversationState) -> ConversationState:
    """Route to welcome message based on button click or text input.

    After the welcome, all visitors enter the same universal conversation
    pipeline. The role_mode is kept for analytics only.
    """
    with create_custom_span(
        name="classify_role_mode",
        inputs={"role": state.get("role", "unknown"), "query": state.get("query", "")}
    ):
        # If role already set (frontend button click), normalize and show welcome
        if state.get("role"):
            raw_role = state.get("role", "").strip().lower()
            normalized = _ROLE_ALIASES.get(raw_role, raw_role.replace(" ", "_"))
            # Map any old HM roles to explorer
            if normalized.startswith("hiring_manager"):
                normalized = "explorer"
            state["role_mode"] = normalized
            state["role"] = _ROLE_DISPLAY.get(normalized, state.get("role", "Just looking around"))

            persona_hints: Dict[str, str] = state["session_memory"].setdefault("persona_hints", {})
            persona_hints.setdefault("role_mode", normalized)

            # Show welcome on first role detection
            if not persona_hints.get("role_welcome_shown"):
                persona_hints["role_welcome_shown"] = True

                # Confession: delegate to crush flow handler
                if normalized == "confession":
                    from assistant.flows.node_logic.stage1_intent_router import handle_crush_confession
                    state["message_intent"] = "crush_confession"
                    return handle_crush_confession(state)

                welcome_msg = _get_welcome_message(normalized)
                if welcome_msg:
                    state["answer"] = welcome_msg
                    state["pipeline_halt"] = True
                    return state

            # Clear stale state from previous turns
            if state.get("pipeline_halt") and persona_hints.get("role_welcome_shown"):
                state.pop("pipeline_halt", None)

            state["answer"] = None
            state["draft_answer"] = None
            if not state.get("pipeline_halt"):
                state["pipeline_halt"] = None

            return state

        # No role set — infer from query text
        query_raw = state.get("query", "")
        query = query_raw.lower().strip()
        normalized = None

        # Number-based selections (1-4)
        selection_key = query if query in _ROLE_SELECTION_MAP else query.replace(" ", "")
        if selection_key in _ROLE_SELECTION_MAP:
            normalized = _ROLE_SELECTION_MAP[selection_key]

        # Exact alias match
        if not normalized and query in _ROLE_ALIASES:
            normalized = _ROLE_ALIASES[query]

        # Partial alias match
        if not normalized:
            for alias_text, alias_key in _ROLE_ALIASES.items():
                if alias_text in query:
                    normalized = alias_key
                    break

        # Confession keywords
        if not normalized and any(kw in query for kw in ["confess", "crush", "secret"]):
            normalized = "confession"

        # Default: explorer (broadest welcome)
        if not normalized:
            normalized = "explorer"

        state["role_mode"] = normalized
        state["role"] = _ROLE_DISPLAY.get(normalized, "Learn more about Noah")

        persona_hints: Dict[str, str] = state["session_memory"].setdefault("persona_hints", {})
        persona_hints.setdefault("role_mode", normalized)

        # Show welcome only for menu selections or direct alias matches
        # Substantive questions skip the welcome and go straight to RAG
        is_menu_number = query_raw.strip() in _ROLE_SELECTION_MAP
        is_role_alias = query in _ROLE_ALIASES
        if not persona_hints.get("role_welcome_shown"):
            persona_hints["role_welcome_shown"] = True
            if is_menu_number or is_role_alias:
                # Confession: delegate to crush flow
                if normalized == "confession":
                    from assistant.flows.node_logic.stage1_intent_router import handle_crush_confession
                    state["message_intent"] = "crush_confession"
                    return handle_crush_confession(state)

                welcome_msg = _get_welcome_message(normalized)
                if welcome_msg:
                    state["answer"] = welcome_msg
                    state["pipeline_halt"] = True
                    return state

    return state


# ============================================================================
# Repeated Query Detection (unchanged — unrelated to role routing)
# ============================================================================

def detect_repeated_query(state: ConversationState) -> ConversationState:
    """Detect if user asked the same question in recent turns."""
    with create_custom_span(
        name="detect_repeated_query",
        inputs={"query": state.get("query", "")[:100]}
    ):
        query = state.get("query", "").lower().strip()
        chat_history = state.get("chat_history", [])

        if not query or len(query) < 5:
            return state

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

        if recent_user_queries:
            for prev_query in recent_user_queries:
                if query == prev_query:
                    state["is_repeated_query"] = True
                    state["repeated_query_count"] = 2
                    logger.info(f"Detected repeated query (exact match): '{query[:50]}...'")
                    return state

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
