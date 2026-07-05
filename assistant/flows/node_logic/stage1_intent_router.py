"""Intent router - classifies user messages into high-level intent categories.

This module runs BEFORE RAG retrieval to route different types of messages appropriately:
- knowledge_query: Normal portfolio questions → proceed to RAG
- crush_confession: User confessing a crush → dedicated flow (no RAG)
- greeting: Simple greetings → warm welcome (already handled by handle_greeting)
- small_talk: Casual conversation → personality response (no RAG)
- off_topic: Outside expertise → graceful redirect (no RAG)

Design Principles:
- Fast and cheap: Uses a simple LLM call with Haiku for cost efficiency
- Early routing: Prevents unnecessary RAG calls for non-knowledge queries
- Clear categories: Explicit intent types make routing decisions transparent

The capture state machines (crush confession, contact/lead capture) live in
assistant/flows/capture/ and are re-exported here for compatibility.
classify_intent() itself is a short dispatcher over ordered _check_* helpers;
each helper returns the handled state (early exit) or None (fall through).
"""

import logging
import re
from typing import Any
from anthropic import Anthropic
import os

from assistant.state.conversation_state import ConversationState

# ── Capture-flow API (moved to assistant/flows/capture in the stage1 split) ──
# Re-exported from this module for compatibility: conversation_nodes.py,
# stage2_role_routing.py, and the tests all import these names from here.
from assistant.flows.capture.constants import (  # noqa: F401
    _BUYING_SIGNAL_PATTERNS,
    _CAPTURE_QUESTION_FRAGMENTS,
    _CAPTURE_TRIGGER_FORM_PROMPT,
    _CONTACT_FORM_DETAILS_MARKER,
    _CONTACT_FORM_MARKER,
    _CONTACT_FORM_PROMPT,
    _CONTACT_FORM_RE,
    _CRUSH_COMPLETE_MARKERS,
    _CRUSH_FORM_MARKER,
    _CRUSH_FORM_PROMPT,
    _HM_CAPTURE_COMPLETE_MARKERS,
    _HM_CAPTURE_DETAILS_MARKER,
    _HM_CAPTURE_MARKER,
    _HM_SOFT_OFFER_MARKER,
    _VISITOR_QUESTION_PATTERNS,
    _compute_buying_signals,
    _compute_message_count,
    _compute_questions_asked_about_visitor,
)
from assistant.flows.capture.crush_flow import (  # noqa: F401
    _detect_crush_flow_from_history,
    _is_anonymous_choice,
    _is_cancel_choice,
    _is_reveal_choice,
    _looks_like_contact_info,
    _parse_crush_form,
    _parse_name_and_message,
    handle_crush_confession,
    handle_crush_flow_continuation,
)
from assistant.flows.capture.lead_capture import (  # noqa: F401
    _detect_hm_capture_flow_from_history,
    _is_capture_question_response,
    _is_connect_intent,
    _parse_hm_contact_info,
    _save_recruiter_lead,
    handle_hm_capture_continuation,
)
from assistant.flows.capture.notifications import (  # noqa: F401
    _send_crush_notifications,
    _send_hm_lead_notifications,
)

logger = logging.getLogger(__name__)


INTENT_CLASSIFICATION_PROMPT = """Classify the user's message into TWO values separated by a pipe (|):

1. Intent (one of):
- knowledge_query: Questions about Noah's background, skills, projects, experience, portfolio, OR questions about Portfolia (the AI assistant itself)
- crush_confession: User expressing romantic interest, asking Noah out, confessing feelings
- greeting: Simple greetings like "hi", "hello", "hey there" (single turn, no substance)
- small_talk: Casual conversation, jokes, commentary not about Noah's portfolio
- off_topic: Questions completely unrelated to Noah or software/data careers, OR personal/sensitive questions

2. Visitor signal (one of):
- hiring: mentions hiring, role, team, position, resume, interviewing, company, "we are building", availability, compensation, "looking for", "our stack", "opening"
- client: wants to hire Noah to BUILD something for them, "build this for us", "need an AI system", "looking for a developer", "freelance", "contract", "upwork", "can you build", "build something like this", "hire noah to build", "need something similar", "looking for someone to build", "consulting"
- crush: romantic language, heart emojis, "you're cute", flirting, attracted
- gatekeeper: screening for someone else, "my boss", "our hiring manager asked", "evaluating for my team", "forwarding this", "sharing with", "on behalf of"
- student: learning, studying, "I'm a student", "learning about RAG", "studying AI", "trying to understand", "building something similar", "class project", "school"
- casual: short messages, browsing language, "just looking", "checking this out", "cool", curiosity without business context
- neutral: can't determine visitor type from this message alone

IMPORTANT: Personal/private questions are off_topic (salary, address, dating status, etc.)
IMPORTANT: Questions about "you" referring to Portfolia ARE knowledge_query.
IMPORTANT: Mentioning a dating app as a traffic source (e.g., "I came from hinge") is NOT a crush confession. It's small_talk.

Examples:
- "Tell me about Noah's projects" → knowledge_query|neutral
- "We're looking for an AI engineer" → knowledge_query|hiring
- "What's his experience with our tech stack?" → knowledge_query|hiring
- "I have a crush on Noah" → crush_confession|crush
- "just checking this out" → small_talk|casual
- "hi" → greeting|neutral
- "How were you built?" → knowledge_query|neutral
- "Is he actively looking for new roles?" → knowledge_query|hiring
- "cool" → small_talk|casual
- "What's the weather like?" → off_topic|neutral
- "I came here from hinge" → small_talk|casual
- "found this on tinder" → small_talk|casual
- "I came here from ig" → small_talk|casual
- "I work for a company and am impressed" → knowledge_query|hiring
- "my manager asked me to look into this" → knowledge_query|gatekeeper
- "I'm learning about RAG and found this" → knowledge_query|student
- "this is for a class project" → knowledge_query|student
- "evaluating this for our hiring manager" → knowledge_query|gatekeeper
- "can Noah build something like this for my company?" → knowledge_query|client
- "I need someone to build an AI assistant" → knowledge_query|client
- "we're looking for someone to build an agentic system" → knowledge_query|client
- "is Noah available for freelance work?" → knowledge_query|client
- "found this on upwork" → knowledge_query|client
- "I want to hire Noah to build something similar" → knowledge_query|client

Respond with ONLY the two values separated by |, nothing else."""


def classify_intent(state: ConversationState) -> ConversationState:
    """Classify user message intent before RAG retrieval.

    Args:
        state: ConversationState with query field

    Returns:
        Updated state with:
        - message_intent: One of [knowledge_query, crush_confession, greeting, small_talk, off_topic]
        - skip_rag: Boolean flag (True for non-knowledge intents)

    Performance:
        - ~150ms (fast model call with Haiku or GPT-3.5)
        - Cached for repeated queries

    Dispatcher: each _check_* helper below runs in strict order and either
    returns the handled state (early exit) or None (fall through).
    """
    result = _check_crush_flow_continuation(state)
    if result is not None:
        return result

    result = _check_capture_continuation(state)
    if result is not None:
        return result

    _update_engagement_counters(state)

    query = state.get("query", "").strip()

    # If no query or already classified, skip
    if not query or state.get("message_intent"):
        return state

    for check in (
        _check_traffic_source,
        _check_crush_keywords,
        _check_direct_contact_request,
        _check_greeting,
        _check_menu_selection,
        _check_continuation,
        _check_self_knowledge_shortcuts,
        _check_topic_keywords,
        _check_self_referential,
        _check_short_reply,
        _check_form_submission,
    ):
        result = check(state, query)
        if result is not None:
            return result

    _classify_with_llm(state, query)

    result = _maybe_trigger_capture_offer(state, query)
    if result is not None:
        return result

    return state


def _check_crush_flow_continuation(state: ConversationState) -> ConversationState | None:
    """Recover crush flow state from chat_history and dispatch if active."""
    # ── Crush flow state recovery ────────────────────────────────────────
    # Reconstruct crush flow state from chat_history if not already set.
    # This is critical for serverless/stateless deployments where
    # awaiting_crush_choice and crush_flow_step don't persist across calls.
    if not state.get("awaiting_crush_choice") and not state.get("crush_flow_step"):
        detected_step = _detect_crush_flow_from_history(state)
        if detected_step:
            state["awaiting_crush_choice"] = True
            state["crush_flow_step"] = detected_step
            logger.info(f"Crush flow state recovered from chat_history: step={detected_step}")

    # Check if user is in crush flow - handle that first
    if state.get("awaiting_crush_choice") or state.get("crush_flow_step"):
        return handle_crush_flow_continuation(state)

    return None


def _check_capture_continuation(state: ConversationState) -> ConversationState | None:
    """Recover HM capture flow state from chat_history and dispatch if active."""
    # ── HM capture flow state recovery ───────────────────────────────────
    if not state.get("hm_capture_step"):
        detected_hm_step = _detect_hm_capture_flow_from_history(state)
        if detected_hm_step:
            state["hm_capture_step"] = detected_hm_step
            logger.info(f"HM capture flow state recovered from chat_history: step={detected_hm_step}")

    if state.get("hm_capture_step"):
        return handle_hm_capture_continuation(state)

    return None


def _update_engagement_counters(state: ConversationState) -> None:
    """Compute engagement counters and the soft-offer flag from chat_history."""
    # ── Compute engagement counters (stateless, from chat_history) ───────
    chat_history = state.get("chat_history", [])
    current_query = state.get("query", "")
    # Include current query in history for counter computation (it hasn't been appended yet)
    chat_with_current = chat_history + [{"role": "user", "content": current_query}] if current_query else chat_history
    counted_from_history = _compute_message_count(chat_with_current)
    # Use persistent counter to survive chat_history truncation (bounded memory).
    # Always increment stored_count — never let it decrease even if chat_history shrinks.
    session_memory = state.get("session_memory") or {}
    stored_count = session_memory.get("lifetime_message_count", 0)
    # Primary: increment stored counter each turn. Floor: counted_from_history.
    real_count = max(counted_from_history, stored_count + 1)
    state["message_count"] = real_count
    state.setdefault("session_memory", {})["lifetime_message_count"] = real_count
    logger.debug(
        f"msg_count: counted_from_history={counted_from_history}, "
        f"stored_count={stored_count}, real_count={real_count}"
    )
    state["questions_asked_about_visitor"] = _compute_questions_asked_about_visitor(chat_history)
    state["buying_signals_count"] = _compute_buying_signals(chat_with_current)

    # Detect soft offer from history (for reach-out offer dedup)
    if not state.get("hm_soft_offer_made"):
        for msg in chat_history:
            content = ""
            if isinstance(msg, dict):
                role = msg.get("role") or msg.get("type", "")
                content = msg.get("content", "")
            elif hasattr(msg, "content"):
                role = getattr(msg, "type", "") or getattr(msg, "role", "")
                content = getattr(msg, "content", "")
            if role in ("assistant", "ai") and _HM_SOFT_OFFER_MARKER in content:
                state["hm_soft_offer_made"] = True
                break


def _check_traffic_source(state: ConversationState, query: str) -> ConversationState | None:
    """Route traffic-source mentions ("I came from hinge") away from crush/capture."""
    # Traffic source detection — BEFORE crush keyword check and LLM classify.
    # "I came here from hinge" is NOT a crush confession.
    # "I am from linkedin" is NOT contact info.
    query_lower = query.lower()
    _traffic_source_phrases = [
        "came from hinge", "came from tinder", "came from bumble",
        "from hinge", "from tinder", "from bumble", "from a dating app",
        "on hinge", "on tinder", "on bumble",
        "from linkedin", "from twitter", "from instagram", "from ig",
        "from reddit", "from github", "from youtube", "from upwork",
        "i am from linkedin", "i'm from linkedin",
        "i am from twitter", "i'm from twitter",
        "i am from instagram", "i'm from instagram",
    ]
    if any(phrase in query_lower for phrase in _traffic_source_phrases):
        # Save traffic source to session_memory regardless of routing
        session_memory = state.get("session_memory") or {}
        session_memory["traffic_source"] = query.strip()
        state["session_memory"] = session_memory

        # If the message also contains a knowledge query, route through RAG
        _knowledge_signals = ("?", "what", "how", "tell me", "show me", "projects", "built")
        if any(sig in query_lower for sig in _knowledge_signals):
            logger.info(f"Traffic source with knowledge query — routing to RAG: {query[:50]}")
            state["message_intent"] = "knowledge_query"
            return state

        logger.info(f"Traffic source detected (not capture/crush): {query[:50]}")
        state["message_intent"] = "small_talk"
        state["skip_rag"] = True
        return state

    return None


def _check_crush_keywords(state: ConversationState, query: str) -> ConversationState | None:
    """Keyword-based crush confession detection (before the LLM call)."""
    query_lower = query.lower()
    # CRITICAL FIX: Keyword-based crush confession detection (before LLM call)
    # This catches cases where the LLM might misclassify obvious crush confessions
    crush_keywords = [
        "confess a crush",
        "confess my crush",
        "have a crush",
        "crush on noah",
        "crush on you",
        "ask noah out",
        "ask you out",
        "go out with",
        "date noah",
        "date you",
        "romantic interest",
        "wanna go out",
        "want to go out",
        "interested in noah",
        "attracted to"
    ]

    if any(keyword in query_lower for keyword in crush_keywords):
        logger.info(f"Crush confession detected via keywords: {query[:50]}")
        state["message_intent"] = "crush_confession"
        state["skip_rag"] = True
        state["visitor_type"] = "crush"
        return state

    return None


def _check_direct_contact_request(state: ConversationState, query: str) -> ConversationState | None:
    """Present the contact form for direct reach-out requests."""
    query_lower = query.lower()
    # ── Direct contact / reach-out request detection ─────────────────────
    _contact_phrases = [
        "reach out", "have noah reach out", "contact", "get in touch",
        "take my info", "take my data", "yes reach out",
    ]
    if any(phrase in query_lower for phrase in _contact_phrases):
        logger.info(f"Direct contact request detected via keywords: {query[:50]}")
        state["message_intent"] = "connect"
        state["skip_rag"] = True
        state["answer"] = _CONTACT_FORM_PROMPT
        state["hm_capture_step"] = "awaiting_hm_details"
        state["pipeline_halt"] = True
        return state

    return None


def _check_greeting(state: ConversationState, query: str) -> ConversationState | None:
    """Handle explicit greeting flags and simple keyword greetings."""
    # Skip if this is explicitly marked as a greeting (e.g., just "hi" or "hello")
    if state.get("is_greeting"):
        state["message_intent"] = "greeting"
        state["skip_rag"] = True
        return state

    # Check for simple greetings via keywords (don't waste LLM call)
    query_lower = query.lower().strip()
    greeting_phrases = ["hi", "hello", "hey", "hey there", "hi there", "hello there", "yo", "sup", "what's up", "howdy"]
    if query_lower in greeting_phrases:
        logger.info(f"Simple greeting detected via keywords: {query}")
        state["message_intent"] = "greeting"
        state["skip_rag"] = True
        return state

    return None


def _check_menu_selection(state: ConversationState, query: str) -> ConversationState | None:
    """Route single-digit menu picks to knowledge_query for stage-2 routing."""
    query_lower = query.lower().strip()
    # ── Menu selection detection ─────────────────────────────────────────
    # Single-digit "1"–"4" (or emoji variants) are menu picks, not off_topic.
    # Detect them here so they pass through to classify_role_mode in stage 2.
    _MENU_SELECTIONS = {"1", "2", "3", "4", "1️⃣", "2️⃣", "3️⃣", "4️⃣"}
    if query_lower in _MENU_SELECTIONS:
        logger.info(f"Menu selection detected pre-Haiku: {query}")
        state["message_intent"] = "knowledge_query"
        state["skip_rag"] = False
        return state

    return None


def _check_continuation(state: ConversationState, query: str) -> ConversationState | None:
    """Expand short continuations ("tell me more", "yes") with prior topic context."""
    query_lower = query.lower().strip()
    # ── Short continuation detection ────────────────────────────────────
    # Phrases like "tell me more", "go deeper", "yes", "continue" have no
    # semantic content and fail retrieval + edge case detection.  When the
    # previous assistant message was about a knowledge topic, treat these as
    # knowledge_query and expand the query with the previous topic.
    continuation_phrases = [
        "tell me more", "go deeper", "more", "continue", "keep going",
        "go on", "more details", "elaborate", "explain more", "yes",
        "yeah", "yep", "sure", "ok", "okay", "absolutely", "definitely",
        "go ahead", "please", "yes please", "tell me", "dig deeper",
        "what else", "and", "more about that", "more on that",
        "explain further", "go deeper into this", "go deeper on this",
        "go deeper on that", "more on this", "keep going on that",
        "tell me more about this", "tell me more about that",
        "and you?", "and you", "what about you",
    ]
    # Also match phrases that START with a continuation prefix,
    # but ONLY when the remainder is filler ("into this", "on that", etc.)
    # NOT when the user provides a real topic ("into the attrition model").
    continuation_prefixes = [
        "tell me more", "go deeper", "go deep", "more about", "more on",
        "explain further", "elaborate on", "dig deeper", "keep going",
        "continue with", "expand on",
    ]
    _filler_suffixes = {
        "", "this", "that", "it", "into this", "on this", "on that",
        "into that", "about this", "about that", "about it",
        "with this", "with that", "on it",
        "you", "your work", "yourself", "your architecture",
        "your pipeline", "your design", "into your work",
        "into you", "deeper into you", "about you",
    }
    # Filler suffixes that indicate the user is asking about Portfolia itself
    _self_referential_fillers = {
        "you", "your work", "yourself", "your architecture",
        "your pipeline", "your design", "into your work",
        "into you", "deeper into you", "about you",
    }
    prefix_match = False
    matched_filler = ""
    for prefix in continuation_prefixes:
        if query_lower.startswith(prefix):
            remainder = query_lower[len(prefix):].strip()
            if remainder in _filler_suffixes:
                prefix_match = True
                matched_filler = remainder
                break
    is_continuation = (
        query_lower in continuation_phrases
        or prefix_match
    )
    # Check if continuation is self-referential (about Portfolia itself)
    _continuation_is_self_ref = (
        matched_filler in _self_referential_fillers
        or query_lower in ("and you?", "and you", "what about you")
    )
    if is_continuation:
        chat_history = state.get("chat_history", [])
        if chat_history:
            # Find the last real user question (not another continuation)
            # to use as topic context for the expanded query
            for msg in reversed(chat_history):
                content = ""
                role = ""
                if isinstance(msg, dict):
                    role = msg.get("role") or msg.get("type", "")
                    content = msg.get("content", "")
                elif hasattr(msg, "content"):
                    role = getattr(msg, "type", None) or getattr(msg, "role", "")
                    content = getattr(msg, "content", "")

                if role in ("user", "human") and content:
                    # Skip if this user message is also a continuation
                    if content.lower().strip() in continuation_phrases:
                        continue
                    expanded = f"Go deeper on this topic: {content}"
                    logger.info(
                        f"Short continuation '{query}' expanded to: '{expanded[:80]}...'"
                    )
                    state["query"] = expanded
                    state["original_query"] = query
                    state["is_continuation"] = True
                    state["message_intent"] = "knowledge_query"
                    state["skip_rag"] = False
                    # Mark self-referential if the continuation targets Portfolia
                    if _continuation_is_self_ref:
                        state["is_self_referential"] = True
                        logger.info(f"Continuation is self-referential (filler='{matched_filler}')")
                    return state

            # Fallback: no previous non-continuation user message found.
            # Check if the last assistant message ended with a follow-up
            # question — if so, the user is responding to it.
            for msg in reversed(chat_history):
                content = ""
                role = ""
                if isinstance(msg, dict):
                    role = msg.get("role") or msg.get("type", "")
                    content = msg.get("content", "")
                elif hasattr(msg, "content"):
                    role = getattr(msg, "type", None) or getattr(msg, "role", "")
                    content = getattr(msg, "content", "")

                if role in ("assistant", "ai") and content:
                    if "?" in content:
                        logger.info(
                            f"Short continuation '{query}' after assistant follow-up, classified as knowledge_query"
                        )
                        state["message_intent"] = "knowledge_query"
                        state["skip_rag"] = False
                        state["is_continuation"] = True
                        if _continuation_is_self_ref:
                            state["is_self_referential"] = True
                        return state
                    break

        # Fallback: self-referential continuation without expandable history.
        # "go deeper into you" is a continuation about Portfolia itself —
        # route to knowledge_query with self-referential flag even without
        # a previous user message to expand from.
        if _continuation_is_self_ref:
            logger.info(
                f"Self-referential continuation '{query}' without expansion history — routing to knowledge_query"
            )
            state["message_intent"] = "knowledge_query"
            state["skip_rag"] = False
            state["is_continuation"] = True
            state["is_self_referential"] = True
            return state

    return None


def _check_self_knowledge_shortcuts(state: ConversationState, query: str) -> ConversationState | None:
    """Direct answers for model/purpose/GitHub questions that fail retrieval."""
    query_lower = query.lower().strip()
    # ── Quick self-knowledge answers ────────────────────────────────
    # Short queries about Portfolia's model/tech that fail pgvector retrieval
    # (too short, get 0 chunks). These deserve a direct answer.
    _model_phrases = [
        "what model", "which model", "what llm", "which llm",
        "what ai", "which ai model",
    ]
    if any(phrase in query_lower for phrase in _model_phrases):
        logger.info(f"Model question detected — direct self-knowledge answer: {query}")
        state["message_intent"] = "self_knowledge"
        state["skip_rag"] = True
        state["answer"] = (
            "I run on Anthropic Claude Sonnet 4.5 for generation and Claude Haiku for intent classification. "
            "My embeddings are OpenAI text-embedding-3-small (1536 dimensions) powering pgvector semantic search in Supabase. "
            "Want me to break down how the whole pipeline works, or curious about something specific?"
        )
        state["pipeline_halt"] = True
        return state

    # ── Purpose / why-built questions ────────────────────────────────
    # "What is your purpose?" and "Why did Noah build you?" fail pgvector
    # retrieval (too abstract, 0 chunks). Give a direct confident answer.
    _purpose_phrases = [
        "what is your purpose", "what's your purpose", "why were you built",
        "why did noah build you", "why did he build you", "what do you do",
        "why do you exist", "what are you for", "what's the point of you",
    ]
    if any(phrase in query_lower for phrase in _purpose_phrases):
        logger.info(f"Purpose question detected — direct self-knowledge answer: {query}")
        state["message_intent"] = "self_knowledge"
        state["skip_rag"] = True
        state["answer"] = (
            "I'm here to show you who Noah is and what he builds. Ask me anything — his work, his projects, "
            "his background. I know it all because he built me from scratch. I'm also a live demo of his "
            "engineering — every answer runs through a 22-node pipeline with semantic search, grounding "
            "validation, and quality gates. So while I'm telling you about Noah, I'm showing you what he can do."
        )
        state["pipeline_halt"] = True
        return state

    # ── GitHub / code link requests ──────────────────────────────────
    # When user asks to see code, GitHub, or the repo, give a direct
    # response with the link instead of running retrieval.
    _github_exact = {"github", "repo", "repository"}
    _github_phrases = [
        "source code", "see the code", "show me the code",
        "see actual code", "show code", "where's the code",
        "wheres the code", "where is the code", "can i see the code",
        "see his code", "check the code", "code link",
        "show me his code", "link to the code", "see your code",
        "see code", "view code", "view the code", "code please",
    ]
    query_stripped = query_lower.rstrip("?!.,")
    if (query_stripped in _github_exact or
            any(phrase in query_lower for phrase in _github_phrases)):
        logger.info(f"GitHub link request detected: {query}")
        state["message_intent"] = "github_link"
        state["skip_rag"] = True
        state["answer"] = (
            "Here's Noah's GitHub: https://github.com/iNoahCodeGuy — "
            "want me to walk through any specific project?"
        )
        state["pipeline_halt"] = True
        return state

    return None


def _check_topic_keywords(state: ConversationState, query: str) -> ConversationState | None:
    """Route single topic words and project names straight to knowledge_query."""
    query_lower = query.lower().strip()
    # ── Single-topic word detection ────────────────────────────────────
    # Short topic words that always map to knowledge_query about Noah.
    # These bypass the LLM classifier to avoid misclassification.
    _single_topic_words = {
        "projects", "tesla", "coaching", "skills", "background",
        "experience", "resume", "education", "certifications", "certs",
        "mma", "bjj", "work", "career", "portfolio",
        # Project-specific single words (prevent Haiku misclassification)
        "attrition", "heatmap", "segmentation", "clustering",
        "regression", "streamlit", "portfolia",
    }
    if query_lower in _single_topic_words:
        logger.info(f"Single-topic word classified as knowledge_query: {query}")
        state["message_intent"] = "knowledge_query"
        state["skip_rag"] = False
        return state

    # ── Project name detection ────────────────────────────────────────
    # Noah's project names must always route to knowledge_query.
    # Without this, Haiku misclassifies them as off_topic when the
    # assistant welcome message (which lists projects) isn't in Haiku's context.
    _project_name_phrases = [
        # Full and partial project names
        "lead response heatmap", "response heatmap", "lead heatmap",
        "employee attrition", "attrition model", "attrition prediction",
        "logistic regression", "naive bayes",
        "customer segmentation", "decision tree", "decision trees",
        "k-means", "kmeans", "k means",
        "response time analysis", "response time",
        "portfolia", "rag pipeline", "rag assistant",
        # Standalone project keywords (catch "the heatmap", "tell me about attrition", etc.)
        "heatmap", "attrition", "segmentation", "clustering",
    ]
    if any(phrase in query_lower for phrase in _project_name_phrases):
        logger.info(f"Project name detected — classified as knowledge_query: {query}")
        state["message_intent"] = "knowledge_query"
        state["skip_rag"] = False
        return state

    return None


def _check_self_referential(state: ConversationState, query: str) -> ConversationState | None:
    """Route queries about Portfolia itself to knowledge_query with the self flag."""
    query_lower = query.lower().strip()
    # ── Self-referential query detection ────────────────────────────────
    # Queries about Portfolia itself ("tell me about you", "your architecture")
    # should route to knowledge_query with skip_rag=False so the self-knowledge
    # injection in stage4's handle_grounding_gap() can provide the answer.
    _self_referential_markers = [
        "about you", "about yourself", "tell me about you", "explain yourself",
        "your architecture", "your pipeline", "your design", "your system",
        "your tech stack", "your retrieval", "your generation", "your nodes",
        "how were you", "how are you built", "how do you work", "how do you",
        "how you work", "how you were built", "how this works", "how does this work",
        "what are you", "who are you", "who built you", "what powers you",
        "describe yourself", "introduce yourself",
        # Personality / behavior / decision-making questions
        "your personality", "your voice", "your tone", "designed after",
        "who designed", "your behavior", "why don't you", "why aren't you",
        "your style", "how do you decide", "your purpose", "what are you for",
        # Meta questions about Portfolia's conversational behavior
        "shouldn't you", "shouldnt you", "why don't you ask",
        "why aren't you asking", "ask me", "ask about me",
        "do you care", "do you even care", "you should ask",
        "aren't you going to ask", "aren't you curious",
        "why haven't you", "why havent you",
        # Data handling / privacy questions
        "collect my data", "collect my information", "collect data",
        "what data", "my data", "my information",
        "do you collect", "do you track", "do you store",
        "privacy", "cookies", "tracking",
        "what do you store", "what do you collect",
        "are you going to collect", "store my data",
    ]
    _self_referential_exact = {"you", "yourself", "portfolia"}
    query_stripped = query_lower.rstrip("?!., ")
    if query_stripped in _self_referential_exact or any(
        marker in query_lower for marker in _self_referential_markers
    ):
        logger.info(f"Self-referential query detected — routing to knowledge_query: {query}")
        state["message_intent"] = "knowledge_query"
        state["skip_rag"] = False
        state["is_self_referential"] = True
        return state

    return None


def _check_short_reply(state: ConversationState, query: str) -> ConversationState | None:
    """Route brief replies to an assistant question straight to generation."""
    chat_history = state.get("chat_history", [])
    # ── Short conversational reply detection ──────────────────────────
    # When the user sends a brief reply (under 3 words) and the last
    # assistant message ended with a question, the user is answering
    # Portfolia — NOT going off-topic.  Route to generation with
    # skip_rag=True so Sonnet can use conversation history naturally.
    if len(query.split()) < 3:
        _last_asst_content = None
        for _msg in reversed(chat_history[-6:] if chat_history else []):
            _r = ""
            _c = ""
            if isinstance(_msg, dict):
                _r = _msg.get("role") or _msg.get("type", "")
                _c = _msg.get("content", "")
            elif hasattr(_msg, "content"):
                _r = getattr(_msg, "type", "") or getattr(_msg, "role", "")
                _c = getattr(_msg, "content", "")
            if _r in ("assistant", "ai") and _c:
                _last_asst_content = _c
                break
        if _last_asst_content and _last_asst_content.rstrip().endswith("?"):
            # Skip greeting/menu messages — their "?" is a menu prompt
            _MENU_INDICATORS = ("1️⃣", "2️⃣", "3️⃣", "4️⃣", "what brings you here")
            if not any(ind in _last_asst_content.lower() for ind in _MENU_INDICATORS):
                logger.info(
                    f"Short reply '{query}' after assistant question — routing to generation (skip_rag)"
                )
                state["message_intent"] = "knowledge_query"
                state["skip_rag"] = True
                return state

    return None


def _check_form_submission(state: ConversationState, query: str) -> ConversationState | None:
    """Detect contact intent / contact-info submissions before the LLM classifier."""
    query_lower = query.lower().strip()
    # ── Contact info submission pre-classifier ────────────────────────
    # Detect messages with email, phone, or explicit contact submission
    # BEFORE the LLM classifier to prevent misclassification as small_talk.
    _contact_submission_phrases = [
        "here is my info", "here's my info", "my info is",
        "my name is", "my email is", "my number is", "my phone is",
        "here is my email", "here's my email", "here is my number",
        "here's my number", "here are my details", "here's my details",
        "reach me at", "contact me at", "you can reach me",
    ]
    _has_email = bool(re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', query))
    _has_phone = bool(re.search(r'(?:\d[\d\s\-\(\)]{6,}\d)', query))
    _has_contact_phrase = any(phrase in query_lower for phrase in _contact_submission_phrases)

    # ── Contact intent detection (offer to collect info) ──────────────
    # "I want to contact Noah" → offer to collect their details,
    # rather than just handing over LinkedIn.
    _contact_intent_phrases = [
        "contact noah", "reach out to noah", "reach out to him",
        "have him contact", "have him reach out", "get in touch",
        "pass along my info", "give him my info", "connect me with noah",
        "i want to contact", "can i talk to", "talk to noah",
        "how do i reach", "how can i reach", "put me in touch",
    ]
    if (any(p in query_lower for p in _contact_intent_phrases)
            and not state.get("hm_capture_step")
            and not _has_email and not _has_phone):
        logger.info(f"Contact intent detected (offering to collect info): {query[:60]}")
        state["answer"] = (
            "Drop your name, email, and whatever context you want Noah to have. "
            "I'll make sure he sees it."
        )
        state["hm_capture_step"] = "awaiting_hm_details"
        state["message_intent"] = "knowledge_query"
        state["skip_rag"] = True
        state["pipeline_halt"] = True
        return state

    if _has_email or _has_phone or _has_contact_phrase:
        logger.info(f"Contact info submission detected: email={_has_email}, phone={_has_phone}, phrase={_has_contact_phrase}")
        session_id = state.get("session_id", "unknown")

        # Parse contact info from the message
        info = _parse_hm_contact_info(query)

        # Save to Supabase recruiter_leads table
        _save_recruiter_lead(session_id, info, state)

        # Send SMS to Noah
        _send_hm_lead_notifications(info)

        # Build confirmation response
        display = info.get("name") or info.get("email") or info.get("phone") or "your info"
        state["answer"] = (
            f"Got it — {display}'s details have been forwarded to Noah. "
            f"He'll follow up directly. Thanks for reaching out."
        )

        # Resume conversation naturally
        chat_history = state.get("chat_history", [])
        if chat_history:
            # Find last assistant message topic to bridge back
            for msg in reversed(chat_history):
                content = ""
                role = ""
                if isinstance(msg, dict):
                    role = msg.get("role") or msg.get("type", "")
                    content = msg.get("content", "")
                elif hasattr(msg, "content"):
                    role = getattr(msg, "type", "") or getattr(msg, "role", "")
                    content = getattr(msg, "content", "")
                if role in ("assistant", "ai") and content and "?" in content:
                    state["answer"] += " Anything else you want to know about Noah's work?"
                    break

        state["message_intent"] = "contact_info_submission"
        state["skip_rag"] = True
        state["pipeline_halt"] = True
        return state

    return None


def _classify_with_llm(state: ConversationState, query: str) -> None:
    """Classify intent + visitor signal with Haiku; mutates state, never halts."""
    chat_history = state.get("chat_history", [])
    # For all other messages, use the LLM classifier to determine intent + visitor signal
    try:
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Include last 3 user messages for visitor signal accumulation
        classification_messages = []
        recent_user_msgs = []
        for msg in reversed(chat_history):
            if isinstance(msg, dict):
                role = msg.get("role") or msg.get("type", "")
                content = msg.get("content", "")
            elif hasattr(msg, "content"):
                role = getattr(msg, "type", "") or getattr(msg, "role", "")
                content = getattr(msg, "content", "")
            else:
                continue
            if role in ("user", "human") and content:
                # Skip contact form submissions — they contain phrases like
                # "I love Noah" that cause Haiku to false-positive on crush.
                if not _CONTACT_FORM_RE.search(content):
                    recent_user_msgs.append(content[:200])
            if len(recent_user_msgs) >= 3:
                break
        # Build context: previous messages then current query
        if recent_user_msgs:
            context = "Previous messages: " + " | ".join(reversed(recent_user_msgs))
            classification_messages.append({"role": "user", "content": context})
            classification_messages.append({"role": "assistant", "content": "Noted. Send me the current message to classify."})
        classification_messages.append({"role": "user", "content": query})

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=30,
            temperature=0,
            system=INTENT_CLASSIFICATION_PROMPT,
            messages=classification_messages,
        )

        raw = response.content[0].text.strip().lower()

        # Parse two-field response: "intent|visitor_signal"
        parts = raw.split("|")
        intent = parts[0].strip()
        visitor_signal = parts[1].strip() if len(parts) > 1 else "neutral"

        # Validate intent
        valid_intents = ["knowledge_query", "crush_confession", "greeting", "small_talk", "off_topic"]
        if intent not in valid_intents:
            logger.warning(f"Unexpected intent classification: {intent}, defaulting to knowledge_query")
            intent = "knowledge_query"

        # Validate visitor signal
        valid_signals = ["hiring", "crush", "gatekeeper", "student", "casual", "neutral"]
        if visitor_signal not in valid_signals:
            visitor_signal = "neutral"

        state["message_intent"] = intent
        state["skip_rag"] = (intent != "knowledge_query")

        logger.info(
            f"Intent classified as: {intent} (skip_rag={state['skip_rag']}) | "
            f"visitor_signal={visitor_signal} | msg_count={state.get('message_count', 0)} | "
            f"buying_signals={state.get('buying_signals_count', 0)}"
        )

    except Exception as e:
        logger.error(f"Intent classification failed: {e}, defaulting to knowledge_query")
        state["message_intent"] = "knowledge_query"
        state["skip_rag"] = False


def _maybe_trigger_capture_offer(state: ConversationState, query: str) -> ConversationState | None:
    """Post-classification capture triggers: contact form and soft offer."""
    # ── Data capture trigger (universal) ─────────────────────────────────
    msg_count = state.get("message_count", 0)
    buying = state.get("buying_signals_count", 0)

    if (_is_connect_intent(query)
            and not state.get("hm_capture_step")
            and msg_count >= 2):
        state["answer"] = _CAPTURE_TRIGGER_FORM_PROMPT
        state["hm_capture_step"] = "awaiting_hm_details"
        state["pipeline_halt"] = True
        state["skip_rag"] = True
        logger.info(f"Capture form triggered: connect intent at msg_count={msg_count}")
        return state

    # ── Contact form capture trigger (all visitor types) ────────────────
    # If last assistant message had a capture question and user responds
    # with hiring/connect intent, ALWAYS present the contact form first.
    # Never try to parse contact info from a hiring intent message.
    if (not state.get("hm_capture_step")
            and msg_count >= 2
            and _is_capture_question_response(query, state)):
        state["answer"] = _CONTACT_FORM_PROMPT
        state["hm_capture_step"] = "awaiting_hm_details"
        state["pipeline_halt"] = True
        state["skip_rag"] = True
        logger.info("Contact form presented: user responded to capture question with intent")
        return state

    # Soft offer at message 5+ for any visitor with no buying signal
    if (msg_count >= 5
            and buying == 0
            and not state.get("hm_soft_offer_made")
            and not state.get("hm_capture_step")):
        state["hm_soft_offer_made"] = True
        logger.info("Soft offer flagged for injection at message %d", msg_count)

    return None


def handle_non_knowledge_intent(state: ConversationState, rag_engine: Any) -> ConversationState:
    """Handle messages that don't require RAG retrieval.

    Routes to appropriate handlers based on message_intent:
    - crush_confession → dedicated crush flow
    - small_talk → personality response with redirect
    - off_topic → graceful redirect to portfolio topics

    Args:
        state: ConversationState with message_intent field
        rag_engine: RAG engine (for potential future use)

    Returns:
        Updated state with answer and pipeline_halt=True
    """
    intent = state.get("message_intent", "knowledge_query")

    if intent == "knowledge_query":
        # Should not reach here, but pass through if it does
        return state

    if intent == "greeting":
        # Warm greeting with personality — works for first turn and mid-conversation
        if not state.get("answer"):
            state["answer"] = (
                "Hey! I'm Portfolia — Noah built me to show what he can do. "
                "Ask me about his projects, his background, or how I work under the hood. "
                "What are you curious about?"
            )
        state["pipeline_halt"] = True
        return state

    if intent == "github_link":
        # Already handled in classify_intent with answer set
        if state.get("answer"):
            return state
        state["answer"] = (
            "Here's Noah's GitHub: https://github.com/iNoahCodeGuy — "
            "want me to walk through any specific project?"
        )
        state["pipeline_halt"] = True
        return state

    if intent == "self_knowledge":
        # Already handled in classify_intent with answer set
        if state.get("answer"):
            return state
        return state

    if intent == "contact_info_submission":
        # Already handled in classify_intent with answer set
        if state.get("answer"):
            return state
        return state

    if intent == "crush_confession":
        # If crush flow continuation already handled this (answer set by
        # handle_crush_flow_continuation inside classify_intent), don't overwrite
        if state.get("pipeline_halt") and state.get("answer"):
            return state
        # New crush confession — show initial options
        return handle_crush_confession(state)

    if intent == "small_talk":
        # Context-aware small talk handling
        query_lower = (state.get("original_query", "") or state.get("query", "") or "").lower()
        chat_history = state.get("chat_history", [])
        msg_count = state.get("message_count", 0)

        # Check if user is answering a question Portfolia asked
        # BUT: skip greeting/menu messages — their "?" is a menu prompt,
        # not a genuine follow-up question. Without this exclusion,
        # every small_talk query on turn 2+ gets rerouted to knowledge_query
        # because the greeting ("What brings you here?") always contains "?".
        _GREETING_MENU_INDICATORS = ("1️⃣", "2️⃣", "3️⃣", "4️⃣", "what brings you here")
        _last_assistant_had_question = False
        for _m in reversed(chat_history[-4:] if chat_history else []):
            _r = ""
            _c = ""
            if isinstance(_m, dict):
                _r = _m.get("role") or _m.get("type", "")
                _c = _m.get("content", "")
            elif hasattr(_m, "content"):
                _r = getattr(_m, "type", "") or getattr(_m, "role", "")
                _c = getattr(_m, "content", "")
            if _r in ("assistant", "ai") and _c and "?" in _c:
                # Skip greeting/menu messages — they always have "?" but
                # the user's reply is not a knowledge answer
                if any(ind in _c.lower() for ind in _GREETING_MENU_INDICATORS):
                    break
                _last_assistant_had_question = True
                break

        if _last_assistant_had_question and "?" not in query_lower:
            # User is answering a question — reroute to generation
            state["message_intent"] = "knowledge_query"
            state["skip_rag"] = False
            return state

        # Traffic source detection — welcome visitors who mention where they came from
        _traffic_sources = [
            "linkedin", " ig", "instagram", "hinge", "tinder", "bumble",
            "upwork", "twitter", "referral", "reddit", "github", "youtube",
            "friend told", "someone told", "sent me", "showed me",
            "saw your", "saw his", "saw it on", "came from",
            "found on", "found this on", "came here from",
        ]
        if any(s in query_lower for s in _traffic_sources):
            state["answer"] = (
                "What caught your eye? I can go deep on Noah's projects, "
                "his background, or how I work under the hood."
            )
            state["pipeline_halt"] = True
            return state

        # Buying signal reroute — "I work for X" should not be small talk
        _buying_reroute = [
            "work for", "work at", "our company", "my company",
            "we're looking", "we are looking", "i'm with",
        ]
        if any(w in query_lower for w in _buying_reroute):
            state["message_intent"] = "knowledge_query"
            state["skip_rag"] = False
            return state

        # Compliment patterns — acknowledge briefly, bridge to uncovered content
        _compliment_words = ["cool", "impressive", "amazing", "awesome", "love this",
                             "great", "nice", "wow", "incredible", "sick", "dope",
                             "impressed", "really good", "this is cool", "i'm impressed"]
        if any(w in query_lower for w in _compliment_words):
            state["answer"] = (
                "Noted. There's more under the hood than what you've seen so far. "
                "the retrieval system and grounding validation are worth a look "
                "if you want to see how the engineering holds up."
            )
            state["pipeline_halt"] = True
            return state

        # Confusion detection — user doesn't know what this is
        _q_stripped = query_lower.strip().rstrip("?!. ")
        _confusion = {"what is this", "what is this thing", "huh",
                      "what am i looking at", "what does this do", "who is this"}
        if _q_stripped in _confusion or _q_stripped == "what":
            state["answer"] = (
                "I'm Portfolia, an AI assistant Noah built from scratch "
                "to demo his engineering. Ask me about his projects, "
                "his background, or how I work."
            )
            state["pipeline_halt"] = True
            return state

        # Early conversation — clean intro
        if msg_count <= 2:
            state["answer"] = (
                "I'm Noah's portfolio assistant. I know his projects, his background, "
                "and how I work under the hood. What are you curious about?"
            )
        else:
            # Mid-conversation — short redirect
            state["answer"] = (
                "My range is Noah's work and my own architecture. "
                "Pick a thread and I'll go as deep as you want."
            )
        state["pipeline_halt"] = True
        return state

    if intent == "off_topic":
        # Check if off_topic message actually contains buying signals — reroute
        query_lower_ot = (state.get("original_query", "") or state.get("query", "") or "").lower()
        _ot_buying = [
            "work for", "work at", "our company", "my company", "i'm with",
            "we're looking", "we are looking", "impressed", "really good",
            "this is cool", "i'm impressed",
        ]
        if any(w in query_lower_ot for w in _ot_buying):
            state["message_intent"] = "knowledge_query"
            state["skip_rag"] = False
            return state

        # Single specific suggestion, no menu
        state["answer"] = (
            "That's outside what I cover, but the retrieval architecture behind this "
            "conversation is worth a look if you're curious about how I work."
        )
        state["pipeline_halt"] = True
        return state

    # Fallback
    return state
