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
"""

import logging
import re
from typing import Any
from anthropic import Anthropic
import os

from assistant.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)


# ── Crush flow chat-history markers ──────────────────────────────────────────
# Used to reconstruct crush flow state from chat_history when state fields
# don't persist across API calls (serverless/stateless architecture).
_CRUSH_INITIAL_MARKER = "What's it gonna be?"
_CRUSH_REVEAL_MARKER = "Go ahead and tell me your name and a message"
_CRUSH_COMPLETE_MARKERS = ["Message sent", "Say less", "Noah knows"]

INTENT_CLASSIFICATION_PROMPT = """Classify the user's message into TWO values separated by a pipe (|):

1. Intent (one of):
- knowledge_query: Questions about Noah's background, skills, projects, experience, portfolio, OR questions about Portfolia (the AI assistant itself)
- crush_confession: User expressing romantic interest, asking Noah out, confessing feelings
- greeting: Simple greetings like "hi", "hello", "hey there" (single turn, no substance)
- small_talk: Casual conversation, jokes, commentary not about Noah's portfolio
- off_topic: Questions completely unrelated to Noah or software/data careers, OR personal/sensitive questions

2. Visitor signal (one of):
- hiring: mentions hiring, role, team, position, resume, interviewing, company, "we are building", availability, compensation, "looking for", "our stack", "opening"
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

Respond with ONLY the two values separated by |, nothing else."""


# ── Hiring manager data capture markers ──────────────────────────────────
_HM_CAPTURE_MARKER = "Would you like me to pass your info along to Noah?"
_HM_CAPTURE_DETAILS_MARKER = "fill this out so we can best assist you"
_HM_CAPTURE_COMPLETE_MARKERS = ["I'll make sure Noah sees this", "Info passed along"]
_HM_SOFT_OFFER_MARKER = "just let me know and I'll set it up"

# ── Contact form markers ─────────────────────────────────────────────────
# FRONTEND SPEC: When capture triggers, the web UI should render Name/Number/Email/Company/Additional
# as interactive form fields below Portfolia's message, not as plain text in the chat bubble.
# Terminal client uses plain text as fallback.
_CONTACT_FORM_MARKER = "fill this out so we can best assist you"
_CONTACT_FORM_DETAILS_MARKER = "Name:\nNumber:\nEmail:\nCompany:\nAdditional information:"

# Phrases in last assistant message that indicate a capture question was asked
_CAPTURE_QUESTION_FRAGMENTS = [
    "what brings you here", "what's your angle",
    "hiring, building, or just curious", "hiring, curiosity",
    "want to share what you're working on",
    "noah can follow up", "want noah to reach out",
    "say the word", "just let me know",
]


# ── Engagement counter computation ──────────────────────────────────────
# These functions compute state from chat_history each turn (stateless).

def _compute_message_count(chat_history: list) -> int:
    """Count user messages in chat_history."""
    count = 0
    for msg in chat_history:
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("type", "")
        elif hasattr(msg, "type"):
            role = getattr(msg, "type", "") or getattr(msg, "role", "")
        else:
            continue
        if role in ("user", "human"):
            count += 1
    return count


_VISITOR_QUESTION_PATTERNS = [
    r"what.*(?:role|position|team|company|building|looking for|hiring)",
    r"are you.*(?:hiring|building|looking|recruiting)",
    r"what brings you",
    r"may i ask",
    r"out of curiosity",
    r"what kind of (?:role|work|team|company)",
    r"who.*(?:team|company|organization)",
    r"are you in tech",
]


def _compute_questions_asked_about_visitor(chat_history: list) -> int:
    """Count how many times Portfolia asked about the VISITOR's context."""
    count = 0
    for msg in chat_history:
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("type", "")
            content = msg.get("content", "")
        elif hasattr(msg, "content"):
            role = getattr(msg, "type", "") or getattr(msg, "role", "")
            content = getattr(msg, "content", "")
        else:
            continue
        if role in ("assistant", "ai") and content and "?" in content:
            content_lower = content.lower()
            if any(re.search(p, content_lower) for p in _VISITOR_QUESTION_PATTERNS):
                count += 1
    return count


_BUYING_SIGNAL_PATTERNS = {
    "mentioned_hiring": r'\b(?:hiring|looking for|need someone|recruiting|open role|opening)\b',
    "described_role": r'\b(?:engineer|developer|architect|specialist|lead|analyst)\b',
    "team_context": r'\b(?:our team|my team|the team|our company|we\'re building|our stack)\b',
    "asked_timeline": r'\b(?:when available|start date|timeline|availability|how soon)\b',
    "budget_mentioned": r'\b(?:salary|compensation|budget|rate|benefits)\b',
    "intent_to_connect": r'\b(?:talk to him|reach him|get in touch|connect with|set something up|can he do a screen|how do i reach|how can i reach|contact him)\b',
    "actively_looking": r'\b(?:actively looking|open to opportunities|is he open|is he looking|available for)\b',
}


def _compute_buying_signals(chat_history: list) -> int:
    """Count distinct buying signals across all user messages."""
    all_user_text = ""
    for msg in chat_history:
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("type", "")
            content = msg.get("content", "")
        elif hasattr(msg, "content"):
            role = getattr(msg, "type", "") or getattr(msg, "role", "")
            content = getattr(msg, "content", "")
        else:
            continue
        if role in ("user", "human") and content:
            all_user_text += " " + content
    all_user_text = all_user_text.lower()

    signals = set()
    for signal_name, pattern in _BUYING_SIGNAL_PATTERNS.items():
        if re.search(pattern, all_user_text):
            signals.add(signal_name)
    return len(signals)


def _resolve_visitor_type(current: str, new_signal: str) -> str:
    """Resolve visitor_type from current state + new signal.

    Sticky rules:
    - Once hiring_manager or crush, never downgrade.
    - gatekeeper is semi-sticky: won't downgrade to casual/student/unknown,
      but can upgrade to hiring_manager.
    - student can upgrade to gatekeeper or hiring_manager.
    - casual can upgrade to hiring_manager, gatekeeper, or student.
    - unknown upgrades to anything.
    """
    if current in ("hiring_manager", "crush"):
        return current
    if new_signal == "hiring":
        return "hiring_manager"
    if new_signal == "crush":
        return "crush"
    if new_signal == "gatekeeper":
        if current in ("unknown", "casual", "student"):
            return "gatekeeper"
        return current  # gatekeeper doesn't override hiring_manager/crush
    if new_signal == "student":
        if current in ("unknown", "casual"):
            return "student"
        return current  # student doesn't override gatekeeper/hiring_manager/crush
    if new_signal == "casual":
        if current == "unknown":
            return "casual"
        return current  # casual doesn't override gatekeeper/student/hiring_manager/crush
    return current


def _detect_visitor_type_from_history(chat_history: list) -> str:
    """Recover visitor_type from chat_history by scanning for signal patterns.

    Used in serverless to reconstruct visitor_type without relying on
    persisted state fields.
    """
    all_user_text = ""
    for msg in chat_history:
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("type", "")
            content = msg.get("content", "")
        elif hasattr(msg, "content"):
            role = getattr(msg, "type", "") or getattr(msg, "role", "")
            content = getattr(msg, "content", "")
        else:
            continue
        if role in ("user", "human") and content:
            all_user_text += " " + content
    all_user_text = all_user_text.lower()

    # Check for crush signals
    crush_phrases = ["crush", "cute", "attractive", "handsome", "hot", "single",
                     "relationship", "❤", "😍", "🥰", "💕", "embarrassing but"]
    if any(phrase in all_user_text for phrase in crush_phrases):
        return "crush"

    # Check for hiring signals (need 2+ to be confident)
    hiring_count = 0
    hiring_phrases = ["hiring", "role", "position", "team", "our stack",
                      "we're building", "opening", "candidate", "looking for",
                      "company", "availability", "start date"]
    for phrase in hiring_phrases:
        if phrase in all_user_text:
            hiring_count += 1
    if hiring_count >= 2:
        return "hiring_manager"

    # Check for gatekeeper signals (before casual — "our team" overlaps with hiring)
    gatekeeper_phrases = ["my boss", "my manager", "our team asked", "forwarding",
                          "sharing with", "screening", "on behalf", "evaluating for",
                          "asked me to look", "asked me to check"]
    if any(phrase in all_user_text for phrase in gatekeeper_phrases):
        return "gatekeeper"

    # Check for student signals
    student_phrases = ["student", "learning about", "studying", "building something similar",
                       "school", "course", "class project", "trying to understand",
                       "learning how"]
    if any(phrase in all_user_text for phrase in student_phrases):
        return "student"

    # Check for casual signals
    casual_phrases = ["just looking", "checking this out", "just browsing", "cool"]
    if any(phrase in all_user_text for phrase in casual_phrases):
        return "casual"

    return "unknown"


def _detect_crush_flow_from_history(state: ConversationState) -> str | None:
    """Detect the current crush flow step by scanning chat_history for markers.

    Because state is rebuilt from scratch on each serverless invocation, we
    cannot rely on awaiting_crush_choice / crush_flow_step persisting. Instead
    we look at the last assistant message to determine where we are.

    Returns:
        'awaiting_choice' | 'awaiting_contact_info' | None
    """
    chat_history = state.get("chat_history", [])
    if not chat_history:
        return None

    # Find the last assistant message
    last_assistant_content = None
    for msg in reversed(chat_history):
        content = None
        role = None
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("type")
            content = msg.get("content", "")
        elif hasattr(msg, "content"):
            role = getattr(msg, "type", None) or getattr(msg, "role", None)
            content = getattr(msg, "content", "")

        if role in ("assistant", "ai"):
            last_assistant_content = content
            break

    if not last_assistant_content:
        return None

    # Check if crush flow already completed (don't re-enter)
    for marker in _CRUSH_COMPLETE_MARKERS:
        if marker in last_assistant_content:
            return None

    # Check if we're awaiting contact info (reveal path)
    if _CRUSH_REVEAL_MARKER in last_assistant_content:
        return "awaiting_contact_info"

    # Check if we're awaiting the 1/2 choice
    if _CRUSH_INITIAL_MARKER in last_assistant_content:
        return "awaiting_choice"

    return None


# ── HM data capture flow ────────────────────────────────────────────────
# Follows the same chat_history-marker pattern as the crush flow.

def _detect_hm_capture_flow_from_history(state: ConversationState) -> str | None:
    """Detect HM data capture step from chat_history markers.

    Returns:
        'awaiting_hm_response' | 'awaiting_hm_details' | None
    """
    chat_history = state.get("chat_history", [])
    if not chat_history:
        return None

    last_assistant_content = None
    for msg in reversed(chat_history):
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("type")
            content = msg.get("content", "")
        elif hasattr(msg, "content"):
            role = getattr(msg, "type", None) or getattr(msg, "role", None)
            content = getattr(msg, "content", "")
        else:
            continue
        if role in ("assistant", "ai"):
            last_assistant_content = content
            break

    if not last_assistant_content:
        return None

    # Check completion first
    for marker in _HM_CAPTURE_COMPLETE_MARKERS:
        if marker in last_assistant_content:
            return None

    if _HM_CAPTURE_DETAILS_MARKER in last_assistant_content:
        return "awaiting_hm_details"

    # Contact form marker (from new capture question flow)
    if _CONTACT_FORM_MARKER in last_assistant_content:
        return "awaiting_hm_details"

    if _HM_CAPTURE_MARKER in last_assistant_content:
        return "awaiting_hm_response"

    return None


def _parse_hm_contact_info(query: str) -> dict:
    """Parse hiring manager contact info from free-form input.

    Returns dict with keys: name, email, phone, company, message.
    Any value can be None if not detected.
    """
    info = {"name": None, "email": None, "phone": None, "company": None, "message": None}

    # Extract email
    email_match = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', query)
    if email_match:
        info["email"] = email_match.group()

    # Extract phone
    phone_match = re.search(r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', query)
    if phone_match:
        info["phone"] = phone_match.group()

    # Extract name — "my name is X", "I'm X", "this is X", "Name, ..."
    name_match = re.match(
        r"(?:my name is|i'm|i am|this is|it's)\s+(\w+(?:\s+\w+)?)",
        query, re.IGNORECASE,
    )
    if name_match:
        info["name"] = name_match.group(1).strip()
    elif not info["email"] and not info["phone"]:
        # If first words look like a name (1-3 capitalized words before a comma)
        parts = query.split(",", 1)
        if len(parts) == 2 and len(parts[0].strip().split()) <= 3:
            info["name"] = parts[0].strip()

    # Extract company — "at X", "from X", "X company"
    company_match = re.search(
        r'(?:at|from|with)\s+([A-Z][\w\s&.-]{1,30}?)(?:\s*[,.]|\s+(?:and|we|our|i|my|looking|hiring)|\s*$)',
        query, re.IGNORECASE,
    )
    if company_match:
        info["company"] = company_match.group(1).strip()

    # Remaining text is the message
    remaining = query
    for val in [info["email"], info["phone"], info["name"]]:
        if val:
            remaining = remaining.replace(val, "").strip()
    remaining = re.sub(r'^[\s,.-]+|[\s,.-]+$', '', remaining)
    if remaining and len(remaining) > 3:
        info["message"] = remaining

    return info


def _save_recruiter_lead(session_id: str, info: dict, state: ConversationState) -> bool:
    """Save hiring manager lead to Supabase recruiter_leads table."""
    try:
        from assistant.config.supabase_config import get_supabase_client
        supabase = get_supabase_client()
        supabase.table('recruiter_leads').insert({
            'session_id': session_id,
            'name': info.get('name'),
            'email': info.get('email'),
            'phone': info.get('phone'),
            'company': info.get('company'),
            'message': info.get('message'),
            'visitor_type': state.get('visitor_type', 'hiring_manager'),
            'buying_signals_count': state.get('buying_signals_count', 0),
            'message_count': state.get('message_count', 0),
            'capture_trigger': 'intent_to_connect',
        }).execute()
        logger.info(f"Recruiter lead saved for session {session_id}: {info.get('name')}")
        return True
    except Exception as e:
        logger.error(f"Failed to save recruiter lead: {e}")
        return False


def _send_hm_lead_sms(info: dict) -> bool:
    """Send Twilio SMS to Noah about a new hiring manager lead."""
    try:
        from assistant.services.twilio_service import get_twilio_service
        twilio = get_twilio_service()
        noah_phone = os.getenv("NOAH_PHONE_NUMBER")

        if not noah_phone or not twilio or not twilio.enabled:
            logger.warning("Twilio not configured — skipping HM lead SMS")
            return False

        name = info.get('name') or 'Unknown'
        company = info.get('company') or 'Unknown company'
        email = info.get('email') or ''
        phone = info.get('phone') or ''
        msg = info.get('message') or ''

        sms_body = f"💼 Portfolia Lead: {name} at {company}"
        if email:
            sms_body += f"\nEmail: {email}"
        if phone:
            sms_body += f"\nPhone: {phone}"
        if msg:
            sms_body += f"\nMessage: {msg[:200]}"

        if len(sms_body) > 1600:
            sms_body = sms_body[:1597] + "..."

        result = twilio.send_sms(to_phone=noah_phone, message=sms_body)
        logger.info(f"HM lead SMS sent to Noah: {result.get('status', 'unknown')}")
        return True
    except Exception as e:
        logger.error(f"Failed to send HM lead SMS: {e}")
        return False


def _is_connect_intent(query: str) -> bool:
    """Check if user expresses intent to connect with Noah."""
    q = query.lower()
    connect_phrases = [
        "talk to him", "reach him", "get in touch", "connect with",
        "set something up", "can he do a screen", "how do i reach",
        "how can i reach", "contact him", "connect directly",
        "let's set up", "schedule", "i'd like to talk",
        "want to connect", "reach out to him", "reach out to noah",
        "set up a call", "put me in touch", "pass my info",
        "yes", "sure", "yeah", "please", "do it",
    ]
    return any(phrase in q for phrase in connect_phrases)


def _is_capture_question_response(query: str, state: ConversationState) -> bool:
    """Check if user is responding to a capture question with connect/hiring intent.

    Returns True if:
    1. The last assistant message contained a capture question fragment, AND
    2. The user's response indicates hiring intent, company mention, or
       explicit interest in connecting.
    """
    # First, check if last assistant message had a capture question
    chat_history = state.get("chat_history", [])
    last_assistant = ""
    for msg in reversed(chat_history):
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("type", "")
            content = msg.get("content", "")
        elif hasattr(msg, "content"):
            role = getattr(msg, "type", "") or getattr(msg, "role", "")
            content = getattr(msg, "content", "")
        else:
            continue
        if role in ("assistant", "ai") and content:
            last_assistant = content.lower()
            break

    if not last_assistant:
        return False

    had_capture_question = any(
        frag in last_assistant for frag in _CAPTURE_QUESTION_FRAGMENTS
    )
    if not had_capture_question:
        return False

    # Now check if user's response signals hiring/connect intent
    q = query.lower().strip()

    # Direct connect intent
    if _is_connect_intent(q):
        return True

    # Hiring/recruitment signals in response
    # Traffic source phrases that should NOT be treated as hiring signals
    _traffic_source_phrases = [
        "from linkedin", "from twitter", "from instagram", "from ig",
        "from reddit", "from github", "from youtube", "from upwork",
        "from hinge", "from tinder", "from bumble",
    ]
    if any(ts in q for ts in _traffic_source_phrases):
        return False

    hiring_signals = [
        "hiring", "recruiter", "recruiting", "we're looking",
        "we are looking", "open role", "open position",
        "evaluating candidates", "data analyst", "data engineer",
        "software engineer", "developer role", "team is hiring",
        "interested in him", "interested in noah",
        "our company", "our team", "my company", "my team",
    ]
    if any(signal in q for signal in hiring_signals):
        return True

    return False


def handle_hm_capture_continuation(state: ConversationState) -> ConversationState:
    """Handle hiring manager data capture flow steps.

    Follows the same state machine pattern as handle_crush_flow_continuation.
    """
    query = state.get("query", "").strip()
    step = state.get("hm_capture_step")
    session_id = state.get("session_id", "unknown")

    state["skip_rag"] = True
    state["message_intent"] = "knowledge_query"

    # ── Cancel detection ─────────────────────────────────────────────────
    cancel_phrases = ["no", "nah", "no thanks", "not right now", "maybe later",
                      "i'm good", "im good", "skip", "pass"]
    if query.lower().strip() in cancel_phrases or "not interested" in query.lower():
        state["answer"] = (
            "No worries at all. If you change your mind, I'm always here. "
            "Anything else you want to know about Noah's work?"
        )
        state["hm_capture_step"] = None
        state["pipeline_halt"] = True
        logger.info("HM capture flow declined by visitor")
        return state

    # ── Step: awaiting_hm_response (user deciding yes/no) ────────────────
    if step == "awaiting_hm_response":
        if _is_connect_intent(query):
            # User said yes — always present the contact form, never parse info here
            state["answer"] = (
                "I can have Noah reach out — fill this out so we can best assist you:\n\n"
                "Name:\nNumber:\nEmail:\nCompany:\nAdditional information:"
            )
            state["hm_capture_step"] = "awaiting_hm_details"
            state["pipeline_halt"] = True
            return state
        else:
            # Not a clear yes or no — treat as normal conversation, exit capture flow
            state["hm_capture_step"] = None
            state["skip_rag"] = False
            state["message_intent"] = None  # Let classify_intent re-classify
            return state

    # ── Step: awaiting_hm_details (collecting name/email/company) ────────
    if step == "awaiting_hm_details":
        info = _parse_hm_contact_info(query)

        if info.get("name") or info.get("email") or info.get("phone"):
            _save_recruiter_lead(session_id, info, state)
            _send_hm_lead_sms(info)
            display = info.get("name") or info.get("email") or "your info"
            state["answer"] = (
                f"I'll make sure Noah sees this. {display}'s details have been forwarded — "
                f"he'll follow up directly. Thanks for checking out his work."
            )
            state["hm_capture_step"] = None
            state["message_intent"] = "contact_info_submission"
            state["pipeline_halt"] = True
        else:
            state["answer"] = (
                "I just need at least a name or email so Noah can reach back out. "
                "Something like: \"Alex, alex@company.com — interested in chatting about the data role.\""
            )
            state["pipeline_halt"] = True

        return state

    # Fallback
    state["hm_capture_step"] = None
    return state


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
    """
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

    # ── HM capture flow state recovery ───────────────────────────────────
    if not state.get("hm_capture_step"):
        detected_hm_step = _detect_hm_capture_flow_from_history(state)
        if detected_hm_step:
            state["hm_capture_step"] = detected_hm_step
            logger.info(f"HM capture flow state recovered from chat_history: step={detected_hm_step}")

    if state.get("hm_capture_step"):
        return handle_hm_capture_continuation(state)

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

    # Infer visitor_type from explicit role selection (most reliable signal)
    if state.get("visitor_type", "unknown") == "unknown":
        role = state.get("role", "").lower()
        if "hiring" in role or "recruiter" in role:
            state["visitor_type"] = "hiring_manager"
            logger.info(f"Visitor type inferred from role: hiring_manager (role='{role}')")
        elif "crush" in role:
            state["visitor_type"] = "crush"
            logger.info(f"Visitor type inferred from role: crush (role='{role}')")
        elif role in ("just looking around", "explorer", "casual"):
            state["visitor_type"] = "casual"
            logger.info(f"Visitor type inferred from role: casual (role='{role}')")

    # Recover visitor_type from session_memory (persists Haiku classifications
    # like gatekeeper/student that history-based recovery can't detect)
    if state.get("visitor_type", "unknown") == "unknown":
        stored_vtype = session_memory.get("visitor_type")
        if stored_vtype and stored_vtype != "unknown":
            state["visitor_type"] = stored_vtype
            logger.info(f"Visitor type recovered from session_memory: {stored_vtype}")

    # Recover visitor_type from history if still unknown (serverless)
    if state.get("visitor_type", "unknown") == "unknown" and chat_history:
        recovered = _detect_visitor_type_from_history(chat_history)
        if recovered != "unknown":
            state["visitor_type"] = recovered
            logger.info(f"Visitor type recovered from chat_history: {recovered}")

    # Detect soft offer from history
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

    query = state.get("query", "").strip()

    # If no query or already classified, skip
    if not query or state.get("message_intent"):
        return state

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
        logger.info(f"Traffic source detected (not capture/crush): {query[:50]}")
        state["message_intent"] = "small_talk"
        state["skip_rag"] = True
        return state

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

    # ── Menu selection detection ─────────────────────────────────────────
    # Single-digit "1"–"4" (or emoji variants) are menu picks, not off_topic.
    # Detect them here so they pass through to classify_role_mode in stage 2.
    _MENU_SELECTIONS = {"1", "2", "3", "4", "1️⃣", "2️⃣", "3️⃣", "4️⃣"}
    if query_lower in _MENU_SELECTIONS:
        logger.info(f"Menu selection detected pre-Haiku: {query}")
        state["message_intent"] = "knowledge_query"
        state["skip_rag"] = False
        return state

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
            "engineering — every answer runs through a 21-node pipeline with semantic search, grounding "
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

    # ── Single-topic word detection ────────────────────────────────────
    # Short topic words that always map to knowledge_query about Noah.
    # These bypass the LLM classifier to avoid misclassification.
    _single_topic_words = {
        "projects", "tesla", "coaching", "skills", "background",
        "experience", "resume", "education", "certifications", "certs",
        "mma", "bjj", "work", "career", "portfolio",
    }
    if query_lower in _single_topic_words:
        logger.info(f"Single-topic word classified as knowledge_query: {query}")
        state["message_intent"] = "knowledge_query"
        state["skip_rag"] = False
        return state

    # ── Self-referential query detection ────────────────────────────────
    # Queries about Portfolia itself ("tell me about you", "your architecture")
    # should route to knowledge_query with skip_rag=False so the self-knowledge
    # injection in stage4's handle_grounding_gap() can provide the answer.
    _self_referential_markers = [
        "about you", "about yourself", "tell me about you", "explain yourself",
        "your architecture", "your pipeline", "your design", "your system",
        "your tech stack", "your retrieval", "your generation", "your nodes",
        "how were you", "how are you built", "how do you work", "how do you",
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
        _send_hm_lead_sms(info)

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

        # Resolve visitor_type with sticky rules
        current_vtype = state.get("visitor_type", "unknown")
        state["visitor_type"] = _resolve_visitor_type(current_vtype, visitor_signal)

        # Also check current query for buying signals
        query_buying = 0
        for pattern in _BUYING_SIGNAL_PATTERNS.values():
            if re.search(pattern, query_lower):
                query_buying += 1
        if query_buying > 0 and state["visitor_type"] != "hiring_manager":
            # Current query alone has hiring signals — upgrade
            state["visitor_type"] = _resolve_visitor_type(state["visitor_type"], "hiring")

        logger.info(
            f"Intent classified as: {intent} (skip_rag={state['skip_rag']}) | "
            f"visitor_type={state['visitor_type']} | msg_count={state.get('message_count', 0)} | "
            f"buying_signals={state.get('buying_signals_count', 0)}"
        )

    except Exception as e:
        logger.error(f"Intent classification failed: {e}, defaulting to knowledge_query")
        state["message_intent"] = "knowledge_query"
        state["skip_rag"] = False

    # ── HM data capture trigger ──────────────────────────────────────────
    # Check if eligible hiring manager is expressing intent to connect
    msg_count = state.get("message_count", 0)
    buying = state.get("buying_signals_count", 0)
    vtype = state.get("visitor_type", "unknown")

    if (vtype == "hiring_manager"
            and msg_count >= 6
            and buying >= 1
            and _is_connect_intent(query)
            and not state.get("hm_capture_step")):
        state["answer"] = (
            "Absolutely. Would you like me to pass your info along to Noah? "
            "He's responsive and usually follows up within a day."
        )
        state["hm_capture_step"] = "awaiting_hm_response"
        state["pipeline_halt"] = True
        state["skip_rag"] = True
        logger.info("HM capture flow triggered: eligible hiring manager expressed connect intent")
        return state

    # ── Contact form capture trigger (all visitor types) ────────────────
    # If last assistant message had a capture question and user responds
    # with hiring/connect intent, ALWAYS present the contact form first.
    # Never try to parse contact info from a hiring intent message.
    if (not state.get("hm_capture_step")
            and msg_count >= 2
            and _is_capture_question_response(query, state)):
        state["answer"] = (
            "I can have Noah reach out — fill this out so we can best assist you:\n\n"
            "Name:\nNumber:\nEmail:\nCompany:\nAdditional information:"
        )
        state["hm_capture_step"] = "awaiting_hm_details"
        state["pipeline_halt"] = True
        state["skip_rag"] = True
        logger.info("Contact form presented: user responded to capture question with intent")
        return state

    # Soft offer at message 10+ for hiring managers with no buying signal
    if (vtype == "hiring_manager"
            and msg_count >= 10
            and buying == 0
            and not state.get("hm_soft_offer_made")
            and not state.get("hm_capture_step")):
        # Don't halt the pipeline — just mark that we should inject the soft offer
        # The soft offer text will be injected in the generation prompt
        state["hm_soft_offer_made"] = True
        logger.info("HM soft offer flagged for injection at message 10+")

    # Persist visitor_type to session_memory so it survives across turns
    # (history-based recovery can't detect gatekeeper/student classifications)
    final_vtype = state.get("visitor_type", "unknown")
    if final_vtype != "unknown":
        state.setdefault("session_memory", {})["visitor_type"] = final_vtype

    return state


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
                "Noted. There's more under the hood than what you've seen so far -- "
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
                "I'm Portfolia -- an AI assistant Noah built from scratch "
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


def handle_crush_confession(state: ConversationState) -> ConversationState:
    """Handle crush confession — Step 1: Show anonymous/reveal options.

    Args:
        state: ConversationState

    Returns:
        Updated state with crush confession prompt and flags
    """
    state["answer"] = (
        "Wait... for real?? 👀 Okay I wasn't expecting anyone to actually pick this but I respect the energy.\n\n"
        "I can let Noah know someone came through with intentions. But first — how do you want to play this?\n\n"
        "**1️⃣ 🕵️ Stay anonymous** — I'll tell him he's got a secret admirer\n"
        "**2️⃣ 😏 Reveal yourself** — drop your name and a message for Noah, and I'll pass it along\n\n"
        "What's it gonna be?"
    )

    # Set flags for next turn handling
    state["awaiting_crush_choice"] = True
    state["crush_flow_step"] = "awaiting_choice"
    state["pipeline_halt"] = True
    state["skip_rag"] = True
    state["message_intent"] = "crush_confession"

    logger.info("Crush confession detected - presented options to user")

    return state


def _is_cancel_choice(query: str) -> bool:
    """Check if the user wants to cancel the crush flow."""
    q = query.lower().strip()
    # Short words — exact match only to avoid false positives ("no" in "anonymous")
    exact_only = {"no", "nah", "nope", "jk", "back", "stop", "exit", "wrong"}
    # Longer phrases — safe for substring matching
    substring_ok = [
        "nevermind", "never mind", "nvm", "cancel", "just kidding",
        "forget it", "changed my mind", "go back", "not that",
        "professionally", "not a crush", "not what i meant",
        "not romantic", "not interested", "wrong option",
    ]
    if q in exact_only:
        return True
    return any(signal in q for signal in substring_ok)


def _is_anonymous_choice(query: str) -> bool:
    """Check if the user chose the anonymous option."""
    q = query.lower().strip()
    # Exact-match-only signals (single chars that cause false positives in substrings)
    exact_only = {"1"}
    # Signals that can use substring matching (multi-word phrases)
    substring_ok = [
        "anonymous", "stay anonymous", "secret", "secret admirer",
        "option 1", "first", "first one", "the first",
    ]
    if q in exact_only:
        return True
    return any(signal == q or signal in q for signal in substring_ok)


def _is_reveal_choice(query: str) -> bool:
    """Check if the user chose the reveal option."""
    q = query.lower().strip()
    # Exact-match-only signals (single chars that cause false positives in substrings)
    exact_only = {"2"}
    # Signals that can use substring matching
    substring_ok = [
        "reveal", "reveal myself", "reveal yourself",
        "option 2", "second", "second one", "the second",
        "drop my name", "full send", "bold",
    ]
    if q in exact_only:
        return True
    return any(signal == q or signal in q for signal in substring_ok)


def _looks_like_contact_info(query: str) -> bool:
    """Check if a message looks like it contains contact info or a direct reveal.

    Used in the awaiting_choice step to detect that the user is providing
    contact info directly (implicitly choosing "reveal") instead of saying
    "1" or "2".

    Matches: phone numbers, emails, "my name is", "tell noah/him", "call me",
    "here is my number", "hit me up", name + number patterns, etc.
    """
    q = query.lower().strip()
    # Phone number pattern (7+ digits, allowing dashes/spaces/parens)
    if re.search(r'\d[\d\s\-\(\)]{6,}', query):
        return True
    # Email pattern
    if re.search(r'\S+@\S+\.\S+', query):
        return True
    # Name introduction patterns (exclude "i'm from/on/just/here" — traffic sources)
    if re.search(r"(?:my name is|i'm (?!from |on |just |here )|i am (?!from |on |just |here )|this is |it's |call me )", q):
        return True
    # Messaging-Noah patterns
    if re.search(r"(?:tell (?:noah|him)|let (?:noah|him) know|hit me up|message him|reach me)", q):
        return True
    # Social media handles
    if re.search(r'@\w{2,}', query):
        return True
    # "here is my number/contact/info"
    if re.search(r"(?:here(?:'s| is) my|my (?:number|phone|contact|email|ig|insta|snap))", q):
        return True
    return False


def _parse_name_and_message(query: str) -> tuple[str | None, str | None]:
    """Extract name and message from free-form user input.

    Accepts formats like:
    - "my name is Sarah and tell him he seems really cool"
    - "Sarah, tell him he's awesome"
    - "I'm Mike and my message is you seem great"
    - "Sarah, 702-555-1234"
    - "Sarah, sarah@email.com"
    - "tell noah to call me 707-319-0951"
    - "noah is so hot, here is my number 707-319-0951"
    - "tell him to hit me up, my name is Sarah 707-319-0951"

    Returns:
        (name, message) tuple. Either or both can be None if not parseable.
    """
    text = query.strip()

    # Try "my name is X and ..." pattern
    match = re.match(
        r"(?:my name is|i'm|i am|this is|it's|its)\s+(\w+(?:\s+\w+)?)\s*(?:and|,|\.)\s*(.*)",
        text, re.IGNORECASE,
    )
    if match:
        return match.group(1).strip(), match.group(2).strip() or None

    # Try "... my name is X ..." anywhere in the message (not just start)
    # [a-zA-Z] ensures we don't capture a phone number after "call me"
    match = re.search(
        r"(?:my name is|i'm|i am|call me)\s+([a-zA-Z]\w*)",
        text, re.IGNORECASE,
    )
    if match:
        name = match.group(1).strip()
        # Rest of the message is the contact/message
        rest = text[:match.start()].strip().rstrip(",").strip()
        rest2 = text[match.end():].strip().lstrip(",").strip()
        message = " ".join(filter(None, [rest, rest2])).strip() or None
        return name, message

    # Try "Name, message" pattern (comma separated)
    parts = text.split(",", 1)
    if len(parts) == 2 and len(parts[0].strip().split()) <= 3:
        return parts[0].strip(), parts[1].strip() or None

    # Try "Name - message" pattern
    parts = text.split(" - ", 1)
    if len(parts) == 2 and len(parts[0].strip().split()) <= 3:
        return parts[0].strip(), parts[1].strip() or None

    # If text is short (likely just a name), treat as name with no message
    if len(text.split()) <= 3 and not any(c in text for c in "?!.@"):
        return text, None

    # Can't parse a name — treat entire thing as a message
    return None, text


def _send_crush_sms(name: str | None = None, message: str | None = None, anonymous: bool = False) -> bool:
    """Send Twilio SMS to NOAH about a crush confession.

    The TO number is ALWAYS NOAH_PHONE_NUMBER from env.
    The user's contact info goes in the MESSAGE BODY, never in the TO field.

    Args:
        name: Confessor's name (None for anonymous).
        message: The user's raw message / contact info (None for anonymous).
        anonymous: True for anonymous confession.

    Returns True if SMS sent successfully, False otherwise.
    """
    try:
        from assistant.services.twilio_service import get_twilio_service

        twilio = get_twilio_service()
        noah_phone = os.getenv("NOAH_PHONE_NUMBER")

        if not noah_phone:
            logger.error("NOAH_PHONE_NUMBER not set in env — crush SMS cannot be sent")
            return False

        if not twilio or not twilio.enabled:
            logger.warning("Twilio not configured — skipping crush SMS")
            return False

        if anonymous:
            sms_body = "Portfolia Alert 💌 You've got a secret admirer browsing your portfolio!"
        else:
            sms_body = f"Portfolia Alert 💌 Someone on your portfolio said: {message}"
            if name and name != "Someone":
                sms_body += f"\nContact: {name}"
        if len(sms_body) > 1600:
            sms_body = sms_body[:1597] + "..."

        logger.info(f"Sending crush SMS to NOAH ({noah_phone})...")
        result = twilio.send_sms(to_phone=noah_phone, message=sms_body)
        logger.info(f"Crush SMS sent to NOAH successfully: {result.get('status', 'unknown')}")
        return True

    except Exception as e:
        logger.error(f"Failed to send crush SMS: {e}")
        return False


def handle_crush_flow_continuation(state: ConversationState) -> ConversationState:
    """Handle subsequent steps in the crush confession flow.

    Called when user is in the middle of the crush flow (after initial prompt).

    Args:
        state: ConversationState with crush_flow_step indicator

    Returns:
        Updated state with appropriate response for current step
    """
    query = state.get("query", "").strip()
    step = state.get("crush_flow_step")
    session_id = state.get("session_id", "unknown")

    # Ensure skip_rag and message_intent are set for all crush flow continuations
    state["skip_rag"] = True
    state["message_intent"] = "crush_confession"

    # ── Cancel / escape detection (any step) ────────────────────────────
    if _is_cancel_choice(query):
        state["answer"] = "No worries — back to the regular conversation. What can I help you with?"
        state["crush_flow_step"] = None
        state["awaiting_crush_choice"] = False
        state["pipeline_halt"] = True
        logger.info("Crush flow cancelled by user")
        return state

    # ── Escape detection: confusion or Noah-related queries ────────────
    # If the user asks about Noah or seems confused, exit crush flow
    # silently and let the pipeline re-classify their message normally.
    _escape_noah_kw = [
        "noah", "background", "projects", "technical", "professional",
        "skills", "experience", "work", "portfolio", "resume", "job",
        "built", "engineering", "architecture", "pipeline",
    ]
    _escape_confusion = {
        "what", "huh", "pick what", "what do you mean",
        "what is this", "what are you", "i don't understand",
    }
    q_low = query.lower().strip()
    is_noah_query = any(kw in q_low for kw in _escape_noah_kw)
    is_confused = (
        q_low in _escape_confusion
        or q_low.startswith("show me")
        or q_low.startswith("tell me about")
        or (len(q_low.split()) <= 4 and "?" in query
            and not any(w in q_low for w in ["anonymous", "reveal", "crush", "admirer"]))
    )
    if is_noah_query or is_confused:
        logger.info(f"Crush flow escape: '{query[:60]}' — exiting to normal pipeline")
        state["crush_flow_step"] = None
        state["awaiting_crush_choice"] = False
        state["message_intent"] = None
        state["skip_rag"] = False
        return state

    # ── Step 2: User choosing anonymous (1) or reveal (2) ────────────────
    if step == "awaiting_choice":
        # SMART DETECTION: If the user provides contact info directly
        # (phone number, name, email, "tell noah to call me ..."), treat
        # that as an implicit "reveal" and jump straight to processing
        # their info — don't make them say "2" first.
        if _looks_like_contact_info(query):
            logger.info(f"Implicit reveal detected (contact info provided): {query[:60]}")
            # Fall through to the awaiting_contact_info handler below
            step = "awaiting_contact_info"
            state["crush_flow_step"] = "awaiting_contact_info"
            # Don't return — let it fall through to step 3 processing

        elif _is_anonymous_choice(query):
            # Anonymous choice — store in Supabase + send SMS
            try:
                from supabase import create_client
                supabase = create_client(
                    os.getenv("SUPABASE_URL"),
                    os.getenv("SUPABASE_SERVICE_KEY")
                )
                supabase.table('crush_confessions').insert({
                    'session_id': session_id,
                    'anonymous': True,
                    'name': None,
                    'contact': None
                }).execute()
                logger.info(f"Anonymous crush confession stored for session {session_id}")
            except Exception as e:
                logger.error(f"Failed to store anonymous crush confession: {e}")

            # Send anonymous SMS to Noah
            _send_crush_sms(anonymous=True)

            state["answer"] = (
                "Say less 🕵️ Noah knows he's got a secret admirer. "
                "If you ever want to come back and reveal yourself, the option is always open 💌\n\n"
                "Want to see what he actually builds? I can walk you through his projects 😄"
            )
            state["crush_flow_step"] = None
            state["awaiting_crush_choice"] = False
            state["pipeline_halt"] = True
            return state

        elif _is_reveal_choice(query):
            # Reveal choice — ask for name + message
            state["answer"] = (
                "Full send. I respect it 💯\n\n"
                "Go ahead and tell me your name and a message for Noah — "
                "whatever you want him to know."
            )
            state["crush_flow_step"] = "awaiting_contact_info"
            state["awaiting_crush_choice"] = True
            state["pipeline_halt"] = True
            return state

        else:
            # Couldn't determine choice
            state["answer"] = (
                "Hmm, I need either **1** (stay anonymous) or **2** (reveal yourself). "
                "Which one sounds good to you?"
            )
            state["pipeline_halt"] = True
            return state

    # ── Step 3: User providing name + message ────────────────────────────
    if step == "awaiting_contact_info":
        name, message = _parse_name_and_message(query)

        # Accept the confession if we got a name OR if the raw message
        # contains contact info (phone, email, etc.) even without a name.
        has_usable_info = name or _looks_like_contact_info(query)

        if has_usable_info:
            display_name = name or "Someone"
            contact_data = message or query  # fall back to raw input

            # Store in Supabase
            try:
                from supabase import create_client
                supabase = create_client(
                    os.getenv("SUPABASE_URL"),
                    os.getenv("SUPABASE_SERVICE_KEY")
                )
                supabase.table('crush_confessions').insert({
                    'session_id': session_id,
                    'anonymous': False,
                    'name': name,
                    'contact': contact_data
                }).execute()
                logger.info(f"Revealed crush confession stored: {display_name}")
            except Exception as e:
                logger.error(f"Failed to store revealed crush confession: {e}")

            # Send SMS to Noah via Twilio
            sms_message = contact_data if not name else (message or f"{name} visited your portfolio and wanted to say hi")
            _send_crush_sms(name=display_name, message=sms_message)

            state["answer"] = (
                "Message sent 📱✨ Noah's been notified. "
                "Want to see what he actually builds? I can walk you through his projects 😄"
            )
            state["crush_flow_step"] = None
            state["awaiting_crush_choice"] = False
            state["pipeline_halt"] = True

        else:
            # Couldn't parse a name or contact info — ask again
            state["answer"] = (
                "I need at least your name so Noah knows who to thank 😄 "
                "Try something like: \"Sarah, tell him he seems really cool\""
            )
            state["pipeline_halt"] = True

        return state

    # Fallback — shouldn't reach here, but reset crush state
    state["crush_flow_step"] = None
    state["awaiting_crush_choice"] = False
    return state
