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
_CRUSH_FORM_MARKER = "Message for Noah:"
_CRUSH_COMPLETE_MARKERS = ["Message sent", "Say less", "Noah knows"]

# Pattern to detect contact form submissions in chat history.
# Used to filter them out of Haiku's classification context.
_CONTACT_FORM_RE = re.compile(
    r"Name:.*(?:Number:|Email:|Company:)", re.DOTALL | re.IGNORECASE
)

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
_CONTACT_FORM_DETAILS_MARKER = "Name:\nNumber:\nEmail:\nCompany:\nHow did you find this website?:\nAdditional information:"

# Phrases in last assistant message that indicate a capture question was asked
_CAPTURE_QUESTION_FRAGMENTS = [
    "what brings you here", "what's your angle",
    "hiring, building, or just curious", "hiring, curiosity",
    "want to share what you're working on",
    "noah can follow up", "want noah to reach out",
    "noah can reach out", "want to connect with noah",
    "say the word", "just let me know",
    "i'll set it up", "i can set that up",
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
        'awaiting_crush_form' | None
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

    # Check if we're awaiting crush form submission
    if _CRUSH_FORM_MARKER in last_assistant_content:
        return "awaiting_crush_form"

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
    info = {"name": None, "email": None, "phone": None, "company": None, "referral_source": None, "message": None}

    # Extract email
    email_match = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', query)
    if email_match:
        info["email"] = email_match.group()

    # Extract phone
    phone_match = re.search(r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', query)
    if phone_match:
        info["phone"] = phone_match.group()

    # Extract name — "Name: X", "my name is X", "I'm X", "this is X", "Name, ..."
    label_name_match = re.search(r'(?:^|\n)\s*name\s*[:=]\s*(.+)', query, re.IGNORECASE)
    name_match = re.match(
        r"(?:my name is|i'm|i am|this is|it's)\s+(\w+(?:\s+\w+)?)",
        query, re.IGNORECASE,
    )
    if label_name_match and label_name_match.group(1).strip():
        info["name"] = label_name_match.group(1).strip()
    elif name_match:
        info["name"] = name_match.group(1).strip()
    elif not info["email"] and not info["phone"]:
        # If first words look like a name (1-3 capitalized words before a comma)
        parts = query.split(",", 1)
        if len(parts) == 2 and len(parts[0].strip().split()) <= 3:
            info["name"] = parts[0].strip()

    # Extract company — "Company: X", "at X", "from X", "X company"
    label_company_match = re.search(r'(?:^|\n)\s*company\s*[:=]\s*(.+)', query, re.IGNORECASE)
    company_match = re.search(
        r'(?:at|from|with)\s+([A-Z][\w\s&.-]{1,30}?)(?:\s*[,.]|\s+(?:and|we|our|i|my|looking|hiring)|\s*$)',
        query, re.IGNORECASE,
    )
    if label_company_match and label_company_match.group(1).strip():
        info["company"] = label_company_match.group(1).strip()
    elif company_match:
        info["company"] = company_match.group(1).strip()

    # Extract referral source — "How did you find this website?: ..."
    referral_match = re.search(
        r'(?:how did you find|found (?:this|you|the site|the website)|referral)[:\s]+(.+?)(?:\n|$)',
        query, re.IGNORECASE,
    )
    if referral_match:
        info["referral_source"] = referral_match.group(1).strip()

    # Remaining text is the message
    remaining = query
    for val in [info["email"], info["phone"], info["name"], info["referral_source"]]:
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
            'referral_source': info.get('referral_source'),
            'message': info.get('message'),
            'visitor_type': state.get('visitor_type', 'hiring_manager'),
            'buying_signals_count': state.get('buying_signals_count', 0),
            'message_count': state.get('message_count', 0),
            'capture_trigger': 'intent_to_connect',
        }).execute()
        logger.info(f"Recruiter lead saved for session {session_id}: {info.get('name')}")

        # Mark capture in conversation analytics
        try:
            from assistant.analytics.supabase_analytics import supabase_analytics
            turn_count = state.get("message_count", 0)
            supabase_analytics.mark_data_captured(
                session_id=session_id,
                capture_turn=turn_count,
                capture_type="recruiter_lead",
                referral_source=info.get("referral_source"),
            )
        except Exception as analytics_err:
            logger.error(f"Failed to mark capture in analytics: {analytics_err}")

        return True
    except Exception as e:
        logger.error(f"Failed to save recruiter lead: {e}")
        return False


def _send_hm_lead_notifications(info: dict) -> bool:
    """Send SMS and email notifications to Noah about a new hiring manager lead."""
    name = info.get('name') or 'Unknown'
    company = info.get('company') or 'Unknown company'
    email = info.get('email') or ''
    phone = info.get('phone') or ''
    referral = info.get('referral_source') or ''
    msg = info.get('message') or ''

    # ── SMS ──
    try:
        from assistant.services.twilio_service import get_twilio_service
        twilio = get_twilio_service()
        noah_phone = os.getenv("NOAH_PHONE_NUMBER")

        if noah_phone and twilio and twilio.enabled:
            sms_body = f"Portfolia Lead: {name} at {company}"
            if email:
                sms_body += f"\nEmail: {email}"
            if phone:
                sms_body += f"\nPhone: {phone}"
            if referral:
                sms_body += f"\nFound via: {referral[:100]}"
            if msg:
                sms_body += f"\nMessage: {msg[:200]}"

            if len(sms_body) > 1600:
                sms_body = sms_body[:1597] + "..."

            result = twilio.send_sms(to_phone=noah_phone, message=sms_body)
            logger.info(f"HM lead SMS sent to Noah: {result.get('status', 'unknown')}")
        else:
            logger.warning("Twilio not configured -- skipping HM lead SMS")
    except Exception as e:
        logger.error(f"HM lead SMS failed: {e}")

    # ── Email ──
    try:
        from assistant.services.resend_service import get_resend_service
        resend_svc = get_resend_service()

        if resend_svc and resend_svc.enabled:
            subject = f"Portfolia Lead: {name} at {company}"
            html = (
                "<h2>New Recruiter/Hiring Manager Lead</h2>"
                f"<p><strong>Name:</strong> {name}</p>"
                f"<p><strong>Company:</strong> {company}</p>"
                f"<p><strong>Email:</strong> {email or '(not provided)'}</p>"
                f"<p><strong>Phone:</strong> {phone or '(not provided)'}</p>"
                f"<p><strong>Referral Source:</strong> {referral or '(not provided)'}</p>"
                f"<p><strong>Message:</strong> {msg or '(none)'}</p>"
                "<p><em>Captured via Portfolia recruiter lead flow</em></p>"
            )
            resend_svc.send_email(
                to_email=resend_svc.admin_email,
                subject=subject,
                html=html,
            )
            logger.info("HM lead email sent to Noah")
        else:
            logger.warning("Resend not configured -- skipping HM lead email")
    except Exception as e:
        logger.error(f"HM lead email failed: {e}")

    return True


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
                "Name:\nNumber:\nEmail:\nCompany:\nHow did you find this website?:\nAdditional information:"
                "\n\nOnce you submit, I can walk you through the rest of Noah's projects."
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
            _send_hm_lead_notifications(info)
            display = info.get("name") or info.get("email") or "your info"
            state["answer"] = (
                f"I'll make sure Noah sees this. {display}'s details have been forwarded — "
                f"he'll follow up directly. Want to see what else Noah has built?"
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

    # ── Direct contact / reach-out request detection ─────────────────────
    _contact_phrases = [
        "reach out", "have noah reach out", "contact", "get in touch",
        "take my info", "take my data", "yes reach out",
    ]
    if any(phrase in query_lower for phrase in _contact_phrases):
        logger.info(f"Direct contact request detected via keywords: {query[:50]}")
        state["message_intent"] = "connect"
        state["skip_rag"] = True
        state["answer"] = (
            "I can have Noah reach out — fill this out so we can best assist you:"
            "\n\nName:\nNumber:\nEmail:\nCompany:\nHow did you find this website?:\nAdditional information:"
            "\n\nOnce you submit, I can walk you through the rest of Noah's projects."
        )
        state["pipeline_halt"] = True
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

    # ── Data capture trigger (universal) ─────────────────────────────────
    msg_count = state.get("message_count", 0)
    buying = state.get("buying_signals_count", 0)

    if (_is_connect_intent(query)
            and not state.get("hm_capture_step")
            and msg_count >= 2):
        state["answer"] = (
            "I can have Noah reach out. Fill this out so we can best assist you:\n\n"
            "Name:\n"
            "Number:\n"
            "Email:\n"
            "Company:\n"
            "How did you find this website?:\n"
            "Additional information:"
        )
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
        state["answer"] = (
            "I can have Noah reach out — fill this out so we can best assist you:\n\n"
            "Name:\nNumber:\nEmail:\nCompany:\nHow did you find this website?:\nAdditional information:"
            "\n\nOnce you submit, I can walk you through the rest of Noah's projects."
        )
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


def handle_crush_confession(state: ConversationState) -> ConversationState:
    """Handle crush confession — show the crush form immediately.

    Args:
        state: ConversationState

    Returns:
        Updated state with crush form prompt and flags
    """
    state["answer"] = (
        "Didn't expect anyone to actually pick this one. Respect the commitment though.\n\n"
        "I can let Noah know someone came through with intentions. Fill this out:\n\n"
        "Name:\n"
        "Number or social:\n"
        "Message for Noah:\n\n"
        "Want to stay anonymous? Just leave contact info blank."
    )

    # Set flags for next turn handling
    state["awaiting_crush_choice"] = True
    state["crush_flow_step"] = "awaiting_crush_form"
    state["pipeline_halt"] = True
    state["skip_rag"] = True
    state["message_intent"] = "crush_confession"

    logger.info("Crush confession detected - presented form to user")

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

    Used to detect that a message contains contact info (name, phone,
    email, social handle, etc.).

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


def _parse_crush_form(query: str, anonymous: bool = True):
    """Parse crush form submission.

    Anonymous form expects: Alias + Message for Noah
    Reveal form expects: Name + Number or social + Message for Noah

    Returns:
        anonymous=True: (alias, message) tuple
        anonymous=False: dict with keys: name, contact, message
    """
    text = query.strip()

    if anonymous:
        # Try field-based parsing: "Alias: X\nMessage for Noah: Y"
        alias_match = re.search(
            r"(?:alias|nickname|call me)\s*[:=]?\s*(.+)", text, re.IGNORECASE
        )
        msg_match = re.search(
            r"(?:message(?:\s+for\s+noah)?)\s*[:=]?\s*(.+)", text, re.IGNORECASE
        )
        if alias_match and msg_match:
            return alias_match.group(1).strip(), msg_match.group(1).strip()

        # Fallback: split on newline — first line alias, rest message
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if len(lines) >= 2:
            return lines[0], " ".join(lines[1:])

        # Single line — treat as message only
        return None, text if text else None

    # ── Reveal form ──
    info: dict[str, str | None] = {"name": None, "contact": None, "message": None}

    name_match = re.search(r"(?:name)\s*[:=]\s*(.+)", text, re.IGNORECASE)
    contact_match = re.search(
        r"(?:number(?:\s+or\s+social)?|social|phone|ig|insta|snap|email|contact(?:\s+info)?)\s*[:=]\s*(.+)",
        text, re.IGNORECASE,
    )
    msg_match = re.search(
        r"(?:message(?:\s+for\s+noah)?)\s*[:=]?\s*(.+)", text, re.IGNORECASE
    )

    if name_match:
        info["name"] = name_match.group(1).strip()
    if contact_match:
        info["contact"] = contact_match.group(1).strip()
    if msg_match:
        info["message"] = msg_match.group(1).strip()

    # Fallback: line-based parsing
    if not any(info.values()):
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if len(lines) >= 1:
            info["name"] = lines[0]
        if len(lines) >= 2:
            info["contact"] = lines[1]
        if len(lines) >= 3:
            info["message"] = " ".join(lines[2:])

    # Final fallback: reuse existing free-form parser
    if not info["name"] and not info["contact"]:
        name, msg = _parse_name_and_message(text)
        info["name"] = name
        info["message"] = msg
        phone = re.search(
            r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text
        )
        if phone:
            info["contact"] = phone.group()
        else:
            handle = re.search(r'@\w{2,}', text)
            if handle:
                info["contact"] = handle.group()

    return info


def _send_crush_notifications(
    anonymous: bool = False,
    alias: str | None = None,
    name: str | None = None,
    contact: str | None = None,
    message: str | None = None,
) -> None:
    """Send both SMS and email notifications to Noah about a crush confession."""

    # ── SMS ──
    try:
        from assistant.services.twilio_service import get_twilio_service
        twilio = get_twilio_service()
        noah_phone = os.getenv("NOAH_PHONE_NUMBER")

        if noah_phone and twilio and twilio.enabled:
            if anonymous:
                sms_body = (
                    f"Portfolia: Secret admirer alert.\n"
                    f"Alias: {alias or 'Anonymous'}\n"
                    f"Message: {(message or '(none)')[:200]}"
                )
            else:
                sms_body = (
                    f"Portfolia: Someone chose the bold option.\n"
                    f"Name: {name or 'Unknown'}\n"
                    f"Contact: {contact or '(none)'}\n"
                    f"Message: {(message or '(none)')[:150]}"
                )
            if len(sms_body) > 1600:
                sms_body = sms_body[:1597] + "..."
            twilio.send_sms(to_phone=noah_phone, message=sms_body)
            logger.info("Crush SMS sent to Noah")
        else:
            logger.warning("Twilio not configured -- skipping crush SMS")
    except Exception as e:
        logger.error(f"Crush SMS failed: {e}")

    # ── Email ──
    try:
        from assistant.services.resend_service import get_resend_service
        resend_svc = get_resend_service()

        if resend_svc and resend_svc.enabled:
            if anonymous:
                subject = "Portfolia: Secret admirer"
                html = (
                    "<h2>Secret Admirer on Portfolia</h2>"
                    f"<p><strong>Alias:</strong> {alias or 'Anonymous'}</p>"
                    f"<p><strong>Message:</strong> {message or '(none)'}</p>"
                    "<p><em>Submitted via crush confession flow</em></p>"
                )
            else:
                subject = f"Portfolia: {name or 'Someone'} chose the bold option"
                html = (
                    "<h2>Crush Confession on Portfolia</h2>"
                    f"<p><strong>Name:</strong> {name or 'Unknown'}</p>"
                    f"<p><strong>Contact:</strong> {contact or '(none)'}</p>"
                    f"<p><strong>Message:</strong> {message or '(none)'}</p>"
                    "<p><em>Submitted via crush confession flow</em></p>"
                )
            resend_svc.send_email(
                to_email=resend_svc.admin_email,
                subject=subject,
                html=html,
            )
            logger.info("Crush email sent to Noah")
        else:
            logger.warning("Resend not configured -- skipping crush email")
    except Exception as e:
        logger.error(f"Crush email failed: {e}")


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
    # Crush/romantic signals — if present, don't escape even if "noah" appears
    _crush_stay_signals = [
        "crush", "love", "like", "cute", "hot", "fine", "sexy",
        "date", "dating", "fuck", "hook up", "hookup", "hit on",
        "admirer", "confession", "confess", "attracted", "marry",
        "tryna", "wanna", "kiss", "flirt",
    ]
    q_low = query.lower().strip()
    has_crush_signal = any(sig in q_low for sig in _crush_stay_signals)
    is_noah_query = not has_crush_signal and any(kw in q_low for kw in _escape_noah_kw)
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

    # ── Step: Crush form submission (unified — name/contact optional) ─────
    if step == "awaiting_crush_form":
        info = _parse_crush_form(query, anonymous=False)
        r_name = info.get("name")
        r_contact = info.get("contact")
        r_message = info.get("message")

        # Determine if anonymous (no name AND no contact provided)
        is_anonymous = not r_name and not r_contact

        # Need at least a message
        if not r_name and not r_contact and not r_message:
            state["answer"] = (
                "I need at least a message for Noah. "
                'Something like: "Tell him his portfolio convinced me."'
            )
            state["pipeline_halt"] = True
            return state

        display_name = r_name or "Anonymous"
        safe_message = r_message or "(no message)"
        safe_contact = r_contact or ""

        try:
            from supabase import create_client
            supabase = create_client(
                os.getenv("SUPABASE_URL"),
                os.getenv("SUPABASE_SERVICE_KEY"),
            )
            supabase.table("crush_confessions").insert({
                "session_id": session_id,
                "anonymous": is_anonymous,
                "name": display_name,
                "contact": safe_contact or None,
            }).execute()
            logger.info(f"Crush stored: name={display_name}, anonymous={is_anonymous}")
        except Exception as e:
            logger.error(f"Failed to store crush: {e}")

        # Mark capture in conversation analytics
        try:
            from assistant.analytics.supabase_analytics import supabase_analytics
            supabase_analytics.mark_data_captured(
                session_id=session_id,
                capture_turn=state.get("message_count", 0),
                capture_type="crush_confession",
            )
        except Exception as analytics_err:
            logger.error(f"Failed to mark crush capture in analytics: {analytics_err}")

        _send_crush_notifications(
            anonymous=is_anonymous,
            alias=display_name if is_anonymous else None,
            name=None if is_anonymous else display_name,
            contact=safe_contact or None,
            message=safe_message,
        )

        if is_anonymous:
            state["answer"] = (
                "Say less. Noah knows he's got a secret admirer, and he got your message.\n\n"
                "Now that we've handled that, want to see what he actually builds?"
            )
        else:
            state["answer"] = (
                f"Done. Noah just got notified that {display_name} visited his portfolio "
                "and chose the bold option. Now that we've handled that, "
                "want to see what he actually builds? Might add context to the decision."
            )

        state["crush_flow_step"] = None
        state["awaiting_crush_choice"] = False
        state["pipeline_halt"] = True
        return state

    # Fallback — shouldn't reach here, but reset crush state
    state["crush_flow_step"] = None
    state["awaiting_crush_choice"] = False
    return state
