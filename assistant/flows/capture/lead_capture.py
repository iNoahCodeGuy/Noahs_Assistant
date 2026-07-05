"""Contact/lead capture flow — a stateless finite-state machine.

Follows the same chat_history-marker pattern as the crush flow (see
capture/constants.py): the contact form embeds a marker string, and
_detect_hm_capture_flow_from_history finds it in the last assistant message
on the next turn to resume at the right step. Submission writes to the
Supabase recruiter_leads table and fires SMS/email via
capture/notifications.py.

Naming note: the "hm" (hiring manager) prefixes are legacy — capture is
universal and offered to any visitor type.
"""

import logging
import re

from assistant.flows.capture.constants import (
    _CAPTURE_QUESTION_FRAGMENTS,
    _CONTACT_FORM_MARKER,
    _CONTACT_FORM_PROMPT,
    _HM_CAPTURE_COMPLETE_MARKERS,
    _HM_CAPTURE_DETAILS_MARKER,
    _HM_CAPTURE_MARKER,
)
from assistant.flows.capture.notifications import _send_hm_lead_notifications
from assistant.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)


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

    # Case-insensitive: one producer emits "…reach out. Fill this out…"
    # (capital F after the period) while the markers are lowercase — an
    # exact-case check made that form's step unrecoverable from history.
    content_lower = last_assistant_content.lower()
    if _HM_CAPTURE_DETAILS_MARKER in content_lower:
        return "awaiting_hm_details"

    # Contact form marker (from new capture question flow)
    if _CONTACT_FORM_MARKER in content_lower:
        return "awaiting_hm_details"

    if _HM_CAPTURE_MARKER in last_assistant_content:
        return "awaiting_hm_response"

    return None


def _parse_hm_contact_info(query: str) -> dict:
    """Parse hiring manager contact info from free-form input.

    Returns dict with keys: name, email, phone, company, referral_source, message.
    Any value can be None if not detected.
    """
    info = {"name": None, "email": None, "phone": None, "company": None, "referral_source": None, "message": None}

    # Extract email
    email_match = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', query)
    if email_match:
        info["email"] = email_match.group()

    # Extract phone (digit-pattern OR "Number/Phone: X" label)
    phone_match = re.search(r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', query)
    if phone_match:
        info["phone"] = phone_match.group()
    else:
        label_phone = re.search(r'(?:^|\n)\s*(?:number|phone)\s*[:=][ \t]*([^\n]+)', query, re.IGNORECASE)
        if label_phone and label_phone.group(1).strip():
            info["phone"] = label_phone.group(1).strip()

    # Extract name — "Name: X", "my name is X", "I'm X", "this is X", "Name, ..."
    # Use [^\n]+ (not .+) and [ \t]* (not \s*) to avoid crossing newlines
    label_name_match = re.search(r'(?:^|\n)\s*name\s*[:=][ \t]*([^\n]+)', query, re.IGNORECASE)
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
    label_company_match = re.search(r'(?:^|\n)\s*company\s*[:=][ \t]*([^\n]+)', query, re.IGNORECASE)
    company_match = re.search(
        r'(?:at|from|with)\s+([A-Z][\w\s&.-]{1,30}?)(?:\s*[,.]|\s+(?:and|we|our|i|my|looking|hiring)|\s*$)',
        query, re.IGNORECASE,
    )
    if label_company_match and label_company_match.group(1).strip():
        info["company"] = label_company_match.group(1).strip()
    elif company_match:
        info["company"] = company_match.group(1).strip()

    # Extract referral source — "How did you find this website?: X"
    # Consume the full label (including "this website?") before capturing the value
    referral_match = re.search(
        r'(?:how did you find[^:\n]*[?]?\s*[:=]|found (?:this|you|the site|the website)\s*[:=]|referral\s*[:=])[ \t]*([^\n]+)',
        query, re.IGNORECASE,
    )
    if referral_match and referral_match.group(1).strip():
        info["referral_source"] = referral_match.group(1).strip()

    # Extract additional info label value
    additional_match = re.search(
        r'(?:^|\n)\s*additional\s*(?:information|info)?\s*[:=][ \t]*([^\n]+)',
        query, re.IGNORECASE,
    )
    additional_text = additional_match.group(1).strip() if additional_match and additional_match.group(1).strip() else None

    # Build message from remaining text, stripping form labels
    remaining = query
    for val in [info["email"], info["phone"], info["name"], info["company"], info["referral_source"]]:
        if val:
            remaining = remaining.replace(val, "", 1).strip()
    # Strip known form label lines (empty or already-extracted)
    remaining = re.sub(
        r'(?m)^\s*(?:name|phone|number|email|company|how did you find[^\n]*?|additional\s*(?:information|info)?)\s*[:=][ \t]*\n?',
        '', remaining, flags=re.IGNORECASE,
    )
    remaining = re.sub(r'^[\s,.-]+|[\s,.-]+$', '', remaining)

    # Prefer additional_text if remaining is mostly form debris
    if additional_text and (not remaining or len(remaining) <= 3):
        info["message"] = additional_text
    elif remaining and len(remaining) > 3:
        info["message"] = remaining
    elif additional_text:
        info["message"] = additional_text

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
            state["answer"] = _CONTACT_FORM_PROMPT
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
