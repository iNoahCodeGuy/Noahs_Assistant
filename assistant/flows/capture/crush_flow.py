"""Crush confession flow — a stateless finite-state machine.

State is recovered from chat_history markers each turn (see capture/constants.py):
the form embeds _CRUSH_FORM_MARKER; _detect_crush_flow_from_history finds it in the
last assistant message to resume at 'awaiting_crush_form'. Submission writes to the
Supabase crush_confessions table and fires SMS/email via capture/notifications.py —
the pipeline, not the LLM, triggers the side effects.
"""

import logging
import os
import re

from assistant.flows.capture.constants import _CRUSH_COMPLETE_MARKERS, _CRUSH_FORM_MARKER, _CRUSH_FORM_PROMPT
from assistant.flows.capture.notifications import _send_crush_notifications
from assistant.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)


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


def handle_crush_confession(state: ConversationState) -> ConversationState:
    """Handle crush confession — show the crush form immediately.

    Args:
        state: ConversationState

    Returns:
        Updated state with crush form prompt and flags
    """
    state["answer"] = _CRUSH_FORM_PROMPT

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
                # SUPABASE_SERVICE_KEY is the legacy name; keep as fallback
                os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY"),
            )
            supabase.table("crush_confessions").insert({
                "session_id": session_id,
                "anonymous": is_anonymous,
                "name": display_name,
                "contact": safe_contact or None,
                "message": r_message or None,
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
