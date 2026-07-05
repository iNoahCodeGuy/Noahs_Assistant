"""State-machine checkpoint strings shared by both capture flows.

The crush confession and contact/lead capture flows are stateless state
machines: nothing persists server-side between turns, so on every request the
current step is recovered by scanning chat_history for the marker strings
below. The code that PRODUCES a message (embedding a marker in Portfolia's
answer text) and the code that DETECTS the step on the next turn must use the
exact same strings — which is why every marker, and every form-text template
that embeds one, lives here in a single module imported by both sides.

Also holds the stateless engagement-counter helpers (recomputed fresh from
chat_history each turn) used by intent classification in stage1_intent_router.
"""

import re

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

# ── Form-text templates ──────────────────────────────────────────────────
# Producer side of the markers above: these answer texts embed the marker
# strings that the detectors scan for on the next turn. Keep in sync.

# Crush confession form (embeds _CRUSH_FORM_MARKER).
_CRUSH_FORM_PROMPT = (
    "Didn't expect anyone to actually pick this one. Respect the commitment though.\n\n"
    "I can let Noah know someone came through with intentions. Fill this out:\n\n"
    "Name:\n"
    "Number or social:\n"
    "Message for Noah:\n\n"
    "Want to stay anonymous? Just leave contact info blank."
)

# Contact form (embeds _CONTACT_FORM_MARKER / _HM_CAPTURE_DETAILS_MARKER).
_CONTACT_FORM_PROMPT = (
    "I can have Noah reach out — fill this out so we can best assist you:\n\n"
    "Name:\nNumber:\nEmail:\nCompany:\nHow did you find this website?:\nAdditional information:"
    "\n\nOnce you submit, I can walk you through the rest of Noah's projects."
)

# Contact form variant used by the universal data-capture trigger.
_CAPTURE_TRIGGER_FORM_PROMPT = (
    "I can have Noah reach out. Fill this out so we can best assist you:\n\n"
    "Name:\n"
    "Number:\n"
    "Email:\n"
    "Company:\n"
    "How did you find this website?:\n"
    "Additional information:"
)


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


# ── Un-underscored aliases for external importers ────────────────────────
CRUSH_FORM_MARKER = _CRUSH_FORM_MARKER
CRUSH_COMPLETE_MARKERS = _CRUSH_COMPLETE_MARKERS
CONTACT_FORM_RE = _CONTACT_FORM_RE
HM_CAPTURE_MARKER = _HM_CAPTURE_MARKER
HM_CAPTURE_DETAILS_MARKER = _HM_CAPTURE_DETAILS_MARKER
HM_CAPTURE_COMPLETE_MARKERS = _HM_CAPTURE_COMPLETE_MARKERS
HM_SOFT_OFFER_MARKER = _HM_SOFT_OFFER_MARKER
CONTACT_FORM_MARKER = _CONTACT_FORM_MARKER
CONTACT_FORM_DETAILS_MARKER = _CONTACT_FORM_DETAILS_MARKER
CAPTURE_QUESTION_FRAGMENTS = _CAPTURE_QUESTION_FRAGMENTS
CRUSH_FORM_PROMPT = _CRUSH_FORM_PROMPT
CONTACT_FORM_PROMPT = _CONTACT_FORM_PROMPT
CAPTURE_TRIGGER_FORM_PROMPT = _CAPTURE_TRIGGER_FORM_PROMPT
