"""Discovery hooks — engagement pacing appended after the pipeline runs.

This module owns the conversational "marketing" layer: curiosity hooks about
uncovered projects, reach-out offers, and the pacing rules for when each is
appended to an answer. It runs as a post-loop step in run_conversation_flow,
after both short-circuit and full-pipeline paths.

Kept out of the orchestrator so conversation_flow.py stays pure pipeline
wiring. Copy fragments here must stay in sync with the capture-flow
markers — the detector scans chat history for the exact strings the
producer emitted.
"""

from __future__ import annotations

import logging
import random
import re

logger = logging.getLogger(__name__)

# Appends a discovery question (or curiosity hook) to early-phase responses
# that don't already end with one.  Works for both hardcoded pipeline_halt
# responses and LLM-generated answers.
_SKIP_INTENTS = frozenset({
    "crush_confession", "greeting", "contact_info_submission",
})

_CAPTURE_QUESTIONS = (
    "What brings you here?",
)

_KNOWLEDGE_HOOKS = (
    "The architecture behind this conversation is the best demo "
    "of his engineering — ask me how I work.",
    "The attrition model hit 94.75% on imbalanced data. "
    "The last 12 points are where the real work happened.",
    "Most portfolio sites are static pages. This one answers back.",
    "The retrieval system behind this conversation uses the same math "
    "that powers the attrition model.",
    "Noah's biology degree is the least obvious and most important part "
    "of the technical foundation.",
    "The MMA fighter story connects to the professional background "
    "in a way most people don't expect.",
    "The Response Time Analysis app is worth asking about if you care "
    "about how he thinks through statistical problems.",
    "This system doesn't just answer questions — it writes to databases, "
    "sends SMS, and fires emails. Ask about the agentic architecture.",
)

# Enterprise bridge hooks — used specifically after architecture/self-knowledge
# responses to bridge from "how I work" to "how this works at enterprise scale."
_ENTERPRISE_HOOKS = (
    "Every pattern in this conversation — intent routing, grounding "
    "validation, deterministic tool execution — runs in enterprise AI "
    "systems at scale. Want to see how they transfer?",
    "The architecture behind this conversation is the same one powering "
    "enterprise voice agents and customer support systems. "
    "Want to see how the patterns map?",
    "These pipeline patterns — classify before you retrieve, validate "
    "before you generate — are exactly how production agentic systems "
    "work at enterprise scale. Ask me how they transfer.",
)

_REACH_OUT_OFFERS = (
    "Want Noah to reach out? Say the word and I'll set it up.",
    "If you want Noah to follow up directly, just let me know.",
    "Noah can reach out if you're interested — just say the word.",
    "Want to connect with Noah directly? I can set that up.",
)


def _is_architecture_response(state: dict) -> bool:
    """Detect if the current response is about Portfolia's architecture or self-knowledge.

    Used to trigger enterprise bridge hooks instead of generic knowledge hooks.
    """
    if state.get("is_self_referential"):
        return True
    if state.get("message_intent") == "self_knowledge":
        return True
    # Check the answer itself for architecture discussion indicators
    answer_lower = (state.get("answer") or "").lower()
    _arch_indicators = [
        "pipeline", "node", "intent classification", "vector search",
        "grounding validation", "hallucination check", "state machine",
        "retrieval", "rag", "pgvector", "haiku", "sonnet",
        "stage 1", "stage 2", "stage 3", "stage 4", "stage 5",
        "22-node", "22 node", "21-node", "21 node", "functional pipeline",
        "deterministic tool", "agentic",
    ]
    indicator_count = sum(1 for ind in _arch_indicators if ind in answer_lower)
    return indicator_count >= 3


def _pick_enterprise_hook(state: ConversationState) -> str:
    """Select an enterprise bridge hook that hasn't been used yet."""
    chat_history = state.get("chat_history", [])
    past_assistant_text = " ".join(
        msg.get("content", "").lower()
        for msg in chat_history
        if isinstance(msg, dict)
        and (msg.get("role") or msg.get("type", "")) in ("assistant", "ai")
    )
    unused = [
        hook for hook in _ENTERPRISE_HOOKS
        if hook[:40].lower() not in past_assistant_text
    ]
    if unused:
        return random.choice(unused)
    # All enterprise hooks used — fall back to a regular knowledge hook
    return _pick_knowledge_hook(state)


def _pick_knowledge_hook(state: ConversationState) -> str:
    """Select a knowledge hook that hasn't been used in the conversation yet.

    Scans assistant messages in chat_history for fragments of each hook.
    Returns a random unused hook, or falls back to a generic one if all
    have been used.
    """
    chat_history = state.get("chat_history", [])
    past_assistant_text = " ".join(
        msg.get("content", "").lower()
        for msg in chat_history
        if isinstance(msg, dict)
        and (msg.get("role") or msg.get("type", "")) in ("assistant", "ai")
    )

    # Filter to hooks whose key phrase hasn't appeared in past messages
    unused = []
    for hook in _KNOWLEDGE_HOOKS:
        # Use first 40 chars as a fingerprint — enough to detect reuse
        fingerprint = hook[:40].lower()
        if fingerprint not in past_assistant_text:
            unused.append(hook)

    if unused:
        return random.choice(unused)
    # All hooks used — return a neutral closing statement
    return "There's more to the story if you're curious."


def _should_include_reach_out(state: ConversationState) -> bool:
    """Determine whether this turn should include a reach-out offer.

    Rules:
    - Starts at message 2 (every message, not just odd ones)
    - Skip if user has explicitly declined twice
    - Skip if a reach-out offer was in the immediately preceding assistant message
    - Skip if already in capture flow or capture is complete
    """
    msg_count = state.get("message_count", 0)

    if msg_count < 2:
        logger.debug("_should_include_reach_out: False (msg_count=%d < 2)", msg_count)
        return False

    # Skip if already in capture flow
    if state.get("hm_capture_step"):
        logger.debug("_should_include_reach_out: False (in capture flow)")
        return False

    chat_history = state.get("chat_history", [])

    # Count explicit declines from user messages that follow a reach-out offer
    _offer_fragments = [
        "want noah to reach out", "noah can follow up",
        "noah can reach out", "want to connect with noah",
        "say the word", "just let me know",
        "fill this out so we can best assist",
    ]
    _decline_phrases = [
        "no thanks", "no thank you", "not right now", "not now",
        "maybe later", "i'm good", "im good", "nah", "pass",
        "not interested", "no need", "i'll pass", "ill pass",
        "just browsing", "just looking", "not yet",
    ]

    decline_count = 0
    prev_was_offer = False
    for msg in chat_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or msg.get("type", "")
        content = (msg.get("content") or "").lower()
        if role in ("assistant", "ai"):
            prev_was_offer = any(frag in content for frag in _offer_fragments)
        elif role in ("user", "human") and prev_was_offer:
            if any(phrase in content for phrase in _decline_phrases):
                decline_count += 1
            prev_was_offer = False

    if decline_count >= 2:
        logger.debug("_should_include_reach_out: False (declined %d times)", decline_count)
        return False

    # Skip if the immediately preceding assistant message already has an offer
    recent_assistant_msgs = [
        msg.get("content", "").lower()
        for msg in chat_history
        if isinstance(msg, dict)
        and (msg.get("role") or msg.get("type", "")) in ("assistant", "ai")
    ][-1:]  # Only check the LAST assistant message (not last 2)

    for prev in recent_assistant_msgs:
        if any(frag in prev for frag in _offer_fragments):
            logger.debug("_should_include_reach_out: False (offer in previous message)")
            return False

    logger.debug("_should_include_reach_out: True (msg_count=%d, declines=%d)", msg_count, decline_count)
    return True


def _pick_reach_out_offer() -> str:
    """Return a random reach-out offer string."""
    return random.choice(_REACH_OUT_OFFERS)


# Patterns that identify menu-style questions (multi-option "or" questions).
# Reused by _maybe_append_discovery_question to detect trailing menu endings
# that _strip_menu_endings may have missed.
_MENU_QUESTION_RE = re.compile(
    r'(?:Would you like|Want|Shall I|Should I|Interested in|Curious about|Wanna|'
    r'Would you rather|What would you like|Which (?:one|topic|area))'
    r'.*?\bor\b.*?\?',
    re.IGNORECASE,
)


def _is_menu_ending(answer: str) -> bool:
    """Check if the answer's trailing question is a menu-style multi-option prompt.

    Extracts the last sentence (from last '. '/'! '/'\\n' boundary to end)
    and checks whether it matches a menu pattern (trigger word + 'or' + '?').
    Single discovery questions like "What brings you here?" return False.
    """
    # Find the last sentence boundary
    last_boundary = 0
    for m in re.finditer(r'(?:[.!?]\s+|\n)', answer):
        last_boundary = m.end()

    last_sentence = answer[last_boundary:].strip()
    if not last_sentence:
        return False

    is_menu = bool(_MENU_QUESTION_RE.search(last_sentence))
    logger.info(
        "Discovery hook _is_menu_ending check | last_sentence='%s' | is_menu=%s",
        last_sentence[:100], is_menu,
    )
    return is_menu


def _strip_trailing_menu(answer: str) -> str:
    """Remove the trailing menu question from the answer text.

    Returns the answer with the last menu-style sentence removed.
    """
    last_boundary = 0
    for m in re.finditer(r'(?:[.!?]\s+|\n)', answer):
        last_boundary = m.end()

    if last_boundary > 0:
        return answer[:last_boundary].rstrip()
    # Entire answer is the menu question — return empty so caller keeps original
    return ""


def _maybe_append_discovery_question(state: dict) -> dict:
    """Append a capture question AND knowledge hook to substantive answers that lack them.

    Every substantive response (not greetings, not crush flow) should end with:
    1. A capture/discovery question — draws out who they are or why they're here
    2. A knowledge hook — a statement that invites curiosity about an uncovered topic
    """
    logger.info(
        "ENTER _maybe_append_discovery_question: injected=%s answer_len=%d intent=%s msg=%d",
        state.get("_discovery_injected"),
        len(state.get("answer", "")),
        state.get("message_intent"),
        state.get("message_count", 0),
    )
    if state.get("_discovery_injected"):
        return state

    answer = (state.get("answer") or "").strip()
    if not answer:
        return state

    # If answer ends with "?", check whether it's a menu ending or a real question.
    # Menu endings ("Want to see X or Y?") get replaced with our two-part ending.
    # Real questions are left alone only if a knowledge hook is also present.
    if answer.endswith("?"):
        if _is_menu_ending(answer):
            stripped = _strip_trailing_menu(answer)
            if stripped:
                answer = stripped
                logger.info(
                    "Discovery hook: replaced menu ending, answer now ends: '...%s'",
                    answer[-80:],
                )
            else:
                # Entire answer was the menu question — leave it alone
                return state

    # Don't touch special flows
    if state.get("message_intent") in _SKIP_INTENTS:
        return state
    if state.get("hm_capture_step"):
        return state
    if state.get("is_greeting"):
        return state

    msg_count = state.get("message_count", 0)
    if msg_count < 1:
        return state

    # Check what's already present at the END of the answer (last 200 chars).
    # Only look at the tail to avoid false-positiving on phrases that appear
    # incidentally in the prose body (e.g., "worth a look" mid-paragraph).
    answer_tail = answer[-200:].lower()
    has_capture = any(
        frag in answer_tail for frag in [
            "what brings you", "what caught your eye", "what's your angle",
            "hiring, building", "hiring, curiosity", "hiring, exploring",
            "want to share what you",
        ]
    )
    has_reach_out = any(
        frag in answer_tail for frag in [
            "want noah to reach out", "noah can follow up",
            "noah can reach out", "want to connect with noah",
            "say the word", "just let me know",
        ]
    )
    has_hook = any(
        frag in answer_tail for frag in [
            "worth a look", "same math that powers",
            "statistical foundation", "architecture behind this",
            "attrition model", "if you want to see",
            "ask me how i work", "worth understanding",
            "worth knowing", "segmentation", "k-means",
            "decision tree", "gap is worth", "patterns the",
            "if you're evaluating", "opposite direction",
            # Enterprise bridge hook fragments
            "enterprise ai", "enterprise scale", "enterprise voice",
            "how they transfer", "how the patterns map",
            "production agentic",
        ]
    )

    # If the LLM wrote content AFTER a capture question, treat it as a hook.
    # This prevents double-appending when the LLM already ended with
    # "What brings you here?\n\n<hook sentence>".
    if not has_hook and has_capture:
        _capture_pos = max(
            answer_tail.rfind("what brings you"),
            answer_tail.rfind("what caught your eye"),
            answer_tail.rfind("what's your angle"),
        )
        if _capture_pos >= 0:
            _after_capture = answer_tail[_capture_pos:].strip()
            # If there's substantial text after the capture question line,
            # the LLM already wrote a hook.
            if "\n" in _after_capture and len(_after_capture.split("\n", 1)[1].strip()) > 20:
                has_hook = True

    # If all three already present, nothing to do
    logger.info(
        "DIAG _maybe_append: ends_with_q=%s has_capture=%s has_reach_out=%s has_hook=%s tail=%r",
        answer.endswith("?"), has_capture, has_reach_out, has_hook, answer_tail,
    )
    if (has_capture or has_reach_out) and has_hook:
        state["_discovery_injected"] = True
        return state

    if msg_count >= 2:
        # ── Message 2+: knowledge hook + reach-out offer ──
        # Architecture/self-knowledge responses get enterprise bridge hooks
        # instead of generic knowledge hooks.
        is_arch = _is_architecture_response(state)
        suffix_parts = []

        if is_arch:
            # Enterprise bridge: capture-aware endings
            has_given_info = bool(state.get("hm_capture_step") == "complete")
            if not has_hook:
                suffix_parts.append(_pick_enterprise_hook(state))
            # If user hasn't given contact info yet, also offer reach-out
            if not has_given_info:
                include_reach_out = _should_include_reach_out(state) and not has_reach_out
                if include_reach_out:
                    suffix_parts.append(_pick_reach_out_offer())
        else:
            # Standard flow: reach-out offer + knowledge hook
            include_reach_out = _should_include_reach_out(state) and not has_reach_out
            if include_reach_out:
                suffix_parts.append(_pick_reach_out_offer())

            # Knowledge hook (always, if not already present)
            if not has_hook:
                suffix_parts.append(_pick_knowledge_hook(state))

        if suffix_parts:
            state["answer"] = answer + "\n\n" + "\n\n".join(suffix_parts)
            state["_discovery_injected"] = True
            logger.info(
                "Discovery hook appended (msg_count=%d): arch=%s reach_out=%s hook=%s",
                msg_count, is_arch, not has_reach_out, not has_hook,
            )
    else:
        # ── Message 1: "What brings you here?" + knowledge hook ──
        capture = _CAPTURE_QUESTIONS[0]  # "What brings you here?"

        if not has_capture:
            _capture_fragments = [
                "what brings you", "what caught your eye",
                "are you exploring for yourself", "what's your angle",
                "hiring, building", "hiring, curiosity", "hiring, exploring",
                "want to share what you", "want noah to reach out",
            ]
            chat_history = state.get("chat_history", [])
            recent_assistant_msgs = [
                msg.get("content", "").lower()
                for msg in chat_history
                if isinstance(msg, dict)
                and (msg.get("role") or msg.get("type", "")) in ("assistant", "ai")
            ][-2:]
            for prev in recent_assistant_msgs:
                if any(frag in prev for frag in _capture_fragments):
                    has_capture = True
                    logger.info("Discovery hook: skipping capture — similar phrase in recent history")
                    break

        hook = _KNOWLEDGE_HOOKS[msg_count % len(_KNOWLEDGE_HOOKS)]

        suffix_parts = []
        if not has_capture:
            suffix_parts.append(capture)
        if not has_hook:
            suffix_parts.append(hook)

        if suffix_parts:
            state["answer"] = answer + "\n\n" + "\n".join(suffix_parts)
            state["_discovery_injected"] = True
            logger.info(
                "Discovery hook appended (msg_count=%d): capture=%s hook=%s",
                msg_count, not has_capture, not has_hook,
            )

    return state
