"""Voice and markdown post-processing helpers (extracted from stage6_formatting_nodes).

Rule-based text transforms applied to generated answers:
- Em-dash removal, chunk-citation removal, markdown header stripping
- Menu-ending and italic-emphasis stripping
- Voice compliance enforcement (banned openers/phrases, exclamation caps,
  inline menu and interest-poll stripping)

Pure regex/string manipulation - no LLM calls, no state access.
"""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def _strip_em_dashes(text: str) -> str:
    """Replace em-dashes and double-hyphens with periods or commas.

    Converts '—', '–', and '--' to '. ' (before uppercase) or ', ' (otherwise).
    Applied as a universal post-processing step on ALL answer paths.
    """
    if not text:
        return text

    def _replace_dash(m: re.Match) -> str:
        after = m.group(1)
        if after and after[0].isupper():
            return ". " + after
        return ", " + (after or "")

    text = re.sub(r'\s*[—–]\s*(\S)', _replace_dash, text)
    text = re.sub(r'\s*--\s*(\S)', _replace_dash, text)
    # Catch trailing em-dashes with no following character
    text = re.sub(r'\s*[—–]\s*$', '.', text, flags=re.MULTILINE)
    text = re.sub(r'\s*--\s*$', '.', text, flags=re.MULTILINE)
    return text


def _remove_chunk_citations(text: str) -> str:
    """Remove citation phrases that reference retrieved chunks verbatim.

    Removes phrases like:
    - "The retrieved notes call out..."
    - "The retrieved chunks mention..."
    - "The context chunks indicate..."
    - "The retrieved context says..."

    These phrases break the first-person narrative and reveal the RAG mechanism.

    Args:
        text: Answer text that may contain chunk citation phrases

    Returns:
        Text with citation phrases removed

    Example:
        >>> _remove_chunk_citations("The retrieved notes call out Streamlit UI.")
        "Streamlit UI."
    """
    # Patterns to remove (case-insensitive)
    citation_patterns = [
        r'\b[Tt]he\s+retrieved\s+notes\s+call\s+out\s+',
        r'\b[Tt]he\s+retrieved\s+chunks?\s+(?:call\s+out|mention|indicate|say|show|contain)\s+',
        r'\b[Tt]he\s+context\s+chunks?\s+(?:call\s+out|mention|indicate|say|show|contain)\s+',
        r'\b[Tt]he\s+retrieved\s+context\s+(?:calls?\s+out|mentions?|indicates?|says?|shows?|contains?)\s+',
        r'\b[Tt]he\s+context\s+(?:calls?\s+out|mentions?|indicates?|says?|shows?|contains?)\s+',
        r'\b[Tt]he\s+notes\s+(?:call\s+out|mention|indicate|say|show|contain)\s+',
        # Additional patterns to catch more citation phrases
        r'\b[Tt]he\s+facts\s+in\s+context\s+mention\s+',
        r'\b[Tt]he\s+materials\s+mention\s+',
        r'\b[Cc]ontext\s+snippets\s+cite\s+',
        r'\b[Tt]he\s+retrieved\s+context\s+highlights\s+',
    ]

    result = text
    for pattern in citation_patterns:
        # Remove the citation phrase, preserving the rest of the sentence
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)

    # Clean up any double spaces or punctuation issues (preserve newlines for markdown)
    result = re.sub(r'[^\S\n]+', ' ', result)
    result = re.sub(r' +([,.!?])', r'\1', result)

    return result.strip()


def _remove_markdown_headers(text: str) -> str:
    """Remove markdown headers (##, ###) and convert to plain text.

    Removes markdown headers like:
    - "## Example Conversation 1" → "Example Conversation 1"
    - "### Turn 1: Enterprise Opening" → "Turn 1: Enterprise Opening"
    - "## 🎯 Hiring Manager" → "Hiring Manager"

    These headers often appear when documentation chunks are copied verbatim.

    Args:
        text: Answer text that may contain markdown headers

    Returns:
        Text with headers removed or converted to plain text

    Example:
        >>> _remove_markdown_headers("## Example Conversation 1\\n**Content**")
        "Example Conversation 1\\n**Content**"
        >>> _remove_markdown_headers("### Turn 1: Enterprise Opening")
        "Turn 1: Enterprise Opening"
    """
    if not text:
        return text

    # Remove markdown headers (##, ###, ####) - capture the header text
    # Pattern: 1-4 hashes followed by whitespace, then optional emojis, then text
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        # Match markdown headers with optional emojis
        # Pattern: ^#{1,4}\s+([🎯🔧⚙️📊🏗️🧪🚀]*\s*)?(.+)$
        header_match = re.match(r'^#{1,4}\s+(?:[🎯🔧⚙️📊🏗️🧪🚀]+\s*)?(.+)$', line)
        if header_match:
            # Extract just the text content after the header
            cleaned_lines.append(header_match.group(1).strip())
        else:
            cleaned_lines.append(line)

    result = '\n'.join(cleaned_lines)

    # Clean up any double newlines created by header removal
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()


def _strip_menu_endings(text: str) -> str:
    """Remove trailing sentences that offer menu-style multiple options.

    Only strips questions that present multiple choices (contain "or" between
    options). Single discovery questions like "What brings you here?" or
    "What's your angle on this?" are preserved.

    Detects and strips endings like:
    - "Want to hear about X or Y?"
    - "Would you like to explore X or Y?"
    - "Shall I go into X or Y?"
    - "Want to see the code, or should I go deeper on one of them?"

    Scans the last 3 sentences backward so trailing emoji / short phrases
    after the menu question don't hide the match.

    Args:
        text: Answer text that may end with a menu-style question

    Returns:
        Text with menu endings removed
    """
    if not text:
        return text

    stripped = text.rstrip()

    # Collect all sentence boundary positions (end of ". " / "! " / "? " / "\n")
    boundaries = []
    for m in re.finditer(r'(?:[.!?]\s+|\n)', stripped):
        boundaries.append(m.end())

    # Menu patterns — ONLY multi-option questions (must contain "or" between choices).
    # Single-topic questions ("What brings you here?", "What's your angle?") are
    # legitimate discovery questions and must NOT be stripped.
    menu_patterns = [
        # trigger word + "or" + question mark
        re.compile(
            r'(?:Would you like|Want|Shall I|Should I|Interested in|Curious about|Wanna)'
            r'.*?\bor\b.*?\?',
            re.IGNORECASE,
        ),
        # Any sentence with "or" between two options followed by question mark
        # e.g. "see the code, or go deeper on one of them?"
        re.compile(r'.*?\bor\b\s+(?:should|shall|would|do you|I)\b.*?\?', re.IGNORECASE),
        # "would you rather" (inherently multi-option)
        re.compile(r'\bwould you rather\b.*?\?', re.IGNORECASE),
        # "What would you like" / "Which" with explicit option list
        re.compile(r'\b(?:What would you like|Which (?:one|topic|area))\b.*?\bor\b.*?\?', re.IGNORECASE),
        # Cross-sentence menu: "Want X? Or Y" — the "or" starts a new sentence
        # e.g. "Want Noah to reach out? Or I can walk you through another project"
        re.compile(
            r'(?:Want|Shall I|Should I|Would you like).*?\?'
            r'\s+[Oo]r\b.*?(?:\.|!|\?|$)',
            re.IGNORECASE,
        ),
    ]

    logger.info(
        "_strip_menu_endings: text_len=%d, boundaries=%d, last_80='%s'",
        len(stripped), len(boundaries), stripped[-80:],
    )

    # Scan backward through the last 3 sentence boundaries.
    # For each boundary, check text from that boundary to end-of-string.
    # This handles cases where emoji or a short phrase follows the menu question
    # (e.g. "Want to see the code? 😄" — the "? " is a boundary, so the
    #  last sentence is just "😄"; we need to also check the preceding sentence).
    check_starts = boundaries[-3:] if boundaries else []
    check_starts.reverse()  # check from latest boundary backward
    # Also check from the very start (no boundary) as the fallback
    if not check_starts:
        check_starts = [0]

    # Label each pattern for diagnostics
    pattern_labels = [
        "trigger+or+?",
        "or+modal+?",
        "would_you_rather",
        "what_which+or+?",
        "cross_sentence_or",
    ]

    for start_pos in check_starts:
        candidate = stripped[start_pos:]
        logger.info(
            "_strip_menu_endings: checking candidate at pos=%d len=%d: '%s'",
            start_pos, len(candidate), candidate[:120],
        )
        for pattern, label in zip(menu_patterns, pattern_labels):
            match = pattern.search(candidate)
            logger.info(
                "_strip_menu_endings:   pattern=%-20s matched=%s%s",
                label,
                bool(match),
                f"  span={match.span()}  text='{match.group()[:80]}'" if match else "",
            )
            if match:
                # Preserve capture-flow endings (discovery questions + reach-out offer)
                # BUT if the match contains "or" after a "?", it's a menu ending
                # even if it starts with a capture phrase like "Want Noah to reach out?"
                matched_text = match.group().lower()
                has_cross_sentence_or = bool(re.search(r'\?\s+or\b', matched_text, re.IGNORECASE))
                _capture_phrases = [
                    "what brings you here",
                    "what caught your eye",
                    "are you exploring",
                    "want noah to reach out",
                    "reach out",
                ]
                if any(p in matched_text for p in _capture_phrases) and not has_cross_sentence_or:
                    logger.info(
                        "_strip_menu_endings: skipping capture-flow ending: '%s'",
                        match.group()[:80],
                    )
                    continue
                # Cut at the boundary where the menu sentence begins
                if start_pos > 0:
                    result = stripped[:start_pos].rstrip()
                    if result:
                        logger.info(
                            "Menu ending stripped at boundary %d: '%s'",
                            start_pos, candidate[:80],
                        )
                        return result
                # Entire text is the menu sentence — return as-is
                logger.info("Menu ending IS the entire text — keeping as-is")
                return text

    logger.info("_strip_menu_endings: NO match in last sentences: '%s'", stripped[-80:])
    return text


def _strip_italic_emphasis(text: str) -> str:
    """Remove italic emphasis formatting while preserving bold.

    Strips single-asterisk italic pairs (*word* -> word) and
    underscore italic pairs (_word_ -> word). Preserves bold (**word**).

    Args:
        text: Answer text that may contain italic formatting

    Returns:
        Text with italic emphasis removed
    """
    if not text:
        return text

    # First, protect bold pairs by replacing them with a placeholder
    bold_placeholder = "\x00BOLD\x00"
    protected = re.sub(r'\*\*(.+?)\*\*', lambda m: f"{bold_placeholder}{m.group(1)}{bold_placeholder}", text)

    # Strip remaining single-asterisk italic pairs
    protected = re.sub(r'\*([^*\n]+?)\*', r'\1', protected)

    # Restore bold pairs
    result = protected.replace(bold_placeholder, "**")

    # Strip underscore italics (_word_ -> word), but not __bold__
    # Protect double underscores first
    dunder_placeholder = "\x00DUNDER\x00"
    result = re.sub(r'__(.+?)__', lambda m: f"{dunder_placeholder}{m.group(1)}{dunder_placeholder}", result)
    result = re.sub(r'(?<!\w)_([^_\n]+?)_(?!\w)', r'\1', result)
    result = result.replace(dunder_placeholder, "__")

    return result


# ── Voice compliance enforcement ──────────────────────────────────
# Rule-based post-processor that catches personality violations the LLM
# ignores from the system prompt. Runs after generation, <5ms.

# Openers that violate the Craig Jones voice (enthusiastic, performative, filler)
_BANNED_OPENERS: List[re.Pattern] = [
    re.compile(r'^Ah\s+cool[,!.]?\s*', re.IGNORECASE),
    re.compile(r'^Oh\s+cool[,!.]?\s*', re.IGNORECASE),
    re.compile(r'^That\'s\s+awesome[.!]?\s*', re.IGNORECASE),
    re.compile(r'^That\'s\s+great[.!]?\s*', re.IGNORECASE),
    re.compile(r'^That\'s\s+(?:a\s+)?(?:really\s+)?(?:great|good|interesting|excellent|fantastic)\s+(?:question|point|observation)[.!]?\s*', re.IGNORECASE),
    re.compile(r'^Great\s+question[.!]?\s*', re.IGNORECASE),
    re.compile(r'^Good\s+question[.!]?\s*', re.IGNORECASE),
    re.compile(r'^Interesting\s+question[.!]?\s*', re.IGNORECASE),
    re.compile(r'^(?:I\'d\s+)?(?:love|be\s+happy)\s+to\s+(?:help|show|walk|break|explain|tell)\b[^.!?]*[.!]?\s*', re.IGNORECASE),
    re.compile(r'^(?:Let\s+me\s+(?:break\s+that\s+down|walk\s+you\s+through|explain))\b[^.!?]*[.!]?\s*', re.IGNORECASE),
    re.compile(r'^Here\'s\s+the\s+(?:breakdown|thing|cool\s+part|magic)[.!:]?\s*', re.IGNORECASE),
    re.compile(r'^(?:Ha|Haha|Hah)[,!.]?\s+', re.IGNORECASE),
    re.compile(r'^LOL[.!,]?\s*', re.IGNORECASE),
    re.compile(r'^Absolutely[.!,]?\s*', re.IGNORECASE),
    re.compile(r'^Of\s+course[.!,]?\s*', re.IGNORECASE),
    re.compile(r'^Sure\s+thing[.!,]?\s*', re.IGNORECASE),
    re.compile(r'^(?:Oh\s+)?(?:I\s+)?(?:appreciate|love)\s+(?:the|that|your)\s+(?:energy|enthusiasm|curiosity|interest)[.!]?\s*', re.IGNORECASE),
]

# Mid-text phrases that break voice (performative enthusiasm)
_BANNED_PHRASES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"That's awesome[.!]?", re.IGNORECASE), ""),
    (re.compile(r"That's great[.!]?", re.IGNORECASE), ""),
    (re.compile(r"I'd love to show you", re.IGNORECASE), "Here's"),
    (re.compile(r"I'd love to", re.IGNORECASE), "I can"),
    (re.compile(r"I'd be happy to", re.IGNORECASE), "I can"),
    (re.compile(r"I appreciate (?:the|that|your) (?:energy|enthusiasm|curiosity|interest)[.!,]?", re.IGNORECASE), ""),
]

# Inline option lists: "whether that's X, Y, or Z" pattern
_INLINE_MENU_PATTERNS: List[re.Pattern] = [
    re.compile(
        r',?\s*whether\s+(?:that\'s|it\'s|you\'re\s+(?:interested\s+in|looking\s+at|curious\s+about))\s+'
        r'.+?,\s+.+?,?\s+or\s+.+?[.?!]',
        re.IGNORECASE,
    ),
    re.compile(
        r',?\s*(?:whether\s+)?(?:that\'s|it\'s)\s+.+?,\s+.+?,?\s+or\s+.+?[.?!]',
        re.IGNORECASE,
    ),
    # "from X to Y to Z" option spread
    re.compile(
        r',?\s*(?:from|ranging\s+from)\s+.+?\s+to\s+.+?\s+to\s+.+?[.?!]',
        re.IGNORECASE,
    ),
]

# "What kind of X interests you?" — generic interest-polling questions
_INTEREST_POLL_PATTERNS: List[re.Pattern] = [
    re.compile(r'What\s+(?:kind|type|sort)\s+of\s+\w+\s+interests?\s+you\s*(?:most|the\s+most)?\s*\?', re.IGNORECASE),
    re.compile(r'What\s+(?:are\s+you|would\s+you\s+be)\s+most\s+interested\s+in\s*\?', re.IGNORECASE),
    re.compile(r'What\s+(?:catches|caught)\s+your\s+(?:eye|interest|attention)\s+(?:most|the\s+most)\s*\?', re.IGNORECASE),
]


def _enforce_voice_rules(text: str) -> str:
    """Post-generation voice compliance enforcer.

    Catches personality violations the LLM produces despite system prompt rules:
    1. Banned openers (enthusiastic filler, performative reactions)
    2. Excess exclamation points (max 1 per response, none in first sentence)
    3. Banned mid-text phrases (enthusiasm that breaks Craig Jones voice)
    4. Inline menu patterns ("whether that's X, Y, or Z")
    5. Generic interest-polling questions

    Runs in <5ms. No LLM call — pure regex/string manipulation.
    """
    if not text:
        return text

    result = text

    # ── 1. Strip banned openers ──
    # Apply repeatedly in case multiple filler phrases stack at the start
    changed = True
    iterations = 0
    while changed and iterations < 5:
        changed = False
        iterations += 1
        for pattern in _BANNED_OPENERS:
            new_result = pattern.sub('', result, count=1)
            if new_result != result:
                result = new_result
                changed = True
                break  # restart from top after each strip

    # Capitalize first letter if we stripped an opener
    if result and result != text:
        result = result[0].upper() + result[1:] if len(result) > 1 else result.upper()
        logger.info("Voice enforcer: stripped banned opener")

    # ── 2. Strip banned mid-text phrases ──
    for pattern, replacement in _BANNED_PHRASES:
        new_result = pattern.sub(replacement, result)
        if new_result != result:
            logger.info("Voice enforcer: replaced banned phrase")
            result = new_result

    # Clean up double spaces / orphaned punctuation from phrase removal
    result = re.sub(r'  +', ' ', result)
    result = re.sub(r'\.\s*\.', '.', result)
    result = re.sub(r'^\s*[,.]?\s*', '', result)

    # ── 3. Kill excess exclamation points ──
    # No exclamation in the first sentence
    first_sentence_end = re.search(r'[.!?]', result)
    if first_sentence_end and result[first_sentence_end.start()] == '!':
        result = result[:first_sentence_end.start()] + '.' + result[first_sentence_end.start() + 1:]
        logger.info("Voice enforcer: replaced ! in first sentence")

    # Max 1 exclamation in entire response (preserve the first one found after sentence 1)
    excl_count = result.count('!')
    if excl_count > 1:
        # Keep the first !, replace the rest with .
        first_excl = result.index('!')
        before = result[:first_excl + 1]
        after = result[first_excl + 1:].replace('!', '.')
        result = before + after
        logger.info("Voice enforcer: capped exclamation points from %d to 1", excl_count)

    # ── 4. Strip inline menu patterns ──
    for pattern in _INLINE_MENU_PATTERNS:
        match = pattern.search(result)
        if match:
            # Remove the inline menu, clean up surrounding text
            before = result[:match.start()].rstrip()
            after = result[match.end():].lstrip()
            if before and after:
                # Join with period if the before part doesn't end with punctuation
                if before[-1] not in '.!?:':
                    result = before + '. ' + after
                else:
                    result = before + ' ' + after
            elif before:
                result = before
            else:
                result = after
            logger.info("Voice enforcer: stripped inline menu pattern")
            break  # one pass is enough

    # ── 5. Strip generic interest-polling questions ──
    for pattern in _INTEREST_POLL_PATTERNS:
        match = pattern.search(result)
        if match:
            # Remove the polling question, keep everything before it
            before = result[:match.start()].rstrip()
            after = result[match.end():].lstrip()
            if before:
                result = before + (' ' + after if after else '')
            logger.info("Voice enforcer: stripped interest-polling question")
            break

    # Final cleanup: collapse whitespace, trim trailing spaces on lines
    result = re.sub(r' +\n', '\n', result)
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = result.strip()

    return result
