"""Link sharing and throttling helpers (extracted from stage6_formatting_nodes).

Owns the portfolio link constants (LinkedIn, GitHub, resume) and the
cooldown-based throttle that caps how often links appear in responses.

Pure string manipulation - no LLM calls.
"""

import re
import logging

logger = logging.getLogger(__name__)


# ── Link throttling constants ──────────────────────────────────────
_LINK_PATTERN = re.compile(
    r'https?://(?:github\.com/iNoahCodeGuy|linkedin\.com/in/noah[^\s)]*)',
    re.IGNORECASE,
)
_LINK_COOLDOWN = 3  # require at least 3 link-free responses between link appearances


def _throttle_links(answer: str, chat_history: list) -> str:
    """Strip GitHub/LinkedIn links if they appeared too recently in the conversation.

    Scans the last N assistant messages. If any of the most recent _LINK_COOLDOWN
    responses already contain a link, all portfolio links are removed from the
    current answer. This caps link frequency to roughly every 3rd–4th response.

    Direct contact/connection requests are exempt — if the user explicitly asks
    for links, they always get them.
    """
    if not _LINK_PATTERN.search(answer):
        return answer  # no links to throttle

    # Collect recent assistant messages (most recent first)
    recent_assistant: list[str] = []
    for msg in reversed(chat_history):
        if isinstance(msg, dict):
            role = msg.get("role", "") or msg.get("type", "")
            content = msg.get("content", "")
        elif hasattr(msg, "type"):
            role = getattr(msg, "type", "") or getattr(msg, "role", "")
            content = getattr(msg, "content", "")
        else:
            continue
        if role in ("assistant", "ai") and content:
            recent_assistant.append(content)
        if len(recent_assistant) >= _LINK_COOLDOWN:
            break

    # If any of the last _LINK_COOLDOWN responses had a link, suppress this one
    if any(_LINK_PATTERN.search(resp) for resp in recent_assistant):
        stripped = _LINK_PATTERN.sub("", answer)
        # Clean up orphaned formatting left behind (e.g. "GitHub:  |")
        stripped = re.sub(r'\|\s*$', '', stripped, flags=re.MULTILINE)
        stripped = re.sub(r':\s*\|', ' |', stripped)
        stripped = re.sub(r'\|\s*\|', '|', stripped)
        stripped = re.sub(r'\s*\|\s*$', '', stripped, flags=re.MULTILINE)
        # Remove lines that are now empty or just whitespace + punctuation
        lines = stripped.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip lines that became just labels with no URL
            if re.match(r'^\s*(GitHub|LinkedIn|GitHub \(.*\)|LinkedIn \(.*\))\s*:?\s*$', line, re.IGNORECASE):
                continue
            cleaned_lines.append(line)
        stripped = '\n'.join(cleaned_lines)
        # Collapse triple+ newlines
        stripped = re.sub(r'\n{3,}', '\n\n', stripped)
        logger.info("Link throttle: suppressed links (cooldown not met)")
        return stripped.strip()

    logger.debug(f"Link throttle: allowed (no links in last {_LINK_COOLDOWN} assistant messages)")
    return answer


# Resume and profile constants
RESUME_DOWNLOAD_URL = "https://noahsaiassistant.vercel.app/resume/Noah_Delacalzada_Resume.pdf"
LINKEDIN_URL = "https://www.linkedin.com/in/noah-delacalzada"
GITHUB_URL = "https://github.com/iNoahCodeGuy"
