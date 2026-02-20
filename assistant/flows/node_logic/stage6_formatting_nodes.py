"""Formatting pipeline nodes - structured answer layout with toggles and enrichments.

This module handles the final presentation layer:
1. format_answer → Structures draft into headings, bullets, toggles, and action blocks
2. Helper functions → Split sources, summarize, build followup suggestions

Design Principles:
- SRP: Only handles presentation, doesn't generate content or retrieve
- Role awareness: Different layouts for technical vs nontechnical audiences
- Depth control: Expandable sections based on user's depth preference
- Action integration: Weaves planned actions (analytics, code, resume) into layout

Performance Characteristics:
- format_answer: ~50-150ms (depends on action count and live API calls)
- Helper functions: <1ms each (simple string manipulation)

Layout Strategy:
- Always: Teaching Takeaways (summary bullets) + Full Walkthrough (toggle)
- Conditional: Live analytics, metrics, diagrams, code, fun facts, resume links
- Always: Sources (toggle) + Where next? (followup prompts)

See: docs/context/DATA_COLLECTION_AND_SCHEMA_REFERENCE.md for presentation rules
"""

import re
import logging
from typing import Dict, Any, List, Set, Tuple, Optional

from assistant.state.conversation_state import ConversationState
from assistant.core.rag_engine import RagEngine
from assistant.flows import content_blocks
from assistant.flows.node_logic.util_code_validation import is_valid_code_snippet

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


def _extract_content_from_message(msg) -> str:
    """Extract content from message (handles dict, LangChain message objects, and strings).

    Args:
        msg: Can be a dict, AIMessage/HumanMessage object, or string

    Returns:
        Content string, or empty string if msg is None/empty
    """
    if not msg:
        return ""
    if isinstance(msg, dict):
        return msg.get("content", "")
    elif hasattr(msg, "content"):
        return getattr(msg, "content", "")
    else:
        return str(msg)


def _retry_generation_if_insufficient(state: ConversationState, rag_engine: RagEngine) -> str:
    """Retry generation with reinforced instructions if answer has quality issues.

    Enhanced to handle:
    1. Original menu option 1 quality checks (comprehensive coverage)
    2. Answer quality warnings (relevance, novelty)
    3. Conversation guidance needs (stuck patterns, progression)

    Args:
        state: Conversation state with draft_answer and quality warnings
        rag_engine: RAG engine for LLM access

    Returns:
        Enhanced answer (or original if retry fails)
    """
    generation_quality_warning = state.get("generation_quality_warning")
    answer_quality_warning = state.get("answer_quality_warning")
    guidance_needed = state.get("conversation_guidance_needed", [])

    # Check if ANY quality issue exists
    has_quality_issue = (generation_quality_warning or
                        answer_quality_warning or
                        guidance_needed)

    if not has_quality_issue:
        # No quality issues - return draft as-is
        draft = state.get("draft_answer")
        return _extract_content_from_message(draft) if draft else ""

    # Determine which type of retry is needed
    original_answer = _extract_content_from_message(state.get("draft_answer", ""))
    retrieved_chunks = state.get("retrieved_chunks", [])

    # CASE 1: Original menu option 1 quality check
    if (generation_quality_warning and
        state.get("query_type") == "menu_selection" and
        state.get("menu_choice") == "1" and
        state.get("role_mode") == "hiring_manager_technical"):

        logger.info(f"🔄 Retrying generation due to menu quality issue: {generation_quality_warning}")

        retry_instructions = (
            "RETRY ATTEMPT - Previous response failed quality check.\n\n"
            f"Issue detected: {generation_quality_warning}\n\n"
            "YOU MUST include ALL 5 layers with these EXACT headings:\n"
            "**Frontend Layer:**\n"
            "**Backend/Orchestration Layer:**\n"
            "**Data Layer:**\n"
            "**Observability Layer:**\n"
            "**Deployment Layer:**\n\n"
            "Each layer needs 2-3 complete sentences. Total target: 300-350 words.\n"
            "DO NOT skip any layer. DO NOT copy verbatim from context."
        )

    # CASE 2: Answer quality warning (relevance or novelty)
    elif answer_quality_warning:
        logger.info(f"🔄 Retrying generation due to answer quality issue: {answer_quality_warning}")

        retry_instructions = "RETRY ATTEMPT - Previous answer had quality issues.\n\n"

        if "answer_relevance_low" in answer_quality_warning:
            # Answer didn't address query properly
            query = state.get("query", "")
            retry_instructions += f"""
CRITICAL: Previous answer was not relevant to the query.

Query: "{query}"

Your answer MUST:
1. Directly address the query's key terms
2. Use specific examples from the retrieved context
3. Stay focused on what the user asked about

DO NOT go off on tangents or repeat previous responses.
"""

        if "answer_too_similar" in answer_quality_warning:
            # Answer was too similar to previous response
            retry_instructions += """
CRITICAL: Previous answer was too similar to an earlier response.

Your answer MUST:
1. Take a DIFFERENT angle or perspective
2. Include NEW information not mentioned before
3. Provide fresh examples or details

DO NOT repeat what you've already said.
"""

    # CASE 3: Conversation guidance needed
    elif guidance_needed:
        logger.info(f"🔄 Retrying generation with conversation guidance: {guidance_needed}")

        retry_instructions = "RETRY ATTEMPT - Conversation guidance needed.\n\n"

        if "stuck_need_redirection" in guidance_needed:
            retry_instructions += """
CRITICAL: User seems stuck in repetitive pattern.

Generate a fresh answer that:
1. Acknowledges the previous discussion briefly
2. Takes the conversation in a NEW direction
3. Offers specific, different followup options

Example: "We've covered the orchestration layer pretty thoroughly. Want to see how this pattern adapts to enterprise use cases like customer support?"
"""

        if "missing_enterprise_guidance" in guidance_needed:
            retry_instructions += """
CRITICAL: User is progressing from orchestration to enterprise adaptation.

Your answer MUST include:
1. Enterprise adaptation examples (customer support, internal docs)
2. How the architecture scales to production
3. Specific guidance on next steps

Followups must guide to customer support, internal docs, or production patterns.
"""

        if "suggest_depth_increase" in guidance_needed:
            retry_instructions += """
CRITICAL: User is ready to go deeper.

Your answer should:
1. Provide more technical depth than previous responses
2. Include code-level details or implementation specifics
3. Offer to show actual node implementations or architecture diagrams
"""

        if "suggest_synthesis" in guidance_needed:
            retry_instructions += """
CRITICAL: User has explored multiple topics - time to synthesize.

Your answer should:
1. Connect the dots between topics they've explored
2. Show the end-to-end flow
3. Explain how the pieces work together
"""

    else:
        # No retry needed for other cases
        return original_answer

    try:
        enhanced_answer = rag_engine.response_generator.generate_contextual_response(
            query=state.get("query", ""),
            context=retrieved_chunks,
            role=state.get("role", ""),
            chat_history=state.get("chat_history", []),
            extra_instructions=retry_instructions,
            model_name=state.get("analytics_metadata", {}).get("selected_model", "claude-sonnet-4-5-20250929")
        )

        # Validate retry succeeded
        word_count = len(enhanced_answer.split())
        required_layers = ["**Frontend Layer:**", "**Backend/Orchestration Layer:**",
                          "**Data Layer:**", "**Observability Layer:**", "**Deployment Layer:**"]
        missing = [l for l in required_layers if l not in enhanced_answer]

        if not missing and word_count >= 250:
            logger.info(f"✅ Retry succeeded: {word_count} words, all 5 layers present")
            return enhanced_answer
        else:
            logger.warning(f"⚠️ Retry still insufficient: {word_count} words, missing {len(missing)} layers")
            return original_answer  # Fall back to original

    except Exception as e:
        logger.error(f"Retry generation failed: {e}")
        return original_answer


# Import retriever functions with graceful fallback
try:
    from assistant.retrieval.import_retriever import (
        detect_import_in_query,
        get_import_explanation,
        search_import_explanations,
    )
    IMPORT_RETRIEVER_AVAILABLE = True
except ImportError:
    logger.warning("Import retriever not available - stack justification disabled")
    IMPORT_RETRIEVER_AVAILABLE = False

    # Stub functions for graceful degradation
    def detect_import_in_query(query: str):
        return None

    def get_import_explanation(import_name: str, role: str):
        return None

    def search_import_explanations(query: str, role: str, top_k: int = 3):
        return []

# Resume and profile constants
RESUME_DOWNLOAD_URL = "https://noahsaiassistant.vercel.app/resume/Noah_Delacalzada_Resume.pdf"
LINKEDIN_URL = "https://www.linkedin.com/in/noah-delacalzada"
GITHUB_URL = "https://github.com/iNoahCodeGuy"


def _format_action_request_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Simple, clean format for resource requests (resume, linkedin, github).

    Returns a minimal response without Teaching Takeaways / Full Walkthrough boilerplate.

    Args:
        state: Conversation state with pending_actions

    Returns:
        Partial state update with formatted answer
    """
    pending_actions = state.get("pending_actions", [])
    action_types = {a.get("type") for a in pending_actions}

    sections = ["Here are Noah's resources:", ""]

    if "send_resume" in action_types:
        resume_link = state.get("resume_signed_url", RESUME_DOWNLOAD_URL)
        sections.append(f"**Resume**: {resume_link}")

    if "send_linkedin" in action_types:
        sections.append(f"**LinkedIn**: {LINKEDIN_URL}")

    if "send_github" in action_types:
        sections.append(f"**GitHub**: {GITHUB_URL}")

    # If only ask_reach_out without any resources, provide context
    if not any(t in action_types for t in ["send_resume", "send_linkedin", "send_github"]):
        sections = ["I'd be happy to help connect you with Noah."]

    if "ask_reach_out" in action_types:
        sections.append("")
        sections.append("Would you like Noah to reach out directly?")

    answer = "\n".join(sections)

    logger.info(f"Formatted action request response: {len(answer)} chars")
    return {"answer": answer}


def _split_answer_and_sources(answer: str) -> tuple[str, str]:
    """Extract sources section from answer if present.

    Args:
        answer: Full answer text (may contain "Sources:" section)

    Returns:
        Tuple of (body_text, sources_text)

    Example:
        >>> _split_answer_and_sources("RAG works...\n\nSources: KB section 1")
        ("RAG works...", "KB section 1")
    """
    if "Sources:" in answer:
        parts = answer.split("Sources:", 1)
        body = parts[0].strip()
        sources = parts[1].strip()
        return body, sources
    return answer.strip(), ""


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

    # Clean up any double spaces or punctuation issues
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s+([,.!?])', r'\1', result)

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


def _add_subcategory_code_blocks(
    sections: List[str],
    active_subcats: List[str],
    rag_engine: RagEngine,
    query: str,
    role: str,
    depth: int
) -> None:
    """Add code blocks and diagrams based on active technical subcategories.

    Subcategory-aware enrichments:
    - state_management_depth → ConversationState dataclass
    - architecture_depth → LangGraph pipeline flow
    - data_pipeline_depth → pgvector retrieval code
    - stack_depth → requirements/dependencies table

    Args:
        sections: List of formatted sections to append to
        active_subcats: Active subcategory names
        rag_engine: RAG engine for code retrieval
        query: User query for context
        role: User role for filtering
        depth: Depth level for toggle defaults
    """
    try:
        # State management: Show ConversationState structure
        if "state_management_depth" in active_subcats:
            results = rag_engine.retrieve_with_code("ConversationState dataclass", role=role)
            snippets = results.get("code_snippets", []) if results else []

            if snippets:
                snippet = snippets[0]
                code_content = snippet.get("content", "")
                if is_valid_code_snippet(code_content):
                    formatted_code = content_blocks.format_code_snippet(
                        code=code_content,
                        file_path="src/state/conversation_state.py",
                        language="python",
                        description="ConversationState schema with all tracked fields",
                    )
                    sections.append("")
                    sections.append(formatted_code)

        # Data pipeline: Show pgvector retrieval
        if "data_pipeline_depth" in active_subcats:
            results = rag_engine.retrieve_with_code("pgvector retrieval", role=role)
            snippets = results.get("code_snippets", []) if results else []

            if snippets:
                snippet = snippets[0]
                code_content = snippet.get("content", "")
                if is_valid_code_snippet(code_content):
                    formatted_code = content_blocks.format_code_snippet(
                        code=code_content,
                        file_path="src/retrieval/pgvector_retriever.py",
                        language="python",
                        description="Vector search with Supabase RPC",
                    )
                    sections.append("")
                    sections.append(formatted_code)

    except Exception as exc:
        logger.warning(f"Subcategory code enrichment failed: {exc}")


def _summarize_answer(text: str, depth: int) -> List[str]:
    """Extract key sentences from answer for summary bullets.

    Handles Q&A format cleanup: If LLM generated answer in Q:/A: format,
    extracts just the answer portions and formats as clean bullets.

    Avoids splitting on numbered list periods (1. 2. 3.) and markdown headers.

    Args:
        text: Answer body text
        depth: User's depth preference (1=brief, 2=detailed, 3=comprehensive)

    Returns:
        List of summary bullet strings

    Example:
        >>> _summarize_answer("First point. Second point. Third point.", depth=1)
        ["- First point.", "- Second point."]
        >>> _summarize_answer("Q: What is RAG? A: Retrieval-Augmented Generation.", depth=1)
        ["- Retrieval-Augmented Generation"]
    """
    # Sentence splitting pattern that avoids:
    # - Numbered lists (1. 2. 3.)
    # - Markdown bold numbered headers (**1. **2.)
    # Uses negative lookbehind to skip periods after digits
    sentence_split_pattern = r'(?<=[.!?])(?<!\d\.)(?<!\*\*\d\.)\s+'

    # Check if answer is in Q&A format and extract just answer portions
    qa_pattern = r'Q:\s*.*?\s*A:\s*(.*?)(?=Q:|$)'
    qa_matches = re.findall(qa_pattern, text, re.DOTALL | re.IGNORECASE)

    if qa_matches:
        # Extract answer portions only, ignore question format
        sentences = []
        for answer_text in qa_matches:
            answer_sentences = [s.strip() for s in re.split(sentence_split_pattern, answer_text.strip()) if s.strip()]
            sentences.extend(answer_sentences)
        logger.debug(f"Detected Q&A format, extracted {len(sentences)} answer sentences")
    else:
        # Normal sentence splitting (avoiding numbered list periods)
        sentences = [s.strip() for s in re.split(sentence_split_pattern, text) if s.strip()]

    # Filter out orphaned markdown headers like "**1." or "**2."
    sentences = [s for s in sentences if not re.match(r'^\*\*\d+\.?\s*$', s)]

    limit = 2 if depth <= 1 else 3
    summary = []
    for sentence in sentences:
        if sentence.lower().startswith("sources:"):
            continue
        # Remove any remaining "Q:" or "A:" prefixes
        sentence = re.sub(r'^[QA]:\s*', '', sentence, flags=re.IGNORECASE).strip()
        # Skip sentences that are just markdown formatting artifacts
        if sentence and not re.match(r'^\*\*\d+\.?\s*$', sentence):
            summary.append(f"- {sentence}")
        if len(summary) >= limit:
            break
    return summary


def _build_subcategory_followups(active_subcats: List[str]) -> List[str]:
    """Generate followup questions based on active technical subcategories.

    Merged from logging_nodes.suggest_followups for inline followup generation.
    Maps subcategory focus to natural next-step questions.

    Args:
        active_subcats: List of active subcategory names

    Returns:
        List of contextual followup questions
    """
    suggestions = []

    if "stack_depth" in active_subcats:
        suggestions.append("Want to compare LangChain vs LlamaIndex trade-offs?")
        suggestions.append("Should I break down the requirements.txt dependencies?")

    if "architecture_depth" in active_subcats:
        suggestions.append("Want the LangGraph node flow diagram?")
        suggestions.append("Should I map this architecture to your team's stack?")

    if "data_pipeline_depth" in active_subcats:
        suggestions.append("Curious about the pgvector RPC implementation?")
        suggestions.append("Want to see the embedding generation flow?")

    if "state_management_depth" in active_subcats:
        suggestions.append("Should I show the ConversationState transitions?")
        suggestions.append("Want to trace a query through the pipeline?")

    # Default fallback if no matches
    if not suggestions:
        suggestions = [
            "Want me to walk through the LangGraph node transitions in detail?",
            "Curious how the Supabase pgvector query works under load?",
            "Should we map this architecture to your internal stack?",
        ]

    return suggestions[:3]  # Return at most 3


def _analyze_conversation_flow(chat_history: List[Dict], session_memory: Dict) -> Dict:
    """Analyze conversation pattern to infer progression.

    Detects conversation patterns like:
    - orchestration → enterprise (Turn 3: orchestration, Turn 4: enterprise)
    - architecture → implementation (architectural discussion → code details)
    - general → specific (broad question → specific follow-up)

    Args:
        chat_history: List of message dicts with 'role' and 'content' keys
        session_memory: Dict with topics, entities, persona_hints, etc.

    Returns:
        Dict with:
        - pattern: Detected pattern name (or None)
        - topics: List of accumulated topics
        - has_progression: Boolean indicating if progression detected

    Example:
        >>> chat_history = [
        ...     {"role": "user", "content": "explain orchestration"},
        ...     {"role": "assistant", "content": "..."},
        ...     {"role": "user", "content": "how is this relevant to enterprise?"}
        ... ]
        >>> session_memory = {"topics": ["orchestration", "technical"]}
        >>> _analyze_conversation_flow(chat_history, session_memory)
        {"pattern": "orchestration_to_enterprise", "topics": [...], "has_progression": True}
    """
    patterns = []
    topics = session_memory.get("topics", []) if session_memory else []
    topics_text = " ".join(topics).lower() if topics else ""

    # Need at least 2 messages to detect patterns
    if not chat_history or len(chat_history) < 2:
        return {
            "pattern": None,
            "topics": topics,
            "has_progression": False
        }

    # Get last few messages for pattern analysis
    recent_messages = chat_history[-4:] if len(chat_history) > 4 else chat_history
    # Extract content from messages (support both dict format and LangGraph message objects)
    recent_contents = []
    for msg in recent_messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
        elif hasattr(msg, "content"):
            content = msg.content if hasattr(msg, "content") else ""
        else:
            content = ""
        recent_contents.append(content.lower())
    recent_content = " ".join(recent_contents)

    # Detect pattern: orchestration → enterprise
    # User asked about orchestration (Turn 3), now asking about enterprise/adaptation (Turn 4)
    if "orchestration" in topics_text or "orchestration" in recent_content:
        # Check recent messages for enterprise/adaptation mentions (support both formats)
        # Expanded keywords to catch "customer support", "adapt", etc.
        enterprise_keywords = ["enterprise", "customer support", "adapt", "use case", "production", "scales"]
        enterprise_mentioned = False
        for msg in recent_messages[-2:]:
            if isinstance(msg, dict):
                content = msg.get("content", "").lower()
            elif hasattr(msg, "content"):
                content = msg.content.lower() if hasattr(msg, "content") else ""
            else:
                content = ""
            if any(kw in content for kw in enterprise_keywords):
                enterprise_mentioned = True
                break
        if enterprise_mentioned:
            patterns.append("orchestration_to_enterprise")

    # Detect pattern: architecture → implementation
    # User asked about architecture, now asking about implementation/code
    if "architecture" in recent_content or "architecture" in topics_text:
        if any(kw in recent_content for kw in ["code", "implementation", "how is this built", "show me"]):
            patterns.append("architecture_to_implementation")

    # Detect pattern: general → specific
    # User started with general question, now asking specific follow-up
    if len(chat_history) >= 4:
        # Compare first query vs recent queries (support both dict format and LangGraph message objects)
        first_msg = chat_history[0]
        recent_msg = chat_history[-1]

        # Extract content and role from first message
        if isinstance(first_msg, dict):
            first_content = first_msg.get("content", "")
            first_role = first_msg.get("role", "") or first_msg.get("type", "")
        else:
            first_content = getattr(first_msg, "content", "") if hasattr(first_msg, "content") else ""
            first_role = getattr(first_msg, "role", "") or getattr(first_msg, "type", "") if hasattr(first_msg, "role") or hasattr(first_msg, "type") else ""
        first_query = first_content.lower() if (first_role == "user" or first_role == "human") else ""

        # Extract content and role from recent message
        if isinstance(recent_msg, dict):
            recent_content = recent_msg.get("content", "")
            recent_role = recent_msg.get("role", "") or recent_msg.get("type", "")
        else:
            recent_content = getattr(recent_msg, "content", "") if hasattr(recent_msg, "content") else ""
            recent_role = getattr(recent_msg, "role", "") or getattr(recent_msg, "type", "") if hasattr(recent_msg, "role") or hasattr(recent_msg, "type") else ""
        recent_query = recent_content.lower() if (recent_role == "user" or recent_role == "human") else ""

        # First query is general (short, simple), recent query is specific (longer, more keywords)
        if first_query and recent_query:
            first_word_count = len(first_query.split())
            recent_word_count = len(recent_query.split())
            if first_word_count <= 5 and recent_word_count > 5:
                patterns.append("general_to_specific")

    return {
        "pattern": patterns[0] if patterns else None,
        "topics": topics,
        "has_progression": len(patterns) > 0
    }


def _should_offer_synthesis(session_memory: Dict, conversation_turn: int, conversation_phase: str) -> bool:
    """Check if we should offer to synthesize across topics.

    Synthesis is offered:
    - In synthesis phase (8+ turns, 4+ topics)
    - OR at 5+ turns with 3+ topics (if not in synthesis phase yet)
    - But not too frequently (wait 3 turns between offers)

    Args:
        session_memory: Session memory with topics
        conversation_turn: Current conversation turn
        conversation_phase: Current conversation phase

    Returns:
        True if synthesis should be offered, False otherwise
    """
    topics = session_memory.get("topics", []) if session_memory else []

    # Always offer in synthesis phase
    if conversation_phase == "synthesis":
        last_synthesis = session_memory.get("last_synthesis_turn", 0) if session_memory else 0
        if conversation_turn - last_synthesis >= 3:
            return True

    # Offer at 5+ turns with 3+ topics
    if conversation_turn >= 5 and len(topics) >= 3:
        last_synthesis = session_memory.get("last_synthesis_turn", 0) if session_memory else 0
        if conversation_turn - last_synthesis >= 3:
            return True

    return False


# Extended conversation arcs for indefinite interactions (100+ turns)
# Each tuple: (current_topic, [next_topic_options])
# Arcs create guided journeys through the knowledge base
CONVERSATION_ARCS = {
    "hiring_manager_technical": [
        # Phase 1: Architecture Foundation (turns 1-10)
        ("tech_stack", ["orchestration", "rag_pipeline", "data_layer"]),
        ("architecture", ["orchestration", "rag_pipeline", "observability"]),
        ("orchestration", ["langgraph_nodes", "state_management", "error_handling"]),
        ("rag_pipeline", ["retrieval", "generation", "grounding"]),
        ("data_layer", ["pgvector", "embeddings", "supabase"]),

        # Phase 2: Deep Dives (turns 10-25)
        ("langgraph_nodes", ["enterprise_adaptation", "cost_analysis", "testing"]),
        ("state_management", ["memory_accumulation", "session_persistence", "scaling"]),
        ("retrieval", ["similarity_search", "query_composition", "reranking"]),
        ("generation", ["prompt_engineering", "model_selection", "streaming"]),
        ("observability", ["langsmith", "analytics", "cost_tracking"]),

        # Phase 3: Enterprise Application (turns 25-50)
        ("enterprise_adaptation", ["customer_support", "internal_docs", "sales_enablement"]),
        ("customer_support", ["role_adaptation", "action_handlers", "metrics"]),
        ("internal_docs", ["knowledge_ingestion", "access_control", "search_quality"]),
        ("sales_enablement", ["crm_integration", "lead_scoring", "personalization"]),
        ("cost_analysis", ["token_optimization", "caching", "batch_processing"]),

        # Phase 4: Production Concerns (turns 50-75)
        ("deployment", ["vercel_serverless", "edge_functions", "cold_starts"]),
        ("scaling", ["horizontal_scaling", "rate_limiting", "load_balancing"]),
        ("testing", ["qa_strategy", "golden_datasets", "regression_testing"]),
        ("monitoring", ["alerting", "dashboards", "incident_response"]),

        # Phase 5: Noah's Background (turns 75+)
        ("noahs_background", ["projects", "philosophy", "tesla_experience"]),
        ("projects", ["portfolia_deep_dive", "other_projects", "github"]),
        ("philosophy", ["ai_ethics", "learning_approach", "career_goals"]),

        # Circular paths for indefinite conversation
        ("synthesis", ["new_territory", "deeper_dive", "noah_connection"]),
        ("new_territory", ["tech_stack", "enterprise_adaptation", "noahs_background"]),
    ],
    "software_developer": [
        # Phase 1: Code Architecture
        ("architecture", ["code_structure", "module_design", "dependencies"]),
        ("code_structure", ["flows", "nodes", "state_management"]),
        ("module_design", ["rag_engine", "response_generator", "retrievers"]),

        # Phase 2: Implementation Details
        ("implementation", ["langgraph_flow", "pgvector_queries", "prompt_templates"]),
        ("langgraph_flow", ["node_logic", "edge_conditions", "state_transitions"]),
        ("pgvector_queries", ["similarity_search", "indexing", "optimization"]),

        # Phase 3: Quality & Testing
        ("debugging", ["tracing", "logging", "error_handling"]),
        ("testing", ["pytest", "mocking", "integration_tests"]),
        ("performance", ["latency", "caching", "async_patterns"]),

        # Phase 4: DevOps
        ("deployment", ["vercel", "environment_config", "secrets_management"]),
        ("monitoring", ["langsmith", "supabase_analytics", "alerts"]),

        # Circular
        ("synthesis", ["architecture", "implementation", "noahs_code"]),
    ],
    "hiring_manager_nontechnical": [
        # Phase 1: Career Overview
        ("career", ["achievements", "growth_trajectory", "unique_value"]),
        ("achievements", ["tesla_impact", "project_outcomes", "metrics"]),

        # Phase 2: Business Value
        ("business_impact", ["roi_examples", "efficiency_gains", "team_value"]),
        ("team_fit", ["collaboration", "communication", "leadership"]),

        # Phase 3: Conversion
        ("resume", ["skills_summary", "experience_highlights", "contact"]),
        ("contact", ["availability", "next_steps", "references"]),

        # Circular
        ("synthesis", ["career", "business_impact", "unique_value"]),
    ],
    "explorer": [
        # Phase 1: Casual Exploration
        ("casual", ["what_is_this", "how_it_works", "who_built_it"]),
        ("what_is_this", ["architecture_overview", "cool_features", "demo"]),

        # Phase 2: Deeper Interest
        ("architecture", ["rag_basics", "ai_explained", "under_the_hood"]),
        ("behind_scenes", ["how_i_think", "my_memory", "my_limitations"]),

        # Phase 3: Fun Stuff
        ("fun_facts", ["mma_background", "chess_ai_story", "hidden_features"]),
        ("mma_background", ["fight_record", "training", "discipline"]),

        # Circular
        ("synthesis", ["casual", "architecture", "fun_facts"]),
    ]
}


# ============================================================================
# MAIN KNOWLEDGE PILLARS - Core topics to guide users through
# ============================================================================

MAIN_PILLARS = {
    "hiring_manager_technical": {
        "pillars": [
            # 5 pillars for technical hiring managers
            ("orchestration", "Explore the orchestration layer — how nodes, states, and safeguards work together"),
            ("tech_stack", "See my full tech stack — frontend, backend, observability"),
            ("enterprise", "See how this architecture adapts for enterprise use cases"),
            ("data_pipeline", "Learn about data pipeline management — embeddings, vector storage, analytics"),
            ("noahs_background", "Learn about Noah's technical background and projects"),
        ],
        "topic_to_pillar": {
            # Orchestration pillar (the "brain")
            "orchestration": "orchestration", "langgraph": "orchestration", "nodes": "orchestration",
            "state_management": "orchestration", "memory": "orchestration", "pipeline": "orchestration",
            "flow": "orchestration", "langgraph_nodes": "orchestration", "state": "orchestration",
            "safeguards": "orchestration", "conversation_flow": "orchestration",

            # Tech stack pillar (architecture, frontend, backend, observability)
            "architecture": "tech_stack", "tech_stack": "tech_stack", "frontend": "tech_stack",
            "backend": "tech_stack", "observability": "tech_stack", "langsmith": "tech_stack",
            "rag_pipeline": "tech_stack", "rag": "tech_stack", "supabase": "tech_stack",

            # Data pipeline pillar (separate from tech_stack)
            "data_pipeline": "data_pipeline", "data_flow": "data_pipeline", "data flow": "data_pipeline",
            "embedding": "data_pipeline", "embeddings": "data_pipeline", "vector": "data_pipeline",
            "pgvector": "data_pipeline", "data_layer": "data_pipeline", "data": "data_pipeline",
            "knowledge_base": "data_pipeline", "knowledge base": "data_pipeline", "migration": "data_pipeline",
            "ingest": "data_pipeline", "ingestion": "data_pipeline", "chunking": "data_pipeline",
            "analytics": "data_pipeline", "logging": "data_pipeline", "tracking": "data_pipeline",
            "metrics": "data_pipeline", "dashboard": "data_pipeline",

            # Enterprise pillar
            "enterprise": "enterprise", "enterprise_adaptation": "enterprise", "customer_support": "enterprise",
            "internal_docs": "enterprise", "sales_enablement": "enterprise", "deployment": "enterprise",
            "cost_analysis": "enterprise", "scaling": "enterprise", "cost": "enterprise",
            "production": "enterprise", "adapt": "enterprise",

            # Noah's background pillar
            "noahs_background": "noahs_background", "career": "noahs_background", "noah": "noahs_background",
            "resume": "noahs_background", "projects": "noahs_background", "tesla": "noahs_background",
            "background": "noahs_background", "skills": "noahs_background", "experience": "noahs_background",
        }
    },
    "software_developer": {
        "pillars": [
            # Lead with implementation (code-focused users want to see code first)
            ("implementation", "Explore the LangGraph implementation — nodes, state, pgvector queries"),
            ("architecture", "See the code architecture — modules, flows, design patterns"),
            ("debugging", "See the debugging and testing strategies"),
            ("noahs_background", "Learn about Noah's technical background"),
        ],
        "topic_to_pillar": {
            # Implementation pillar (first - show the code)
            "implementation": "implementation", "langgraph_flow": "implementation", "pgvector_queries": "implementation",
            "code": "implementation", "langgraph": "implementation", "stategraph": "implementation",
            # Data pipeline keywords for developers
            "data_pipeline": "implementation", "embedding": "implementation", "retrieval": "implementation",
            "vector_search": "implementation", "similarity": "implementation",

            # Architecture pillar
            "architecture": "architecture", "code_structure": "architecture", "module_design": "architecture",
            "design_patterns": "architecture", "modules": "architecture", "flows": "architecture",

            # Debugging pillar
            "debugging": "debugging", "testing": "debugging", "performance": "debugging",
            "pytest": "debugging", "error_handling": "debugging", "tracing": "debugging",

            # Noah's background
            "noahs_background": "noahs_background", "career": "noahs_background", "noah": "noahs_background",
        }
    },
    "hiring_manager_nontechnical": {
        "pillars": [
            ("career", "See Noah's career journey and achievements"),
            ("business_impact", "Understand the business value Noah delivers"),
            ("team_fit", "See how Noah works with teams"),
            ("resume", "Get Noah's resume and contact info"),
        ],
        "topic_to_pillar": {
            "career": "career", "achievements": "career", "growth_trajectory": "career",
            "business_impact": "business_impact", "roi_examples": "business_impact", "efficiency_gains": "business_impact",
            "team_fit": "team_fit", "collaboration": "team_fit", "communication": "team_fit",
            "resume": "resume", "contact": "resume", "availability": "resume",
        }
    },
    "explorer": {
        "pillars": [
            ("what_is_this", "Discover what I am and how I work"),
            ("architecture", "See the technology behind me"),
            ("fun_facts", "Learn some fun facts about Noah"),
            ("noahs_background", "Learn about Noah's background"),
        ],
        "topic_to_pillar": {
            "casual": "what_is_this", "what_is_this": "what_is_this", "how_it_works": "what_is_this",
            "architecture": "architecture", "behind_scenes": "architecture", "rag_basics": "architecture",
            "fun_facts": "fun_facts", "mma_background": "fun_facts", "hidden_features": "fun_facts",
            "noahs_background": "noahs_background", "who_built_it": "noahs_background",
        }
    },
}


def _get_explored_pillars(session_memory: Dict, role_mode: str) -> Set[str]:
    """Get which main pillars have been explored based on topics discussed.

    Args:
        session_memory: Session memory dict containing topics list
        role_mode: User's role mode for pillar mapping

    Returns:
        Set of explored pillar keys
    """
    topics = session_memory.get("topics", [])
    pillar_config = MAIN_PILLARS.get(role_mode, {})
    topic_to_pillar = pillar_config.get("topic_to_pillar", {})

    explored = set()
    for topic in topics:
        topic_lower = topic.lower()
        if topic_lower in topic_to_pillar:
            explored.add(topic_to_pillar[topic_lower])

    return explored


def _get_unexplored_pillars(session_memory: Dict, role_mode: str) -> List[Tuple[str, str]]:
    """Get unexplored pillars with user-friendly prompts.

    Args:
        session_memory: Session memory dict containing topics list
        role_mode: User's role mode for pillar mapping

    Returns:
        List of (pillar_key, user_friendly_prompt) tuples for unexplored pillars
    """
    pillar_config = MAIN_PILLARS.get(role_mode, {})
    all_pillars = pillar_config.get("pillars", [])
    explored = _get_explored_pillars(session_memory, role_mode)

    return [(key, prompt) for key, prompt in all_pillars if key not in explored]


def _get_depth_options_for_pillar(pillar: str, role_mode: str = "hiring_manager_technical") -> List[str]:
    """Get specific depth options for each pillar.

    Used when user is exploring a pillar to offer meaningful next steps
    that go deeper into that specific area.

    Args:
        pillar: The pillar key (orchestration, tech_stack, enterprise, data_pipeline, noahs_background)
        role_mode: User's role mode for context

    Returns:
        List of specific depth prompts for the pillar
    """
    depth_map = {
        "orchestration": [
            "Dive into specific nodes like retrieve_chunks or generate_draft",
            "See how state flows through the pipeline"
        ],
        "tech_stack": [
            "Explore the Supabase setup in detail",
            "See the LangSmith observability implementation"
        ],
        "enterprise": [
            "See customer support adaptation patterns",
            "Explore production safeguards"
        ],
        "data_pipeline": [
            "See how embeddings are generated and stored",
            "Explore the analytics logging system"
        ],
        "noahs_background": [
            "See Noah's GitHub projects",
            "Learn about his certifications and training path"
        ],
    }
    return depth_map.get(pillar, [f"Go deeper into {pillar}"])


def _extract_current_topic_from_query(query: str, role_mode: str = "hiring_manager_technical") -> Optional[str]:
    """Extract the pillar being discussed in current query.

    This is used to:
    1. Filter the current topic OUT of follow-up suggestions (don't suggest what user just asked)
    2. Mark the current pillar as explored BEFORE generating follow-ups

    Args:
        query: User's current query
        role_mode: User's role mode for context

    Returns:
        Pillar key if detected (orchestration, tech_stack, enterprise, data_pipeline, noahs_background), None otherwise
    """
    query_lower = query.lower()

    # Noah's background pillar keywords
    if any(kw in query_lower for kw in [
        "noah", "background", "career", "resume", "tesla", "experience",
        "skills", "certifications", "projects", "github", "linkedin",
        "who built", "about noah", "noah's"
    ]):
        return "noahs_background"

    # Orchestration pillar keywords
    if any(kw in query_lower for kw in [
        "orchestration", "nodes", "state management", "memory", "langgraph",
        "pipeline", "flow", "safeguards", "conversation flow", "state graph",
        "how do you think", "how does the pipeline"
    ]):
        return "orchestration"

    # Data pipeline pillar keywords (separate from tech_stack)
    if any(kw in query_lower for kw in [
        "data pipeline", "embedding", "embeddings", "vector", "pgvector",
        "data flow", "chunking", "analytics", "ingestion", "ingest",
        "knowledge base", "migration", "metrics", "logging", "tracking"
    ]):
        return "data_pipeline"

    # Tech stack pillar keywords (architecture, frontend, backend, observability)
    if any(kw in query_lower for kw in [
        "tech stack", "architecture", "frontend", "backend", "rag",
        "supabase", "observability", "langsmith", "full stack", "retrieval"
    ]):
        return "tech_stack"

    # Enterprise pillar keywords
    if any(kw in query_lower for kw in [
        "enterprise", "adapt", "customer support", "scaling", "production",
        "deployment", "cost", "internal docs", "sales enablement",
        "how would this scale", "enterprise use"
    ]):
        return "enterprise"

    # For software_developer role, check implementation-specific keywords
    if role_mode == "software_developer":
        if any(kw in query_lower for kw in ["code", "implementation", "show me the code"]):
            return "implementation"
        if any(kw in query_lower for kw in ["debug", "test", "error"]):
            return "debugging"

    return None


# ============================================================================
# OUT-OF-SCOPE DETECTION - Detect queries outside Portfolia's knowledge base
# ============================================================================

OUT_OF_SCOPE_INDICATORS = [
    # Generic tech topics Portfolia doesn't cover in depth
    "kubernetes", "docker compose", "terraform", "ansible", "helm",
    "react native", "flutter", "swift", "kotlin", "java", "c++", "rust",
    "machine learning theory", "neural network math", "deep learning theory",
    "aws lambda", "azure functions", "gcp cloud run",
    # Personal opinions
    "what do you think about", "your opinion on", "best programming language",
    "should i use", "which is better",
    # Completely unrelated
    "weather", "sports", "news", "politics", "stock market", "crypto",
]

IN_SCOPE_BRIDGES = {
    # Map out-of-scope topics to in-scope alternatives with bridge prompts
    "kubernetes": ("tech_stack", "My deployment uses Vercel serverless rather than Kubernetes. Want to see how the deployment architecture works?"),
    "docker": ("tech_stack", "I'm deployed on Vercel Edge Functions rather than Docker containers. Want to see the serverless architecture?"),
    "machine learning": ("tech_stack", "I use RAG instead of fine-tuning - it's more cost-effective for knowledge-grounded responses. Want to see how the RAG pipeline works?"),
    "react native": ("tech_stack", "My frontend is Next.js for web. Want to see the chat interface architecture?"),
    "aws": ("tech_stack", "I'm deployed on Vercel with Supabase. Want to see how the cloud architecture works?"),
    "azure": ("tech_stack", "I'm deployed on Vercel with Supabase. Want to see the deployment architecture?"),
    "gcp": ("tech_stack", "I'm deployed on Vercel with Supabase. Want to see the cloud architecture?"),
    "fine-tuning": ("tech_stack", "I use RAG instead of fine-tuning - want to see why that's a better fit for many enterprise use cases?"),
    "best programming language": ("noahs_background", "I'm built with Python, TypeScript, and SQL. Want to learn about Noah's technical skills?"),
}

def detect_out_of_scope(query: str, role_mode: str = "explorer") -> Tuple[bool, Optional[str], Optional[str]]:
    """Detect if query is outside Portfolia's knowledge base and suggest bridge topic.

    Args:
        query: User's query string
        role_mode: User's role mode for context

    Returns:
        Tuple of (is_out_of_scope, bridge_pillar, bridge_prompt)
        - is_out_of_scope: True if query is outside knowledge base
        - bridge_pillar: Suggested pillar to redirect to (or None)
        - bridge_prompt: User-friendly prompt to redirect (or None)
    """
    query_lower = query.lower()

    # Check for out-of-scope indicators
    for indicator in OUT_OF_SCOPE_INDICATORS:
        if indicator in query_lower:
            # Check for bridge to in-scope topic
            bridge = IN_SCOPE_BRIDGES.get(indicator)
            if bridge:
                return True, bridge[0], bridge[1]

            # Generic out-of-scope response
            return True, None, (
                "That's outside my knowledge base - I'm focused on demonstrating "
                "Noah's GenAI engineering skills through my own architecture. "
                "Want to explore how I'm built instead?"
            )

    return False, None, None


def _predict_next_topics(role_mode: str, current_topics: List[str], conversation_phase: str) -> List[str]:
    """Predict likely next topics based on conversation arc and phase.

    Args:
        role_mode: User's role mode
        current_topics: List of current topics (use last 3-5)
        conversation_phase: Current conversation phase

    Returns:
        List of predicted next topics
    """
    arcs = CONVERSATION_ARCS.get(role_mode, [])

    # Use most recent topic to predict next
    if current_topics:
        recent_topic = current_topics[-1].lower()
        for topic, next_topics in arcs:
            if topic in recent_topic:
                # Filter based on phase
                if conversation_phase == "synthesis":
                    # In synthesis phase, suggest connecting topics
                    return ["synthesis", "big_picture", "connect_dots"]
                return next_topics

    return []


def _generate_synthesis_response(state: ConversationState) -> str:
    """Generate synthesis response connecting multiple turns.

    Uses conversational style (not teaching) to connect dots.

    Args:
        state: Conversation state with chat_history, session_memory

    Returns:
        Synthesis response string
    """
    session_memory = state.get("session_memory", {})
    topics = session_memory.get("topics", [])
    chat_history = state.get("chat_history", [])

    # Extract turn-by-turn summary
    turn_summaries = []
    turn_num = 1
    for i in range(0, len(chat_history), 2):
        if i < len(chat_history):
            user_msg = chat_history[i]
            if isinstance(user_msg, dict):
                query = user_msg.get("content", "")[:100]
            else:
                query = getattr(user_msg, "content", "")[:100] if hasattr(user_msg, "content") else ""

            if query:
                turn_summaries.append(f"Turn {turn_num}: {query}")
                turn_num += 1

    # Synthesize topics
    topics_text = ", ".join(topics[-5:]) if topics else "various topics"

    response = f"""**Connecting the Dots**

We've covered a lot of ground! Let me synthesize:

{chr(10).join(turn_summaries[-5:]) if turn_summaries else "We've explored several topics."}

**The Big Picture:**
We've explored {topics_text}. These topics connect because [synthesis explanation].

**What's Left to Explore:**
- [Suggest 3 natural next steps based on conversation arc]

Want me to dive deeper into any of these?"""

    return response


def _build_followups(variant: str, intent: str = "general", active_subcats: List[str] = None, role_mode: str = "explorer", chat_history: List[Dict] = None, session_memory: Dict = None, quality_warnings: str = None, guidance_flags: List[str] = None, conversation_phase: str = None, query: str = None) -> List[str]:
    """Generate role-specific followup prompt suggestions with conversation guidance and phase awareness.

    Enhanced with conversation guidance for quality issues, natural progression, and phase-aware suggestions.
    Uses sliding window analysis (last 6 messages) for scalability with 100+ turns.

    CRITICAL: Filters out the current topic from suggestions - we don't suggest what user just asked about!

    Args:
        variant: "engineering" | "business" | "mixed" (layout variant)
        intent: Query intent/type (technical, data, career, etc.)
        active_subcats: Active technical subcategories for precision targeting
        role_mode: User's role (explorer, software_developer, etc.)
        chat_history: Conversation history (only last 6 messages analyzed)
        session_memory: Session memory with topics (only last 10 topics used)
        quality_warnings: Quality warning string from validate_answer_quality
        guidance_flags: List of guidance flags from validate_conversation_guidance
        conversation_phase: Current phase: discovery/exploration/synthesis/extended
        query: Current user query (used to filter current topic from suggestions)

    Returns:
        List of 3 followup prompt strings

    Example:
        >>> _build_followups("engineering", "technical", guidance_flags=["stuck_need_redirection"])
        ["Want to explore a different aspect?", ...]
    """
    active_subcats = active_subcats or []
    chat_history = chat_history or []
    session_memory = session_memory or {}
    guidance_flags = guidance_flags or []

    # Confession mode gets no followups (privacy)
    if role_mode == "confession":
        return []

    # =========================================================================
    # PRIORITY 0: PILLAR-AWARE GUIDANCE (guide users through 4 main pillars)
    # =========================================================================

    # CRITICAL: Get current topic from query to EXCLUDE from suggestions
    # We should never suggest what the user just asked about!
    current_topic_pillar = _extract_current_topic_from_query(query or "", role_mode) if query else None

    unexplored_pillars = _get_unexplored_pillars(session_memory, role_mode)
    explored_pillars = _get_explored_pillars(session_memory, role_mode)
    topics = session_memory.get("topics", []) if session_memory else []

    # Filter OUT the current topic from unexplored pillars
    # Don't suggest "Learn about Noah's background" when user just asked about Noah!
    if current_topic_pillar:
        unexplored_pillars = [(key, prompt) for key, prompt in unexplored_pillars if key != current_topic_pillar]
        logger.debug(f"Filtered current topic '{current_topic_pillar}' from follow-up suggestions")

    # If current topic WAS the only unexplored pillar, user is about to complete all 5
    # Offer depth on current topic + synthesis instead of falling through to generic
    if not unexplored_pillars and current_topic_pillar:
        logger.info(f"User exploring last pillar '{current_topic_pillar}' - offer depth + synthesis")
        depth_options = _get_depth_options_for_pillar(current_topic_pillar, role_mode)
        return [
            depth_options[0] if depth_options else f"Go deeper into {current_topic_pillar}",
            "See how all these pieces connect end-to-end",
            "Ready to see Noah's resume and LinkedIn?"
        ]

    # Check for pillar depth warning (user has been in same pillar for 3+ turns)
    pillar_depth = session_memory.get("pillar_depth", {}) if session_memory else {}
    deep_pillar = None
    for pillar, depth in pillar_depth.items():
        if depth >= 3 and pillar != current_topic_pillar:
            deep_pillar = pillar
            break

    # If user has gone deep (3+ turns) on one pillar, actively guide to others
    if deep_pillar and unexplored_pillars:
        logger.info(f"User deep in '{deep_pillar}' pillar - actively guiding to other pillars")
        pillar_prompts = []
        pillar_prompts.append(f"You've explored {deep_pillar} in depth! {unexplored_pillars[0][1]}")
        if len(unexplored_pillars) > 1:
            pillar_prompts.append(unexplored_pillars[1][1])
        pillar_prompts.append("See how all the pieces connect end-to-end")
        return pillar_prompts[:3]

    # =========================================================================
    # FIX 6: SMART FOLLOWUPS - Always: 1 depth option + 1-2 unexplored pillars
    # Per spec: "would you like me to list all my nodes and states or would you
    # like to see how enterprises use similar programs for customer support?"
    # =========================================================================
    if current_topic_pillar or topics:
        smart_followups = []

        # 1. First option: Depth into current topic
        if current_topic_pillar:
            depth_options = _get_depth_options_for_pillar(current_topic_pillar, role_mode)
            if depth_options:
                smart_followups.append(depth_options[0])
            else:
                smart_followups.append(f"Go deeper into {current_topic_pillar}")
        elif topics:
            # Fallback to most recent topic
            smart_followups.append(f"Go deeper into {topics[-1]}")

        # 2. Second option: First unexplored pillar
        if unexplored_pillars:
            smart_followups.append(unexplored_pillars[0][1])

        # 3. Third option: Second unexplored pillar OR synthesis
        if len(unexplored_pillars) > 1:
            smart_followups.append(unexplored_pillars[1][1])
        elif len(explored_pillars) >= 3:
            smart_followups.append("See how all these pieces connect end-to-end")
        elif current_topic_pillar != "noahs_background":
            smart_followups.append("Learn about Noah's technical background and projects")

        logger.debug(f"Smart follow-ups (Fix 6): depth + unexplored pillars: {smart_followups}")
        return smart_followups[:3]

    # If all pillars explored (5), offer synthesis and resume
    if len(explored_pillars) >= 5:
        logger.info("All 5 pillars explored - offering synthesis and resume")
        return [
            "Want to see how all these pieces connect end-to-end?",
            "Ready to see Noah's resume and LinkedIn?",
            "Should we go deeper into any specific area?"
        ]

    # =========================================================================
    # PRIORITY 1: Quality-based guidance (if user is stuck or answer was repetitive)
    # =========================================================================
    if quality_warnings and "answer_too_similar" in quality_warnings:
        # User is stuck - suggest unexplored pillars instead of generic topics
        if unexplored_pillars:
            return [prompt for _, prompt in unexplored_pillars[:3]]
        return [
            "Want to explore a different aspect of the system?",
            "Curious about the data layer or deployment instead?",
            "Should we look at production patterns or enterprise use cases?"
        ]

    # =========================================================================
    # PRIORITY 1.5: Context-aware follow-ups based on current topic
    # Guide back to unexplored pillars instead of going deeper into sub-topics
    # =========================================================================
    if chat_history:
        last_user_msg = None
        for msg in reversed(chat_history[-6:]):  # Check last 6 messages
            if isinstance(msg, dict):
                if msg.get("role") == "user" or msg.get("type") == "human":
                    last_user_msg = msg.get("content", "")
                    break
            elif hasattr(msg, "type") and (msg.type == "human" or getattr(msg, "role", None) == "user"):
                last_user_msg = getattr(msg, "content", "")
                break

        if last_user_msg:
            query_lower = last_user_msg.lower()

            # Check if user is going deep into enterprise - guide back to other pillars
            is_enterprise_query = any(kw in query_lower for kw in ["adapt", "adapts", "adaptation", "customer support", "enterprise", "use case"])

            if is_enterprise_query and unexplored_pillars:
                # Instead of going deeper into enterprise, offer unexplored pillars
                pillar_prompts = [prompt for _, prompt in unexplored_pillars[:2]]
                pillar_prompts.append("Go deeper into enterprise adaptation patterns")
                return pillar_prompts[:3]

    # =========================================================================
    # PRIORITY 2: Conversation guidance flags (from validate_conversation_guidance)
    # =========================================================================
    if guidance_flags:
        if "stuck_need_redirection" in guidance_flags:
            return [
                "Want to explore a different topic area?",
                "Curious about the backend architecture or data pipeline?",
                "Should we look at testing strategies or deployment patterns?"
            ]

        if "missing_enterprise_guidance" in guidance_flags:
            return [
                "Want to see how this adapts to customer support?",
                "Curious about enterprise deployment patterns?",
                "Should I show the production safeguards that make this enterprise-ready?"
            ]

        if "suggest_depth_increase" in guidance_flags:
            return [
                "Want to go deeper into the implementation details?",
                "Curious about the code-level architecture?",
                "Should I show the actual node implementations?"
            ]

        if "suggest_synthesis" in guidance_flags:
            return [
                "Want to see how these pieces connect together?",
                "Curious about the end-to-end flow across all these topics?",
                "Should I synthesize how the architecture, data, and deployment work together?"
            ]

        if "suggest_new_territory" in guidance_flags:
            return [
                "We've covered a lot! Want to explore something completely different?",
                "Curious about an area we haven't touched yet?",
                "Should we revisit any topic with a fresh perspective?"
            ]

    # PRIORITY 3: Phase-aware suggestions (when no specific guidance flags)
    if conversation_phase == "synthesis":
        return [
            "Want to see how all these pieces connect end-to-end?",
            "Ready for a synthesis of what we've covered?",
            "Should I map out how the architecture, data, and deployment work together?"
        ]

    if conversation_phase == "extended":
        return [
            "We've covered a lot! Want to explore something completely different?",
            "Curious about an area we haven't touched yet?",
            "Should we revisit any topic with fresh perspective?"
        ]

    # PRIORITY 3: Conversation flow analysis for natural progression
    # Use sliding window: only analyze last 6 messages (scalable for 100+ turns)
    if chat_history and len(chat_history) >= 2:
        # Analyze conversation pattern to infer progression (scalable version)
        flow_analysis = _analyze_conversation_flow(chat_history, session_memory)

        # Use conversation arc prediction for proactive follow-ups
        topics = session_memory.get("topics", []) if session_memory else []
        predicted_topics = _predict_next_topics(role_mode, topics[-3:], conversation_phase)

        if predicted_topics:
            # Generate follow-ups based on predicted next topics
            if "synthesis" in predicted_topics or "big_picture" in predicted_topics:
                return [
                    "Want to see how all these pieces connect end-to-end?",
                    "Ready for a synthesis of what we've covered?",
                    "Should I map out how everything works together?"
                ]
            elif "enterprise_adaptation" in predicted_topics:
                return [
                    "Want to see how this adapts to enterprise use cases?",
                    "Curious about customer support or internal documentation applications?",
                    "Should I show how the architecture scales to different domains?"
                ]
            elif "implementation" in predicted_topics or "code_walkthrough" in predicted_topics:
                return [
                    "Want to see the actual code implementation?",
                    "Curious about the code-level architecture?",
                    "Should I show the node implementations?"
                ]
            elif "cost_analysis" in predicted_topics:
                return [
                    "Want to see the cost breakdown at scale?",
                    "Curious about the pricing model?",
                    "Should I show the ROI calculations?"
                ]

        # Generate context-aware followups based on progression pattern
        if flow_analysis.get("pattern") == "orchestration_to_enterprise":
            return [
                "Want to see how orchestration scales in enterprise deployments?",
                "Curious about the enterprise patterns built on this architecture?",
                "Should I show the production safeguards that make this enterprise-ready?"
            ]
        elif flow_analysis.get("pattern") == "architecture_to_implementation":
            return [
                "Want to see the code behind this architecture?",
                "Curious about the specific implementation details?",
                "Should I walk through the technical implementation?"
            ]
        elif flow_analysis.get("pattern") == "general_to_specific":
            # User is getting more specific - offer deeper dives
            return [
                "Want to dive deeper into the technical details?",
                "Curious about how this works under the hood?",
                "Should I show you the implementation or walk through the code?"
            ]

    # Subcategory-specific followups take priority for technical queries
    if intent in {"technical", "engineering"} and active_subcats:
        return _build_subcategory_followups(active_subcats)

    # Intent-based followups
    if intent in {"technical", "engineering"}:
        return [
            "Want me to walk through the LangGraph node transitions in detail?",
            "Curious how the Supabase pgvector query works under load?",
            "Should we map this architecture to your internal stack?",
        ]
    elif intent in {"data", "analytics"}:
        return [
            "Need the retrieval accuracy metrics for last week?",
            "Want the cost-per-query breakdown?",
            "Should we compare grounding confidence across roles?",
        ]
    elif intent in {"career", "general"}:
        return [
            "Want the story behind building this assistant end to end?",
            "Should I outline Noah's production launch checklist?",
            "Curious how this adapts to your team's workflow?",
        ]

    # Variant-based fallback (original logic)
    if variant == "engineering":
        return [
            "Walk through the LangGraph node transitions",
            "Inspect the pgvector retrieval implementation",
            "Map this pattern onto your stack",
        ]
    if variant == "business":
        return [
            "Review the rollout checklist for enterprise teams",
            "Estimate cost savings for your workflow",
            "Explore adoption risks and mitigation steps",
        ]
    return [
        "See how the architecture could be adapted to customer support",
        "Peek at the analytics dashboard",
        "Ask for the testing and QA strategy",
    ]


def format_answer(state: ConversationState, rag_engine: RagEngine) -> Dict[str, Any]:
    """Structure the draft answer using headings, bullets, and toggles.

    This is the final presentation layer that transforms a plain LLM-generated
    answer into a rich, structured response with:
    - Teaching Takeaways (summary bullets)
    - Full Walkthrough (expandable detailed explanation)
    - Conditional enrichments based on pending actions:
        * Live analytics snapshot
        * Cost/latency/grounding metrics
        * Engineering sequence diagram
        * Enterprise adaptation diagram
        * Code references
        * Import explanations
        * Fun facts
        * MMA fight link
        * LinkedIn/resume links
        * Confession prompt
    - Sources (expandable citations)
    - Where next? (followup prompts)

    Layout control:
    - depth_level: Controls which toggles are open by default
    - layout_variant: "mixed" | "engineering" | "business"
    - pending_actions: List of action dicts with "type" field

    Performance:
    - Baseline: ~20ms (string manipulation only)
    - With live analytics: ~300ms (API call to /api/analytics)
    - With code retrieval: ~150ms (pgvector search)

    Design Principles:
    - SRP: Only handles formatting, doesn't generate or retrieve content
    - Extensibility: Easy to add new action types and enrichments
    - Role awareness: Different blocks for technical vs nontechnical
    - Observability: Logs failures gracefully without crashing pipeline

    Args:
        state: ConversationState with draft_answer and pending_actions
        rag_engine: RAG engine for code retrieval (if needed)

    Returns:
        Updated state with:
        - answer: Fully formatted answer with all enrichments
        - followup_prompts: List of suggested next questions

    Example:
        >>> state = {
        ...     "draft_answer": "RAG works by...",
        ...     "pending_actions": [{"type": "include_code_reference"}],
        ...     "depth_level": 2
        ... }
        >>> format_answer(state, rag_engine)
        >>> "**Teaching Takeaways**" in state["answer"]
        True
    """
    # #region agent log
    with open('/Users/noahdelacalzada/NoahsAIAssistant/NoahsAIAssistant-/.cursor/debug.log', 'a') as f:
        import json
        import time
        f.write(json.dumps({
            "location": "stage6_formatting_nodes.py:652",
            "message": "format_answer: Entry",
            "data": {
                "has_draft_answer": bool(state.get("draft_answer")),
                "draft_answer_len": len(_extract_content_from_message(state.get("draft_answer"))) if state.get("draft_answer") else 0,
                "has_answer": bool(state.get("answer")),
                "answer_len": len(_extract_content_from_message(state.get("answer"))) if state.get("answer") else 0,
                "draft_answer_preview": _extract_content_from_message(state.get("draft_answer"))[:100] if state.get("draft_answer") else None
            },
            "timestamp": int(time.time() * 1000),
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "C"
        }) + "\n")
    # #endregion

    # GUARDRAIL: Retry generation if quality check failed
    base_answer = _retry_generation_if_insufficient(state, rag_engine)

    if base_answer is None:
        # Use draft_answer from generate_draft() - do NOT fall back to old answer
        # Falling back to old answer would preserve welcome messages from previous turns
        base_answer = state.get("draft_answer")

    # Extract content from base_answer if it's an AIMessage object
    base_answer = _extract_content_from_message(base_answer)

    if base_answer is None or not base_answer:
        logger.error("format_answer called without draft_answer")

        # #region agent log
        with open('/Users/noahdelacalzada/NoahsAIAssistant/NoahsAIAssistant-/.cursor/debug.log', 'a') as f:
            import json
            import time
            f.write(json.dumps({
                "location": "stage6_formatting_nodes.py:681",
                "message": "format_answer: Clearing answer because draft_answer is None",
                "data": {
                    "has_draft_answer": bool(state.get("draft_answer")),
                    "has_answer": bool(state.get("answer")),
                    "base_answer_after_retry": base_answer
                },
                "timestamp": int(time.time() * 1000),
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "C"
            }) + "\n")
        # #endregion

        # Explicitly clear answer instead of returning empty dict
        # Returning {} means no state update, which would preserve old answer
        return {"answer": None, "draft_answer": None}

    if not base_answer:
        # Empty answer - clear it explicitly
        return {"answer": None, "draft_answer": None}

    # PRIORITY: Skip enrichments if answer is already a role welcome message
    # Prevents duplicate content when menu selection shows role-specific welcome
    # BUT: Only skip if this is actually the CURRENT turn's answer, not an old one
    # Check if draft_answer exists - if not, this is an old answer and should be cleared
    welcome_indicators = [
        "Since you selected",
        "You can choose where to start:",
        "What brings you here?",
        "I can focus on the areas most relevant to you"
    ]
    # Only skip formatting if draft_answer exists (meaning this is a new welcome message)
    # If draft_answer is None/missing, this is an old answer and should not be returned
    if state.get("draft_answer") and any(indicator in base_answer for indicator in welcome_indicators):
        # This is a welcome/menu message from generate_draft() - don't add formatting
        # Return partial update dict (not full state) to avoid preserving old fields
        return {"answer": base_answer}

    # SIMPLIFIED FORMAT FOR ACTION REQUESTS (resume, linkedin, github)
    # Skip the full Teaching Takeaways / Full Walkthrough template for simple resource requests
    if state.get("query_type") == "action_request":
        is_repeated = state.get("is_repeated_action_request", False)
        # #region agent log
        debug_trace = state.get("_debug_trace", [])
        debug_trace.append({"loc": "format_answer:action_request", "is_repeated": is_repeated, "base_answer_preview": base_answer[:100] if base_answer else None})
        # #endregion

        # If this is a repeated action request, just return the acknowledgment
        if is_repeated:
            debug_trace.append({"loc": "format_answer:returning_ack", "msg": "Returning repeated action acknowledgment"})
            return {"answer": base_answer, "_debug_trace": debug_trace}

        result = _format_action_request_response(state)
        result["_debug_trace"] = debug_trace
        return result

    depth = state.get("depth_level", 1)
    layout_variant = state.get("layout_variant", "mixed")
    pending_actions = state.get("pending_actions", [])
    action_types = {action["type"] for action in pending_actions}
    query = state.get("query", "")
    role = state.get("role", "Just looking around")

    body_text, sources_text = _split_answer_and_sources(base_answer)
    # Remove chunk citation phrases that break first-person narrative
    body_text = _remove_chunk_citations(body_text)
    # Remove markdown headers that leak from KB chunks
    body_text = _remove_markdown_headers(body_text)
    # Strip italic emphasis (*word* -> word) while preserving bold
    body_text = _strip_italic_emphasis(body_text)
    # Strip menu-style endings ("Want X or Y?")
    logger.info(
        "DIAG format_answer BEFORE _strip_menu_endings | last_120: '%s'",
        body_text[-120:] if body_text else "(empty)",
    )
    body_text = _strip_menu_endings(body_text)
    logger.info(
        "DIAG format_answer AFTER  _strip_menu_endings | last_120: '%s'",
        body_text[-120:] if body_text else "(empty)",
    )

    # Clean, conversational output — no HTML wrappers or "Teaching Takeaways" template
    sections: List[str] = []
    sections.append(body_text)

    # Extract active technical subcategories for smart artifact selection
    active_subcats = state.get("analytics_metadata", {}).get("technical_subcategories", [])
    show_technical_depth = state.get("show_technical_depth", False)

    # Live analytics snapshot (data_display_requested action)
    if "render_live_analytics" in action_types:
        try:
            import requests
            from assistant.config.supabase_config import supabase_settings
            if supabase_settings.is_vercel:
                analytics_url = "https://noahsaiassistant.vercel.app/api/analytics"
            else:
                analytics_url = "http://localhost:3000/api/analytics"

            response = requests.get(analytics_url, timeout=3)
            response.raise_for_status()
            analytics_data = response.json()
            from assistant.flows.node_logic.util_analytics_renderer import render_live_analytics

            analytics_report = render_live_analytics(analytics_data, state.get("role"), focus=None)
            sections.append("")
            sections.append(f"**Live Analytics Snapshot**\n{analytics_report}")
        except Exception as exc:
            logger.error(f"Failed to fetch live analytics: {exc}")
            sections.append("")
            sections.append("Live analytics are temporarily unavailable. I can share the cached summary if you like.")

    # Cost/latency/grounding metrics
    if "include_metrics_block" in action_types:
        metrics, source = content_blocks.cost_latency_grounded_block()
        metrics_lines = "\n".join(f"- {m}" for m in metrics)
        sections.append("")
        sections.append(f"**Cost · Latency · Grounding**\n{metrics_lines}\n_Source: {source}_")

    # Engineering sequence diagram
    if "include_sequence_diagram" in action_types or (show_technical_depth and "architecture_depth" in active_subcats):
        sections.append("")
        sections.append(content_blocks.engineering_sequence_diagram())

    # Enterprise adaptation diagram
    if "include_adaptation_diagram" in action_types:
        sections.append("")
        sections.append(content_blocks.enterprise_adaptation_diagram())

    # Subcategory-aware code enrichments
    if show_technical_depth and active_subcats:
        _add_subcategory_code_blocks(sections, active_subcats, rag_engine, query, role, depth)

    # Code reference (retrieves from code index) - for explicit actions or general technical queries
    if "include_code_reference" in action_types:
        # Check if this is an architecture-specific code request
        code_actions = [a for a in pending_actions if a.get("type") == "include_code_reference"]
        architecture_context = any(a.get("context") == "architecture" for a in code_actions)

        if architecture_context:
            # Special handling for architecture queries - show conversation flow structure
            architecture_code = """# LangGraph conversation pipeline (src/flows/conversation_flow.py)
from langgraph.graph import StateGraph
from assistant.state.conversation_state import ConversationState

def build_conversation_graph():
    \"\"\"Build the conversation flow as a state graph.\"\"\"
    graph = StateGraph(ConversationState)

    # Stage 0: Session initialization
    graph.add_node("initialize_conversation_state", initialize_conversation_state)

    # Stage 1: Role classification and greeting
    graph.add_node("handle_greeting", handle_greeting)
    graph.add_node("classify_role_mode", classify_role_mode)

    # Stage 2: Query understanding (intent + entities)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("extract_entities", extract_entities)

    # Stage 3: Query refinement (composition + clarification)
    graph.add_node("assess_clarification_need", assess_clarification_need)
    graph.add_node("compose_query", compose_query)

    # Stage 4: Retrieval pipeline (pgvector + grounding)
    graph.add_node("retrieve_chunks", retrieve_chunks)
    graph.add_node("validate_grounding", validate_grounding)

    # Stage 5: Generation pipeline (LLM response)
    graph.add_node("generate_draft", generate_draft)
    graph.add_node("hallucination_check", hallucination_check)

    # Stage 6: Action planning and formatting
    graph.add_node("plan_actions", plan_actions)
    graph.add_node("format_answer", format_answer)

    # Stage 7: Logging and followups
    graph.add_node("log_and_notify", log_and_notify)

    # Define edges (control flow)
    graph.set_entry_point("initialize_conversation_state")
    graph.add_edge("initialize_conversation_state", "handle_greeting")
    # ... (18 nodes total with conditional routing)

    return graph.compile()"""

            formatted_code = content_blocks.format_code_snippet(
                code=architecture_code,
                file_path="src/flows/conversation_flow.py",
                language="python",
                description="LangGraph StateGraph orchestration with 18 nodes across 7 pipeline stages",
            )
            sections.append("")
            sections.append(formatted_code)
        else:
            # Normal code retrieval from code index
            try:
                results = rag_engine.retrieve_with_code(query, role=role)
                snippets = results.get("code_snippets", []) if results else []
            except Exception as exc:
                logger.warning(f"Code retrieval failed: {exc}")
                snippets = []

            if snippets:
                snippet = snippets[0]
                code_content = snippet.get("content", "")
                citation = snippet.get("citation", "codebase")
                if is_valid_code_snippet(code_content):
                    formatted_code = content_blocks.format_code_snippet(
                        code=code_content,
                        file_path=citation,
                        language="python",
                        description="Core logic referenced in this explanation",
                    )
                    sections.append("")
                    sections.append(formatted_code)
            elif "include_code_reference" in action_types:
                sections.append("")
                sections.append("Code index is refreshing; happy to walk through the architecture instead.")

    # Import explanations (stack justifications)
    if "explain_imports" in action_types:
        import_name = detect_import_in_query(query)
        if import_name:
            explanation_data = get_import_explanation(import_name, role)
            if explanation_data:
                formatted = content_blocks.format_import_explanation(
                    import_name=explanation_data["import"],
                    tier=explanation_data["tier"],
                    explanation=explanation_data["explanation"],
                    enterprise_concern=explanation_data.get("enterprise_concern", ""),
                    enterprise_alternative=explanation_data.get("enterprise_alternative", ""),
                    when_to_switch=explanation_data.get("when_to_switch", ""),
                )
                sections.append("")
                sections.append(f"**Why {import_name}?**\n{formatted}")
        else:
            relevant_imports = search_import_explanations(query, role, top_k=3)
            if relevant_imports:
                bullets = "\n".join(f"- **{imp_data['import']}**: {imp_data['explanation']}" for imp_data in relevant_imports)
                sections.append("")
                sections.append(f"**Stack Justifications**\n{bullets}")

    # Fun facts
    if "share_fun_facts" in action_types:
        sections.append("")
        sections.append(content_blocks.fun_facts_block())

    # MMA fight link
    if "share_mma_link" in action_types or state.get("query_type") == "mma":
        sections.append("")
        sections.append(content_blocks.mma_fight_link())

    # LinkedIn link
    if "send_linkedin" in action_types:
        sections.append("")
        sections.append(f"LinkedIn profile: {LINKEDIN_URL}")

    # GitHub link
    if "send_github" in action_types:
        sections.append("")
        sections.append(f"GitHub: {GITHUB_URL}")

    # Resume download link
    if "send_resume" in action_types:
        resume_link = state.get("resume_signed_url", RESUME_DOWNLOAD_URL)
        sections.append("")
        sections.append(f"Résumé download: {resume_link}")

    # Resume offer prompt (before sending)
    if "offer_resume_prompt" in action_types and not state.get("offer_sent"):
        sections.append("")
        sections.append("If it would help, I can share Noah's résumé or LinkedIn—just let me know.")

    # Reach out prompt
    if "ask_reach_out" in action_types:
        sections.append("")
        sections.append("Would you like Noah to reach out directly?")

    # Confession prompt
    if "collect_confession" in action_types:
        sections.append("")
        sections.append(
            "💌 Your message is safe. Share it anonymously or add contact info and I'll pass it privately to Noah."
        )

    # Sources — omit from displayed answer to keep output clean.
    # The sources are still available in the retrieved_chunks state field.

    # Merged: Generate followup prompts (from suggest_followups node)
    # Use subcategory-aware logic for precision targeting
    intent = state.get("query_intent") or state.get("query_type") or "general"
    role_mode = state.get("role_mode", "explorer")
    followup_variant = state.get("followup_variant", "mixed")
    chat_history = state.get("chat_history", [])
    session_memory = state.get("session_memory", {})

    # =========================================================================
    # CRITICAL: Update pillar exploration BEFORE generating follow-ups
    # This ensures current topic is marked explored BEFORE we suggest what's next
    # =========================================================================
    current_pillar = _extract_current_topic_from_query(query, role_mode)
    if current_pillar and session_memory:
        explored = set(session_memory.get("explored_pillars", []))
        if current_pillar not in explored:
            explored.add(current_pillar)
            session_memory["explored_pillars"] = list(explored)
            logger.debug(f"Marked pillar '{current_pillar}' as explored BEFORE generating follow-ups")

        # Also update pillar depth counter
        pillar_depth = session_memory.setdefault("pillar_depth", {})
        pillar_depth[current_pillar] = pillar_depth.get(current_pillar, 0) + 1
        logger.debug(f"Pillar depth for '{current_pillar}': {pillar_depth[current_pillar]}")

    # Pass quality warnings, guidance flags, and conversation phase for context-aware followups
    quality_warnings = state.get("answer_quality_warning")
    guidance_flags = state.get("conversation_guidance_needed", [])
    conversation_phase = state.get("conversation_phase")

    followups = _build_followups(
        variant=followup_variant,
        intent=intent,
        active_subcats=active_subcats,
        role_mode=role_mode,
        chat_history=chat_history,
        session_memory=session_memory,
        quality_warnings=quality_warnings,
        guidance_flags=guidance_flags,
        conversation_phase=conversation_phase,
        query=query  # Pass query to filter current topic from suggestions
    )

    # Store followup prompts in state for the frontend to use as suggestion chips,
    # but don't append them to the answer text — keeps the response conversational.
    if followups:
        state["followup_prompts"] = followups

    # SUBTLE AVAILABILITY MENTION - Natural transition to resume when engagement is high
    # Only for hiring managers who have demonstrated engagement but haven't been offered resume yet
    if (role in ["Hiring Manager (technical)", "Hiring Manager (nontechnical)"] and
        not state.get("offer_sent") and
        "offer_resume_prompt" not in action_types and
        "send_resume" not in action_types):

        # Check engagement criteria for subtle mention
        engagement_score = state.get("engagement_score", 0)
        depth_level = state.get("depth_level", 1)
        hiring_signals_strong = state.get("hiring_signals_strong", False)

        # Conditions for subtle availability mention:
        # - High engagement (score >= 12) but not quite at offer threshold (15)
        # - OR medium depth (2) with some hiring signals
        should_add_subtle_mention = (
            (12 <= engagement_score < 15) or
            (depth_level >= 2 and hiring_signals_strong)
        )

        # Only mention once per session
        availability_mentioned = session_memory.get("persona_hints", {}).get("availability_mentioned", False)

        if should_add_subtle_mention and not availability_mentioned:
            sections.append("")
            sections.append(
                "_By the way, Noah built all of this while working at Tesla. "
                "If you're curious about bringing this kind of GenAI thinking to your team, "
                "I'd be happy to share his resume and LinkedIn._"
            )
            # Mark that we've mentioned availability
            state.setdefault("session_memory", {}).setdefault("persona_hints", {})["availability_mentioned"] = True
            logger.info(f"📌 Added subtle availability mention (engagement={engagement_score}, depth={depth_level})")

    enriched_answer = "\n".join(section for section in sections if section is not None)

    # ── Link throttling ───────────────────────────────────────────────
    # Count recent assistant responses that already contained a link.
    # If a link appeared in the last 2 responses, strip links from this one.
    # This ensures links appear on at most every 3rd–4th response.
    enriched_answer = _throttle_links(enriched_answer, chat_history)

    # Return partial update dict (not full state) to avoid preserving old fields
    # In LangGraph StateGraph, nodes should return partial updates
    partial_update: Dict[str, Any] = {
        "answer": enriched_answer.strip()
    }
    if followups:
        partial_update["followup_prompts"] = followups
    return partial_update


def _detect_topic_from_query(query: str) -> Optional[str]:
    """Extract topic from a user query.

    Args:
        query: User's query string

    Returns:
        Topic string or None if no clear topic detected
    """
    if not query:
        return None

    query_lower = query.lower()

    # Map query keywords to topics
    topic_keywords = {
        "background": ["resume", "cv", "linkedin", "github", "career", "background", "noah", "experience", "skills", "certifications"],
        "data_pipeline": ["data pipeline", "embeddings", "vector", "pgvector", "chunking", "analytics", "ingestion", "data flow"],
        "orchestration": ["orchestration", "langgraph", "nodes", "pipeline", "state", "safeguards", "flow"],
        "architecture": ["architecture", "tech stack", "frontend", "backend", "system design"],
        "enterprise": ["enterprise", "customer support", "adaptation", "scaling", "production", "adapt"],
    }

    for topic, keywords in topic_keywords.items():
        if any(kw in query_lower for kw in keywords):
            return topic

    return None


def _detect_previous_topic(chat_history: List[Dict], current_query: str = None) -> str:
    """Detect the topic from the previous turn's conversation.

    Excludes "Where next?" followup sections and <details> blocks to avoid
    false topic detection from suggested next steps.

    Fix 5: Only returns a topic if it's relevant to the current query.
    This prevents "Building on our architecture discussion" when user asks about resume.

    Args:
        chat_history: List of conversation messages
        current_query: Optional current query to check topic relevance

    Returns:
        Topic string (e.g., "orchestration", "architecture", "data_pipeline") or empty string
    """
    if not chat_history or len(chat_history) < 2:
        return ""

    # Look at the last assistant message to extract topic
    # Go backwards to find the most recent assistant message
    previous_topic = None
    for msg in reversed(chat_history[-6:]):  # Check last 6 messages
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("type", "")
            content = msg.get("content", "")
        elif hasattr(msg, "type"):
            role = msg.type
            content = getattr(msg, "content", "")
        else:
            continue

        if role in ["assistant", "ai"]:
            # EXCLUDE followup sections from topic detection
            # These often mention topics as suggestions, not as the actual discussed topic
            content_for_detection = content

            # Remove "Where next?" section and everything after
            content_for_detection = re.sub(
                r'\*\*Where next\?\*\*.*$', '', content_for_detection,
                flags=re.DOTALL | re.IGNORECASE
            )

            # Remove <details> blocks (collapsed sections with suggestions)
            content_for_detection = re.sub(
                r'<details>.*?</details>', '', content_for_detection,
                flags=re.DOTALL
            )

            # Remove citation blocks
            content_for_detection = re.sub(
                r'<details>\s*<summary>Show citations</summary>.*?</details>', '',
                content_for_detection, flags=re.DOTALL
            )

            # Log cleaned content for debugging
            logger.debug(f"Topic detection input (cleaned): '{content_for_detection[:200]}...'")

            content_lower = content_for_detection.lower()

            # Extract key topics from the assistant's core response (not suggestions)
            # Order matters - check more specific topics first
            topic_keywords = {
                "data_pipeline": ["data pipeline", "embeddings", "vector storage", "pgvector", "chunking", "analytics", "ingestion"],
                "orchestration": ["orchestration", "langgraph", "nodes", "pipeline stages", "state management"],
                "architecture": ["architecture", "tech stack", "system design", "frontend", "backend"],
                "enterprise": ["enterprise", "customer support", "adaptation", "scaling", "production"],
                "retrieval": ["retrieval", "rag", "semantic search", "chunks"],
                "background": ["career", "tesla", "mma", "professional background", "github", "noah's", "resume", "linkedin"]
            }

            for topic, keywords in topic_keywords.items():
                if any(kw in content_lower for kw in keywords):
                    logger.debug(f"Detected previous topic: '{topic}'")
                    previous_topic = topic
                    break

            if previous_topic:
                break

    # Fix 5: Only return topic if it matches current query's topic
    # Don't say "Building on our architecture discussion" when user asks about resume
    if previous_topic and current_query:
        current_topic = _detect_topic_from_query(current_query)
        if current_topic and current_topic != previous_topic:
            logger.debug(f"Skipping turn reference: previous='{previous_topic}', current='{current_topic}' (mismatch)")
            return ""  # Topics don't match, skip the turn reference

    return previous_topic or ""


def _validate_cross_turn_references(answer: str, chat_history: List[Dict]) -> Dict[str, Any]:
    """Check if answer references previous turns.

    Args:
        answer: The formatted answer text
        chat_history: List of conversation messages

    Returns:
        Dict with validation results
    """
    if not chat_history or len(chat_history) < 4:
        return {"has_references": False, "reference_count": 0, "chat_history_length": len(chat_history)}

    answer_lower = answer.lower()
    reference_phrases = [
        "building on", "as we discussed", "turn 1", "turn 2", "turn 3", "turn 4",
        "earlier", "previous conversation", "as we explored", "mentioned earlier",
        "before", "previously", "we talked about"
    ]

    reference_count = sum(1 for phrase in reference_phrases if phrase in answer_lower)
    has_references = reference_count > 0

    if len(chat_history) >= 6 and not has_references:
        logger.warning(
            f"Answer missing cross-turn references: chat_history_len={len(chat_history)}, "
            f"answer_preview={answer[:100]}"
        )

    return {
        "has_references": has_references,
        "reference_count": reference_count,
        "chat_history_length": len(chat_history)
    }


def _validate_memory_demonstration(answer: str) -> Dict[str, Any]:
    """Check if answer demonstrates memory accumulation (when asked).

    Args:
        answer: The formatted answer text

    Returns:
        Dict with validation results
    """
    answer_lower = answer.lower()
    memory_phrases = [
        "turn 1", "turn 2", "turn 3", "accumulated", "improved",
        "similarity", "because", "enabled by", "depth_level",
        "topics", "chat_history", "session_memory", "progressive",
        "each turn", "over time", "as conversation", "builds on"
    ]

    memory_demo_count = sum(1 for phrase in memory_phrases if phrase in answer_lower)
    has_demonstration = memory_demo_count >= 3  # At least 3 memory-related phrases

    if not has_demonstration:
        logger.warning(
            f"Memory query missing demonstration: answer_preview={answer[:100]}, "
            f"phrase_count={memory_demo_count}"
        )

    return {
        "has_demonstration": has_demonstration,
        "phrase_count": memory_demo_count,
        "is_memory_query": True
    }
