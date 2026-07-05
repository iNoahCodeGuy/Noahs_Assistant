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


# Moved helpers - re-exported so existing
# `from assistant.flows.node_logic.stage6_formatting_nodes import X` call sites
# (conversation_flow, stage3, stage5, stage7, tests) keep working unchanged.
from assistant.flows.node_logic.util_voice import (
    _strip_em_dashes,
    _remove_chunk_citations,
    _remove_markdown_headers,
    _strip_menu_endings,
    _strip_italic_emphasis,
    _enforce_voice_rules,
)
from assistant.flows.node_logic.util_links import (
    _throttle_links,
    RESUME_DOWNLOAD_URL,
    LINKEDIN_URL,
    GITHUB_URL,
)
from assistant.flows.node_logic.util_followups import (
    CONVERSATION_ARCS,
    MAIN_PILLARS,
    _analyze_conversation_flow,
    _build_followups,
    _build_subcategory_followups,
    _extract_current_topic_from_query,
    _generate_synthesis_response,
    _get_depth_options_for_pillar,
    _get_explored_pillars,
    _get_unexplored_pillars,
    _predict_next_topics,
    _should_offer_synthesis,
)


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

    if "send_linkedin" in action_types:
        sections.append(f"**LinkedIn**: {LINKEDIN_URL}")

    if "send_github" in action_types:
        sections.append(f"**GitHub**: {GITHUB_URL}")

    # If only ask_reach_out without any resources, provide context
    if not any(t in action_types for t in ["send_linkedin", "send_github"]):
        sections = ["I'd be happy to help connect you with Noah."]

    if "ask_reach_out" in action_types:
        sections.append("")
        sections.append("Would you like Noah to reach out directly?")

    answer = "\n".join(sections)

    answer = _strip_em_dashes(answer)
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
                        file_path="assistant/state/conversation_state.py",
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
                        file_path="assistant/retrieval/pgvector_retriever.py",
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
        return {"answer": _strip_em_dashes(base_answer)}

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
            return {"answer": _strip_em_dashes(base_answer), "_debug_trace": debug_trace}

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
    # Markdown headers preserved for frontend react-markdown renderer
    # Strip italic emphasis (*word* -> word) while preserving bold
    body_text = _strip_italic_emphasis(body_text)
    # Strip menu-style endings ("Want X or Y?")
    logger.info(
        "DIAG format_answer BEFORE _strip_menu_endings | last_120: '%s'",
        body_text[-120:] if body_text else "(empty)",
    )
    body_text = _strip_menu_endings(body_text)

    # Second pass: catch ANY trailing text with "? Or" or ", or" menu pattern.
    # The boundary-based stripper above splits on sentence boundaries and misses
    # cross-sentence patterns like "Want X? Or I can do Y."
    if body_text:
        # Find the last occurrence of "? Or " or ", or " near the end
        # and strip from the sentence start before it
        for or_pattern in [r'\?\s+[Oo]r\s', r',\s+or\s+(?:want|would|shall|should|I can|hear|see)\b']:
            or_match = re.search(or_pattern, body_text[max(0, len(body_text)-300):], re.IGNORECASE)
            if or_match:
                # Found "? Or" or ", or" in the last 300 chars
                abs_pos = max(0, len(body_text)-300) + or_match.start()
                # For "? Or", find the start of the sentence containing the "?"
                if '?' in or_pattern:
                    # Walk backward from the "?" to find the sentence start
                    q_pos = abs_pos  # position of the "?"
                    # Find the previous sentence-ending punctuation + space or newline
                    search_region = body_text[:q_pos]
                    last_boundary = max(
                        search_region.rfind('. '),
                        search_region.rfind('.\n'),
                        search_region.rfind('!\n'),
                        search_region.rfind('! '),
                    )
                    cut_pos = last_boundary + 1 if last_boundary >= 0 else q_pos
                else:
                    # For ", or", cut at the comma
                    cut_pos = abs_pos
                trimmed = body_text[:cut_pos].rstrip()
                if trimmed and len(trimmed) > 50:
                    logger.info(
                        "Cross-sentence menu stripped at pos %d: '%s'",
                        cut_pos, body_text[cut_pos:cut_pos+80],
                    )
                    body_text = trimmed
                    break

    logger.info(
        "DIAG format_answer AFTER  _strip_menu_endings | last_120: '%s'",
        body_text[-120:] if body_text else "(empty)",
    )

    # Voice compliance: strip banned openers, excess !, inline menus, polling questions
    body_text = _enforce_voice_rules(body_text)
    logger.info(
        "DIAG format_answer AFTER  _enforce_voice_rules | last_120: '%s'",
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
            # Architecture queries: show the real runtime structure — a
            # faithful condensation of the pipeline tuple in
            # assistant/flows/conversation_flow.py (22 nodes, stages 0-7).
            architecture_code = """# Functional conversation pipeline (assistant/flows/conversation_flow.py)
# Nodes run in order; each returns a partial dict merged into shared state.

pipeline = (
    # Stage 0: initialization
    initialize_conversation_state,
    # Stage 1: first contact (menu prompt, deterministic greetings)
    prompt_for_role_selection,
    handle_greeting,
    # Stage 1.5: intent routing — classify BEFORE retrieval (Claude Haiku)
    classify_message_intent,
    handle_non_knowledge_intent,
    # Stage 2: understanding
    classify_role_mode,
    classify_intent,
    detect_conversation_phase,
    extract_entities,
    # Stage 3: query preparation
    assess_clarification_need,
    ask_clarifying_question,
    presentation_controller,
    compose_query,
    # Stage 4: retrieval (pgvector, 0.50 strict / 0.30 fallback)
    retrieve_chunks,
    validate_grounding,
    handle_grounding_gap,
    # Stage 5: generation (Claude Sonnet 4.5) + claim verification
    generate_draft,
    hallucination_check,
    # Stage 6: enrichment
    plan_actions,
    format_answer,
    # Stage 7: side effects + memory
    execute_actions,
    update_memory,
)

for node in pipeline:
    state.update(node(state) or {})
    if state.get("pipeline_halt") or state.get("is_greeting"):
        break  # short-circuits: greetings, forms, clarifications

log_and_notify(state)  # analytics — always runs, even on short-circuit"""

            formatted_code = content_blocks.format_code_snippet(
                code=architecture_code,
                file_path="assistant/flows/conversation_flow.py",
                language="python",
                description="Functional pipeline: 22 nodes across 8 stages, deterministic short-circuits",
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

    # Resume offer prompt (before sending)
    if "offer_resume_prompt" in action_types and not state.get("offer_sent"):
        sections.append("")
        sections.append("If it would help, I can share Noah's résumé or LinkedIn. Just let me know.")

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
        "offer_resume_prompt" not in action_types):

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

    # ── Em-dash removal ───────────────────────────────────────────────
    enriched_answer = _strip_em_dashes(enriched_answer)

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
