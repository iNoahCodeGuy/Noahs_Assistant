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
from typing import Dict, Any, List

from src.state.conversation_state import ConversationState
from src.core.rag_engine import RagEngine
from src.flows import content_blocks
from src.flows.node_logic.code_validation import is_valid_code_snippet

logger = logging.getLogger(__name__)

# Import retriever functions with graceful fallback
try:
    from src.retrieval.import_retriever import (
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

# Resume constants
RESUME_DOWNLOAD_URL = "https://noahsaiassistant.vercel.app/resume/Noah_Delacalzada_Resume.pdf"
LINKEDIN_URL = "https://www.linkedin.com/in/noah-delacalzada"


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


def _summarize_answer(text: str, depth: int) -> List[str]:
    """Extract key sentences from answer for summary bullets.

    Args:
        text: Answer body text
        depth: User's depth preference (1=brief, 2=detailed, 3=comprehensive)

    Returns:
        List of summary bullet strings

    Example:
        >>> _summarize_answer("First point. Second point. Third point.", depth=1)
        ["- First point.", "- Second point."]
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    limit = 2 if depth <= 1 else 3
    summary = []
    for sentence in sentences:
        if sentence.lower().startswith("sources:"):
            continue
        summary.append(f"- {sentence}")
        if len(summary) >= limit:
            break
    return summary


def _build_followups(variant: str) -> List[str]:
    """Generate role-specific followup prompt suggestions.

    Args:
        variant: "engineering" | "business" | default (general)

    Returns:
        List of 3 followup prompt strings

    Example:
        >>> _build_followups("engineering")
        ["Walk through the LangGraph node transitions", ...]
    """
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
        "See how the architecture adapts to customer support",
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
    base_answer = state.get("draft_answer") or state.get("answer")
    if base_answer is None:
        logger.error("format_answer called without draft_answer")
        return state

    if not base_answer:
        return {}

    depth = state.get("depth_level", 1)
    layout_variant = state.get("layout_variant", "mixed")
    pending_actions = state.get("pending_actions", [])
    action_types = {action["type"] for action in pending_actions}
    query = state.get("query", "")
    role = state.get("role", "Just looking around")

    body_text, sources_text = _split_answer_and_sources(base_answer)
    summary_lines = _summarize_answer(body_text, depth)

    sections: List[str] = []
    sections.append("**Teaching Takeaways**")
    sections.extend(summary_lines or ["- I pulled the relevant context and kept the answer grounded."])

    details_block = content_blocks.render_block(
        "Full Walkthrough",
        body_text,
        summary="Expand for the detailed explanation",
        open_by_default=depth >= 2,
    )
    sections.append("")
    sections.append(details_block)

    # Live analytics snapshot (data_display_requested action)
    if "render_live_analytics" in action_types:
        try:
            import requests
            from src.config.supabase_config import supabase_settings
            if supabase_settings.is_vercel:
                analytics_url = "https://noahsaiassistant.vercel.app/api/analytics"
            else:
                analytics_url = "http://localhost:3000/api/analytics"

            response = requests.get(analytics_url, timeout=3)
            response.raise_for_status()
            analytics_data = response.json()
            from src.flows.node_logic.analytics_renderer import render_live_analytics

            analytics_report = render_live_analytics(analytics_data, state.get("role"), focus=None)
            sections.append("")
            sections.append(
                content_blocks.render_block(
                    "Live Analytics Snapshot",
                    analytics_report,
                    summary="View Supabase analytics",
                    open_by_default=depth >= 3,
                )
            )
        except Exception as exc:
            logger.error(f"Failed to fetch live analytics: {exc}")
            sections.append("")
            sections.append("Live analytics are temporarily unavailable. I can share the cached summary if you like.")

    # Cost/latency/grounding metrics
    if "include_metrics_block" in action_types:
        metrics, source = content_blocks.cost_latency_grounded_block()
        metrics_body = list(metrics) + [f"Source: {source}"]
        sections.append("")
        sections.append(
            content_blocks.render_block(
                "Cost · Latency · Grounding",
                metrics_body,
                summary="Metrics snapshot",
                open_by_default=depth >= 3,
            )
        )

    # Engineering sequence diagram
    if "include_sequence_diagram" in action_types:
        sections.append("")
        sections.append(
            content_blocks.render_block(
                "Engineering Sequence",
                content_blocks.engineering_sequence_diagram(),
                summary="See the LangGraph handoff",
                open_by_default=depth >= 2,
            )
        )

    # Enterprise adaptation diagram
    if "include_adaptation_diagram" in action_types:
        sections.append("")
        sections.append(
            content_blocks.render_block(
                "Enterprise Adaptation",
                content_blocks.enterprise_adaptation_diagram(),
                summary="Show the adaptation map",
                open_by_default=False,
            )
        )

    # Code reference (retrieves from code index)
    if "include_code_reference" in action_types:
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
                sections.append(
                    content_blocks.render_block(
                        "Code Reference",
                        formatted_code,
                        summary="Peek at the implementation",
                        open_by_default=depth >= 3,
                    )
                )
                sections.append(content_blocks.code_display_guardrails())
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
                sections.append(
                    content_blocks.render_block(
                        f"Why {import_name}",
                        formatted,
                        summary=f"Stack choice: {import_name}",
                        open_by_default=False,
                    )
                )
        else:
            relevant_imports = search_import_explanations(query, role, top_k=3)
            if relevant_imports:
                bullets = []
                for imp_data in relevant_imports:
                    bullets.append(
                        f"{imp_data['import']}: {imp_data['explanation']}"
                    )
                sections.append("")
                sections.append(
                    content_blocks.render_block(
                        "Stack Justifications",
                        bullets,
                        summary="Why these libraries?",
                        open_by_default=False,
                    )
                )

    # Fun facts
    if "share_fun_facts" in action_types:
        sections.append("")
        fun_fact_lines = [
            line.lstrip("- ").strip()
            for line in content_blocks.fun_facts_block().split("\n")
            if line.strip()
        ]
        sections.append(
            content_blocks.render_block(
                "Fun Facts",
                fun_fact_lines,
                summary="Quick facts about Noah",
                open_by_default=False,
            )
        )

    # MMA fight link
    if "share_mma_link" in action_types or state.get("query_type") == "mma":
        sections.append("")
        sections.append(content_blocks.mma_fight_link())

    # LinkedIn link
    if "send_linkedin" in action_types:
        sections.append("")
        sections.append(f"LinkedIn profile: {LINKEDIN_URL}")

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

    # Sources (citations from retrieval)
    if sources_text:
        sections.append("")
        sections.append(
            content_blocks.render_block(
                "Sources",
                [line.strip() for line in sources_text.splitlines() if line.strip()],
                summary="Show citations",
                open_by_default=False,
            )
        )

    # Followup prompts (Where next?)
    followups = _build_followups(state.get("followup_variant", "mixed"))
    sections.append("")
    sections.append("**Where next?**")
    sections.extend(f"- {item}" for item in followups)
    state["followup_prompts"] = followups

    enriched_answer = "\n".join(section for section in sections if section is not None)
    state["answer"] = enriched_answer.strip()
    return state
