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

from assistant.state.conversation_state import ConversationState
from assistant.core.rag_engine import RagEngine
from assistant.flows import content_blocks
from assistant.flows.node_logic.util_code_validation import is_valid_code_snippet

logger = logging.getLogger(__name__)


def _retry_generation_if_insufficient(state: ConversationState, rag_engine: RagEngine) -> str:
    """Retry generation with reinforced instructions if answer too short or missing sections.

    This is a guardrail for menu option 1 (full tech stack) to ensure comprehensive coverage.
    Only triggers if generation_quality_warning is set by generate_draft.

    Args:
        state: Conversation state with draft_answer and quality warning
        rag_engine: RAG engine for LLM access

    Returns:
        Enhanced answer (or original if retry fails)
    """
    # GUARD: Don't retry if this is the initial greeting
    if state.get("is_greeting") or state.get("pipeline_halt"):
        persona_hints = state.get("session_memory", {}).get("persona_hints", {})
        if persona_hints.get("initial_greeting_shown"):
            return state.get("answer", "")

    quality_warning = state.get("generation_quality_warning")

    # Only retry for menu option 1 if quality warning exists
    if not quality_warning:
        return state.get("draft_answer", state.get("answer", ""))

    if not (state.get("query_type") == "menu_selection" and
            state.get("menu_choice") == "1" and
            state.get("role_mode") == "hiring_manager_technical"):
        return state.get("draft_answer", state.get("answer", ""))

    logger.info(f"🔄 Retrying generation due to quality issue: {quality_warning}")

    original_answer = state.get("draft_answer", "")
    retrieved_chunks = state.get("retrieved_chunks", [])

    # Build reinforced prompt
    retry_instructions = (
        "RETRY ATTEMPT - Previous response failed quality check.\n\n"
        f"Issue detected: {quality_warning}\n\n"
        "YOU MUST include ALL 5 layers with these EXACT headings:\n"
        "Frontend Layer:\n"
        "Backend/Orchestration Layer:\n"
        "Data Layer:\n"
        "Observability Layer:\n"
        "Deployment Layer:\n\n"
        "Each layer needs 2-3 complete sentences. Total target: 300-350 words.\n"
        "DO NOT skip any layer. DO NOT copy verbatim from context."
    )

    try:
        enhanced_answer = rag_engine.response_generator.generate_contextual_response(
            query=state.get("query", ""),
            context=retrieved_chunks,
            role=state.get("role", ""),
            chat_history=state.get("chat_history", []),
            extra_instructions=retry_instructions,
            model_name=state.get("analytics_metadata", {}).get("selected_model", "gpt-4o-mini")
        )

        # Validate retry succeeded
        word_count = len(enhanced_answer.split())
        required_layers = ["Frontend Layer:", "Backend/Orchestration Layer:",
                          "Data Layer:", "Observability Layer:", "Deployment Layer:"]
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


def _remove_markdown_headers(text: str) -> str:
    """Remove raw markdown headers and formatting from text.

    Removes documentation-style markdown that shouldn't appear in conversational responses:
    - Lines starting with ## or ### (markdown headers)
    - Markdown bold syntax **text** (handled elsewhere but ensure cleanup)
    - Standalone markdown list markers when they're part of raw documentation

    Args:
        text: Text that may contain raw markdown

    Returns:
        Text with markdown headers and formatting removed
    """
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        # Skip lines that are markdown headers (## or ###)
        if stripped.startswith('##') or stripped.startswith('###'):
            continue
        # Keep the line
        cleaned_lines.append(line)

    result = '\n'.join(cleaned_lines)

    # Remove markdown bold (already done elsewhere, but ensure it's clean)
    result = re.sub(r'\*\*([^*]+?)\*\*', r'\1', result)
    result = re.sub(r'\*\*', '', result)

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
                    sections.append(
                        content_blocks.render_block(
                            "State Management Schema",
                            formatted_code,
                            summary="See the ConversationState dataclass",
                            open_by_default=depth >= 3,
                        )
                    )

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
                    sections.append(
                        content_blocks.render_block(
                            "RAG Pipeline Code",
                            formatted_code,
                            summary="See the pgvector retrieval implementation",
                            open_by_default=depth >= 3,
                        )
                    )

    except Exception as exc:
        logger.warning(f"Subcategory code enrichment failed: {exc}")


def _summarize_answer(text: str, depth: int) -> List[str]:
    """Extract key sentences from answer for summary bullets.

    Handles Q&A format cleanup: If LLM generated answer in Q:/A: format,
    extracts just the answer portions and formats as clean bullets.

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
    # Check if answer is in Q&A format and extract just answer portions
    qa_pattern = r'Q:\s*.*?\s*A:\s*(.*?)(?=Q:|$)'
    qa_matches = re.findall(qa_pattern, text, re.DOTALL | re.IGNORECASE)

    if qa_matches:
        # Extract answer portions only, ignore question format
        sentences = []
        for answer_text in qa_matches:
            answer_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text.strip()) if s.strip()]
            sentences.extend(answer_sentences)
        logger.debug(f"Detected Q&A format, extracted {len(sentences)} answer sentences")
    else:
        # Normal sentence splitting
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    limit = 2 if depth <= 1 else 3
    summary = []
    for sentence in sentences:
        if sentence.lower().startswith("sources:"):
            continue
        # Remove any remaining "Q:" or "A:" prefixes
        sentence = re.sub(r'^[QA]:\s*', '', sentence, flags=re.IGNORECASE).strip()
        if sentence:  # Only add if non-empty after cleanup
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


def _build_followups(variant: str, intent: str = "general", active_subcats: List[str] = None, role_mode: str = "explorer", chart_would_help: bool = False) -> List[str]:
    """Generate role-specific followup prompt suggestions with subcategory awareness.

    Merged logic from logging_nodes.suggest_followups for inline followup generation.

    Args:
        variant: "engineering" | "business" | "mixed" (layout variant)
        intent: Query intent/type (technical, data, career, etc.)
        active_subcats: Active technical subcategories for precision targeting
        role_mode: User's role (explorer, software_developer, etc.)
        chart_would_help: Whether a chart/graph would enhance understanding

    Returns:
        List of 3 followup prompt strings

    Example:
        >>> _build_followups("engineering", "technical", ["architecture_depth"])
        ["Want the LangGraph node flow diagram?", ...]
    """
    active_subcats = active_subcats or []

    # Confession mode gets no followups (privacy)
    if role_mode == "confession":
        return []

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
        suggestions = []

        # Add chart offer if chart would help (contextual and role-aware)
        if chart_would_help:
            if role_mode in ["software_developer", "hiring_manager_technical"]:
                # Technical users: performance/technical charts
                suggestions.append("Would you like to see a chart visualizing this data?")
            elif role_mode == "hiring_manager_nontechnical":
                # Business users: business value charts
                suggestions.append("Would you like to see a chart showing these trends?")
            else:
                # General users: accessible charts
                suggestions.append("Would you like to see a chart of this data?")

        # Add other data followups
        if len(suggestions) < 3:
            suggestions.extend([
                "Need the retrieval accuracy metrics for last week?",
                "Want the cost-per-query breakdown?",
                "Should we compare grounding confidence across roles?",
            ])

        return suggestions[:3]
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
    # GUARD: Skip formatting if initial greeting is being shown
    # Return immediately if this is the initial greeting - don't process it at all
    if state.get("is_greeting") or state.get("pipeline_halt"):
        persona_hints = state.get("session_memory", {}).get("persona_hints", {})
        if persona_hints.get("initial_greeting_shown"):
            logger.info("format_answer: Skipping - initial greeting shown, preserving answer")
            return state

    # CRITICAL GUARD: Never process if answer already contains initial greeting
    current_answer = state.get("answer", "")
    if current_answer and "1️⃣ Hiring Manager" in current_answer:
        logger.warning("format_answer: Initial greeting detected - skipping all processing")
        return state

    # GUARDRAIL: Retry generation if quality check failed
    base_answer = _retry_generation_if_insufficient(state, rag_engine)

    if base_answer is None:
        # Fallback to original answer
        base_answer = state.get("draft_answer") or state.get("answer")

    if base_answer is None:
        logger.error("format_answer called without draft_answer")
        return state

    if not base_answer:
        return {}

    # Strip markdown bold formatting from base answer before processing
    base_answer = re.sub(r'\*\*([^*]+?)\*\*', r'\1', base_answer)
    base_answer = re.sub(r'\*\*', '', base_answer)

    # PRIORITY: Skip enrichments if answer is already a role welcome message
    # Prevents duplicate content when menu selection shows role-specific welcome
    welcome_indicators = [
        "Since you selected",
        "You can choose where to start:",
        "Before we dive in, what best describes you?",
        "I can focus on the areas most relevant to you",
        "1️⃣ Hiring Manager",  # Initial greeting indicator
        "Looking to Confess Crush 💌"  # Initial greeting indicator
    ]
    if any(indicator in base_answer for indicator in welcome_indicators):
        # This is a welcome/menu message - don't add formatting
        logger.info("format_answer: Detected welcome message, skipping formatting")
        state["answer"] = base_answer
        return state

    # Skip formatting for menu option 1 (brief overview only, no Teaching Takeaways or expandable sections)
    if (state.get("query_type") == "menu_selection" and
        state.get("menu_choice") == "1"):
        logger.info("format_answer: Detected menu option 1, skipping formatting - returning brief overview only")
        # Strip any remaining markdown, sources, and return clean answer
        clean_answer = re.sub(r'\*\*([^*]+?)\*\*', r'\1', base_answer)
        clean_answer = re.sub(r'\*\*', '', clean_answer)
        # Remove Sources section if present
        clean_answer = re.sub(r'\nSources:.*$', '', clean_answer, flags=re.DOTALL)
        clean_answer = re.sub(r'\nSources\s*:.*$', '', clean_answer, flags=re.DOTALL)
        state["answer"] = clean_answer.strip()
        return state

    depth = state.get("depth_level", 1)
    layout_variant = state.get("layout_variant", "mixed")
    pending_actions = state.get("pending_actions", [])
    action_types = {action["type"] for action in pending_actions}
    query = state.get("query", "")
    role = state.get("role", "Just looking around")

    body_text, sources_text = _split_answer_and_sources(base_answer)

    # Remove markdown headers from body text (documentation cleanup)
    body_text = _remove_markdown_headers(body_text)

    summary_lines = _summarize_answer(body_text, depth)

    sections: List[str] = []

    # Layer 3: File Content Display in Formatting
    # Show file content prominently when explicitly requested
    if state.get("file_request") and state.get("file_content", {}).get("success"):
        file_data = state["file_content"]
        file_path = file_data.get("file_path", "unknown")
        file_content_text = file_data.get("content", "")

        # Determine file language for syntax highlighting
        if file_path.endswith(".py"):
            lang = "python"
        elif file_path.endswith((".ts", ".tsx")):
            lang = "typescript"
        elif file_path.endswith((".js", ".jsx")):
            lang = "javascript"
        elif file_path.endswith(".md"):
            lang = "markdown"
        else:
            lang = "text"

        # Format file section with code block
        file_section = f"## File: `{file_path}`\n\n```{lang}\n{file_content_text}\n```"
        sections.append(file_section)
        sections.append("")  # Empty line for spacing
        logger.info(f"File content displayed in response: {file_path}")

    sections.append("Teaching Takeaways")
    sections.extend(summary_lines or ["- I pulled the relevant context and kept the answer grounded."])

    details_block = content_blocks.render_block(
        "Full Walkthrough",
        body_text,
        summary="Expand for the detailed explanation",
        open_by_default=depth >= 2,
    )
    sections.append("")
    sections.append(details_block)

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
    if "include_sequence_diagram" in action_types or (show_technical_depth and "architecture_depth" in active_subcats):
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
            sections.append(
                content_blocks.render_block(
                    "Architecture Code Reference",
                    formatted_code,
                    summary="View the conversation pipeline structure",
                    open_by_default=depth >= 3,
                )
            )
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
                    sections.append(
                        content_blocks.render_block(
                            "Code Reference",
                            formatted_code,
                            summary="Peek at the implementation",
                            open_by_default=depth >= 3,
                        )
                    )
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

    # Merged: Generate followup prompts (from suggest_followups node)
    # Use subcategory-aware logic for precision targeting
    intent = state.get("query_intent") or state.get("query_type") or "general"
    role_mode = state.get("role_mode", "explorer")
    followup_variant = state.get("followup_variant", "mixed")
    chart_would_help = state.get("chart_would_help", False)

    followups = _build_followups(
        variant=followup_variant,
        intent=intent,
        active_subcats=active_subcats,
        role_mode=role_mode,
        chart_would_help=chart_would_help
    )

    if followups:
        sections.append("")
        sections.append("Where next?")
        sections.extend(f"- {item}" for item in followups)
        state["followup_prompts"] = followups

    enriched_answer = "\n".join(section for section in sections if section is not None)

    # Strip all markdown bold formatting (**text**) from the final answer
    # Handle multiple occurrences, edge cases, and nested formatting
    enriched_answer = re.sub(r'\*\*([^*]+?)\*\*', r'\1', enriched_answer)
    # Also remove any remaining single asterisks that might be left over
    enriched_answer = re.sub(r'\*\*', '', enriched_answer)
    # Remove any standalone asterisks used for emphasis
    enriched_answer = re.sub(r'\s+\*\s+', ' ', enriched_answer)
    enriched_answer = re.sub(r'\*\s+', '', enriched_answer)

    state["answer"] = enriched_answer.strip()
    return state
