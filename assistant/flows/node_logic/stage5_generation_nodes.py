"""Generation pipeline nodes - LLM response creation and hallucination prevention.

This module handles answer generation with the LLM:
1. generate_draft → LLM creates answer using retrieved context
2. hallucination_check → Attaches citations and flags hallucination risk

Design Principles:
- SRP: Each function handles one generation concern
- Runtime awareness: Technical users see self-referential architecture content
- Defensibility: Graceful degradation on LLM failures
- Observability: Detailed logging for generation quality

Performance Characteristics:
- generate_draft: ~800-1500ms (LLM call)
- hallucination_check: <10ms (in-memory citation attachment)

See: docs/context/CONVERSATION_PERSONALITY.md for generation personality
"""

import logging
import re
from typing import Dict, Any, List, Optional

from assistant.state.conversation_state import ConversationState
from assistant.core.rag_engine import RagEngine
from assistant.flows import content_blocks
from assistant.flows.node_logic.util_code_validation import sanitize_generated_answer
from assistant.config.supabase_config import supabase_settings

logger = logging.getLogger(__name__)

MENU_OPTION_ONE_MIN_WORDS = 100
MENU_OPTION_ONE_MAX_WORDS = 150
MENU_OPTION_ONE_MAX_RETRIES = 2


def _validate_menu_option_one_answer(answer: str) -> Dict[str, Any]:
    """Validate menu option 1 output structure and length (brief overview format)."""

    required_layers = [
        "Frontend",
        "Backend",
        "Data",
        "Observability"
    ]

    # Check if layers are mentioned (case-insensitive, partial match)
    answer_lower = answer.lower()
    missing_layers = [layer for layer in required_layers if layer.lower() not in answer_lower]
    word_count = len(answer.split())

    # Check for follow-up question
    has_followup_question = any(phrase in answer.lower() for phrase in [
        "full detailed",
        "end-to-end walkthrough",
        "specific part",
        "would you like",
        "would you prefer"
    ])

    # Check for Sources section (should not be present)
    has_sources = "sources:" in answer_lower or "sources " in answer_lower

    # Check for repetitive phrases (same phrase appearing 2+ times)
    words = answer.lower().split()
    # Look for repeated 3-word phrases
    phrase_counts = {}
    for i in range(len(words) - 2):
        phrase = " ".join(words[i:i+3])
        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

    repetitive_phrases = [phrase for phrase, count in phrase_counts.items() if count >= 2 and len(phrase) > 10]
    has_repetition = len(repetitive_phrases) > 0

    # Stricter word count validation (reject if >160 words)
    word_count_valid = word_count >= MENU_OPTION_ONE_MIN_WORDS and word_count <= 160

    is_valid = (
        not missing_layers and
        word_count_valid and
        has_followup_question and
        not has_sources and
        not has_repetition
    )

    validation_details = {
        "missing_layers": missing_layers,
        "word_count": word_count,
        "word_count_valid": word_count_valid,
        "has_followup_question": has_followup_question,
        "has_sources": has_sources,
        "has_repetition": has_repetition,
        "repetitive_phrases": repetitive_phrases[:3] if repetitive_phrases else [],
        "is_valid": is_valid
    }

    if not is_valid:
        logger.warning(
            f"Menu option 1 validation failed: "
            f"missing_layers={missing_layers}, "
            f"word_count={word_count} (valid={word_count_valid}), "
            f"has_followup={has_followup_question}, "
            f"has_sources={has_sources}, "
            f"has_repetition={has_repetition}"
        )

    return validation_details


def _format_outline_hint(layer_outline: Optional[Dict[str, str]]) -> str:
    """Convert the layer outline dict into a human readable hint."""

    if not layer_outline:
        return ""

    parts: List[str] = []
    for layer, facts in layer_outline.items():
        if facts and "No specific facts" not in facts:
            parts.append(f"{layer.title()}: {facts}")

    if not parts:
        return ""

    return "Layer cues pulled from retrieved chunks → " + " | ".join(parts)


def _build_retry_instruction(validation: Dict[str, Any], attempt: int, layer_outline: Optional[Dict[str, str]]) -> str:
    """Create a corrective instruction block for retry attempts."""

    missing_layers = validation.get("missing_layers", [])
    word_count = validation.get("word_count", 0)
    word_count_valid = validation.get("word_count_valid", False)
    has_followup = validation.get("has_followup_question", False)
    has_sources = validation.get("has_sources", False)
    has_repetition = validation.get("has_repetition", False)
    repetitive_phrases = validation.get("repetitive_phrases", [])

    missing_text = ", ".join(missing_layers) if missing_layers else "All required layers are present."
    outline_hint = _format_outline_hint(layer_outline)

    retry_block = [
        f"REWRITE DIRECTIVE #{attempt} FOR MENU OPTION 1 (BRIEF OVERVIEW):",
        f"- Previous attempt length: {word_count} words (target: 100-150 words, valid: {word_count_valid}).",
        f"- Missing layers: {missing_text}.",
        f"- Follow-up question present: {has_followup} (REQUIRED).",
    ]

    if has_sources:
        retry_block.append("- CRITICAL: Remove 'Sources:' section. Do not include citations or sources in your response.")

    if has_repetition:
        retry_block.append(f"- CRITICAL: Avoid repetition. The following phrases appeared multiple times: {', '.join(repetitive_phrases[:2])}")
        retry_block.append("- Each layer must have unique wording. Do not repeat the same phrase across layers.")

    retry_block.extend([
        "- Start over and deliver a BRIEF overview that lists the stack with purpose explanations.",
        "- EACH layer (Frontend, Backend, Data Pipeline, Observability) must be mentioned with 1-2 sentences explaining purpose only.",
        f"- Keep it between {MENU_OPTION_ONE_MIN_WORDS} and 160 words (strict limit).",
        "- MUST end with: 'Would you like a full detailed end-to-end walkthrough of how I work, or would you prefer detail on a specific part of my stack or architecture?'",
        "- DO NOT copy verbatim from context. Synthesize into unique, concise sentences.",
        "- DO NOT repeat the same phrase across layers."
    ])

    if outline_hint:
        retry_block.append(f"- {outline_hint}.")

    retry_block.append("- Do not reuse wording from the previous attempt; synthesize anew with the provided context.")

    return "\n".join(retry_block)


def _compute_retrieval_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collect simple stats from retrieved chunks for observability copy."""

    similarities = [chunk.get("similarity") for chunk in chunks if isinstance(chunk, dict) and chunk.get("similarity") is not None]
    avg_similarity = sum(similarities) / len(similarities) if similarities else None

    return {
        "chunk_count": len(chunks),
        "avg_similarity": avg_similarity
    }


def _formatted_layer_fact(layer_outline: Optional[Dict[str, str]], layer: str, fallback: str) -> str:
    """Return a readable fact string for a layer."""

    if not layer_outline:
        return fallback

    facts = layer_outline.get(layer)
    if not facts or "No specific facts" in facts:
        return fallback

    return facts.replace(" | ", "; ")


def _build_deterministic_menu_option_response(layer_outline: Optional[Dict[str, str]], chunks: List[Dict[str, Any]]) -> str:
    """Fallback narrative when the LLM refuses to follow instructions - brief overview format."""

    outline = layer_outline or _extract_layer_outline(chunks)

    frontend_facts = _formatted_layer_fact(
        outline,
        "frontend",
        "Streamlit UI, Next.js"
    )
    backend_facts = _formatted_layer_fact(
        outline,
        "backend",
        "LangGraph, Python 3.11"
    )
    data_facts = _formatted_layer_fact(
        outline,
        "data",
        "Supabase Postgres, pgvector, OpenAI embeddings"
    )
    observability_facts = _formatted_layer_fact(
        outline,
        "observability",
        "LangSmith tracing, Supabase analytics"
    )

    opening = "Here's my tech stack at a glance:"

    frontend = (
        f"Frontend: {frontend_facts}. Purpose: Handles user interactions and provides the chat interface for conversations."
    )

    backend = (
        f"Backend: {backend_facts}. Purpose: Orchestrates conversation flow through modular nodes that handle intent classification, retrieval, generation, and logging."
    )

    data_layer = (
        f"Data Pipeline: {data_facts}. Purpose: Stores knowledge base chunks as vector embeddings and enables semantic search for grounded responses."
    )

    observability = (
        f"Observability: {observability_facts}. Purpose: Tracks performance metrics, traces LLM calls, and logs analytics for debugging and optimization."
    )

    closing_question = (
        "Would you like a full detailed end-to-end walkthrough of how I work, or would you prefer detail on a specific part of my stack or architecture?"
    )

    paragraphs = [opening, frontend, backend, data_layer, observability, closing_question]
    return "\n\n".join(paragraphs)


def _log_instruction_preview(tag: str, instructions: Optional[str]) -> None:
    """Emit concise logs about instruction payload size and preview."""

    if not instructions:
        logger.info(f"{tag}: no extra instructions provided")
        return

    logger.info(f"{tag}: instruction length = {len(instructions)} characters")
    logger.debug(f"{tag}: first 200 chars = {instructions[:200]}")


def _log_answer_snapshot(tag: str, answer: str) -> None:
    """Log word count and preview of a generated answer."""

    if not answer:
        logger.warning(f"{tag}: empty answer returned")
        return

    word_count = len(answer.split())
    logger.info(f"{tag}: answer word count = {word_count}")
    logger.debug(f"{tag}: first 200 chars = {answer[:200]}")


def _extract_layer_outline(chunks: list) -> Dict[str, str]:
    """Extract layer-specific facts from retrieved chunks to guide synthesis.

    Prevents verbatim copying by pre-organizing facts into layer buckets,
    forcing the LLM to synthesize across sources rather than echo one chunk.

    Args:
        chunks: Retrieved knowledge base chunks

    Returns:
        Dict mapping layer names to extracted facts/technologies
    """
    outline = {
        "frontend": [],
        "backend": [],
        "data": [],
        "observability": [],
        "deployment": []
    }

    # Keywords to identify layer-specific content
    layer_keywords = {
        "frontend": ["streamlit", "next.js", "react", "frontend", "ui", "interface"],
        "backend": ["langgraph", "langchain", "python", "orchestration", "pipeline", "nodes"],
        "data": ["supabase", "postgres", "pgvector", "embedding", "vector", "database", "ivfflat"],
        "observability": ["langsmith", "tracing", "monitoring", "analytics", "observability"],
        "deployment": ["vercel", "serverless", "deployment", "stateless", "scaling"]
    }

    for chunk in chunks:
        content = chunk.get("content", "").lower()

        # Extract sentences mentioning each layer
        sentences = content.split(". ")
        for layer, keywords in layer_keywords.items():
            for sentence in sentences:
                if any(keyword in sentence for keyword in keywords):
                    # Store original case sentence
                    original_sentence = sentence.strip()
                    if original_sentence and original_sentence not in outline[layer]:
                        outline[layer].append(original_sentence)

    # Build concise fact strings for each layer
    layer_facts = {}
    for layer, facts in outline.items():
        if facts:
            # Take top 2-3 facts per layer (avoid overwhelming the prompt)
            layer_facts[layer] = " | ".join(facts[:3])
        else:
            layer_facts[layer] = "(No specific facts found - synthesize from general context)"

    return layer_facts


def _detect_verbatim_copying(answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect if the LLM copied chunks verbatim instead of synthesizing.

    Checks for:
    - Same sentence appearing in multiple layers
    - Excessive repetition of phrases like "this ai assistant is built on"
    - Direct chunk content appearing in answer

    Args:
        answer: Generated answer text
        chunks: Retrieved chunks that were used as context

    Returns:
        Dict with:
        - has_verbatim_copying: bool
        - detected_phrases: List of repeated phrases found
        - severity: "low" | "medium" | "high"
    """
    answer_lower = answer.lower()
    detected_phrases = []
    severity = "low"

    # Check for common verbatim patterns
    verbatim_patterns = [
        "this ai assistant is built on",
        "this ai assistant is built",
        "this ai assistant uses",
        "this ai assistant implements",
    ]

    for pattern in verbatim_patterns:
        count = answer_lower.count(pattern)
        if count > 1:
            detected_phrases.append(f"'{pattern}' appears {count} times")
            if count >= 3:
                severity = "high"
            elif count >= 2:
                severity = "medium"

    # Check for same sentence appearing multiple times (simple heuristic)
    sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 20]
    if len(sentences) > 0:
        # Find sentences that appear more than once
        from collections import Counter
        sentence_counts = Counter(sentences)
        repeated = [s for s, count in sentence_counts.items() if count > 1]
        if repeated:
            detected_phrases.append(f"Repeated sentences found: {len(repeated)}")
            severity = "high" if len(repeated) >= 2 else "medium"

    # Check if answer contains direct chunk content (simple substring match)
    chunk_texts = [chunk.get("content", "").lower()[:100] for chunk in chunks[:3]]  # Check first 3 chunks
    for chunk_text in chunk_texts:
        if len(chunk_text) > 30 and chunk_text in answer_lower:
            detected_phrases.append("Direct chunk content detected")
            severity = "medium" if severity == "low" else severity

    has_verbatim = len(detected_phrases) > 0

    if has_verbatim:
        logger.warning(
            f"⚠️ Verbatim copying detected (severity: {severity}): {', '.join(detected_phrases)}"
        )

    return {
        "has_verbatim_copying": has_verbatim,
        "detected_phrases": detected_phrases,
        "severity": severity
    }


def select_model_for_task(state: ConversationState) -> str:
    """Select appropriate OpenAI model based on query complexity.

    Uses different models for different reasoning depths:
    - Reasoning model (o1-preview): Complex architecture, multi-step reasoning, planning
    - Technical model (gpt-4o-mini): Technical queries for hiring managers/developers
    - Default model (gpt-4): Most queries requiring balanced quality/speed
    - Fast model (gpt-3.5-turbo): Simple factual queries, greetings

    Args:
        state: Current conversation state with query and classification metadata

    Returns:
        Model name string (e.g., "o1-preview", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo")
    """
    query = state.get("query", "").lower()
    role_mode = state.get("role_mode", "")
    query_type = state.get("query_type", "")

    # Use GPT-4o for structured technical responses requiring comprehensive explanations
    # gpt-4o-mini struggles with structured output at temperature=0.9
    technical_query_types = ["menu_selection", "technical", "architecture", "code_related"]
    technical_roles = ["hiring_manager_technical", "software_developer"]

    if (query_type in technical_query_types or role_mode in technical_roles):
        # For menu option 1 (brief tech stack overview), gpt-4o-mini is sufficient for concise output
        if query_type == "menu_selection" and state.get("menu_choice") == "1":
            logger.info(f"Using gpt-4o-mini for brief tech stack overview (menu option 1)")
            return "gpt-4o-mini"
        # For other technical queries, gpt-4o-mini is sufficient
        logger.info(f"Using gpt-4o-mini for technical query (type={query_type}, role={role_mode})")
        return "gpt-4o-mini"

    # Use reasoning model for complex tasks requiring extended thinking
    complex_keywords = [
        "compare", "tradeoffs", "why choose", "planning", "strategy",
        "optimization", "scaling", "enterprise", "evaluate", "recommend"
    ]

    # Check for complex reasoning needs (but not basic architecture explanations)
    if any(kw in query for kw in complex_keywords) and "architecture" not in query:
        logger.info(f"Using reasoning model for complex query: {query[:50]}...")
        return supabase_settings.openai_reasoning_model

    # Use fast model for simple queries
    simple_keywords = [
        "hello", "hi", "hey", "thanks", "thank you",
        "what is", "who is", "when", "where"
    ]

    if any(kw in query for kw in simple_keywords) and len(query.split()) < 8:
        logger.info(f"Using fast model for simple query: {query[:50]}...")
        return supabase_settings.openai_fast_model

    # Default to standard model for most queries
    return supabase_settings.openai_model


def generate_draft(state: ConversationState, rag_engine: RagEngine) -> Dict[str, Any]:
    """Generate a draft assistant response using retrieved context.

    This is where the LLM creates the actual answer to the user's question.
    It uses the chunks we retrieved in the previous step as context.

    Special cases:
    - For data display requests, we skip LLM generation and fetch live analytics
    - For vague queries with insufficient context, we provide a helpful fallback

    Runtime awareness (Software Developer role only):
    - Architecture queries → Show conversation flow or full stack diagrams
    - Performance queries → Show metrics table
    - Code queries → Show actual retrieval implementation
    - SQL queries → Show pgvector query example
    - Cost queries → Show cost analysis table
    - Scaling queries → Show enterprise scaling strategy

    Design Principles:
    - **SRP**: Only generates answer, doesn't retrieve or log
    - **Defensibility**: Fail-fast on missing query, fail-safe on LLM errors
    - **Maintainability**: Separates fallback logic from generation logic
    - **Simplicity (KISS)**: Clear flow - validate → check special cases → generate

    Args:
        state: Current conversation state with query + retrieved chunks
        rag_engine: RAG engine with response generator

    Returns:
        Partial state update dict with 'answer' and optional 'fallback_used' flag

    Raises:
        KeyError: If required 'query' field missing from state

    Example:
        >>> state = ConversationState(query="How does RAG work?", retrieved_chunks=[...])
        >>> generate_draft(state, rag_engine)
        >>> len(state["draft_answer"])  # Should have LLM-generated answer
        342
    """
    # Fail-fast: Validate required fields (Defensibility)
    try:
        query = state["query"]
    except KeyError as e:
        logger.error("generate_draft called without query in state")
        raise KeyError("State must contain 'query' field for generation") from e

    # Log entry for debugging
    current_answer = state.get("answer", "")
    logger.info(f"generate_draft ENTERED - query='{query}', pipeline_halt={state.get('pipeline_halt')}, is_greeting={state.get('is_greeting')}, answer_preview={current_answer[:80] if current_answer else 'None'}")

    # GUARD: Skip generation if this is a greeting or pipeline is halted
    if state.get("pipeline_halt") or state.get("is_greeting"):
        logger.info("generate_draft: Skipping - pipeline_halt or is_greeting is True")
        return state

    # GUARD: Skip generation if query is empty (initial greeting case)
    if not query or not query.strip():
        logger.info("generate_draft: Skipping - empty query")
        return state

    # GUARD: Never call LLM if initial greeting is already set (check before any processing)
    if current_answer and "1️⃣ Hiring Manager" in current_answer:
        logger.warning("generate_draft: Initial greeting detected in answer - skipping LLM call entirely")
        return state

    # Access optional fields safely (Defensibility)
    retrieved_chunks = state.get("retrieved_chunks", [])
    role = state.get("role", "Just looking around")
    chat_history = state.get("chat_history", [])

    # Layer 2: File Content Integration in Generation
    # Include file content in context when available (supplements retrieved chunks)
    file_content = state.get("file_content")
    if file_content and file_content.get("success") and file_content.get("content"):
        # Prepend file content as a special chunk at the beginning of retrieved_chunks
        # This ensures it's included in the context passed to the LLM
        file_chunk = {
            "content": f"[File: {file_content['file_path']}]\n{file_content['content']}",
            "section": f"File: {file_content['file_path']}",
            "similarity": 1.0,  # High similarity since it's explicitly requested
            "doc_id": "file_content",
            "metadata": {
                "file_path": file_content['file_path'],
                "line_count": file_content.get('line_count', 0),
                "source": "file_reading"
            }
        }
        # Insert at beginning so file content appears first in context
        retrieved_chunks = [file_chunk] + retrieved_chunks
        logger.info(f"File content included in context: {file_content['file_path']} ({file_content['line_count']} lines)")

    # Initialize update dict (Loose Coupling)
    update: Dict[str, Any] = {}
    state.setdefault("analytics_metadata", {})

    grounding_status = state.get("grounding_status")
    if grounding_status and grounding_status not in {"ok", "unknown"}:
        return state

    # For data display requests, we'll fetch live analytics later
    # Just set a placeholder for now
    if state.get("data_display_requested", False):
        placeholder = "Fetching live analytics data from Supabase..."
        update["answer"] = placeholder
        update["draft_answer"] = placeholder
        state.update(update)
        return state

    # RUNTIME AWARENESS: Detect technical deep dive requests (SOFTWARE DEVELOPER ONLY)
    # Based on PORTFOLIA_LANGGRAPH_CONTEXT.md - Section: "When User Asks Technical Questions"
    runtime_awareness_triggered = False
    runtime_content_block = None

    if role == "Software Developer":
        query_lower = query.lower()

        # Architecture questions → Show conversation flow diagram or full stack
        if any(kw in query_lower for kw in ["architecture", "how do you work", "how does this work", "system design", "how are you built"]):
            if "rag" in query_lower or "retrieval" in query_lower or "search" in query_lower:
                # RAG-specific architecture
                runtime_content_block = content_blocks.rag_pipeline_explanation()
                runtime_awareness_triggered = True
                logger.info("Runtime awareness: RAG pipeline explanation triggered")
            elif "flow" in query_lower or "pipeline" in query_lower or "nodes" in query_lower:
                # Conversation flow
                runtime_content_block = content_blocks.conversation_flow_diagram()
                runtime_awareness_triggered = True
                logger.info("Runtime awareness: Conversation flow diagram triggered")
            else:
                # General architecture
                runtime_content_block = content_blocks.architecture_stack_explanation()
                runtime_awareness_triggered = True
                logger.info("Runtime awareness: Architecture stack explanation triggered")

        # Performance questions → Show metrics table
        elif any(kw in query_lower for kw in ["performance", "latency", "speed", "how fast", "metrics", "p95", "p99"]):
            runtime_content_block = content_blocks.performance_metrics_table()
            runtime_awareness_triggered = True
            logger.info("Runtime awareness: Performance metrics table triggered")

        # Code questions → Show actual retrieval code
        elif any(kw in query_lower for kw in ["show me code", "show code", "show me the code", "retrieval code", "how do you retrieve"]):
            runtime_content_block = content_blocks.code_example_retrieval_method()
            runtime_awareness_triggered = True
            logger.info("Runtime awareness: Code example triggered")

        # SQL/query questions → Show pgvector query
        elif any(kw in query_lower for kw in ["sql", "query", "vector search", "pgvector", "how do you search"]):
            runtime_content_block = content_blocks.pgvector_query_example()
            runtime_awareness_triggered = True
            logger.info("Runtime awareness: pgvector query example triggered")

        # Cost questions → Show cost analysis
        elif any(kw in query_lower for kw in ["cost", "expensive", "pricing", "how much", "budget"]):
            runtime_content_block = content_blocks.cost_analysis_table()
            runtime_awareness_triggered = True
            logger.info("Runtime awareness: Cost analysis table triggered")

        # Scaling questions → Show enterprise scaling strategy
        elif any(kw in query_lower for kw in ["scale", "scaling", "enterprise", "100k users", "production", "deployment"]):
            runtime_content_block = content_blocks.enterprise_scaling_strategy()
            runtime_awareness_triggered = True
            logger.info("Runtime awareness: Enterprise scaling strategy triggered")

    # Use the LLM to generate a response with retrieved context
    # Add display intelligence based on query classification
    extra_instructions = []

    is_menu_option_one = (
        state.get("query_type") == "menu_selection" and
        state.get("menu_choice") == "1" and
        state.get("role_mode") == "hiring_manager_technical"
    )

    layer_outline: Optional[Dict[str, str]] = None

    # Special handling for menu option 1 (full tech stack) - ensure comprehensive coverage
    if is_menu_option_one:

        # Extract layer-specific facts from chunks to force synthesis
        retrieved_chunks = state.get("retrieved_chunks", [])
        layer_outline = _extract_layer_outline(retrieved_chunks)

        extra_instructions.append(
            "CRITICAL INSTRUCTION - BRIEF TECH STACK OVERVIEW:\n\n"
            "You must provide a BRIEF overview of my tech stack. This should be concise and clean.\n\n"
            "CRITICAL RULES:\n"
            "- DO NOT repeat the same phrase across layers. Each layer must use DIFFERENT wording - if you find yourself repeating the same phrase, rewrite it.\n"
            "- DO NOT copy verbatim from context. Synthesize the information into concise, unique sentences.\n"
            "- NEVER say 'this ai assistant' or 'This AI assistant' - always use 'I' or 'my'.\n"
            "- Example: ✅ 'I'm built on Python 3.11+' ❌ 'this ai assistant is built on Python 3.11+'\n"
            "- Example: ✅ 'My architecture uses...' ❌ 'this ai assistant uses...'\n"
            "- DO NOT include 'Sources:' or citations in your response.\n"
            "- CRITICAL: Your response MUST be between 100-150 words. Count your words carefully.\n\n"
            "WHAT NOT TO DO (VERBATIM COPYING EXAMPLES):\n"
            "❌ BAD: Repeating 'this ai assistant is built on a modern, scalable tech stack' in every layer\n"
            "❌ BAD: Copying the exact same sentence structure across Frontend, Backend, Data Pipeline, Observability\n"
            "❌ BAD: Using identical phrasing like 'this ai assistant is built on' multiple times\n"
            "✅ GOOD: Each layer has unique wording: 'I use Streamlit for...' vs 'My backend runs on...' vs 'I store data in...'\n\n"
            "REQUIRED OUTPUT FORMAT:\n\n"
            "[Brief opening: 1 sentence introducing the stack overview]\n\n"
            "Frontend: [1-2 sentences listing technologies and brief purpose. Technologies from context: " + layer_outline.get("frontend", "Streamlit UI, Next.js") + "]\n\n"
            "Backend: [1-2 sentences listing technologies and brief purpose. Technologies from context: " + layer_outline.get("backend", "LangGraph, Python 3.11") + "]\n\n"
            "Data Pipeline: [1-2 sentences listing technologies and brief purpose. Technologies from context: " + layer_outline.get("data", "Supabase Postgres, pgvector, OpenAI embeddings") + "]\n\n"
            "Observability: [1-2 sentences listing technologies and brief purpose. Technologies from context: " + layer_outline.get("observability", "LangSmith tracing, Supabase analytics") + "]\n\n"
            "[Closing question: End with this exact question: 'Would you like more detail on a specific layer, or a deeper dive into how my architecture works?']\n\n"
            "EXAMPLE OF GOOD FORMAT:\n"
            "Here's my tech stack at a glance:\n\n"
            "Frontend: Streamlit UI handles user interactions and provides the chat interface.\n\n"
            "Backend: LangGraph orchestrates conversation flow through modular nodes that handle intent classification and generation.\n\n"
            "Data Pipeline: Supabase Postgres with pgvector stores knowledge base chunks as vector embeddings for semantic search.\n\n"
            "Observability: LangSmith traces LLM calls and tracks performance metrics for debugging and optimization.\n\n"
            "Would you like more detail on a specific layer, or a deeper dive into how my architecture works?\n\n"
            "TARGET: 100-150 words total | Brief and concise | Plain text, no asterisks | "
            "List the stack: Frontend → Backend → Data Pipeline → Observability | "
            "Each layer: 1-2 sentences explaining purpose only | "
            "MUST end with the closing question about specific layer or architecture"
        )
        logger.info(f"📝 Layer outline extracted for synthesis: {list(layer_outline.keys())}")
        logger.warning(f"🚨 DEBUG: extra_instructions length = {len(''.join(extra_instructions))} characters")
        logger.warning(f"🚨 DEBUG: First 200 chars of extra_instructions = {(''.join(extra_instructions))[:200]}")

    # Detect documentation chunks and add synthesis instructions
    has_documentation_chunks = any(
        chunk.get("doc_id") == "documentation" for chunk in retrieved_chunks
    )

    # Detect walkthrough requests
    walkthrough_keywords = [
        "walkthrough", "walk through", "end to end", "end-to-end",
        "full stack", "comprehensive", "complete overview", "full overview"
    ]
    is_walkthrough_query = any(
        kw in query.lower()
        for kw in walkthrough_keywords
    )

    if has_documentation_chunks:
        extra_instructions.append(
            "CRITICAL DOCUMENTATION SYNTHESIS - YOU MUST FOLLOW THESE RULES:\n\n"
            "You retrieved documentation chunks with markdown headers. These are RAW STRUCTURE, not conversational text.\n\n"
            "MANDATORY TRANSFORMATIONS:\n"
            "1. Remove ALL markdown syntax: ##, ###, **, -\n"
            "2. Transform headers into natural sentences:\n"
            "   ❌ '## Example Conversation 1' → ✅ 'Here's an example conversation...'\n"
            "   ❌ '## 🔧 Hiring Manager (Technical)' → ✅ 'For technical hiring managers, I focus on...'\n"
            "3. Synthesize content into flowing first-person narrative\n"
            "4. DO NOT copy-paste documentation structure - rewrite it conversationally\n"
            "5. If you see 'query=\"What's the tech stack?\"', transform it into natural language\n"
            "FAILURE TO DO THIS WILL RESULT IN POOR USER EXPERIENCE."
        )
        logger.info("📚 Documentation chunks detected - added synthesis instructions")

    if is_walkthrough_query:
        if has_documentation_chunks:
            extra_instructions.append(
                "CRITICAL: User requested a FULL WALKTHROUGH. You must provide a comprehensive, "
                "end-to-end explanation (300-500 words) that flows naturally from frontend → backend → data → observability. "
                "DO NOT just list documentation chunks - synthesize them into a cohesive narrative. "
                "Transform all markdown headers into conversational explanations. "
                "Provide detailed explanation, not just brief snippets. "
                "If the documentation contains examples or structured content, weave it into your narrative naturally."
            )
        else:
            extra_instructions.append(
                "CRITICAL: User requested a FULL WALKTHROUGH. Provide a comprehensive, "
                "end-to-end explanation (300-500 words) covering all layers of the architecture. "
                "Flow naturally from frontend → backend → data → observability with detailed explanations."
            )
        logger.info("Walkthrough query detected - comprehensive response mode enabled")

    # Detect enterprise adaptation questions
    enterprise_keywords = [
        "enterprise", "adapt", "scale", "business use case",
        "customer support", "internal docs", "how would this work for",
        "enterprise use", "enterprise deployment", "enterprise adaptation",
        "business scenario", "production deployment", "large scale"
    ]
    query_lower = query.lower()
    is_enterprise_adaptation = any(kw in query_lower for kw in enterprise_keywords)

    if is_enterprise_adaptation:
        extra_instructions.append(
            "ENTERPRISE ADAPTATION FRAMEWORK - CRITICAL INSTRUCTION:\n\n"
            "Structure your response using the systematic enterprise adaptation framework:\n\n"
            "1. Business Use Case Example (concrete scenario):\n"
            "   - Must be specific, not generic. Include scale, context, and business requirements.\n"
            "   - Example: 'Consider a customer support bot handling 10,000 tickets per day for a SaaS company. Users ask about billing, feature availability, and technical troubleshooting. The bot must integrate with Zendesk, authenticate via SSO, and maintain conversation context across multiple channels.'\n"
            "   - Avoid vague statements - be concrete with numbers and context.\n\n"
            "2. Required Changes (specific programming/architecture modifications):\n"
            "   - Authentication: Specify exact implementation (e.g., 'Add SSO layer using OAuth2/OIDC. Implement token validation middleware that intercepts requests before they reach the orchestration layer.')\n"
            "   - Integration: Specify code changes (e.g., 'Swap email action node for Zendesk API integration. Replace execute_actions node's email handler with Zendesk ticket creation logic.')\n"
            "   - Scaling: Specify infrastructure changes (e.g., 'Add Redis caching layer for vector search results. Cache embeddings for common queries to reduce pgvector load from 850ms to 50ms.')\n"
            "   - Security: Specify policy changes (e.g., 'Implement row-level security policies for multi-tenant data. Add tenant_id to all database queries and filter results by organization.')\n"
            "   - Be specific about code/architecture changes, not just concepts.\n\n"
            "3. Unchanged Components (what stays the same):\n"
            "   - 'LangGraph orchestration logic remains identical. The 18-node pipeline structure is unchanged.'\n"
            "   - 'RAG pipeline (embedding → retrieval → generation) unchanged. Same pgvector search, same similarity threshold, same chunking strategy.'\n"
            "   - 'Vector search algorithm preserved. The cosine similarity calculation and IVFFLAT indexing logic stay the same.'\n"
            "   - 'State management pattern consistent. ConversationState structure and node update logic remain unchanged.'\n"
            "   - Explain why these components don't need modification.\n\n"
            "4. Integration Reasoning (why these decisions matter):\n"
            "   - Connect changes to business requirements: 'SSO is required for enterprise security compliance. Without it, customers cannot integrate with their identity providers.'\n"
            "   - Explain tradeoffs and benefits: 'Adding Redis caching increases complexity but reduces database load by 60%, enabling horizontal scaling without database bottlenecks.'\n"
            "   - Show how unchanged components enable rapid adaptation: 'Because the LangGraph orchestration is modular, swapping the action node takes 2 hours instead of 2 weeks. The RAG pipeline remains untouched, preserving all existing knowledge base investments.'\n"
            "   - Demonstrate understanding of enterprise constraints: 'Multi-tenant RLS is non-negotiable for SaaS deployments. The unchanged vector search logic means we can add tenant filtering without rewriting retrieval code.'\n\n"
            "Use systematic paragraphs, NOT bullet points. Include quantitative details where relevant. "
            "Start with the fundamental problem enterprise deployments must solve, then work through each section methodically."
        )
        logger.info("🏢 Enterprise adaptation framework triggered for query")

    # When teaching/explaining, provide comprehensive depth
    elif state.get("needs_longer_response", False) or state.get("teaching_moment", False):
        extra_instructions.append(
            "This is a teaching moment - provide a comprehensive, well-structured explanation. "
            "Break down concepts clearly, connect technical details to business value, and "
            "help the user truly understand. Use examples where helpful."
        )

    # Runtime awareness: Add content block to context if triggered
    if runtime_awareness_triggered and runtime_content_block:
        extra_instructions.append(
            f"RUNTIME AWARENESS: The user asked a technical question about Portfolia's architecture/performance. "
            f"Below is a self-referential teaching block showing live data. Reference this in your explanation, "
            f"weave it into your narrative naturally, and expand on it conversationally. "
            f"Maintain warmth and curiosity while being technically precise.\n\n{runtime_content_block}"
        )

    # EXPLICIT code request - user specifically asked
    if state.get("code_display_requested", False) and role in [
        "Software Developer",
        "Hiring Manager (technical)"
    ]:
        extra_instructions.append(
            "The user has requested code. After your explanation, include relevant code snippets "
            "with comments explaining key decisions. Keep code blocks under 40 lines and focus "
            "on the most interesting parts."
        )
    # PROACTIVE code suggestion - code would clarify but wasn't explicitly requested
    elif state.get("code_would_help", False) and role in [
        "Software Developer",
        "Hiring Manager (technical)"
    ]:
        extra_instructions.append(
            "This technical concept would benefit from a code example. After your explanation, "
            "include a relevant code snippet (≤40 lines) with comments to clarify the implementation. "
            "This is proactive - the user didn't explicitly ask but code will help understanding."
        )

    # EXPLICIT data request - user specifically asked
    if state.get("data_display_requested", False):
        extra_instructions.append(
            "The user wants data/analytics. Be brief with narrative - focus on presenting clean "
            "tables with proper formatting. Include source attribution."
        )
    # PROACTIVE data suggestion - metrics would clarify but weren't explicitly requested
    elif state.get("data_would_help", False):
        extra_instructions.append(
            "This question would benefit from actual metrics/data. After your explanation, "
            "include relevant analytics in table format if available. Be concise with tables, "
            "include source attribution. This is proactive - help the user with concrete numbers."
        )

    # Job details gathering (AFTER resume sent) - Task 9
    # Import here to avoid circular dependency
    from assistant.flows.node_logic.util_resume_distribution import should_gather_job_details, get_job_details_prompt

    if should_gather_job_details(state):
        extra_instructions.append(get_job_details_prompt())

    # Build the instruction suffix
    instruction_suffix = " ".join(extra_instructions) if extra_instructions else None
    base_instruction_suffix = instruction_suffix

    # CRITICAL GUARD: Never call LLM if initial greeting is already set
    # Check again right before LLM call (defense in depth)
    current_answer_check = state.get("answer", "")
    if current_answer_check and "1️⃣ Hiring Manager" in current_answer_check:
        logger.warning("generate_draft: Initial greeting detected before LLM call - aborting generation")
        return state  # Don't call LLM, don't modify answer

    # Select appropriate model for this task
    selected_model = select_model_for_task(state)

    # Store model selection in state for analytics
    state.setdefault("analytics_metadata", {})["selected_model"] = selected_model

    # Generate response with LLM (Encapsulation - delegates to response generator)
    attempt_label = "menu-option-1-brief" if is_menu_option_one else "generic-response"
    attempt_counter = 0

    if is_menu_option_one:
        _log_instruction_preview(f"{attempt_label}-attempt-{attempt_counter}", instruction_suffix)

    try:
        answer = rag_engine.response_generator.generate_contextual_response(
            query=query,
            context=retrieved_chunks,
            role=role,
            chat_history=chat_history,
            extra_instructions=instruction_suffix,
            model_name=selected_model  # Pass selected model
        )
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        # Fail-safe: Provide graceful error message (Defensibility)
        answer = (
            "I'm having trouble generating a response right now. "
            "Please try rephrasing your question or ask something else!"
        )

    if is_menu_option_one:
        _log_answer_snapshot(f"{attempt_label}-attempt-{attempt_counter}", answer)

        # Check for verbatim copying
        verbatim_check = _detect_verbatim_copying(answer, retrieved_chunks)
        if verbatim_check["has_verbatim_copying"]:
            logger.warning(
                f"⚠️ Verbatim copying detected in menu option 1 answer (severity: {verbatim_check['severity']})"
            )
            # Apply stronger first-person enforcement
            answer = rag_engine.response_generator._enforce_first_person(answer)
            update["generation_quality_warning"] = f"Verbatim copying detected: {', '.join(verbatim_check['detected_phrases'])}"

        # Force the correct closing question - detect and replace old patterns
        old_patterns = [
            r"full detailed end-to-end walkthrough",
            r"end-to-end walkthrough",
            r"full end to end walkthrough",
            r"end to end walkthrough"
        ]

        for pattern in old_patterns:
            if re.search(pattern, answer, flags=re.IGNORECASE):
                # Replace the entire closing question
                answer = re.sub(
                    r'Would you like.*\?',
                    "Would you like more detail on a specific layer, or a deeper dive into how my architecture works?",
                    answer,
                    flags=re.IGNORECASE | re.DOTALL
                )
                logger.info("Fixed closing question to match required format")
                break

    # Validate structured output for menu option 1 (brief tech stack overview)
    if is_menu_option_one:
        validation = _validate_menu_option_one_answer(answer)
        retries = 0

        while not validation["is_valid"] and retries < MENU_OPTION_ONE_MAX_RETRIES:
            retries += 1
            logger.warning(
                f"⚠️ Menu option 1 validation failed (attempt {retries}): "
                f"missing_layers={validation['missing_layers']}, words={validation['word_count']}"
            )
            retry_instructions = _build_retry_instruction(validation, retries, layer_outline)
            combined_instructions = "\n\n".join(filter(None, [base_instruction_suffix, retry_instructions]))

            _log_instruction_preview(f"{attempt_label}-attempt-{retries}", combined_instructions)

            try:
                answer = rag_engine.response_generator.generate_contextual_response(
                    query=query,
                    context=retrieved_chunks,
                    role=role,
                    chat_history=chat_history,
                    extra_instructions=combined_instructions,
                    model_name=selected_model
                )
            except Exception as retry_error:
                logger.error(f"Retry generation failed: {retry_error}")
                break

            _log_answer_snapshot(f"{attempt_label}-attempt-{retries}", answer)
            validation = _validate_menu_option_one_answer(answer)

        if not validation["is_valid"]:
            logger.error("❌ Menu option 1 output failed validation after retries — using deterministic fallback")
            answer = _build_deterministic_menu_option_response(layer_outline, retrieved_chunks)
            _log_answer_snapshot(f"{attempt_label}-fallback", answer)
            validation = _validate_menu_option_one_answer(answer)
            update["generation_quality_warning"] = "LLM fallback used for menu option 1"
        else:
            if retries:
                state.setdefault("analytics_metadata", {})["menu_option_retries"] = retries

        missing_layers = validation["missing_layers"]
        word_count = validation["word_count"]
        has_followup = validation.get("has_followup_question", False)

        if missing_layers:
            logger.warning(f"⚠️ Generated answer missing {len(missing_layers)} layers: {missing_layers}")
            logger.warning(f"⚠️ Word count: {word_count} (target: 100-150)")
            update["generation_quality_warning"] = f"Missing layers: {', '.join(missing_layers)}"
        elif word_count < MENU_OPTION_ONE_MIN_WORDS:
            logger.warning(f"⚠️ Generated answer too short: {word_count} words (target: 100-150)")
            update["generation_quality_warning"] = f"Too short: {word_count} words"
        elif word_count > MENU_OPTION_ONE_MAX_WORDS:
            logger.warning(f"⚠️ Generated answer too long: {word_count} words (target: 100-150)")
            update["generation_quality_warning"] = f"Too long: {word_count} words"
        elif not has_followup:
            logger.warning(f"⚠️ Generated answer missing follow-up question")
            update["generation_quality_warning"] = "Missing follow-up question"
        else:
            logger.info(f"✅ Brief overview validated: All layers present, {word_count} words, follow-up question included")

    cleaned_answer = sanitize_generated_answer(answer)

    # CRITICAL GUARD: Never overwrite the initial greeting if it's already set
    current_answer = state.get("answer", "")
    if current_answer and "1️⃣ Hiring Manager" in current_answer:
        logger.warning("generate_draft: Attempted to overwrite initial greeting - preserving original")
        return state  # Don't modify the answer at all

    update["draft_answer"] = cleaned_answer
    update["answer"] = cleaned_answer

    # Return partial update - LangGraph will merge into state
    return update


def hallucination_check(state: ConversationState) -> ConversationState:
    """Attach lightweight citations and flag hallucination risk.

    This node adds source attribution to the draft answer to help users
    understand where information came from and detect potential hallucinations.

    Citation strategy:
    - Extract section metadata from retrieved chunks
    - Append "Sources: ..." to answer if not already present
    - Limit to top 3 sources to avoid clutter

    Performance: <10ms (in-memory string manipulation)

    Design Principles:
    - SRP: Only handles citation attachment, doesn't modify answer content
    - Defensibility: Gracefully handles missing chunks or draft
    - Simplicity (KISS): Simple string concatenation, no complex formatting

    Args:
        state: ConversationState with draft_answer and retrieved_chunks

    Returns:
        Updated state with:
        - draft_answer: Answer with "Sources: ..." appended
        - hallucination_safe: bool flag indicating citation status

    Example:
        >>> state = {
        ...     "draft_answer": "RAG works by retrieving context first...",
        ...     "retrieved_chunks": [
        ...         {"section": "RAG Architecture"},
        ...         {"section": "Vector Search"}
        ...     ]
        ... }
        >>> hallucination_check(state)
        >>> "Sources:" in state["draft_answer"]
        True
    """
    draft = state.get("draft_answer")
    chunks = state.get("retrieved_chunks", [])

    if not draft or not chunks:
        state["hallucination_safe"] = False if not chunks else state.get("hallucination_safe", True)
        return state

    citations = []
    for idx, chunk in enumerate(chunks, start=1):
        section = chunk.get("section") or f"knowledge chunk {idx}"
        citations.append(f"{idx}. {section}")

    citation_text = "; ".join(citations[:3])
    if citation_text and "Sources:" not in draft:
        state["draft_answer"] = f"{draft}\n\nSources: {citation_text}"

    state["hallucination_safe"] = True
    return state
