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
from typing import Dict, Any

from src.state.conversation_state import ConversationState
from src.core.rag_engine import RagEngine
from src.flows import content_blocks
from src.flows.node_logic.util_code_validation import sanitize_generated_answer
from src.config.supabase_config import supabase_settings

logger = logging.getLogger(__name__)


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

    # Use GPT-4o-mini for technical queries from technical audiences
    # This model provides better comprehension than gpt-3.5-turbo at similar cost
    technical_query_types = ["menu_selection", "technical", "architecture", "code_related"]
    technical_roles = ["hiring_manager_technical", "software_developer"]

    if (query_type in technical_query_types or role_mode in technical_roles):
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

    # Access optional fields safely (Defensibility)
    retrieved_chunks = state.get("retrieved_chunks", [])
    role = state.get("role", "Just looking around")
    chat_history = state.get("chat_history", [])

    # Initialize update dict (Loose Coupling)
    update: Dict[str, Any] = {}
    state.setdefault("analytics_metadata", {})

    if state.get("pipeline_halt"):
        return state

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

    # Special handling for menu option 1 (full tech stack) - ensure comprehensive coverage
    if (state.get("query_type") == "menu_selection" and
        state.get("menu_choice") == "1" and
        state.get("role_mode") == "hiring_manager_technical"):

        # Extract layer-specific facts from chunks to force synthesis
        retrieved_chunks = state.get("retrieved_chunks", [])
        layer_outline = _extract_layer_outline(retrieved_chunks)

        extra_instructions.append(
            "CRITICAL INSTRUCTION - FULL TECH STACK WALKTHROUGH:\n\n"
            "⚠️ ANTI-PLAGIARISM RULE: DO NOT copy text verbatim from context chunks. "
            "Synthesize the facts below into a cohesive narrative. Copying = automatic failure.\n\n"
            "📋 REQUIRED OUTPUT FORMAT (Follow this structure exactly):\n\n"
            "[Opening paragraph: 2-3 sentences introducing the 5-layer architecture and why it matters]\n\n"
            "**Frontend Layer:** [2-3 sentences synthesizing these facts: " + layer_outline.get("frontend", "Streamlit UI, Next.js migration planned") + "]\n\n"
            "**Backend/Orchestration Layer:** [2-3 sentences synthesizing these facts: " + layer_outline.get("backend", "LangGraph StateGraph, Python 3.11+, modular pipeline") + "]\n\n"
            "**Data Layer:** [2-3 sentences synthesizing these facts: " + layer_outline.get("data", "Supabase PostgreSQL, pgvector extension, OpenAI embeddings") + "]\n\n"
            "**Observability Layer:** [2-3 sentences synthesizing these facts: " + layer_outline.get("observability", "LangSmith tracing, Supabase analytics") + "]\n\n"
            "**Deployment Layer:** [2-3 sentences synthesizing these facts: " + layer_outline.get("deployment", "Vercel serverless, stateless design, horizontal scaling") + "]\n\n"
            "[Closing paragraph: 2-3 sentences on how these 5 layers work together]\n\n"
            "🎯 TARGET: 300-350 words total | Technical but accessible tone | "
            "Each layer MUST appear with the exact heading shown above."
        )
        logger.info(f"📝 Layer outline extracted for synthesis: {list(layer_outline.keys())}")

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
    from src.flows.node_logic.util_resume_distribution import should_gather_job_details, get_job_details_prompt

    if should_gather_job_details(state):
        extra_instructions.append(get_job_details_prompt())

    # Build the instruction suffix
    instruction_suffix = " ".join(extra_instructions) if extra_instructions else None

    # Select appropriate model for this task
    selected_model = select_model_for_task(state)

    # Store model selection in state for analytics
    state.setdefault("analytics_metadata", {})["selected_model"] = selected_model

    # Generate response with LLM (Encapsulation - delegates to response generator)
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

    # Validate structured output for menu option 1 (full tech stack)
    if (state.get("query_type") == "menu_selection" and
        state.get("menu_choice") == "1" and
        state.get("role_mode") == "hiring_manager_technical"):

        required_layers = [
            "**Frontend Layer:**",
            "**Backend/Orchestration Layer:**",
            "**Data Layer:**",
            "**Observability Layer:**",
            "**Deployment Layer:**"
        ]

        missing_layers = [layer for layer in required_layers if layer not in answer]
        word_count = len(answer.split())

        if missing_layers:
            logger.warning(f"⚠️ Generated answer missing {len(missing_layers)} layers: {missing_layers}")
            logger.warning(f"⚠️ Word count: {word_count} (target: 300-350)")
            # Flag for retry or append missing layers prompt
            update["generation_quality_warning"] = f"Missing layers: {', '.join(missing_layers)}"
        elif word_count < 250:
            logger.warning(f"⚠️ Generated answer too short: {word_count} words (target: 300-350)")
            update["generation_quality_warning"] = f"Too short: {word_count} words"
        else:
            logger.info(f"✅ Structured output validated: All 5 layers present, {word_count} words")

    cleaned_answer = sanitize_generated_answer(answer)
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
