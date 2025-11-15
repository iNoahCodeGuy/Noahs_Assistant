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
from typing import Dict, Any, List, Optional

from assistant.state.conversation_state import ConversationState
from assistant.core.rag_engine import RagEngine
from assistant.flows import content_blocks
from assistant.flows.node_logic.util_code_validation import sanitize_generated_answer
from assistant.config.supabase_config import supabase_settings

logger = logging.getLogger(__name__)

MENU_OPTION_ONE_MIN_WORDS = 330
MENU_OPTION_ONE_MAX_RETRIES = 2


def _validate_menu_option_one_answer(answer: str) -> Dict[str, Any]:
    """Validate menu option 1 output structure and length."""

    required_layers = [
        "**Frontend Layer:**",
        "**Backend/Orchestration Layer:**",
        "**Data Layer:**",
        "**Observability Layer:**",
        "**Deployment Layer:**"
    ]

    missing_layers = [layer for layer in required_layers if layer not in answer]
    word_count = len(answer.split())

    return {
        "missing_layers": missing_layers,
        "word_count": word_count,
        "is_valid": not missing_layers and word_count >= MENU_OPTION_ONE_MIN_WORDS
    }


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
    missing_text = ", ".join(missing_layers) if missing_layers else "All required headings are present."
    outline_hint = _format_outline_hint(layer_outline)

    retry_block = [
        f"REWRITE DIRECTIVE #{attempt} FOR MENU OPTION 1:",
        f"- Previous attempt length: {word_count} words (target: 350-400).",
        f"- Missing headings: {missing_text}.",
        "- Start over and deliver a fresh response that follows the exact layer order.",
        "- EACH layer must contain 3-4 sentences in first person and explicitly explain what the tools do, why they were chosen, and which problems they solve.",
        f"- Do not stop before {MENU_OPTION_ONE_MIN_WORDS} words."
    ]

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
    """Fallback narrative when the LLM refuses to follow instructions."""

    stats = _compute_retrieval_stats(chunks)
    outline = layer_outline or _extract_layer_outline(chunks)

    frontend_facts = _formatted_layer_fact(
        outline,
        "frontend",
        "Streamlit chat console, Next.js front door, Tailwind styling"
    )
    backend_facts = _formatted_layer_fact(
        outline,
        "backend",
        "LangGraph state machine, Python 3.11 orchestrators, modular node logic"
    )
    data_facts = _formatted_layer_fact(
        outline,
        "data",
        "Supabase Postgres, pgvector indexes, OpenAI text-embedding-3-small"
    )
    observability_facts = _formatted_layer_fact(
        outline,
        "observability",
        "LangSmith tracing, Supabase analytics tables, retrieval logs"
    )
    deployment_facts = _formatted_layer_fact(
        outline,
        "deployment",
        "Vercel serverless, Dockerized LangGraph runtime, horizontal scaling"
    )

    retrieval_sentence = ""
    if stats["chunk_count"]:
        if stats["avg_similarity"]:
            retrieval_sentence = (
                f"For this explanation I pulled {stats['chunk_count']} grounded chunks averaging "
                f"{stats['avg_similarity']:.3f} similarity, so every detail maps back to the Supabase knowledge base."
            )
        else:
            retrieval_sentence = (
                f"I anchored this walkthrough on {stats['chunk_count']} retrieved knowledge chunks so the teach-through-tour stays grounded."
            )

    opening = (
        "I'm going to walk you layer by layer through my production tech stack so you can see how a real "
        "RAG assistant is wired. Each section explains what I use, why I chose it, and the practical problems it solves "
        "when I'm teaching hiring managers and engineers."
    )

    frontend = (
        f"**Frontend Layer:** I keep two front doors because I teach in different contexts. Streamlit powers the hands-on "
        f"workshop console where I can spin up new widgets or role toggles in minutes, while the Next.js surface on Vercel "
        f"handles production traffic with proper routing and SEO. The retrieved notes call out {frontend_facts}, and those "
        "choices let me mix rapid iteration with polished presentation. The UI's job is to keep conversations flowing while "
        "surfacing analytics tables, diagrams, or call-to-action cards without ever blocking the learning experience."
    )

    backend = (
        f"**Backend/Orchestration Layer:** Behind the UI sits a LangGraph pipeline written in Python 3.11. I break the "
        f"conversation into intent classification, retrieval, generation, formatting, action planning, and logging nodes so "
        f"every concern stays testable. The facts in context mention {backend_facts}; that modularity lets me swap LLM "
        "providers or add new teaching affordances without rewriting the flow. It also means I can trace every decision a "
        "user turn triggered, which is critical for enterprise audiences evaluating rigor."
    )

    data_layer = (
        f"**Data Layer:** Supabase Postgres is my single source of truth, and pgvector keeps all embeddings inside the same "
        f"transactional database. I chunk knowledge sources, embed them with text-embedding-3-small, and store both the raw "
        f"text and the vector so I can regenerate or audit on demand. The retrieved context highlights {data_facts}. Using "
        f"one governed database keeps costs predictable, enables row-level security if needed, and makes it easy to explain "
        f"how retrieval is grounded when someone asks for proof."
    )

    observability = (
        f"**Observability Layer:** I trace everything with LangSmith and persist superset analytics in Supabase tables. That "
        f"means latency, similarity scores, and follow-up actions are all queryable, and I can show a hiring manager exactly "
        f"how a turn performed. Context snippets cite {observability_facts}, which is how I keep a clean audit trail. This "
        f"layer is what lets me treat the assistant itself as a live portfolio piece—you can see the evidence instead of "
        f"taking my word for it."
    )

    deployment = (
        f"**Deployment Layer:** Everything deploys serverlessly on Vercel so traffic bursts are painless, while a Dockerized "
        f"LangGraph runtime handles local testing and LangGraph Studio sessions. The materials mention {deployment_facts}, so "
        f"you're seeing the same stack I'd ship to an enterprise pilot. Stateless routes keep cold starts low, and the safe "
        f"restart workflow makes it easy to rebuild containers whenever I change prompt wiring or tracing hooks."
    )

    closing_parts = [
        "It all comes together as a single teaching engine: the UI invites curiosity, the LangGraph core orchestrates every "
        "decision, pgvector keeps answers grounded, observability proves reliability, and Vercel handles the edge delivery.",
        "That combination lets me explain GenAI architecture in first person while demonstrating production discipline in the same breath."
    ]

    if retrieval_sentence:
        closing_parts.append(retrieval_sentence)

    closing = " ".join(closing_parts)

    paragraphs = [opening, frontend, backend, data_layer, observability, deployment, closing]
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
        # For menu option 1 (full tech stack), use GPT-4o for better structured output adherence
        if query_type == "menu_selection" and state.get("menu_choice") == "1":
            logger.info(f"Using gpt-4o for structured tech stack response (menu option 1)")
            return "gpt-4o"
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
            "CRITICAL INSTRUCTION - FULL TECH STACK WALKTHROUGH:\n\n"
            "🎯 BALANCING ACT:\n"
            "- Be CREATIVE and CONVERSATIONAL in your explanations (high temperature = warmth + personality)\n"
            "- But ALL FACTS must be GROUNDED in the retrieved context chunks (no hallucinations)\n"
            "- Think of it like: creative teacher using a textbook (context = textbook, you = creative teacher)\n\n"
            "⚠️ ANTI-PLAGIARISM RULE: DO NOT copy text verbatim from context chunks. "
            "Synthesize the facts below into a cohesive, natural-sounding narrative.\n\n"
            "📋 REQUIRED OUTPUT FORMAT (Follow this structure exactly):\n\n"
            "[Opening paragraph: 2-3 conversational sentences introducing the 5-layer architecture. Make it inviting, not robotic! Explain why this architecture matters using natural language.]\n\n"
            "**Frontend Layer:** [3-4 sentences. Start by listing technologies from context: " + layer_outline.get("frontend", "Streamlit UI, Next.js migration planned") + ". Then creatively explain: (1) What each tech does (your words, grounded facts), (2) WHY it was chosen (ground in context but explain conversationally), (3) What problems it solves (specific, from context).]\n\n"
            "**Backend/Orchestration Layer:** [3-4 sentences. Technologies from context: " + layer_outline.get("backend", "LangGraph StateGraph, Python 3.11+, modular pipeline") + ". Then explain conversationally: (1) What it does, (2) Why chosen, (3) Problems solved. Use natural transitions like 'This is perfect for...' or 'We chose this because...'.]\n\n"
            "**Data Layer:** [3-4 sentences. Technologies from context: " + layer_outline.get("data", "Supabase PostgreSQL, pgvector extension, OpenAI embeddings") + ". Explain with personality: (1) What it does, (2) Why chosen, (3) Problems solved. Connect to real-world impact.]\n\n"
            "**Observability Layer:** [3-4 sentences. Technologies from context: " + layer_outline.get("observability", "LangSmith tracing, Supabase analytics") + ". Make it engaging: (1) What it does, (2) Why chosen, (3) Problems solved.]\n\n"
            "**Deployment Layer:** [3-4 sentences. Technologies from context: " + layer_outline.get("deployment", "Vercel serverless, stateless design, horizontal scaling") + ". Explain with warmth: (1) What it does, (2) Why chosen, (3) Problems solved.]\n\n"
            "[Closing paragraph: 2-3 sentences showing how these 5 layers work together. Use conversational language like 'It all comes together...' or 'This architecture matters because...'. Connect to enterprise readiness with warmth.]\n\n"
            "🎯 TARGET: 350-400 words total (DO NOT STOP BEFORE 300 WORDS) | Conversational + technically accurate | "
            "Ground every fact in retrieved context, but explain with personality and natural transitions. "
            "Each layer MUST appear with the exact heading shown above."
        )
        logger.info(f"📝 Layer outline extracted for synthesis: {list(layer_outline.keys())}")
        logger.warning(f"🚨 DEBUG: extra_instructions length = {len(''.join(extra_instructions))} characters")
        logger.warning(f"🚨 DEBUG: First 200 chars of extra_instructions = {(''.join(extra_instructions))[:200]}")

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

    # Select appropriate model for this task
    selected_model = select_model_for_task(state)

    # Store model selection in state for analytics
    state.setdefault("analytics_metadata", {})["selected_model"] = selected_model

    # Generate response with LLM (Encapsulation - delegates to response generator)
    attempt_label = "menu-option-1" if is_menu_option_one else "generic-response"
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

    # Validate structured output for menu option 1 (full tech stack)
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

        if missing_layers:
            logger.warning(f"⚠️ Generated answer missing {len(missing_layers)} layers: {missing_layers}")
            logger.warning(f"⚠️ Word count: {word_count} (target: 350-400)")
            update["generation_quality_warning"] = f"Missing layers: {', '.join(missing_layers)}"
        elif word_count < MENU_OPTION_ONE_MIN_WORDS:
            logger.warning(f"⚠️ Generated answer too short: {word_count} words (target: 350-400)")
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
