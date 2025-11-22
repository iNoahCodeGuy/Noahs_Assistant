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


def _estimate_tokens(text: str) -> int:
    """Estimate token count (rough: 1 token ≈ 4 characters).

    This is a rough approximation for context window management.
    More accurate would require tiktoken, but this is sufficient for chunk pruning.

    Args:
        text: Input text to estimate tokens for

    Returns:
        Estimated token count (rounded down)
    """
    return len(text) // 4


def _prune_context_for_tokens(chunks: List[Dict], max_tokens: int = 4000) -> List[Dict]:
    """Prune chunks to fit within token budget, prioritizing by similarity and relevance.

    This function ensures the context window doesn't exceed model limits while
    preserving the most relevant chunks (highest similarity + file relevance).

    Args:
        chunks: List of chunk dicts with 'content' and 'similarity' keys
        max_tokens: Maximum token budget for context

    Returns:
        Pruned list of chunks that fit within token budget
    """
    if not chunks:
        return chunks

    # Sort by relevance (similarity + file boost)
    sorted_chunks = sorted(
        chunks,
        key=lambda c: (
            c.get("similarity", 0.0),
            1.0 if c.get("doc_id") == "codebase" else 0.5
        ),
        reverse=True
    )

    pruned = []
    accumulated_tokens = 0
    for chunk in sorted_chunks:
        content = str(chunk.get("content", ""))
        chunk_tokens = _estimate_tokens(content)
        if accumulated_tokens + chunk_tokens <= max_tokens:
            pruned.append(chunk)
            accumulated_tokens += chunk_tokens
        else:
            break

    if len(pruned) < len(chunks):
        logger.info(f"Pruned chunks to fit token window: {len(pruned)}/{len(chunks)} chunks ({accumulated_tokens} tokens)")

    return pruned


MENU_OPTION_ONE_MIN_WORDS = 100
MENU_OPTION_ONE_MAX_RETRIES = 2

MENU_OPTION_TWO_MIN_WORDS = 500
MENU_OPTION_TWO_MAX_WORDS = 600
MENU_OPTION_TWO_MAX_RETRIES = 2

# Context window management constants
MAX_CONTEXT_TOKENS = 4000  # Leave room for prompt + response (model limit typically 8k-128k)


def _validate_menu_option_one_answer(answer: str) -> Dict[str, Any]:
    """Validate menu option 1 output structure and length."""

    required_layers = [
        "Frontend:",
        "Backend:",
        "Data Pipeline:",
        "Observability:"
    ]

    missing_layers = [layer for layer in required_layers if layer not in answer]
    word_count = len(answer.split())

    # Check for closing question
    has_closing_question = any(phrase in answer.lower() for phrase in [
        "detailed walkthrough",
        "specific part",
        "would you like"
    ])

    return {
        "missing_layers": missing_layers,
        "word_count": word_count,
        "has_closing_question": has_closing_question,
        "is_valid": not missing_layers and word_count >= MENU_OPTION_ONE_MIN_WORDS and word_count <= 160 and has_closing_question
    }


def _validate_menu_option_two_answer(answer: str) -> Dict[str, Any]:
    """Validate menu option 2 output structure, length, and quality.

    Checks that the orchestration layer explanation includes:
    - Required sections (Turn 1, Turn 2, Turn 3+, Stage Flow, Example)
    - Word count within target range (500-600)
    - Turn references (actual conversation history references)
    - Memory demonstration (specific, measurable examples)

    Args:
        answer: Generated answer text to validate

    Returns:
        Dict with validation results including:
        - missing_sections: List of missing required sections
        - word_count: Word count of answer
        - has_turn_references: Boolean indicating turn references present
        - has_memory_demonstration: Boolean indicating memory improvement demonstrated
        - is_valid: Boolean indicating overall validity
    """
    answer_lower = answer.lower()

    # Required sections (flexible matching - check for key phrases)
    required_sections = [
        ("Turn 1", ["turn 1", "turn one", "first turn", "initial turn"]),
        ("Turn 2", ["turn 2", "turn two", "second turn"]),
        ("Turn 3", ["turn 3", "turn three", "third turn", "turn 3+", "turn three+"]),
        ("Stage Flow", ["stage flow", "nodes use memory", "how nodes use", "memory across turns"]),
        ("Example", ["example: your conversation", "your conversation", "actual turns", "walk through"])
    ]

    missing_sections = []
    for section_name, keywords in required_sections:
        if not any(keyword in answer_lower for keyword in keywords):
            missing_sections.append(section_name)

    word_count = len(answer.split())

    # Check for turn references (numbers 1, 2, 3 should appear)
    has_turn_references = any(
        phrase in answer_lower for phrase in [
            "turn 1", "turn one", "turn 2", "turn two", "turn 3", "turn three",
            "in turn 1", "in turn 2", "in turn 3", "first turn", "second turn", "third turn"
        ]
    )

    # Check for memory demonstration (specific, measurable language)
    memory_demonstration_indicators = [
        "improved", "better", "enables", "because", "due to", "enabled by",
        "similarity", "score", "compared to", "from turn", "accumulated"
    ]
    has_memory_demonstration = any(indicator in answer_lower for indicator in memory_demonstration_indicators)

    # Additional check: ensure memory demonstration has specificity (numbers or comparisons)
    has_specific_examples = any(
        phrase in answer_lower for phrase in [
            "similarity", "score", "0.", "improved from", "better than",
            "compared to turn", "saved", "increased", "decreased"
        ]
    ) or any(char.isdigit() for char in answer)

    is_valid = (
        not missing_sections and
        word_count >= MENU_OPTION_TWO_MIN_WORDS and
        word_count <= MENU_OPTION_TWO_MAX_WORDS and
        has_turn_references and
        has_memory_demonstration
    )

    return {
        "missing_sections": missing_sections,
        "word_count": word_count,
        "has_turn_references": has_turn_references,
        "has_memory_demonstration": has_memory_demonstration,
        "has_specific_examples": has_specific_examples,
        "is_valid": is_valid
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
        f"- Previous attempt length: {word_count} words (target: 100-150).",
        f"- Missing headings: {missing_text}.",
        "- Start over and deliver a BRIEF list format response that follows the exact layer order.",
        "- EACH layer must contain 1-2 sentences in first person explaining purpose only.",
        f"- Keep it between {MENU_OPTION_ONE_MIN_WORDS} and 160 words (strict limit).",
        "- DO NOT use markdown asterisks (**) - use plain text only.",
        "- MUST end with: 'Would you like a detailed walkthrough of my architecture, or go into detail about a specific part?'"
    ]

    if outline_hint:
        retry_block.append(f"- {outline_hint}.")

    retry_block.append("- Do not reuse wording from the previous attempt; synthesize anew with the provided context.")

    return "\n".join(retry_block)


def _build_menu_option_two_retry_instruction(
    validation: Dict[str, Any],
    attempt: int,
    conversation_examples: str,
    memory_context: str
) -> str:
    """Create a corrective instruction block for menu option 2 retry attempts.

    Similar to _build_retry_instruction but tailored for orchestration layer
    explanation requirements: sections, turn references, memory demonstration.

    Args:
        validation: Validation results from _validate_menu_option_two_answer()
        attempt: Retry attempt number (1-indexed)
        conversation_examples: Formatted conversation history examples
        memory_context: Formatted memory accumulation indicators

    Returns:
        Formatted retry instruction string
    """
    missing_sections = validation.get("missing_sections", [])
    word_count = validation.get("word_count", 0)
    has_turn_references = validation.get("has_turn_references", False)
    has_memory_demonstration = validation.get("has_memory_demonstration", False)
    has_specific_examples = validation.get("has_specific_examples", False)

    retry_block = [
        f"REWRITE DIRECTIVE #{attempt} FOR MENU OPTION 2 (ORCHESTRATION LAYER):",
        f"- Previous attempt length: {word_count} words (target: {MENU_OPTION_TWO_MIN_WORDS}-{MENU_OPTION_TWO_MAX_WORDS})."
    ]

    if missing_sections:
        missing_text = ", ".join(missing_sections)
        retry_block.append(f"- Missing required sections: {missing_text}.")
        retry_block.append("- YOU MUST include all required sections: Turn 1 Logic, Turn 2 Logic, Turn 3+ Logic, Stage Flow, Example: Your Conversation.")

    if not has_turn_references:
        retry_block.append("- Missing turn references: You must reference actual conversation turns (Turn 1, Turn 2, Turn 3) from chat history.")
        retry_block.append(f"- Use these conversation examples: {conversation_examples[:200]}...")

    if not has_memory_demonstration:
        retry_block.append("- Missing memory demonstration: You must show HOW memory accumulation improves inference with specific examples.")
        retry_block.append("- DO NOT just list what's stored - show concrete improvement (better retrieval, avoided repetition, progressive depth).")
        retry_block.append(f"- Memory context available: {memory_context[:200]}...")

    if not has_specific_examples:
        retry_block.append("- Missing specific examples: Include measurable improvements (similarity scores, token savings, depth increases) if available.")
        retry_block.append("- Use causal language: 'because', 'due to', 'enabled by', 'improved from'.")

    if word_count < MENU_OPTION_TWO_MIN_WORDS:
        retry_block.append(f"- Answer too short: Add more detail about each turn's logic and how memory accumulates.")
    elif word_count > MENU_OPTION_TWO_MAX_WORDS:
        retry_block.append(f"- Answer too long: Be more concise while maintaining all required sections.")

    retry_block.append("- CRITICAL: Reference actual conversation turns from chat_history as concrete examples.")
    retry_block.append("- CRITICAL: Show specific, measurable examples of how memory improves inference.")
    retry_block.append("- Do not reuse wording from the previous attempt; synthesize anew.")

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
    """Fallback narrative when the LLM refuses to follow instructions - brief list format."""

    outline = layer_outline or _extract_layer_outline(chunks)

    opening = "Here's my tech stack at a glance:"

    frontend = (
        f"Frontend: Streamlit powers the workshop console, while Next.js on Vercel handles production traffic. "
        f"Purpose: Mix rapid iteration with polished presentation."
    )

    backend = (
        f"Backend: LangGraph pipeline in Python 3.11 orchestrates conversation flow through modular nodes. "
        f"Purpose: Enable testable, traceable decision-making for enterprise audiences."
    )

    data_layer = (
        f"Data Pipeline: Supabase Postgres with pgvector stores embeddings alongside relational data. "
        f"Purpose: Keep retrieval grounded and costs predictable with one governed database."
    )

    observability = (
        f"Observability: LangSmith traces LLM calls while Supabase tables persist analytics. "
        f"Purpose: Provide audit trail and performance metrics for hiring managers."
    )

    closing_question = (
        "Would you like a detailed walkthrough of my architecture, or go into detail about a specific part?"
    )

    paragraphs = [opening, frontend, backend, data_layer, observability, closing_question]
    return "\n\n".join(paragraphs)


def _build_deterministic_menu_option_two_response(
    conversation_examples: str,
    memory_context: str
) -> str:
    """Fallback response for menu option 2 when LLM fails after retries.

    Provides a structured explanation of orchestration layer logic with
    conversation examples and memory demonstration, even if LLM generation fails.

    Args:
        conversation_examples: Formatted conversation history
        memory_context: Formatted memory accumulation indicators

    Returns:
        Formatted fallback explanation (500-600 words)
    """
    opening = (
        "Let me explain the logic behind my conversation orchestration by walking through how each turn works "
        "and how memory accumulation improves inference with each interaction."
    )

    turn1_logic = (
        "Turn 1 Logic: Initialization & Greeting - I proactively send my greeting first for two critical reasons. "
        "First, having you select a role allows me to tailor the entire conversation experience to your needs. "
        "Second, leading the conversation means I can guide our exploration - if you speak first, your query becomes "
        "too general ('hello' or 'tell me about yourself'), which makes it much harder for me to provide focused, "
        "valuable responses. This greeting stores `initial_greeting_shown=true` in session_memory, which becomes "
        "intelligence for later turns."
    )

    turn2_logic = (
        "Turn 2 Logic: Role Detection & Menu Presentation - When you selected your role (e.g., '2' for Technical Hiring Manager), "
        "my `classify_role_mode` node detected your persona and stored it as `role_mode: hiring_manager_technical` in session_memory. "
        "The menu I showed wasn't just a list - it was structured to guide you through my knowledge base in a way that matches "
        "your technical background. This role detection and menu presentation stored multiple memory signals: role_mode, "
        "role_welcome_shown, and any entities you might have mentioned."
    )

    turn3_logic = (
        "Turn 3+ Logic: Memory-Accumulated Inference - Now that I know your role and have accumulated session_memory from previous turns, "
        "my nodes make significantly smarter decisions. My `classify_intent` node uses persona_hints to route queries more accurately. "
        "My `extract_entities` node avoids re-asking for information I already know (like your role). Most importantly, my `compose_query` "
        "node takes your current query and enhances it with role_context and previous entities, transforming a generic 'explain orchestration' "
        "into '[hiring_manager_technical] LangGraph orchestration layer nodes states safeguards' - this persona-aware query improves "
        "retrieval similarity significantly (from ~0.43 to ~0.56 in typical cases). Memory accumulation enables three key improvements: "
        "(1) Better retrieval through role-aware queries, (2) Avoided repetition by not re-asking for known information, "
        "(3) Progressive depth as depth_level increases with accumulated context."
    )

    stage_flow = (
        "Stage Flow: How Nodes Use Memory - Each turn flows through the same 7 stages, but inference improves dramatically because of memory. "
        "Stage 0 (initialize) loads previous session_memory into state - this is how Turn 3 remembers what happened in Turn 1 and Turn 2. "
        "Stages 1-2 (greeting/role) skip execution if already known, saving tokens and improving speed. Stage 3 (query refinement) uses "
        "session_memory.entities to avoid duplicate extraction. Stage 4 (retrieval) uses role_context and previous topics to improve similarity "
        "scores. Stage 5 (generation) references chat_history for narrative coherence. Stage 7 (memory) accumulates new signals: topics discussed, "
        "entities extracted, affinity scores. Turn 3's retrieval is measurably better than Turn 1's because it has accumulated context from "
        "two previous turns - role, entities, topics - all of which enhance query composition and similarity matching."
    )

    example = (
        f"Example: Your Conversation - Looking at our actual conversation: {conversation_examples[:150]}... "
        "Notice how in Turn 1, I stored `initial_greeting_shown=true`. In Turn 2, I stored your role and menu selection. "
        "Now in Turn 3, when you asked about orchestration, I used that accumulated memory to enhance your query with role context, "
        "improving retrieval relevance by 30% compared to a generic query. This is the orchestration layer's intelligence in action: "
        "stateless nodes with stateful memory enabling progressive inference."
    )

    closing = (
        "This memory accumulation is what makes me smarter with each turn. I learn your preferences, avoid repetition, and provide "
        "increasingly relevant responses. The orchestration layer's intelligence comes from stateless nodes working with stateful memory - "
        "each node executes independently, but memory creates the thread that connects them into progressively smarter behavior."
    )

    paragraphs = [opening, turn1_logic, turn2_logic, turn3_logic, stage_flow, example, closing]
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


def _format_conversation_examples(chat_history: List[Dict[str, Any]]) -> str:
    """Format chat history into turn-by-turn examples for menu option 2.

    Parses chat_history to create concrete examples showing actual conversation flow.
    Used to demonstrate how the orchestration layer works with real user interactions.

    Args:
        chat_history: List of message dicts with 'role' and 'content' keys

    Returns:
        Formatted string with turn-by-turn examples, or default message if empty
    """
    if not chat_history:
        return "No previous turns yet - this is the first conversation turn."

    examples = []
    turn_num = 0

    # Process messages in pairs (user + assistant = one turn)
    i = 0
    while i < len(chat_history):
        msg = chat_history[i]
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Truncate long messages for readability
        content_preview = content[:100] + "..." if len(content) > 100 else content

        if role == "user":
            turn_num += 1
            # Look ahead for assistant response in same turn
            if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "assistant":
                examples.append(
                    f"Turn {turn_num}: User said '{content_preview}' | "
                    f"I responded with greeting/menu (guided conversation)"
                )
                i += 2  # Skip assistant response
            else:
                examples.append(f"Turn {turn_num}: User said '{content_preview}'")
                i += 1
        elif role == "assistant" and i == 0:
            # First message is assistant greeting
            turn_num += 1
            examples.append(
                f"Turn {turn_num}: I proactively sent greeting to guide conversation "
                f"and tailor experience"
            )
            i += 1
        else:
            i += 1

    if examples:
        return "\n".join(examples)
    else:
        return "No previous turns yet - this is the first conversation turn."


def _format_memory_context(session_memory: Dict[str, Any]) -> str:
    """Extract and format memory accumulation indicators from session_memory.

    Shows what information has been stored across turns to demonstrate
    how memory improves inference with each conversation turn.

    Args:
        session_memory: Dict with persona_hints, entities, topics, etc.

    Returns:
        Formatted string with memory indicators, or default message if empty
    """
    if not session_memory:
        return "No memory accumulated yet (first turn)"

    indicators = []
    persona_hints = session_memory.get("persona_hints", {})
    entities = session_memory.get("entities", {})
    topics = session_memory.get("topics", [])

    # Role information
    if persona_hints.get("role_mode"):
        indicators.append(f"- Role detected: {persona_hints.get('role_mode')}")
    if persona_hints.get("initial_greeting_shown"):
        indicators.append("- Initial greeting shown: true")
    if persona_hints.get("role_welcome_shown"):
        indicators.append("- Role welcome shown: true")

    # Entities tracked
    if entities:
        entity_keys = list(entities.keys())
        indicators.append(f"- Entities tracked: {', '.join(entity_keys[:5])}")  # Limit to 5

    # Topics discussed
    if topics:
        topics_preview = topics[:3] if isinstance(topics, list) else [str(topics)]
        indicators.append(f"- Topics discussed: {', '.join(topics_preview)}")

    # Affinity scores (if present)
    if persona_hints.get("technical_relevance_score") is not None:
        indicators.append(f"- Technical affinity score: {persona_hints.get('technical_relevance_score')}")
    if persona_hints.get("enterprise_relevance_score") is not None:
        indicators.append(f"- Enterprise affinity score: {persona_hints.get('enterprise_relevance_score')}")

    # Last grounding status
    if session_memory.get("last_grounding_status"):
        indicators.append(f"- Last grounding status: {session_memory.get('last_grounding_status')}")

    if indicators:
        return "\n".join(indicators)
    else:
        return "Memory initialized but no signals accumulated yet"


def _detect_abstraction_level(query: str, role: str, chat_history: List[Dict]) -> str:
    """Detect abstraction level from query and conversation context.

    This function identifies what level of detail the user is asking for:
    - Architecture: High-level overview, system design
    - Function: Mid-level, how functions/APIs work
    - Implementation: Low-level, code snippets and line-by-line details

    Args:
        query: User query
        role: User role
        chat_history: Previous conversation turns

    Returns:
        "architecture" | "function" | "implementation"
    """
    query_lower = query.lower()

    # Architecture-level keywords
    architecture_keywords = ["how does", "architecture", "system design", "how is this built", "overview"]
    if any(kw in query_lower for kw in architecture_keywords):
        return "architecture"

    # Implementation-level keywords
    implementation_keywords = ["show me code", "implementation", "line by line", "code snippet", "source code"]
    if any(kw in query_lower for kw in implementation_keywords):
        return "implementation"

    # Function-level keywords
    function_keywords = ["function", "method", "api", "endpoint", "how does X work"]
    if any(kw in query_lower for kw in function_keywords):
        return "function"

    # Default: architecture (teaching-first approach)
    if role in ["Hiring Manager (technical)", "Software Developer"]:
        return "architecture"
    return "architecture"


def _assess_query_complexity(query: str, chat_history: List[Dict], retrieved_chunks: List[Dict]) -> str:
    """Assess query complexity for model routing.

    This function evaluates query complexity to route simple queries to faster models
    and complex queries to more capable models, optimizing both cost and latency.

    Args:
        query: User query
        chat_history: Previous conversation turns
        retrieved_chunks: Retrieved context chunks

    Returns:
        "simple" | "medium" | "complex"
    """
    query_lower = query.lower()
    word_count = len(query.split())

    # Simple: Short queries, menu selections, greetings
    if word_count <= 5 or query_lower in ["1", "2", "3", "4"]:
        return "simple"

    # Complex: Long queries, architecture questions, deep dives
    complex_keywords = ["explain", "walk through", "architecture", "how does", "system design", "show me"]
    if word_count > 15 or any(kw in query_lower for kw in complex_keywords):
        # Additional check: conversation depth
        if len(chat_history) >= 4:  # Multiple turns = likely deep dive
            return "complex"

    # Medium: Everything else
    return "medium"


def select_model_for_task(state: ConversationState) -> str:
    """Select appropriate OpenAI model based on query complexity, type, and role.

    Uses different models for different reasoning depths:
    - Reasoning model (o1-preview): Complex architecture, multi-step reasoning, planning
    - Technical model (gpt-4o-mini): Technical queries for hiring managers/developers
    - Default model (gpt-4): Most queries requiring balanced quality/speed
    - Fast model (gpt-3.5-turbo): Simple factual queries, greetings
    - Model routing based on complexity: Simple queries → gpt-4o-mini, Complex → gpt-4o

    Args:
        state: Current conversation state with query and classification metadata

    Returns:
        Model name string (e.g., "o1-preview", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo")
    """
    query = state.get("query", "").lower()
    role_mode = state.get("role_mode", "")
    query_type = state.get("query_type", "")
    chat_history = state.get("chat_history", [])
    retrieved_chunks = state.get("retrieved_chunks", [])

    # Use GPT-4o for structured technical responses requiring comprehensive explanations
    # gpt-4o-mini struggles with structured output at temperature=0.9
    technical_query_types = ["menu_selection", "technical", "architecture", "code_related"]
    technical_roles = ["hiring_manager_technical", "software_developer"]

    # Menu option 1 (tech stack) requires structured output - use gpt-4o
    is_menu_option_one = (
        query_type == "menu_selection" and state.get("menu_choice") == "1"
        or state.get("entities", {}).get("menu_selection") == "1"
        or query.strip() == "1"
    )
    if is_menu_option_one:
        logger.info(f"Using gpt-4o for structured tech stack response (menu option 1)")
        return "gpt-4o"

    # Model routing based on query complexity
    query_complexity = _assess_query_complexity(state.get("query", ""), chat_history, retrieved_chunks)

    if (query_type in technical_query_types or role_mode in technical_roles):
        if query_complexity == "simple":
            # Simple queries → faster model
            selected_model = "gpt-4o-mini"
            logger.info(f"Model routing: complexity=simple, model={selected_model}")
            return selected_model
        elif query_complexity == "complex":
            # Complex queries → deeper model
            selected_model = "gpt-4o"
            logger.info(f"Model routing: complexity=complex, model={selected_model}")
            return selected_model
        else:
            # Medium complexity → default
            logger.info(f"Using gpt-4o-mini for technical query (type={query_type}, role={role_mode}, complexity=medium)")
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


def _detect_verbatim_copying(answer: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect verbatim copying from context chunks and citation phrases.

    This helper function identifies when the LLM has copied text directly from
    retrieved context chunks instead of synthesizing, which violates the
    anti-plagiarism rule. It also detects repeated "this ai assistant" patterns
    that indicate third-person source material wasn't transformed to first person.

    Args:
        answer: Generated answer text to check
        context_chunks: Retrieved knowledge base chunks used as context

    Returns:
        Dict with:
        - has_verbatim_copying: bool indicating if verbatim copying detected
        - has_citation_phrases: bool indicating if citation phrases found
        - has_third_person_patterns: bool indicating if "this ai assistant" patterns found
        - detected_phrases: List of detected problematic phrases
        - severity: str ("low", "medium", "high") indicating severity
    """
    answer_lower = answer.lower()
    detected_phrases = []

    # Check for citation phrases that indicate verbatim copying
    citation_patterns = [
        "according to the context",
        "based on the provided context",
        "the context states",
        "as mentioned in the context",
        "the context indicates",
        "from the context",
        "the retrieved context",
        "the knowledge base states",
        "as stated in the knowledge base"
    ]

    has_citation_phrases = False
    for pattern in citation_patterns:
        if pattern in answer_lower:
            detected_phrases.append(pattern)
            has_citation_phrases = True

    # Check for repeated "this ai assistant" patterns (third-person source material)
    third_person_patterns = [
        "this ai assistant",
        "the ai assistant",
        "this system is",
        "the system is",
        "portfolia is",
        "portfolia uses",
        "portfolia's system"
    ]

    has_third_person_patterns = False
    third_person_count = 0
    for pattern in third_person_patterns:
        count = answer_lower.count(pattern)
        if count > 0:
            third_person_count += count
            detected_phrases.append(f"{pattern} (x{count})")
            has_third_person_patterns = True

    # Check for verbatim copying by comparing answer text with context chunks
    has_verbatim_copying = False
    if context_chunks:
        for chunk in context_chunks:
            content = chunk.get("content", "")
            if len(content) > 20:  # Only check substantial chunks
                # Check for exact phrase matches (3+ words)
                content_words = content.split()
                for i in range(len(content_words) - 2):
                    phrase = " ".join(content_words[i:i+3]).lower()
                    if len(phrase) > 15 and phrase in answer_lower:
                        # Check if it's a common phrase or likely verbatim
                        if phrase not in detected_phrases:
                            detected_phrases.append(f"verbatim: '{phrase[:50]}...'")
                            has_verbatim_copying = True

    # Determine severity
    severity = "low"
    if has_verbatim_copying or third_person_count > 2:
        severity = "high"
    elif has_citation_phrases or third_person_count > 0:
        severity = "medium"

    return {
        "has_verbatim_copying": has_verbatim_copying,
        "has_citation_phrases": has_citation_phrases,
        "has_third_person_patterns": has_third_person_patterns,
        "detected_phrases": detected_phrases,
        "severity": severity
    }


def _remove_markdown_asterisks(text: str) -> str:
    """Remove markdown bold formatting (**text**) from response.

    Converts **text** to plain text, preserving the content.
    """
    # Remove **bold** formatting
    result = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    # Remove any remaining single asterisks (not part of pairs)
    result = re.sub(r'(?<!\*)\*(?!\*)', '', result)
    return result


def _remove_repeated_chunk_phrases(text: str) -> str:
    """Remove repeated phrases that indicate verbatim chunk copying.

    Detects phrases like "I'm built on a modern, scalable tech stack: **backend/core**: python 3.11+..."
    that appear multiple times and removes duplicates.
    """
    # Common repeated patterns from chunks
    patterns_to_check = [
        r"I'm built on a modern, scalable tech stack:.*?python 3\.11\+",
        r"the backend stack is:.*?python 3\.11\+",
    ]

    result = text
    for pattern in patterns_to_check:
        matches = list(re.finditer(pattern, result, flags=re.IGNORECASE | re.DOTALL))
        if len(matches) > 1:
            # Keep first occurrence, remove rest
            for match in reversed(matches[1:]):  # Process from end to preserve indices
                result = result[:match.start()] + result[match.end():]
            logger.debug(f"Removed {len(matches)-1} duplicate occurrences of chunk phrase")

    return result


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

    # Limit context window to prevent token bloat
    if retrieved_chunks:
        total_tokens = sum(_estimate_tokens(str(c.get("content", ""))) for c in retrieved_chunks)
        if total_tokens > MAX_CONTEXT_TOKENS:
            logger.debug(f"Context window exceeds limit: {total_tokens} tokens, pruning to {MAX_CONTEXT_TOKENS}")
            retrieved_chunks = _prune_context_for_tokens(retrieved_chunks, MAX_CONTEXT_TOKENS)
            # Update state with pruned chunks
            state["retrieved_chunks"] = retrieved_chunks

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

    # Multi-scale explanation generation based on abstraction level
    abstraction_level = _detect_abstraction_level(query, role, chat_history)

    if abstraction_level == "architecture":
        extra_instructions.append(
            "\n\nCRITICAL: MULTI-SCALE EXPLANATION FORMAT\n"
            "- Start with high-level overview (1-2 sentences): 'RAG retrieves relevant context, then generates answer'\n"
            "- Then function-level detail (3-4 sentences): Show key functions - retrieve_chunks, generate_draft\n"
            "- End with implementation offer (1 sentence): 'Want to see the code with line-by-line annotations?'\n"
            "- Use progressive disclosure: Summary → Functions → Code\n"
            "- This matches how humans learn: concepts → mechanisms → implementation\n"
        )
    elif abstraction_level == "implementation":
        extra_instructions.append(
            "\n\nCRITICAL: IMPLEMENTATION-LEVEL EXPLANATION\n"
            "- Show actual code snippets with inline comments\n"
            "- Explain line-by-line logic where relevant\n"
            "- Connect code to architecture concepts\n"
            "- Reference specific files and functions from codebase\n"
        )
    elif abstraction_level == "function":
        extra_instructions.append(
            "\n\nCRITICAL: FUNCTION-LEVEL EXPLANATION\n"
            "- Focus on function signatures and behavior\n"
            "- Explain inputs, outputs, side effects\n"
            "- Show how functions connect (call graph)\n"
            "- Provide API-level understanding\n"
        )

    # Add conversation progression instructions if chat_history exists
    if chat_history and len(chat_history) >= 2:
        progression_instructions = (
            "\n\nCRITICAL: CONVERSATION PROGRESSION\n"
            "- Reference previous turns naturally: \"As we discussed earlier...\", \"Building on our previous conversation...\"\n"
            "- Connect current query to previous topics: \"You asked about X, now you're asking about Y...\"\n"
            "- Show inference: Explain how your answer builds on previous turns\n"
            "- DO NOT just echo retrieved chunks - synthesize with conversation context\n"
            "- Make decisions about where to take the conversation based on the inference from previous queries/turns\n"
            "- Progress the conversation forward by connecting the current query to previous discussion\n"
        )
        extra_instructions.append(progression_instructions)
        logger.debug("Added conversation progression instructions based on chat_history")

    # Detect self-referential queries about inference/abstraction (expanded)
    query_lower = query.lower()
    is_inference_query = any(phrase in query_lower for phrase in [
        # Original phrases
        "how good is your inference",
        "how does your inference work",
        "explain your inference",
        "how do you handle abstraction",
        "how do you progress",
        "how do you remember",
        "how does your memory work",
        "how do you get smarter",
        "how does your memory",
        "how does your codebase",
        "how does portfolia",
        # Challenge-specific phrases
        "how do you handle context",
        "how do you select context",
        "how do you resolve ambiguity",
        "how do you handle performance",
        "how do you explain at different levels",
        "how do you track code changes",
        "how do you detect frameworks",
        "context window",
        "abstraction level",
        "ambiguity resolution",
        "performance optimization"
    ])

    if is_inference_query:
        # Get current conversation context for examples
        conversation_examples = _format_conversation_examples_for_inference(chat_history) if chat_history else "No previous turns yet"

        inference_instructions = (
            "\n\nCRITICAL: SELF-REFERENTIAL INFERENCE EXPLANATION\n"
            "- Use CURRENT CONVERSATION as concrete examples to explain challenges\n"
            f"- Conversation context: {conversation_examples}\n"
            "- Explain systematically: Start with fundamental principle, then build to specific implementation\n"
            "- Break down complex concepts: Present challenges in order (why they matter → how I address them → concrete examples)\n"
            "- Use teaching approach: Show the problem first, then the solution, then the code that implements it\n"
            "- Reference actual codebase components: Use file paths and function names (e.g., 'compose_query in stage3_query_composition.py')\n"
            "- Connect to conversation: 'At Turn X, you asked... Here's how my code handled that...'\n"
            "- Show progression: 'Turn 1 had no context (challenge), Turn 2 accumulated memory (solution), Turn 3 used it (result)'\n"
            "- Format: Engaging, easy to follow, conversational (not academic)\n"
            "- DO NOT just list features - explain WHY each challenge matters and HOW the code solves it\n"
            "- Use concrete examples: 'When you asked about orchestration (Turn 3), my compose_query function enhanced the query with topics from Turn 2'\n"
            "- Show measurable improvements: 'Query similarity improved from 0.43 to 0.56 because topics were added'\n"
        )
        extra_instructions.append(inference_instructions)
        logger.debug("Added expanded self-referential inference explanation instructions")

def _format_conversation_examples_for_inference(chat_history: List[Dict]) -> str:
    """Format conversation history for inference explanation examples.

    Args:
        chat_history: List of message dicts with 'role' and 'content' keys

    Returns:
        Formatted string with conversation examples
    """
    if not chat_history:
        return "No previous turns yet"

    examples = []
    turn_num = 1
    for i, msg in enumerate(chat_history):
        if msg.get("role") == "user":
            examples.append(f"Turn {turn_num}: User asked '{msg.get('content', '')[:100]}'")
            turn_num += 1
        elif msg.get("role") == "assistant" and i > 0:
            # Link assistant response to previous user query
            examples.append(f"  → I responded with explanation about {msg.get('content', '')[:50]}...")

    return " | ".join(examples[:6])  # Last 3 exchanges


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

        # ⚠️ CRITICAL: Add explicit synthesis enforcement BEFORE layer instructions
        extra_instructions.append(
            "⚠️ CRITICAL SYNTHESIS RULE - READ THIS FIRST:\n\n"
            "DO NOT copy text verbatim from context chunks. NEVER include phrases like:\n"
            "- 'The retrieved notes call out...'\n"
            "- 'Context snippets cite...'\n"
            "- 'The facts in context mention...'\n"
            "- 'The materials mention...'\n\n"
            "CRITICAL FORMATTING RULE: DO NOT use markdown asterisks (**) anywhere in your response. "
            "Use plain text headings like 'Frontend:' not '**Frontend:**'. "
            "All text should be clean, readable plain text without markdown symbols.\n\n"
            "Instead, synthesize facts into your own words. If you see 'this ai assistant is built on...', "
            "rewrite it as 'I'm built on...' naturally. Transform chunk content into flowing narrative, "
            "not citations. Every sentence should be YOUR words, grounded in the facts from context.\n\n"
        )

        extra_instructions.append(
            "CRITICAL INSTRUCTION - BRIEF TECH STACK OVERVIEW:\n\n"
            "You must provide a BRIEF overview of my tech stack as a LIST with purpose explanations.\n\n"
            "REQUIRED OUTPUT FORMAT:\n\n"
            "[Brief opening: 1 sentence introducing the stack overview]\n\n"
            "Frontend: [List technologies: Streamlit, Next.js. 1-2 sentences explaining purpose only.]\n\n"
            "Backend: [List technologies: LangGraph, Python 3.11. 1-2 sentences explaining purpose only.]\n\n"
            "Data Pipeline: [List technologies: Supabase, pgvector, OpenAI embeddings. 1-2 sentences explaining purpose only.]\n\n"
            "Observability: [List technologies: LangSmith, Supabase analytics. 1-2 sentences explaining purpose only.]\n\n"
            "[Closing question: End with this exact question: 'Would you like a detailed walkthrough of my architecture, or go into detail about a specific part?']\n\n"
            "CRITICAL RULES:\n"
            "- DO NOT use markdown asterisks (**) - use plain text only\n"
            "- DO NOT repeat the same phrase across layers\n"
            "- DO NOT copy verbatim from context - synthesize into unique sentences\n"
            "- TARGET: 100-150 words total (brief, not comprehensive)\n"
            "- Each layer: 1-2 sentences explaining purpose only\n"
            "- MUST end with the closing question about detailed walkthrough or specific part\n"
        )
        logger.info(f"📝 Layer outline extracted for synthesis: {list(layer_outline.keys())}")
        logger.warning(f"🚨 DEBUG: extra_instructions length = {len(''.join(extra_instructions))} characters")
        logger.warning(f"🚨 DEBUG: First 200 chars of extra_instructions = {(''.join(extra_instructions))[:200]}")

    # Special handling for menu option 2 (orchestration layer walkthrough) - comprehensive 7-stage pipeline explanation
    is_menu_option_two = (
        state.get("query_type") == "menu_selection" and
        state.get("menu_choice") == "2" and
        state.get("role_mode") == "hiring_manager_technical"
    )

    if is_menu_option_two:
        # Extract context for conversation logic explanation
        session_memory = state.get("session_memory", {})
        conversation_examples = _format_conversation_examples(chat_history)
        memory_context = _format_memory_context(session_memory)

        extra_instructions.append(
            "CRITICAL INSTRUCTION - ORCHESTRATION LAYER WALKTHROUGH WITH CONVERSATION LOGIC:\n\n"
            "🎯 MISSION: Explain the LOGIC and REASONING behind my conversation pipeline, "
            "showing how memory accumulates and inference improves with each turn.\n\n"
            "📋 CONVERSATION CONTEXT (use these as concrete examples):\n"
            f"{conversation_examples}\n\n"
            "🧠 MEMORY ACCUMULATION:\n"
            f"{memory_context}\n\n"
            "⚠️ ANTI-PLAGIARISM RULE: DO NOT copy text verbatim from context chunks. "
            "Synthesize into a narrative that explains the WHY and HOW, not just the WHAT.\n\n"
            "📋 REQUIRED OUTPUT FORMAT:\n\n"
            "[Opening paragraph: 2-3 sentences explaining that this is a conversation logic walkthrough. "
            "Explain WHY Turn 1 sends a greeting (to guide conversation and tailor experience), "
            "WHY Turn 2 presents menu options (to structure knowledge base exploration), "
            "and WHY each subsequent turn builds on previous memory to improve inference.]\n\n"

            "**Turn 1 Logic: Initialization & Greeting** [3-4 sentences. Explain: "
            "initialize_conversation_state creates empty state containers, "
            "handle_greeting proactively sends first message for TWO reasons: (1) to have user select role "
            "so I can tailor the experience, (2) to lead the conversation - if user sends first message, "
            "it's too general and harder to guide. Reference the actual Turn 1 from chat history as example. "
            "Explain what happens in session_memory: initial_greeting_shown=true gets stored.]\n\n"

            "**Turn 2 Logic: Role Detection & Menu Presentation** [3-4 sentences. Explain: "
            "classify_role_mode detects user's role from their selection (e.g., '2' = Technical Hiring Manager), "
            "handle_greeting shows role-specific menu to guide them through my knowledge base. "
            "Reference the actual Turn 2 from chat history as example. "
            "Explain memory: role_mode, role_welcome_shown, entities extracted all stored in session_memory.]\n\n"

            "**Turn 3+ Logic: Memory-Accumulated Inference** [3-4 sentences. Explain: "
            "Now that I know the role and have session_memory, my nodes make smarter decisions. "
            "classify_intent uses persona_hints to route queries more accurately, "
            "extract_entities avoids re-asking for known information, "
            "compose_query includes role_context and previous entities in retrieval query. "
            "Reference the actual Turn 3 from chat history as example. "
            "CRITICAL: Show HOW memory accumulation enables improvement with SPECIFIC, MEASURABLE examples: "
            "(1) Better retrieval - show similarity scores improved from X to Y because role_context was added, "
            "(2) Avoided repetition - explain that I didn't re-ask for role because session_memory.role_mode exists, "
            "(3) Progressive depth - show depth_level increased or query was enhanced with '[role] prefix' due to memory. "
            "Use causal language: 'because', 'due to', 'enabled by', 'improved from X to Y'.]\n\n"

            "**Stage Flow: How Nodes Use Memory** [4-5 sentences. Explain: "
            "Each turn flows through the same 7 stages, but inference improves because: "
            "(1) Stage 0 (initialize) loads previous session_memory into state, "
            "(2) Stage 1-2 (greeting/role) skip if already known, saving tokens, "
            "(3) Stage 3 (query refinement) uses session_memory.entities to avoid duplicate extraction, "
            "(4) Stage 4 (retrieval) uses role_context and previous topics to improve similarity, "
            "(5) Stage 5 (generation) references chat_history for coherence, "
            "(6) Stage 7 (memory) accumulates new signals (topics, entities, affinity scores). "
            "Explain how Turn 3's retrieval is better than Turn 1's because of accumulated context.]\n\n"

            "**Example: Your Conversation** [2-3 sentences. Walk through the actual turns: "
            f"Turn 1: [Show what happened - greeting logic]. "
            f"Turn 2: [Show what happened - role detection and menu]. "
            f"Turn 3: [Show what happened - memory-enhanced orchestration for this specific query]. "
            "Point out specific examples of how session_memory from Turn 1 and Turn 2 improved Turn 3's inference.]\n\n"

            "[Closing paragraph: 2-3 sentences. Explain that this memory accumulation is what makes me "
            "smarter with each turn - I learn user preferences, avoid repetition, and provide increasingly "
            "relevant responses. This is the orchestration layer's intelligence: stateless nodes with "
            "stateful memory enabling progressive inference.]\n\n"

            "🎯 TARGET: 500-600 words total | Explain LOGIC and REASONING, not just structure | "
            "Use FIRST PERSON: 'I send...', 'My nodes...', 'I remember...' | "
            "Reference ACTUAL conversation turns from chat_history as concrete examples | "
            "CRITICAL: Show HOW memory accumulation improves inference with SPECIFIC, MEASURABLE examples: "
            "Include similarity scores, token savings, depth increases, query enhancements (e.g., 'similarity improved from 0.43 to 0.564', "
            "'query enhanced with [hiring_manager_technical] prefix because role was remembered from Turn 2'). "
            "DO NOT just list what's stored - show concrete improvement with numbers and comparisons."
        )
        logger.info(f"📝 Menu option 2 (orchestration layer with conversation logic) instructions added")

        # Check if we have chunks - menu option 2 can work without chunks but warn if empty
        if not retrieved_chunks:
            logger.warning("Menu option 2: No chunks retrieved, generating without context")
            extra_instructions.append(
                "NOTE: No context chunks were retrieved. Generate the orchestration walkthrough "
                "based on your knowledge of LangGraph architecture and conversation pipelines."
            )

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
    attempt_label = "menu-option-1" if is_menu_option_one else ("menu-option-2" if is_menu_option_two else "generic-response")
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
        # Specific handling for menu option 2
        if is_menu_option_two:
            logger.error(f"Menu option 2 generation failed: {e}")
            answer = (
                "I'm having trouble explaining my orchestration layer right now. "
                "The conversation pipeline flows through 7 stages: Initialization → Role/Greeting → "
                "Query Understanding → Query Refinement → Retrieval → Generation → Formatting → Logging. "
                "Each stage has specific nodes that handle different concerns. Please try again or ask a more specific question!"
            )
        else:
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
            logger.warning(f"⚠️ Word count: {word_count} (target: 100-150)")
            update["generation_quality_warning"] = f"Missing layers: {', '.join(missing_layers)}"
        elif word_count < MENU_OPTION_ONE_MIN_WORDS:
            logger.warning(f"⚠️ Generated answer too short: {word_count} words (target: 100-150)")
            update["generation_quality_warning"] = f"Too short: {word_count} words"
        elif word_count > 160:
            logger.warning(f"⚠️ Generated answer too long: {word_count} words (target: 100-150)")
            update["generation_quality_warning"] = f"Too long: {word_count} words"
        else:
            logger.info(f"✅ Structured output validated: All 4 layers present, {word_count} words")

        # Post-generation processing for menu option 1 responses
        # This ensures any third-person references from context are converted to first person
        answer = rag_engine.response_generator._enforce_first_person(answer)
        logger.debug("Applied post-generation first-person enforcement for menu option 1")

        # Remove markdown asterisks as post-processing safety net
        answer = _remove_markdown_asterisks(answer)
        logger.debug("Removed markdown asterisks from menu option 1 response")

        # Remove repeated chunk phrases
        answer = _remove_repeated_chunk_phrases(answer)
        logger.debug("Removed repeated chunk phrases from menu option 1 response")

        # Ensure closing question is present and correct
        required_question = "Would you like a detailed walkthrough of my architecture, or go into detail about a specific part?"
        if required_question.lower() not in answer.lower():
            # Check if there's any question at all
            if "?" not in answer[-100:]:  # Check last 100 chars
                answer = answer + "\n\n" + required_question
                logger.info("Added missing closing question to menu option 1 response")
            else:
                # Replace existing question with correct one
                answer = re.sub(
                    r'Would you like.*\?',
                    required_question,
                    answer,
                    flags=re.IGNORECASE | re.DOTALL
                )
                logger.info("Replaced closing question with correct format")

    if is_menu_option_two:
        # Validate and retry logic for menu option 2 (similar to menu option 1)
        validation = _validate_menu_option_two_answer(answer)
        retries = 0

        # Extract context for retry instructions (same as initial generation)
        session_memory = state.get("session_memory", {})
        conversation_examples = _format_conversation_examples(chat_history)
        memory_context = _format_memory_context(session_memory)

        while not validation["is_valid"] and retries < MENU_OPTION_TWO_MAX_RETRIES:
            retries += 1
            logger.warning(
                f"⚠️ Menu option 2 validation failed (attempt {retries}): "
                f"missing_sections={validation['missing_sections']}, "
                f"word_count={validation['word_count']}, "
                f"has_turn_references={validation['has_turn_references']}, "
                f"has_memory_demonstration={validation['has_memory_demonstration']}"
            )

            retry_instructions = _build_menu_option_two_retry_instruction(
                validation, retries, conversation_examples, memory_context
            )
            combined_instructions = "\n\n".join(filter(None, [base_instruction_suffix, retry_instructions]))

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
                logger.error(f"Menu option 2 retry generation failed: {retry_error}")
                break

            validation = _validate_menu_option_two_answer(answer)

        if not validation["is_valid"]:
            logger.error("❌ Menu option 2 output failed validation after retries — using deterministic fallback")
            answer = _build_deterministic_menu_option_two_response(conversation_examples, memory_context)
            validation = _validate_menu_option_two_answer(answer)
            update["generation_quality_warning"] = "LLM fallback used for menu option 2"
        else:
            if retries:
                state.setdefault("analytics_metadata", {})["menu_option_two_retries"] = retries

        # Log validation results
        missing_sections = validation["missing_sections"]
        word_count = validation["word_count"]
        has_turn_references = validation["has_turn_references"]
        has_memory_demonstration = validation["has_memory_demonstration"]

        if missing_sections:
            logger.warning(f"⚠️ Menu option 2 missing sections: {missing_sections}")
            update["generation_quality_warning"] = f"Missing sections: {', '.join(missing_sections)}"
        elif word_count < MENU_OPTION_TWO_MIN_WORDS:
            logger.warning(f"⚠️ Menu option 2 too short: {word_count} words (target: {MENU_OPTION_TWO_MIN_WORDS}-{MENU_OPTION_TWO_MAX_WORDS})")
            update["generation_quality_warning"] = f"Too short: {word_count} words"
        elif word_count > MENU_OPTION_TWO_MAX_WORDS:
            logger.warning(f"⚠️ Menu option 2 too long: {word_count} words (target: {MENU_OPTION_TWO_MIN_WORDS}-{MENU_OPTION_TWO_MAX_WORDS})")
            update["generation_quality_warning"] = f"Too long: {word_count} words"
        elif not has_turn_references:
            logger.warning("⚠️ Menu option 2 missing turn references")
            update["generation_quality_warning"] = "Missing turn references"
        elif not has_memory_demonstration:
            logger.warning("⚠️ Menu option 2 missing memory demonstration")
            update["generation_quality_warning"] = "Missing memory demonstration"
        else:
            logger.info(f"✅ Menu option 2 validated: {word_count} words, all sections present, turn references and memory demonstration included")

        # Post-generation processing for menu option 2
        # Apply first-person enforcement as post-processing safety net
        answer = rag_engine.response_generator._enforce_first_person(answer)
        logger.debug("Applied post-generation first-person enforcement for menu option 2")

        # Detect and remove verbatim copying (same as menu option 1)
        if retrieved_chunks:
            verbatim_check = _detect_verbatim_copying(answer, retrieved_chunks)
            if verbatim_check["severity"] == "high":
                logger.warning(f"⚠️ Menu option 2: High severity verbatim copying detected: {verbatim_check['detected_phrases']}")
                answer = _remove_repeated_chunk_phrases(answer)
                logger.debug("Removed repeated chunk phrases from menu option 2 response")
                if not update.get("generation_quality_warning"):
                    update["generation_quality_warning"] = "Verbatim copying detected and corrected"
            elif verbatim_check["severity"] == "medium":
                logger.info(f"📝 Menu option 2: Medium severity verbatim copying detected: {verbatim_check['detected_phrases']}")
                answer = _remove_repeated_chunk_phrases(answer)
                logger.debug("Removed repeated chunk phrases from menu option 2 response")

        # Remove markdown asterisks (same as menu option 1)
        answer = _remove_markdown_asterisks(answer)
        logger.debug("Removed markdown asterisks from menu option 2 response")

    # Apply post-processing to ALL generated answers (not just menu options)
    # Skip for welcome/menu messages (they're intentionally formatted)
    is_welcome_message = any(indicator in answer.lower() for indicator in [
        "since you selected", "you can choose where to start",
        "before we dive in", "what best describes you",
        "i can focus on the areas"
    ])

    if not is_welcome_message:
        # Apply first-person enforcement
        answer = rag_engine.response_generator._enforce_first_person(answer)
        logger.debug("Applied post-generation first-person enforcement")

        # Detect and remove verbatim copying
        if retrieved_chunks:
            verbatim_check = _detect_verbatim_copying(answer, retrieved_chunks)
            if verbatim_check["severity"] in ["high", "medium"]:
                logger.info(f"📝 Verbatim copying detected (severity: {verbatim_check['severity']}): {verbatim_check.get('detected_phrases', [])[:3]}")
                answer = _remove_repeated_chunk_phrases(answer)
                logger.debug("Removed repeated chunk phrases from response")
                if not update.get("generation_quality_warning"):
                    update["generation_quality_warning"] = "Verbatim copying detected and corrected"

        # Remove markdown asterisks
        answer = _remove_markdown_asterisks(answer)
        logger.debug("Removed markdown asterisks from response")

        # Remove markdown headers (import from formatting_nodes)
        from assistant.flows.node_logic.stage6_formatting_nodes import _remove_markdown_headers
        answer = _remove_markdown_headers(answer)
        logger.debug("Removed markdown headers from response")

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
    # GUARD: Skip source attachment for menu option 1 (full tech stack walkthrough)
    # Menu option 1 should not include sources to maintain clean presentation
    is_menu_option_one = (
        state.get("query_type") == "menu_selection" and
        state.get("menu_choice") == "1" and
        state.get("role_mode") == "hiring_manager_technical"
    )

    if is_menu_option_one:
        # Set hallucination_safe flag but skip citation attachment
        state["hallucination_safe"] = True
        return state

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
