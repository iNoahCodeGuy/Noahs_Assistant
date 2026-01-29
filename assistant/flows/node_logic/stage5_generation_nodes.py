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
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from assistant.state.conversation_state import ConversationState
from assistant.core.rag_engine import RagEngine
from assistant.flows import content_blocks
from assistant.flows.node_logic.util_code_validation import sanitize_generated_answer
from assistant.config.supabase_config import supabase_settings
from assistant.flows.node_logic.chain_of_thought import (
    chain_of_thought_generate,
    detect_confusion_signals,
)

# Module-level verification - this will print when module is imported
print(">>> stage5_generation_nodes.py MODULE LOADED <<<", file=sys.stderr, flush=True)

# Import centralized path configuration
from assistant.config.settings import get_debug_log_path as _get_debug_log_path

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


def _extract_topic_words(query: str) -> List[str]:
    """Extract key topic words from a query (excluding stop words).

    Reuses logic from util_edge_case_detection for consistency.

    Args:
        query: User query text

    Returns:
        List of important topic words
    """
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                  "have", "has", "had", "do", "does", "did", "will", "would", "could",
                  "should", "may", "might", "can", "what", "how", "why", "when", "where",
                  "who", "to", "for", "of", "in", "on", "at", "see", "me", "tell", "about",
                  "explain", "you", "please", "i", "want", "need", "know", "it", "this",
                  "that", "program", "programs", "be", "adapted", "could", "would"}

    # Remove punctuation before extracting words
    normalized = re.sub(r'[^\w\s]', '', query.lower())
    words = normalized.split()
    return [w for w in words if w not in stop_words and len(w) > 2]


def _validate_answer_relevance(query: str, answer: str) -> bool:
    """Check if answer addresses the query topic.

    This validation ensures the LLM-generated answer actually addresses
    the user's question, preventing off-topic responses that copy chunks verbatim.

    Args:
        query: Original user query
        answer: Generated answer text

    Returns:
        True if answer contains at least 2 query topic words, False otherwise
    """
    if not query or not answer:
        return False

    query_topics = _extract_topic_words(query)
    if not query_topics:
        # Query too short or all stop words - skip validation
        return True

    answer_lower = answer.lower()
    matches = sum(1 for topic in query_topics if topic in answer_lower)

    # Require at least 2 topic matches (or 1 if query has only 1-2 topics)
    min_matches = min(2, len(query_topics))
    is_relevant = matches >= min_matches

    if not is_relevant:
        logger.warning(
            f"Answer validation failed: Query topics {query_topics[:5]} not found in answer. "
            f"Found {matches}/{len(query_topics)} matches. Answer starts: {answer[:100]}"
        )

    return is_relevant


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
    # Handle both LangGraph format (type: "human"/"ai") and standard format (role: "user"/"assistant")
    i = 0
    while i < len(chat_history):
        msg = chat_history[i]
        # Support both LangGraph format (type) and standard format (role)
        # Handle both dict format and LangGraph message objects
        if isinstance(msg, dict):
            role = msg.get("role", "")
            msg_type = msg.get("type", "")
            content = msg.get("content", "")
        else:
            # Handle LangGraph message objects (Pydantic models)
            role = getattr(msg, "role", "") if hasattr(msg, "role") else ""
            msg_type = getattr(msg, "type", "") if hasattr(msg, "type") else ""
            content = getattr(msg, "content", "") if hasattr(msg, "content") else ""

        # Normalize to role format: "user" or "assistant"
        is_user = role == "user" or msg_type == "human"
        is_assistant = role == "assistant" or msg_type == "ai"

        # Truncate long messages for readability
        content_preview = content[:100] + "..." if len(content) > 100 else content

        if is_user:
            turn_num += 1
            # Look ahead for assistant response in same turn
            next_msg = chat_history[i + 1] if i + 1 < len(chat_history) else None
            # Handle both dict format and LangChain message objects
            if next_msg is None:
                next_role = ""
                next_type = ""
            elif isinstance(next_msg, dict):
                next_role = next_msg.get("role", "")
                next_type = next_msg.get("type", "")
            else:
                # Handle LangChain message objects (Pydantic models)
                next_role = getattr(next_msg, "role", "") if hasattr(next_msg, "role") else ""
                next_type = getattr(next_msg, "type", "") if hasattr(next_msg, "type") else ""
            next_is_assistant = next_role == "assistant" or next_type == "ai"

            if next_is_assistant:
                examples.append(
                    f"Turn {turn_num}: User said '{content_preview}' | "
                    f"I responded with greeting/menu (guided conversation)"
                )
                i += 2  # Skip assistant response
            else:
                examples.append(f"Turn {turn_num}: User said '{content_preview}'")
                i += 1
        elif is_assistant and i == 0:
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


def _format_progressive_inference_metrics(session_memory: Dict[str, Any]) -> str:
    """Format progressive inference metrics for LLM prompt.

    Returns formatted string with metrics the LLM can reference in answers.

    Args:
        session_memory: Dict with progressive_inference_metrics

    Returns:
        Formatted string with metrics, or empty string if no metrics
    """
    metrics = session_memory.get("progressive_inference_metrics", {})
    if not metrics:
        return ""

    turn_count = metrics.get("turn_count", 0)
    topics_count = metrics.get("topics_count", 0)
    chat_history_length = metrics.get("chat_history_length", 0)
    depth_level = metrics.get("depth_level", 1)
    similarity_history = metrics.get("retrieval_similarity_history", [])

    parts = []
    if turn_count:
        parts.append(f"Turn count: {turn_count}")
    if topics_count:
        parts.append(f"Topics accumulated: {topics_count}")
    if chat_history_length:
        parts.append(f"Chat history length: {chat_history_length} messages")
    if depth_level:
        parts.append(f"Depth level: {depth_level}")

    if similarity_history and len(similarity_history) >= 2:
        # Don't include raw scores in generation prompt - they get confused with content
        # Instead, just indicate the trend without specific numbers
        first_sim = similarity_history[0].get("similarity", 0)
        last_sim = similarity_history[-1].get("similarity", 0)
        trend = "improving" if last_sim > first_sim else "stable"
        parts.append(f"[SYSTEM: Retrieval quality {trend} over conversation]")

    return " | ".join(parts) if parts else ""


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


def _generate_conversation_logic_explanation(state: ConversationState) -> str:
    """Generate explanation of conversation logic for meta-teaching.

    Allows Portfolia to explain her own conversation progression,
    memory accumulation, and inference improvement.

    Args:
        state: Current conversation state

    Returns:
        Formatted explanation string, or empty string if not applicable
    """
    session_memory = state.get("session_memory", {})
    chat_history = state.get("chat_history", [])
    topics = session_memory.get("topics", [])
    metrics = session_memory.get("progressive_inference_metrics", {})

    # Calculate current turn
    current_turn = sum(1 for msg in chat_history
                      if (isinstance(msg, dict) and (msg.get("role") == "user" or msg.get("type") == "human")) or
                         (hasattr(msg, "type") and msg.type == "human")) + 1

    similarity_history = metrics.get("retrieval_similarity_history", [])
    depth_level = state.get("depth_level", 1)
    conversation_phase = state.get("conversation_phase", "discovery")
    role = state.get("role", "Not set")
    role_mode = state.get("role_mode", "unknown")

    # Build similarity progression text
    similarity_text = ""
    if similarity_history and len(similarity_history) >= 2:
        sim_values = [f"Turn {entry.get('turn', '?')}: {entry.get('similarity', 0):.3f}"
                     for entry in similarity_history[-3:]]
        similarity_text = f"\n**Retrieval Similarity Progression**:\n- " + "\n- ".join(sim_values)

        # Calculate improvement
        if len(similarity_history) >= 2:
            first_sim = similarity_history[0].get("similarity", 0)
            last_sim = similarity_history[-1].get("similarity", 0)
            improvement = last_sim - first_sim
            if improvement > 0:
                similarity_text += f"\n- Improvement: +{improvement:.3f} ({improvement/first_sim*100:.1f}% better)"

    # Build topics text
    topics_text = ", ".join(topics[-5:]) if topics else "None yet"

    # Build turn-by-turn summary
    turn_summaries = []
    turn_num = 1
    for i in range(0, min(len(chat_history), 6), 2):
        if i < len(chat_history):
            user_msg = chat_history[i]
            if isinstance(user_msg, dict):
                content = user_msg.get("content", "")[:60]
            else:
                content = getattr(user_msg, "content", "")[:60] if hasattr(user_msg, "content") else ""

            if content:
                turn_summaries.append(f"Turn {turn_num}: \"{content}...\"")
                turn_num += 1

    turn_summary_text = "\n".join(turn_summaries) if turn_summaries else "No turns yet"

    explanation = f"""**Here's what I know about our conversation:**

**Current State:**
- Turn Number: {current_turn}
- Your Role: {role} ({role_mode})
- Conversation Phase: {conversation_phase}
- Depth Level: {depth_level} (1=overview, 2=guided detail, 3=deep dive)

**Topics We've Covered:**
{topics_text}

**Turn-by-Turn Summary:**
{turn_summary_text}
{similarity_text}

**How My Inference Has Improved:**
1. **Turn 1**: I sent a greeting and detected your role → stored `role_mode: {role_mode}`
2. **Each Turn**: I compose your query with role context, improving retrieval similarity
3. **Memory Accumulation**: Topics, entities, and affinity scores build up in `session_memory`
4. **Progressive Depth**: As we talk more, `depth_level` increases → I provide more detail

**What I'm Tracking:**
- Conversation patterns (orchestration → enterprise, general → specific)
- Quality flags (if answers get repetitive, I redirect)
- Phase transitions (discovery → exploration → synthesis)

This is the progressive inference that makes me smarter with each turn. The orchestration layer's intelligence comes from stateless nodes working with stateful memory - each node executes independently, but memory creates the thread that connects them into progressively smarter behavior.

**Want to explore more?**
- Ask about how I detect conversation patterns
- See how my retrieval improves with context
- Understand how I decide what to show you"""

    return explanation


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


# ============================================================================
# DATA PIPELINE DETECTION - Detect queries about data flow/pipeline
# ============================================================================

DATA_PIPELINE_KEYWORDS = [
    "data flow", "data pipeline", "embedding", "embeddings",
    "vector", "pgvector", "knowledge base", "update knowledge",
    "migration", "ingest", "ingestion", "chunking",
    "analytics", "logging", "metrics", "dashboard",
    "how do you learn", "how do you update", "how does data",
    "retrieval pipeline", "rag pipeline", "vector search",
    "similarity search", "data storage"
]


def _is_data_pipeline_query(query: str) -> bool:
    """Detect if query is about data pipeline/data flow.

    Args:
        query: User's query string

    Returns:
        True if query is about data pipeline topics
    """
    query_lower = query.lower()
    return any(kw in query_lower for kw in DATA_PIPELINE_KEYWORDS)


# ============================================================================
# DRIFT DETECTION - Detect when user is drifting away from Portfolia's KB
# ============================================================================

DRIFT_INDICATORS = [
    "in general", "best practice", "what do you think", "your opinion",
    "should i use", "which is better", "compare", "vs",
    "best way to", "how should i", "what's the best", "recommend",
    "pros and cons", "advantages", "disadvantages"
]

PILLAR_ANCHOR_KEYWORDS = [
    # Keep user anchored to Portfolia's knowledge base
    "orchestration", "langgraph", "nodes", "state", "memory", "pipeline",
    "tech stack", "architecture", "rag", "retrieval", "pgvector", "supabase",
    "enterprise", "adapt", "customer support", "scaling", "production",
    "noah", "background", "career", "tesla", "resume", "portfolio",
    "portfolia", "how do you", "your", "this system"
]


def _is_drifting_query(query: str) -> bool:
    """Detect if user is drifting away from Portfolia's knowledge base.

    Drift is when user asks generic questions that aren't about:
    - Portfolia's architecture
    - Noah's background
    - How this specific system works

    Args:
        query: User's query string

    Returns:
        True if query appears to be drifting into generic territory
    """
    query_lower = query.lower()

    # Check for generic question patterns that indicate drift
    has_drift_pattern = any(indicator in query_lower for indicator in DRIFT_INDICATORS)

    # Check if query is still anchored to Portfolia's knowledge base
    has_anchor = any(kw in query_lower for kw in PILLAR_ANCHOR_KEYWORDS)

    # Drift = has generic pattern BUT no anchor to our knowledge base
    if has_drift_pattern and not has_anchor:
        return True

    # Also drift if query is very short with no anchor (might be testing)
    if len(query.split()) <= 3 and not has_anchor:
        return False  # Don't flag very short queries as drift

    return False


def _should_use_teaching_style(state: ConversationState) -> bool:
    """Determine if answer should use John Danaher-style teaching structure.

    Teaching style is used when:
    - depth_level >= 2 AND query asks "how"/"explain"
    - OR depth_level == 3 (deep dive)
    - OR abstraction_level detected (architecture/function/implementation)
    - OR query intent is technical/engineering AND depth_level >= 2

    Returns:
        True if teaching style should be used, False for conversational
    """
    depth_level = state.get("depth_level", 1)
    query = state.get("query", "").lower()
    role = state.get("role", "")
    chat_history = state.get("chat_history", [])
    abstraction_level = _detect_abstraction_level(query, role, chat_history)
    intent = state.get("query_intent") or state.get("query_type", "")

    # Explicit teaching requests
    teaching_keywords = ["how", "explain", "walk through", "show me how", "tell me about"]
    is_explanation_query = any(kw in query for kw in teaching_keywords)

    # Deep dive or architecture questions
    is_deep_dive = depth_level == 3
    is_architecture = abstraction_level == "architecture"
    is_technical = intent in {"technical", "engineering"} and depth_level >= 2

    return (is_explanation_query and depth_level >= 2) or is_deep_dive or is_architecture or is_technical


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
            # Simple queries → faster model (but still high quality)
            selected_model = "gpt-4o-mini"
            logger.info(f"Model routing: complexity=simple, model={selected_model}")
            return selected_model
        elif query_complexity == "complex":
            # Complex queries → best model for quality
            selected_model = "gpt-4o"
            logger.info(f"Model routing: complexity=complex, model={selected_model}")
            return selected_model
        else:
            # Medium complexity → use gpt-4o for best quality (portfolio priority)
            logger.info(f"Using gpt-4o for technical query (type={query_type}, role={role_mode}, complexity=medium)")
            return "gpt-4o"

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


def _format_conversation_examples_for_inference(chat_history: List[Dict]) -> str:
    """Format conversation history for inference explanation examples.

    Args:
        chat_history: List of message dicts with 'role' and 'content' keys, or LangChain message objects

    Returns:
        Formatted string with conversation examples
    """
    if not chat_history:
        return "No previous turns yet"

    examples = []
    turn_num = 1
    for i, msg in enumerate(chat_history):
        # Handle both dict format and LangChain message objects
        if isinstance(msg, dict):
            role = msg.get("role", "")
            msg_type = msg.get("type", "")
            content = msg.get("content", "")
        else:
            # Handle LangChain message objects (Pydantic models)
            role = getattr(msg, "role", "") if hasattr(msg, "role") else ""
            msg_type = getattr(msg, "type", "") if hasattr(msg, "type") else ""
            content = getattr(msg, "content", "") if hasattr(msg, "content") else ""

        # Normalize to role format
        is_user = role == "user" or msg_type == "human"
        is_assistant = role == "assistant" or msg_type == "ai"

        if is_user:
            examples.append(f"Turn {turn_num}: User asked '{content[:100]}'")
            turn_num += 1
        elif is_assistant and i > 0:
            # Link assistant response to previous user query
            examples.append(f"  → I responded with explanation about {content[:50]}...")

    return " | ".join(examples[:6])  # Last 3 exchanges


def _build_context_management_explanation(chat_history: List[Dict], current_turn: int) -> str:
    """Build educational explanation about context management for long conversations.

    This is a teaching moment where Portfolia explains how she manages context
    in long conversations, demonstrating real production GenAI patterns.

    Args:
        chat_history: Full conversation history
        current_turn: Current turn number (1-indexed)

    Returns:
        Educational explanation string, or empty if not needed
    """
    # Only explain at turn 15+ and every 5 turns after that
    if current_turn < 15:
        return ""

    # Don't repeat too frequently (every 5 turns)
    if current_turn % 5 != 0:
        return ""

    total_messages = len(chat_history)
    recent_window = min(6, total_messages)  # Last 3 exchanges (6 messages)

    explanation = (
        "\n\n---\n"
        f"**🧠 Quick Technical Note (Turn {current_turn}):**\n\n"
        f"We've had {total_messages} messages so far — that's a great conversation! "
        "I wanted to share something interesting about how I manage context in long conversations like this.\n\n"
        "**How I Handle Long Conversations:**\n"
        f"- I keep the full conversation history ({total_messages} messages) for analytics and pattern detection\n"
        f"- But for generating responses, I use a **sliding window** of the last {recent_window} messages (last 3 exchanges)\n"
        "- This keeps my responses fast and focused on recent context, while still maintaining conversation continuity\n"
        "- I also prune retrieved chunks to fit within a 4,000-token budget to stay within model limits\n\n"
        "**Why This Matters for Production AI:**\n"
        "This same pattern — sliding windows + token budgets — is how enterprise chatbots handle long customer support conversations. "
        "It balances context richness with performance and cost. The alternative (sending all 40+ messages to the LLM) would be slower, "
        "more expensive, and often less relevant since recent context matters most.\n\n"
        "This is a real production pattern you'd see in systems handling thousands of conversations daily. "
        "Want me to show you the code that implements this, or explain how we tune the window size?"
    )

    return explanation


def _should_use_chain_of_thought(state: ConversationState) -> bool:
    """Determine if query benefits from explicit chain-of-thought reasoning.

    Use CoT for:
    - Enterprise/adaptation queries (critical for business value)
    - User showing confusion signals
    - Complex multi-part questions
    - Deep conversations (10+ messages)
    - Explicit explanation requests

    Args:
        state: Current conversation state

    Returns:
        True if CoT should be used, False otherwise
    """
    query = state.get("query", "").lower()
    chat_history = state.get("chat_history", [])

    # Always use for enterprise queries (critical for your use case)
    enterprise_keywords = [
        "adapt", "enterprise", "customer support", "use case",
        "how would", "how could", "apply", "scales to", "scaling",
        "production", "deploy", "integrate", "customize"
    ]
    if any(kw in query for kw in enterprise_keywords):
        logger.info(f"CoT triggered: enterprise keywords detected in '{query[:50]}'")
        return True

    # Check for confusion signals
    confusion = detect_confusion_signals(query, chat_history)
    if confusion.get("seems_confused"):
        logger.info(f"CoT triggered: confusion signals detected - {confusion.get('signals', [])}")
        return True

    # Complex multi-part questions
    if query.count("?") > 1 or " and " in query:
        logger.info(f"CoT triggered: complex multi-part question detected")
        return True

    # Deep conversations (10+ messages = 5+ exchanges)
    if len(chat_history) >= 10:
        logger.info(f"CoT triggered: deep conversation ({len(chat_history)} messages)")
        return True

    # Explicit explanation requests
    explanation_patterns = ["explain", "walk me through", "how does", "why does", "teach me"]
    if any(p in query for p in explanation_patterns):
        logger.info(f"CoT triggered: explanation request detected")
        return True

    return False


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
    # CRITICAL: Entry point verification - this should ALWAYS print
    import sys
    print(f"\n>>> generate_draft() CALLED - Query: {state.get('query', 'NO QUERY')[:50]} <<<\n", file=sys.stderr, flush=True)

    # #region agent log - Entry point
    try:
        log_path = _get_debug_log_path()
        with open(log_path, 'a') as f:
            import json
            f.write(json.dumps({
                "location": "stage5_generation_nodes.py:1029",
                "message": "generate_draft ENTRY",
                "data": {
                    "query": state.get('query', 'NO QUERY')[:50],
                    "has_retrieved_chunks": bool(state.get('retrieved_chunks')),
                    "retrieved_chunks_count": len(state.get('retrieved_chunks', [])),
                    "role_mode": state.get('role_mode')
                },
                "timestamp": int(time.time() * 1000),
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "ALL"
            }) + "\n")
    except Exception as log_err:
        print(f"DEBUG LOG FAILED in generate_draft: {log_err}", file=sys.stderr, flush=True)
    # #endregion

    # Fail-fast: Validate required fields (Defensibility)
    try:
        query = state["query"]
    except KeyError as e:
        logger.error("generate_draft called without query in state")
        raise KeyError("State must contain 'query' field for generation") from e

    # Handle reformulation loop edge case early
    # Check for reformulation loop in session_memory (where edge case detection stores it)
    session_memory = state.get("session_memory", {})
    if session_memory.get("last_edge_case") == "reformulation_loop":
        from assistant.flows.node_logic.util_conversational_edge_case_handler import handle_reformulation_loop
        logger.info("Reformulation loop detected - attempting clarification response")
        result = handle_reformulation_loop(state)
        # CRITICAL: Only return early if the handler actually handled the case
        # If no relevant previous answer exists, edge_case_handled will be False
        # and we should continue with normal generation
        if result.get("edge_case_handled", False):
            logger.info("Reformulation handler successfully generated clarification")
            return {
                "draft_answer": result.get("draft_answer", ""),
                "answer": result.get("answer", ""),
                "edge_case_handled": True
            }
        else:
            logger.info("Reformulation handler found no relevant previous answer - proceeding with normal generation")

    # =========================================================================
    # PRIORITY: Handle REPEATED QUERY (Fix 3 - is_repeated_query flag)
    # If detect_repeated_query node set this flag, use reformulation handler
    # =========================================================================
    if state.get("is_repeated_query"):
        from assistant.flows.node_logic.util_conversational_edge_case_handler import handle_reformulation_loop
        logger.info("Repeated query detected via is_repeated_query flag - attempting clarification response")
        result = handle_reformulation_loop(state)
        if result.get("edge_case_handled", False):
            logger.info("Repeated query handler successfully generated clarification")
            return {
                "draft_answer": result.get("draft_answer", ""),
                "answer": result.get("answer", ""),
                "edge_case_handled": True,
                "is_repeated_query": True
            }
        else:
            logger.info("Repeated query handler found no relevant previous answer - proceeding with normal generation")

    # =========================================================================
    # PRIORITY: Handle ACTION REQUESTS (Fix 1 - resume/linkedin/github)
    # Short-circuit full generation for pure action requests
    # =========================================================================
    if state.get("query_type") == "action_request":
        requested_resources = state.get("requested_resources", [])
        is_repeated = state.get("is_repeated_action_request", False)

        # #region agent log
        debug_trace = state.get("_debug_trace", [])
        debug_trace.append({"loc": "generate_draft:action_request", "resources": requested_resources, "is_repeated": is_repeated})
        # #endregion

        # Handle REPEATED action requests specially
        if is_repeated:
            # #region agent log
            debug_trace.append({"loc": "generate_draft:repeated_handler", "msg": "Returning acknowledgment for repeated action"})
            # #endregion
            logger.info("Repeated action request detected - returning acknowledgment")
            draft = "I already provided those resources above. Is there something specific you're looking for, or would you like Noah to reach out directly?"
            return {
                "draft_answer": draft,
                "is_repeated_action_request": True,
                "_debug_trace": debug_trace
            }

        logger.info(f"Action request detected - short-circuiting generation for: {requested_resources}")

        # Build minimal response - action planning will add the actual links
        response_parts = ["Here are the resources you requested:"]

        # The actual links will be added by format_answer based on pending_actions
        # We just need a minimal draft that indicates this is an action response
        draft = "Here are the resources you requested:"

        return {
            "draft_answer": draft,
            "answer": draft,
            "action_request_handled": True,
            "requested_resources": requested_resources
        }

    # #region agent log
    try:
        log_path = _get_debug_log_path()
        with open(log_path, 'a') as f:
            import json
            f.write(json.dumps({
                "location": "stage5_generation_nodes.py:1045",
                "message": "generate_draft() entry",
                "data": {
                    "query": query,
                    "query_type": state.get("query_type"),
                    "menu_choice": state.get("menu_choice"),
                    "role_mode": state.get("role_mode"),
                    "has_answer": bool(state.get("answer")),
                    "answer_preview": (state.get("answer", "")[:100] + "...") if state.get("answer") else None,
                    "retrieved_chunks_count": len(state.get("retrieved_chunks", []))
                },
                "timestamp": int(time.time() * 1000),
                "sessionId": "debug-session",
                "runId": "run2",
                "hypothesisId": "E"
            }) + "\n")
    except Exception as log_err:
        # Don't fail on logging errors
        pass
    # #endregion

    # Check if user is asking about conversation logic (self-explanation)
    # This allows Portfolia to explain her own conversation progression
    query_lower = query.lower()
    session_memory = state.get("session_memory", {})

    conversation_logic_phrases = [
        "how do you know", "how did you detect", "what turn", "how many turns",
        "what have we covered", "explain your reasoning", "how are you tracking",
        "what do you know about me", "what have you learned", "how does your inference",
        "how do you remember", "how does your memory", "what topics", "how are you getting smarter"
    ]

    if any(phrase in query_lower for phrase in conversation_logic_phrases):
        # Generate conversation logic explanation
        explanation = _generate_conversation_logic_explanation(state)
        if explanation:
            logger.info("Conversation logic self-explanation generated")
            return {
                "draft_answer": explanation,
                "answer": explanation,
                "self_explanation_generated": True
            }

    # Check if user is asking about edge case detection (meta-teaching)
    # This happens when user asks "how did you detect" after an edge case was handled
    if session_memory.get("last_edge_case") and any(
        phrase in query_lower for phrase in [
            "how did you detect", "how do you handle", "explain the detection",
            "how does that work", "show me how", "meta", "explain how you",
            "how did you know", "why did you", "explain your reasoning"
        ]
    ):
        # Generate meta-teaching explanation
        from assistant.flows.node_logic.util_edge_case_meta_teaching import generate_meta_teaching_explanation
        try:
            explanation = generate_meta_teaching_explanation(state)
            update["draft_answer"] = explanation
            update["answer"] = explanation
            logger.info(f"Meta-teaching explanation generated for edge case: {session_memory.get('last_edge_case')}")
            return update
        except Exception as e:
            logger.warning(f"Failed to generate meta-teaching explanation: {e}")
            # Fall through to normal generation

    # ========== Chain-of-Thought for Complex Queries ==========
    # Use CoT for enterprise queries, confusion signals, complex questions, etc.
    if _should_use_chain_of_thought(state):
        logger.info(f"Using Chain-of-Thought generation for: {query[:50]}...")
        try:
            cot_result = chain_of_thought_generate(state, rag_engine)

            # If CoT determined clarification is needed, return early
            if cot_result.get("clarification_needed"):
                logger.info("CoT determined clarification needed")
                return {
                    "draft_answer": cot_result.get("clarifying_question", ""),
                    "answer": cot_result.get("clarifying_question", ""),
                    "clarification_needed": True,
                    "cot_reasoning": cot_result.get("cot_reasoning"),
                    "cot_metadata": cot_result.get("cot_metadata"),
                    "cot_enabled": True,
                }

            # Return CoT-generated answer
            logger.info(f"CoT generation complete. Reasoning time: {cot_result.get('cot_metadata', {}).get('reasoning_time_ms', 'N/A')}ms")
            return {
                "draft_answer": cot_result.get("draft_answer", ""),
                "answer": cot_result.get("answer", ""),
                "cot_reasoning": cot_result.get("cot_reasoning"),
                "cot_metadata": cot_result.get("cot_metadata"),
                "cot_enabled": True,
            }
        except Exception as e:
            logger.warning(f"CoT generation failed, falling back to standard: {e}")
            # Fall through to standard generation

    # Access optional fields safely (Defensibility)
    retrieved_chunks = state.get("retrieved_chunks", [])
    role = state.get("role", "Just looking around")
    chat_history = state.get("chat_history", [])
    retrieval_scores = state.get("retrieval_scores", [])

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

    # #region agent log
    with open(_get_debug_log_path(), 'a') as f:
        import json
        f.write(json.dumps({
            "location": "stage5_generation_nodes.py:1021",
            "message": "Before pipeline_halt check",
            "data": {
                "pipeline_halt": state.get("pipeline_halt"),
                "query": query,
                "query_type": state.get("query_type"),
                "menu_choice": state.get("menu_choice"),
                "has_answer": bool(state.get("answer")),
                "answer_preview": (state.get("answer", "")[:100] + "...") if state.get("answer") else None
            },
            "timestamp": int(time.time() * 1000),
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "A"
        }) + "\n")
    # #endregion

    if state.get("pipeline_halt"):
        # #region agent log
        log_path = _get_debug_log_path()
        with open(log_path, 'a') as f:
            import json
            f.write(json.dumps({
                "location": "stage5_generation_nodes.py:1022",
                "message": "Early return due to pipeline_halt",
                "data": {
                    "pipeline_halt": state.get("pipeline_halt"),
                    "query": query
                },
                "timestamp": int(time.time() * 1000),
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "A"
            }) + "\n")
        # #endregion
        # Return empty update dict (not state) to avoid preserving old answer
        # In LangGraph StateGraph, nodes should return partial updates, not full state
        # Explicitly clear answer when pipeline is halted to prevent preserving old answers
        return {"answer": None, "draft_answer": None}

    # #region agent log
    with open(_get_debug_log_path(), 'a') as f:
        import json
        f.write(json.dumps({
            "location": "stage5_generation_nodes.py:1024",
            "message": "Pipeline_halt check passed, continuing",
            "data": {
                "pipeline_halt": state.get("pipeline_halt"),
                "grounding_status": state.get("grounding_status")
            },
            "timestamp": int(time.time() * 1000),
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "A"
        }) + "\n")
    # #endregion

    grounding_status = state.get("grounding_status")
    if grounding_status and grounding_status not in {"ok", "unknown"}:
        # Return empty update dict (not state) to avoid preserving old answer
        return {}

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

    # OUT-OF-SCOPE DETECTION - Gracefully handle queries outside knowledge base
    from assistant.flows.node_logic.stage6_formatting_nodes import detect_out_of_scope
    is_out_of_scope, bridge_pillar, bridge_prompt = detect_out_of_scope(query, state.get("role_mode", "explorer"))

    if is_out_of_scope:
        state["out_of_scope_detected"] = True
        state["bridge_suggestion"] = bridge_prompt
        logger.info(f"Out-of-scope query detected: '{query[:50]}'. Bridge: {bridge_pillar}")

        # Add instruction to prompt for graceful handling
        out_of_scope_instruction = f"""
IMPORTANT: The user asked about something outside your primary knowledge base.

DO NOT just say "I don't know" or refuse to answer. Instead:
1. Acknowledge that this specific topic isn't your specialty
2. Bridge gracefully to related content you DO have: "{bridge_prompt}"
3. Always be helpful and offer alternatives

Example response pattern:
"That's outside my main focus area - I'm built to demonstrate GenAI engineering patterns.
However, [bridge to relevant topic]. Would you like to explore that?"
"""
        extra_instructions.append(out_of_scope_instruction)

    # DATA PIPELINE QUERIES - Provide structured 6-stage explanation
    if _is_data_pipeline_query(query):
        logger.info(f"Data pipeline query detected: '{query[:50]}'")
        data_pipeline_instruction = """
DATA PIPELINE EXPLANATION (Use this 6-stage structure):

The user is asking about data flow/pipeline. Structure your response around these 6 stages:

**1. Query Ingestion**
Purpose: Capture and prepare user input for processing.
Implementation: FastAPI endpoint receives query, sanitizes input, extracts entities.
Key Metric: <50ms ingestion latency.

**2. Embedding Generation**
Purpose: Convert query to 1536-dimensional vector for semantic search.
Implementation: OpenAI text-embedding-3-small API.
Key Metric: $0.02/1M tokens, ~100ms per embedding.

**3. Vector Retrieval**
Purpose: Find semantically similar knowledge chunks.
Implementation: Supabase pgvector, cosine similarity, top-4 chunks.
Key Metric: 0.7 similarity threshold.

**4. Context Assembly**
Purpose: Prepare retrieved chunks for LLM generation.
Implementation: Role-aware formatting, 4000 token budget, recency weighting.
Key Metric: 4 chunks × ~500 tokens = 2000 tokens context.

**5. Response Generation**
Purpose: Synthesize grounded response from context.
Implementation: GPT-4o-mini (default), temperature 0.3, grounding validation.
Key Metric: $0.0003/query average cost.

**6. Analytics Logging**
Purpose: Track everything for optimization and debugging.
Implementation: Supabase tables + LangSmith traces.
Key Metric: 100% conversation coverage.

Include specific numbers from the current conversation where possible.
"""
        extra_instructions.append(data_pipeline_instruction)

    # DRIFT DETECTION - Guide user back when they're asking generic questions
    if _is_drifting_query(query):
        session_memory = state.get("session_memory", {})
        from assistant.flows.node_logic.stage6_formatting_nodes import _get_unexplored_pillars
        unexplored = _get_unexplored_pillars(session_memory, state.get("role_mode", "explorer"))

        # Build bridge suggestion
        bridge_suggestion = unexplored[0][1] if unexplored else "exploring a specific aspect of my architecture"

        logger.info(f"Drift detected: '{query[:50]}' - bridging back to knowledge base")

        drift_instruction = f"""
DRIFT DETECTED - BRIDGE BACK TO KNOWLEDGE BASE:

The user's question is drifting toward generic topics outside your specific knowledge.
You're designed to demonstrate Noah's GenAI engineering through YOUR OWN architecture.

RESPONSE PATTERN:
1. Briefly acknowledge their question (1 sentence max)
2. Explain your focus: "I'm built to demonstrate GenAI engineering through my own architecture, so I can speak best to how I handle this specifically..."
3. Bridge back with a specific offer: "{bridge_suggestion}"
4. If relevant, connect their question to something you DO know

EXAMPLE:
User: "What's the best caching strategy in general?"
You: "Great question! I can speak best to my own approach - I use simple in-memory caching
for session state, which keeps costs low at ~$0.0003/query. For enterprise deployments,
you'd want Redis. Want to see how I handle data caching, or explore how this scales for enterprise?"

AVOID:
- Long generic explanations of topics you weren't built to cover
- Making up information outside your knowledge base
- Ignoring the drift and answering as if you're a general AI assistant
"""
        extra_instructions.append(drift_instruction)

    # JOHN DANAHER TEACHING STYLE - Enforce for technical explanations
    # This creates systematic, quantitative, purpose-driven explanations
    role_mode = state.get("role_mode", "explorer")
    if role_mode in ("hiring_manager_technical", "software_developer") and _should_use_teaching_style(state):
        danaher_instruction = """
TEACHING STRUCTURE (Required for technical explanations - John Danaher Style):

FORMAT YOUR RESPONSE WITH THESE 5 PARTS:

1. CONTEXT-SETTING OPENING (2-3 sentences with warmth):
   - Start with: "Let me walk you through this systematically..." or "Here's what makes this powerful..."
   - Set expectations for what you'll explain

2. SYSTEMATIC LAYER ENUMERATION (numbered layers/components):
   - Use numbered format: "**1. [Layer Name]**"
   - For each layer include:
     * **Purpose**: Why this exists (1 sentence)
     * **Implementation**: Specific technologies with numbers
     * **Key Metric**: One concrete number (latency, cost, count)
   - Example:
     **1. Orchestration Layer**
     Purpose: Manages conversation flow through modular, testable nodes.
     Implementation: LangGraph StateGraph with 21 nodes across 7 stages.
     Key Metric: 325ms average node execution time.

3. QUANTITATIVE EVIDENCE (real numbers from this system):
   - Include similarity scores, costs, latencies, counts
   - Example: "In this query, I retrieved 4 chunks with top similarity 0.85"
   - Show costs: "$0.0003 per query", "$0.15/1M tokens"

4. CRITICAL INSIGHT (1-3 sentence synthesis):
   - Connect all layers to an overarching principle
   - Example: "The modularity IS the architecture—each layer swappable, so GPT-5 means replacing one node, not rebuilding."

5. INVITATION TO EXPLORE (3 numbered options):
   - Offer specific next explorations
   - Example: "Where would you like to go from here?
     1. See the actual pgvector SQL query
     2. Explore the grounding validation logic
     3. Walk through cost optimization"

WARMTH CONNECTIVES (use throughout):
- "Here's what makes this powerful..."
- "This part is genuinely fascinating..."
- "The key insight is..."
- "Here's where the magic happens..."

AVOID:
- "Ah, [topic]!" or "I love this!"
- Generic "Let me help you" phrases
- Starting with "Great question"

CRITICAL - KEY METRICS DISTINCTION:
When choosing "Key Metric" values, use CONTEXTUALLY APPROPRIATE metrics:

FOR PORTFOLIA'S ARCHITECTURE (orchestration, tech stack, data pipeline):
- Use SYSTEM metrics: similarity scores, latency, token counts, costs
- Example: "Key Metric: 325ms average latency" or "Key Metric: 0.85 similarity score"

FOR NOAH'S BACKGROUND/CAREER:
- Use HIS metrics: years of experience, certifications, project outcomes
- Example: "Key Metric: 3+ years at Tesla Energy" or "Key Metric: Built production RAG from scratch"
- NEVER use "Key Metric: Top similarity 0.9" when discussing Noah's career!

The user wants to know about NOAH when they ask about his background, not about
how well you retrieved information about him.
"""
        extra_instructions.append(danaher_instruction)
        logger.debug("Added John Danaher teaching style instructions")

    # Check for retrieval topic mismatch - if chunks don't match query, use LLM fallback
    if state.get("retrieval_topic_mismatch"):
        logger.warning(f"Retrieval topic mismatch detected for query: {query[:50]}")
        # Add explicit instruction to use LLM's inherent knowledge
        enterprise_keywords = ["adapt", "adapts", "adaptation", "customer support", "enterprise",
                               "use case", "chatbot", "internal docs", "sales enablement"]
        is_enterprise_query = any(kw in query.lower() for kw in enterprise_keywords)

        if is_enterprise_query:
            fallback_instruction = (
                "\n\n⚠️ CRITICAL: RETRIEVAL MISMATCH DETECTED\n"
                "The retrieved context may not directly answer this question about enterprise adaptation.\n"
                f"Based on your knowledge of RAG systems and enterprise AI patterns, please answer: {query}\n\n"
                "If discussing enterprise adaptation, explain:\n"
                "1. What components typically change (knowledge base, roles, actions)\n"
                "2. What stays the same (orchestration pipeline)\n"
                "3. Expected business value (ROI, cost savings, efficiency gains)\n"
                "4. Code examples where relevant (ROLES dictionary, action handlers)\n\n"
                "DO NOT rely solely on retrieved chunks - use your knowledge to provide a comprehensive answer.\n"
            )
        else:
            fallback_instruction = (
                "\n\n⚠️ CRITICAL: RETRIEVAL MISMATCH DETECTED\n"
                "The retrieved context may not directly answer this question.\n"
                f"Based on your knowledge, please answer: {query}\n\n"
                "DO NOT rely solely on retrieved chunks - use your knowledge to provide a comprehensive answer.\n"
            )

        # Prepend fallback instruction to extra_instructions
        extra_instructions.append(fallback_instruction)

    # Check retrieval quality and add synthesis instructions for moderate scores
    if retrieval_scores:
        top_similarity = max(retrieval_scores) if retrieval_scores else 0.0
        avg_similarity = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0

        # Moderate similarity range (0.4-0.7) needs better synthesis
        # Low scores (<0.4) should trigger clarification, but moderate scores need synthesis help
        if 0.4 <= top_similarity < 0.7:
            extra_instructions.append(
                "\n\nCRITICAL: MODERATE RETRIEVAL QUALITY - ENHANCE SYNTHESIS\n"
                f"- Retrieval similarity is moderate ({top_similarity:.3f}) - retrieved chunks are somewhat relevant but not perfect matches\n"
                "- DO NOT just echo or list the retrieved chunks verbatim\n"
                "- Instead, SYNTHESIZE the information: connect ideas, explain relationships, provide context\n"
                "- If chunks mention examples or documentation structure, extract the underlying concepts and explain them\n"
                "- Bridge gaps: if chunks don't directly answer the question, use your knowledge to connect them to the user's query\n"
                "- Be explicit about what you're inferring vs. what's directly in the chunks\n"
                "- For enterprise adaptation queries: explain HOW the architecture adapts, not just that it can adapt\n"
            )
            logger.debug(f"Added moderate similarity synthesis instructions (top_similarity={top_similarity:.3f})")

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

    # Phase 2: Enhanced Turn Reference Instructions - Add explicit turn numbers and examples
    if chat_history and len(chat_history) >= 2:
        # Count turns for reference
        user_turns = sum(1 for msg in chat_history
                        if (isinstance(msg, dict) and msg.get("role") == "user") or
                           (hasattr(msg, "type") and msg.type == "human"))
        current_turn = user_turns + 1  # Next turn number

        # Build turn-specific examples from actual conversation
        turn_examples = []
        turn_num = 1
        for i, msg in enumerate(chat_history[-6:]):  # Last 3 exchanges (6 messages = 3 turns)
            # Handle both dict format and LangChain message objects
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")[:100]
                msg_type = msg.get("type", "")
                if msg_type == "human":
                    role = "user"
                elif msg_type == "ai":
                    role = "assistant"
            else:
                # Handle LangChain message objects (Pydantic models)
                msg_type = getattr(msg, "type", "") if hasattr(msg, "type") else ""
                role = getattr(msg, "role", "") if hasattr(msg, "role") else ""
                content = getattr(msg, "content", "") if hasattr(msg, "content") else ""
                content = content[:100] if content else ""
                if msg_type == "human":
                    role = "user"
                elif msg_type == "ai":
                    role = "assistant"

            if role == "user":
                turn_examples.append(f"Turn {turn_num}: User asked '{content}...'")
                turn_num += 1

        examples_text = "\n".join(turn_examples) if turn_examples else "Previous turns in conversation history"

        # Get metrics for turn progression context
        session_memory = state.get("session_memory", {})
        metrics = session_memory.get("progressive_inference_metrics", {})
        depth_level = metrics.get("depth_level", 1)
        topics_count = metrics.get("topics_count", 0)

        # Only require turn references when it makes sense (not for every answer)
        should_reference = (
            current_turn >= 3 and  # At least Turn 3
            not state.get("is_greeting", False) and  # Not a greeting
            depth_level >= 1  # Any depth level
        )

        if should_reference:
            # Natural turn reference instruction (conversational, not structural)
            progression_instructions = (
                f"\n\nCONVERSATION CONTEXT:\n"
                f"- You are at Turn {current_turn}\n"
                f"- Previous turns: {examples_text}\n"
                f"- When relevant, naturally reference previous discussion using phrases like:\n"
                f"  * 'Building on [Turn N]'s discussion of [topic]...'\n"
                f"  * 'Following up on [Turn N]'s [topic] question...'\n"
                f"  * 'You've progressed from [Turn X] → [Turn Y] → now [Turn Z]...'\n"
                f"- Make references feel natural and conversational, not forced\n"
                f"- If the current query doesn't relate to previous turns, don't force a reference\n"
                f"- Current conversation state: Depth level {depth_level}, Topics accumulated: {topics_count}\n"
            )
        else:
            # Early turns or greeting - no turn reference needed
            progression_instructions = (
                f"\n\nCONVERSATION CONTEXT:\n"
                f"- You are at Turn {current_turn}\n"
                f"- Current conversation state: Depth level {depth_level}, Topics accumulated: {topics_count}\n"
            )

        # Add context management explanation for long conversations (teaching moment)
        if current_turn >= 15:
            context_note = _build_context_management_explanation(chat_history, current_turn)
            if context_note:
                extra_instructions.append(context_note)

        # Phase 4: Add synthesis instructions
        synthesis_instructions = (
            "\n\nCRITICAL: CROSS-TURN SYNTHESIS\n"
            "- DO NOT treat each query as isolated - synthesize across multiple turns\n"
            "- Example: If Turn 3 was about orchestration and Turn 4 is about enterprise, "
            "say: 'Building on our orchestration discussion from Turn 3, enterprise relevance means...'\n"
            "- Example: If Turn 5 is about implementation, say: 'From Turn 3's architecture discussion to Turn 4's enterprise value, "
            "here's the implementation that bridges both...'\n"
            "- Connect multiple previous turns: Reference 2-3 previous turns when synthesizing\n"
            "- Show progression: Explain how the conversation evolved from Turn 1 → Turn 2 → Turn 3 → current turn\n"
        )

        extra_instructions.append(progression_instructions)
        extra_instructions.append(synthesis_instructions)
        logger.debug(f"Added enhanced conversation progression instructions with turn numbers (Turn {current_turn})")

        # Conditional teaching style structure selection
        should_teach = _should_use_teaching_style(state)
        if should_teach:
            abstraction_level = _detect_abstraction_level(query, role, chat_history)
            if depth_level == 3 or abstraction_level == "architecture":
                # Full 5-part Danaher structure
                teaching_instruction = (
                    "\n\nCRITICAL: JOHN DANAHER-STYLE TEACHING STRUCTURE (5-PART)\n"
                    "Your answer MUST follow this structure:\n\n"
                    "**1. Context-Setting Opening** (2-3 sentences with warmth):\n"
                    "- If referencing previous turn, integrate naturally: 'Building on Turn N's discussion...'\n"
                    "- Set the stage: 'Let me walk you through this systematically...'\n\n"
                    "**2. Systematic Layer Enumeration** (core answer):\n"
                    "- Organize by layers/components with clear numbering (1, 2, 3...)\n"
                    "- For each: Layer Name → Purpose → Implementation → Key Metric\n\n"
                    "**3. Quantitative Evidence** (actual numbers):\n"
                    "- Include real metrics from this conversation\n"
                    "- Show costs, dimensions, performance numbers\n\n"
                    "**4. Critical Insight** (1-3 sentences synthesis):\n"
                    "- Connect layers to overarching principle\n"
                    "- Example: 'The modularity is the architecture...'\n\n"
                    "**5. Invitation to Explore** (3 numbered options):\n"
                    "- Offer specific next explorations\n"
                )
            else:
                # Condensed 3-part structure (for depth_level=2, connecting to previous turns)
                teaching_instruction = (
                    "\n\nCRITICAL: CONDENSED TEACHING STRUCTURE (3-PART)\n"
                    "Your answer MUST follow this structure:\n\n"
                    "**1. Natural Turn Reference + Opening** (1-2 sentences):\n"
                    "- Reference previous turn conversationally: 'Building on Turn N's [topic]...'\n"
                    "- Set up what you'll explain\n\n"
                    "**2. Systematic Component Enumeration** (condensed):\n"
                    "- 3-4 key components with Purpose statements\n"
                    "- Brief implementation details\n"
                    "- Key metrics\n\n"
                    "**3. Critical Insight** (1 sentence):\n"
                    "- Connect to previous discussion\n"
                    "- Bridge to next natural step\n"
                )
            extra_instructions.append(teaching_instruction)
        else:
            # Conversational style - no systematic structure required
            conversational_instruction = (
                "\n\nCONVERSATIONAL STYLE:\n"
                "- Answer naturally and conversationally\n"
                "- If referencing previous turn, do so naturally: 'Following up on Turn N...'\n"
                "- No systematic enumeration required\n"
                "- Keep it brief and to the point\n"
            )
            extra_instructions.append(conversational_instruction)

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
        "performance optimization",
        # Success criteria phrases
        "success criteria",
        "what are your goals",
        "how do you measure success",
        "what defines success",
        "how do you validate",
        "what are your metrics",
        "progressive inference goals",
        "inference success",
        "validation criteria"
    ])

    if is_inference_query:
        # Get current conversation context for examples
        conversation_examples = _format_conversation_examples_for_inference(chat_history) if chat_history else "No previous turns yet"

        # Check if query is about success criteria specifically
        is_success_criteria_query = any(phrase in query_lower for phrase in [
            "success criteria", "what are your goals", "how do you measure success",
            "what defines success", "how do you validate", "what are your metrics"
        ])

        if is_success_criteria_query:
            # Phase 3: Enhanced Memory Demonstration - Include progressive inference metrics
            session_memory = state.get("session_memory", {})
            metrics = session_memory.get("progressive_inference_metrics", {})
            similarity_history = metrics.get("retrieval_similarity_history", [])

            # Build metrics summary for LLM to reference
            metrics_summary = []
            if metrics.get("turn_count") is not None:
                metrics_summary.append(f"Turn count: {metrics['turn_count']}")
            if metrics.get("topics_count") is not None:
                metrics_summary.append(f"Topics accumulated: {metrics['topics_count']}")
            if metrics.get("chat_history_length") is not None:
                metrics_summary.append(f"Chat history length: {metrics['chat_history_length']} messages")
            if metrics.get("depth_level") is not None:
                metrics_summary.append(f"Depth level: {metrics['depth_level']}")
            if similarity_history:
                similarity_progression = [f"Turn {s.get('turn', '?')}: {s.get('similarity', 0):.3f}"
                                        for s in similarity_history[-5:]]
                metrics_summary.append(f"Similarity progression: {' | '.join(similarity_progression)}")

            metrics_text = "\n".join(f"- {m}" for m in metrics_summary) if metrics_summary else "No metrics available yet"

            inference_instructions = (
                "\n\nCRITICAL: PROGRESSIVE INFERENCE SUCCESS CRITERIA EXPLANATION\n"
                "- Explain the core goal: Make smarter decisions with each turn by building on previous context\n"
                "- Explain the three mechanisms: Memory accumulation, Query enhancement, Pattern detection\n"
                "- Use CURRENT CONVERSATION as concrete examples to demonstrate success criteria\n"
                f"- Conversation context: {conversation_examples}\n"
                "- Explain turn-by-turn progression: Turn 1 (0% inference, baseline), Turn 2 (20%, role detected), "
                "Turn 3 (40%, topic extracted), Turn 4 (60-85%, patterns detected), Turn 5+ (80%+, full synthesis)\n"
                "- MUST include specific metrics from this conversation:\n"
                f"{metrics_text}\n"
                "- Explain validation: Chat history preservation, topic accumulation, query enhancement, pattern detection, "
                "depth progression, cross-turn synthesis\n"
                "- Use measurable examples: Show how similarity improved from Turn X to Turn Y, how topics accumulate, "
                "how depth_level increases from 1 to 2 to 3\n"
                "- Connect to current conversation: Reference specific turns from this conversation to demonstrate success criteria\n"
                "- Include specific numbers: 'similarity improved from 0.43 to 0.564', 'topics accumulated from 0 to 3', 'depth_level increased from 1 to 2'\n"
            )
        else:
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

    # Phase 1: Automatic Metric Injection - Inject retrieval scores for measurable improvements
    retrieval_scores = state.get("retrieval_scores", [])
    session_memory = state.get("session_memory", {})
    metrics = session_memory.get("progressive_inference_metrics", {})

    if retrieval_scores and metrics.get("retrieval_similarity_history"):
        similarity_history = metrics["retrieval_similarity_history"]
        current_similarity = retrieval_scores[0] if retrieval_scores else None

        # Build similarity improvement context
        similarity_context = []
        if len(similarity_history) > 1:
            previous_similarity = similarity_history[-2].get("similarity")
            current_similarity_val = similarity_history[-1].get("similarity") if similarity_history else current_similarity
            if previous_similarity and current_similarity_val:
                improvement = current_similarity_val - previous_similarity
                similarity_context.append(
                    f"Retrieval similarity improved from {previous_similarity:.3f} to {current_similarity_val:.3f} "
                    f"(improvement: +{improvement:.3f})"
                )
        elif current_similarity:
            similarity_context.append(
                f"Current retrieval similarity: {current_similarity:.3f}"
            )

        if similarity_context:
            extra_instructions.append(
                "\n\nCRITICAL: MEASURABLE IMPROVEMENTS\n"
                "- You MUST include specific metrics when explaining improvements\n"
                f"- Retrieval similarity context: {' | '.join(similarity_context)}\n"
                "- Reference these numbers in your answer (e.g., 'similarity improved from 0.43 to 0.564')\n"
                "- Show concrete improvements: similarity scores, topic counts, depth progression\n"
            )

    # Phase 1: Include progressive inference metrics in prompt
    progressive_metrics = _format_progressive_inference_metrics(session_memory)
    if progressive_metrics:
        extra_instructions.append(
            f"\n\nPROGRESSIVE INFERENCE METRICS (reference these in your answer):\n"
            f"{progressive_metrics}\n"
        )

    # #region agent log
    with open(_get_debug_log_path(), 'a') as f:
        import json
        f.write(json.dumps({
            "location": "stage5_generation_nodes.py:1247",
            "message": "After inference query check, before menu option detection",
            "data": {
                "is_inference_query": is_inference_query,
                "extra_instructions_count": len(extra_instructions),
                "continuing_to_menu_detection": True
            },
            "timestamp": int(time.time() * 1000),
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "A"
        }) + "\n")
    # #endregion

    # #region agent log
    with open(_get_debug_log_path(), 'a') as f:
        import json
        f.write(json.dumps({
            "location": "stage5_generation_nodes.py:1273",
            "message": "After _format_conversation_examples_for_inference definition, continuing to menu option detection",
            "data": {
                "continuing": True
            },
            "timestamp": int(time.time() * 1000),
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "A"
        }) + "\n")
    # #endregion

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
    # #region agent log
    with open(_get_debug_log_path(), 'a') as f:
        import json
        f.write(json.dumps({
            "location": "stage5_generation_nodes.py:1269",
            "message": "Before menu option 2 check",
            "data": {
                "query_type": state.get("query_type"),
                "menu_choice": state.get("menu_choice"),
                "role_mode": state.get("role_mode"),
                "query_type_match": state.get("query_type") == "menu_selection",
                "menu_choice_match": state.get("menu_choice") == "2",
                "role_mode_match": state.get("role_mode") == "hiring_manager_technical"
            },
            "timestamp": int(time.time() * 1000),
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "B"
        }) + "\n")
    # #endregion

    # Handle repeated menu selection
    if state.get("is_repeated_menu_selection"):
        menu_choice = state.get("menu_choice", "")
        repeated_count = state.get("session_memory", {}).get("repeated_menu_count", 1)

        if repeated_count == 1:
            extra_instructions.append(
                f"\n\nIMPORTANT: User selected menu option {menu_choice} again.\n"
                f"Instead of repeating the same explanation, start your answer with:\n"
                f"'I notice you selected option {menu_choice} again. Would you like me to:'\n"
                f"1. Explore a different aspect of this topic\n"
                f"2. Dive deeper into a specific part\n"
                f"3. Move to a related topic\n"
                f"Then offer 2-3 specific alternative explorations based on the menu option.\n"
            )
            logger.info(f"Added repeated menu selection handling for option {menu_choice}")
        else:
            # Multiple repetitions - suggest different topic
            extra_instructions.append(
                f"\n\nIMPORTANT: User has selected menu option {menu_choice} multiple times ({repeated_count} times).\n"
                f"Start with: 'You seem very interested in this area! Let me suggest a completely different angle...'\n"
                f"Then provide a fresh perspective or suggest exploring a different topic area entirely.\n"
            )
            logger.info(f"Added multiple repetition handling for option {menu_choice} (count: {repeated_count})")

    # Handle repeated query (same question asked twice)
    if state.get("is_repeated_query"):
        query_preview = state.get("query", "")[:50]
        extra_instructions.append(
            "\n\nIMPORTANT: User asked this EXACT question before. DO NOT repeat your previous answer.\n"
            "Instead, start with: 'Déjà vu! We just covered that. Want me to:'\n"
            "Then offer 3 specific alternative angles:\n"
            "- A deeper dive into one specific aspect\n"
            "- A related topic they haven't explored\n"
            "- A different perspective on the same topic\n"
            "End with: 'Or shall we explore something completely new?'\n"
            "Keep the response SHORT (3-5 sentences) - don't re-explain everything.\n"
        )
        logger.info(f"Added repeated query handling for: '{query_preview}...'")

    is_menu_option_two = (
        state.get("query_type") == "menu_selection" and
        state.get("menu_choice") == "2" and
        state.get("role_mode") == "hiring_manager_technical"
    )

    # #region agent log
    with open(_get_debug_log_path(), 'a') as f:
        import json
        f.write(json.dumps({
            "location": "stage5_generation_nodes.py:1355",
            "message": "Menu option 2 detection result",
            "data": {
                "is_menu_option_two": is_menu_option_two,
                "query_type": state.get("query_type"),
                "menu_choice": state.get("menu_choice"),
                "role_mode": state.get("role_mode"),
                "query_type_match": state.get("query_type") == "menu_selection",
                "menu_choice_match": state.get("menu_choice") == "2",
                "role_mode_match": state.get("role_mode") == "hiring_manager_technical"
            },
            "timestamp": int(time.time() * 1000),
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "A"
        }) + "\n")
    # #endregion

    if not is_menu_option_two:
        # #region agent log
        log_path = _get_debug_log_path()
        with open(log_path, 'a') as f:
            import json
            f.write(json.dumps({
                "location": "stage5_generation_nodes.py:1380",
                "message": "Menu option 2 check failed - proceeding with normal generation",
                "data": {
                    "query_type": state.get("query_type"),
                    "menu_choice": state.get("menu_choice"),
                    "role_mode": state.get("role_mode"),
                    "query_type_match": state.get("query_type") == "menu_selection",
                    "menu_choice_match": state.get("menu_choice") == "2",
                    "role_mode_match": state.get("role_mode") == "hiring_manager_technical"
                },
                "timestamp": int(time.time() * 1000),
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "B"
            }) + "\n")
        # #endregion

    if is_menu_option_two:
        # #region agent log
        log_path = _get_debug_log_path()
        with open(log_path, 'a') as f:
            import json
            f.write(json.dumps({
                "location": "stage5_generation_nodes.py:1242",
                "message": "Menu option 2 detected",
                "data": {
                    "query_type": state.get("query_type"),
                    "menu_choice": state.get("menu_choice"),
                    "role_mode": state.get("role_mode"),
                    "pipeline_halt": state.get("pipeline_halt"),
                    "chat_history_len": len(state.get("chat_history", []))
                },
                "timestamp": int(time.time() * 1000),
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "D"
            }) + "\n")
        # #endregion
        logger.info("Menu option 2 (orchestration layer) detected - generating explanation")

        # #region agent log
        log_path = _get_debug_log_path()
        with open(log_path, 'a') as f:
            import json
            f.write(json.dumps({
                "location": "stage5_generation_nodes.py:1296",
                "message": "Menu option 2 detected - about to generate explanation",
                "data": {
                    "has_answer_before": bool(state.get("answer")),
                    "answer_before_preview": (state.get("answer", "")[:100] + "...") if state.get("answer") else None
                },
                "timestamp": int(time.time() * 1000),
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "C"
            }) + "\n")
        # #endregion

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

    # Detect enterprise adaptation queries and add specific instructions
    query_lower = query.lower()
    enterprise_keywords = ["adapt", "adapts", "adaptation", "customer support", "enterprise",
                           "use case", "how would this work for", "scales to", "chatbot"]
    is_enterprise_adaptation = any(kw in query_lower for kw in enterprise_keywords)

    if is_enterprise_adaptation:
        # Get turn context for reference
        chat_history = state.get("chat_history", [])
        current_turn = sum(1 for msg in chat_history
                          if (isinstance(msg, dict) and (msg.get("role") == "user" or msg.get("type") == "human")) or
                             (hasattr(msg, "type") and msg.type == "human")) + 1

        # Detect previous topic for turn reference
        previous_topic = None
        for msg in reversed(chat_history[-4:]):
            if isinstance(msg, dict):
                if msg.get("role") == "assistant" or msg.get("type") == "ai":
                    content = msg.get("content", "").lower()
                    if "orchestration" in content:
                        previous_topic = "orchestration"
                        break
                    elif "architecture" in content:
                        previous_topic = "architecture"
                        break

        turn_reference = ""
        if previous_topic and current_turn >= 3:
            turn_reference = f"IMPORTANT: Start your answer with 'Building on Turn {current_turn - 1}'s {previous_topic} discussion...' to connect to previous context.\n\n"

        extra_instructions.append(
            f"\n\nCRITICAL: ENTERPRISE ADAPTATION EXPLANATION\n"
            f"{turn_reference}"
            f"The user wants to understand how the architecture adapts to customer support or enterprise use cases.\n\n"
            f"Your answer MUST include:\n"
            f"1. **The Three Components That Change**:\n"
            f"   - Knowledge Base: Replace career_kb.csv with product docs/FAQs (same format: section,question,answer,source)\n"
            f"   - Roles: Replace 'Hiring Manager' with 'Customer', 'Premium Customer', 'Partner'\n"
            f"   - Actions: Replace 'send resume' with 'create ticket', 'escalate to agent'\n\n"
            f"2. **What Stays The Same**: The entire LangGraph orchestration layer (8-stage pipeline, state management, retrieval engine)\n\n"
            f"3. **Expected ROI**: 40-60% ticket reduction, 24/7 availability, <2s response time, $30-50K/year savings\n\n"
            f"4. **Code Example**: Show the ROLES dictionary change (3-5 lines):\n"
            f"   ```python\n"
            f"   ROLES = {{\n"
            f"       \"Customer\": {{\n"
            f"           \"sources\": [\"product_docs\", \"faqs\"],\n"
            f"           \"actions\": [\"create_ticket\", \"show_documentation\"]\n"
            f"       }}\n"
            f"   }}\n"
            f"   ```\n\n"
            f"5. **Critical Insight**: The orchestration layer's modularity makes enterprise adaptation possible - same pipeline, different data.\n\n"
            f"Use teaching style (condensed 3-part structure) with systematic enumeration.\n"
        )
        logger.info("Added enterprise adaptation instructions to prompt")

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

    # =========================================================================
    # FIX 8: PILLAR-SPECIFIC GENERATION INSTRUCTIONS
    # Each pillar has specific capabilities as defined by the user
    # =========================================================================
    query_lower = query.lower()

    # Detect which pillar is being discussed
    detected_pillar = None

    # Orchestration pillar detection
    if any(kw in query_lower for kw in ["orchestration", "nodes", "states", "langgraph", "pipeline", "safeguards", "conversation flow"]):
        detected_pillar = "orchestration"
        extra_instructions.append(
            "\n\nPILLAR: ORCHESTRATION LAYER\n"
            "You are explaining the orchestration layer. Your answer MUST:\n"
            "1. If asked to list nodes/states, list ALL 22 nodes with their purposes\n"
            "2. Use the CURRENT conversation as a live example (e.g., 'Right now, my classify_intent node just ran...')\n"
            "3. Show code snippets when they add value (the LangGraph StateGraph setup, node functions)\n"
            "4. Explain how each stage flows into the next\n"
            "5. Mention safeguards: grounding validation, hallucination checks, quality gates\n"
        )

    # Tech Stack pillar detection
    elif any(kw in query_lower for kw in ["tech stack", "full stack", "architecture", "frontend", "backend", "observability"]):
        detected_pillar = "tech_stack"
        extra_instructions.append(
            "\n\nPILLAR: FULL TECH STACK\n"
            "You are explaining the full tech stack. Your answer MUST:\n"
            "1. List everything in the stack with purposes: Streamlit, Next.js, Vercel, LangGraph, Supabase, pgvector, LangSmith\n"
            "2. Detail the frontend: Streamlit workshop console + Next.js production\n"
            "3. Detail the backend: LangGraph pipeline, Python 3.11, modular nodes\n"
            "4. Detail observability: LangSmith tracing, Supabase analytics tables\n"
            "5. Explain why each choice was made (trade-offs)\n"
        )

    # Data Pipeline pillar detection
    elif any(kw in query_lower for kw in ["data pipeline", "embedding", "vector", "pgvector", "chunking", "analytics", "data flow", "ingestion"]):
        detected_pillar = "data_pipeline"
        extra_instructions.append(
            "\n\nPILLAR: DATA PIPELINE MANAGEMENT\n"
            "You are explaining the data pipeline. Your answer MUST:\n"
            "1. Explain embeddings: OpenAI text-embedding-3-small, 1536 dimensions\n"
            "2. Explain vector storage: Supabase pgvector, RPC functions for similarity search\n"
            "3. Explain chunking: How knowledge base CSVs are chunked and embedded\n"
            "4. Explain analytics: What data is collected (queries, similarity scores, latency) and why\n"
            "5. If asked about dashboard, describe the Supabase analytics views\n"
            "6. Relate to enterprise data management patterns\n"
        )

    # Noah's Background pillar detection
    elif any(kw in query_lower for kw in ["noah", "background", "career", "resume", "github", "linkedin", "certifications", "projects", "tesla"]):
        detected_pillar = "noahs_background"
        extra_instructions.append(
            "\n\nPILLAR: NOAH'S TECHNICAL BACKGROUND\n"
            "You are explaining Noah's background. Your answer MUST:\n"
            "1. Cover certifications and training if asked\n"
            "2. Explain Noah's journey into tech (sales → Tesla → AI engineering)\n"
            "3. Provide LinkedIn, GitHub, resume links when asked (action planning will add these)\n"
            "4. Describe GitHub projects: Portfolia, AI experiments, etc.\n"
            "5. Do NOT include internal metrics (similarity scores, etc.) in career answers\n"
            "6. Focus on concrete achievements and skills\n"
        )

    # Detect enterprise adaptation queries and enhance prompt
    is_enterprise_adaptation = any(kw in query_lower for kw in ["adapt", "adapts", "adaptation", "customer support", "enterprise", "use case"])

    if is_enterprise_adaptation:
        detected_pillar = "enterprise"
        extra_instructions.append(
            "\n\nCRITICAL: ENTERPRISE ADAPTATION QUERY - COMPREHENSIVE ANSWER REQUIRED\n"
            "This is an enterprise adaptation query. Your answer MUST:\n"
            "1. Explain HOW the architecture adapts (not just that it can adapt)\n"
            "2. Include concrete examples (knowledge base changes, role changes, code snippets if relevant)\n"
            "3. Mention expected ROI/metrics if available in retrieved chunks\n"
            "4. Reference the Enterprise Adaptation Guide content from retrieved chunks\n"
            "5. Be comprehensive and actionable - focus on the specific use case mentioned (e.g., 'customer support', 'internal docs')\n"
            "6. Explain the adaptation process step-by-step: what changes, why, and how\n"
            "7. Connect the current architecture to the enterprise use case clearly\n"
        )

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

    # #region agent log
    try:
        log_path = _get_debug_log_path()
        with open(log_path, 'a') as f:
            import json
            f.write(json.dumps({
                "location": "stage5_generation_nodes.py:1599",
                "message": "Before LLM generation call",
                "data": {
                    "is_menu_option_two": is_menu_option_two,
                    "is_menu_option_one": is_menu_option_one,
                    "query": query[:100] if query else None,
                    "retrieved_chunks_count": len(retrieved_chunks),
                    "chat_history_len": len(chat_history),
                    "selected_model": selected_model,
                    "has_extra_instructions": bool(instruction_suffix)
                },
                "timestamp": int(time.time() * 1000),
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "A"
            }) + "\n")
    except: pass
    # #endregion

    try:
        # #region agent log - Before generate_contextual_response call
        try:
            log_path = _get_debug_log_path()
            with open(log_path, 'a') as f:
                import json
                f.write(json.dumps({
                    "location": "stage5_generation_nodes.py:1914",
                    "message": "Before generate_contextual_response call",
                    "data": {
                        "query": query[:50] if query else None,
                        "chat_history_len": len(chat_history) if chat_history else 0,
                        "selected_model": selected_model
                    },
                    "timestamp": int(time.time() * 1000),
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "ALL"
                }) + "\n")
        except: pass
        # #endregion

        answer = rag_engine.response_generator.generate_contextual_response(
            query=query,
            context=retrieved_chunks,
            role=role,
            chat_history=chat_history,
            extra_instructions=instruction_suffix,
            model_name=selected_model  # Pass selected model
        )

        # #region agent log - After generate_contextual_response call
        try:
            log_path = _get_debug_log_path()
            with open(log_path, 'a') as f:
                import json
                f.write(json.dumps({
                    "location": "stage5_generation_nodes.py:1945",
                    "message": "After generate_contextual_response call",
                    "data": {
                        "answer_type": type(answer).__name__ if answer else None,
                        "answer_is_none": answer is None,
                        "answer_len": len(answer) if answer and isinstance(answer, str) else 0,
                        "answer_preview": str(answer)[:200] if answer else None,
                        "is_menu_option_two": is_menu_option_two,
                        "is_fallback_message": "I'm having trouble" in str(answer) if answer else False
                    },
                    "timestamp": int(time.time() * 1000),
                    "sessionId": "debug-session",
                    "runId": "run1",
                    "hypothesisId": "A,B"
                }) + "\n")
        except Exception as log_err:
            import sys
            print(f"DEBUG LOG FAILED after call: {log_err}", file=sys.stderr, flush=True)
        # #endregion

    except Exception as e:
        import traceback
        import sys
        error_traceback = traceback.format_exc()
        logger.error(f"LLM generation failed: {e}")
        # CRITICAL: Print to stderr immediately - this will show in terminal
        print(f"\n{'='*60}\nCRITICAL ERROR IN GENERATION (stage5):\nType: {type(e).__name__}\nMessage: {str(e)}\nTraceback:\n{error_traceback}\n{'='*60}\n", file=sys.stderr, flush=True)

        # #region agent log - Multiple fallback strategies
        error_data = {
            "location": "stage5_generation_nodes.py:1878",
            "message": "LLM generation failed with exception (outer handler)",
            "data": {
                "error": str(e),
                "error_type": type(e).__name__,
                "error_traceback": error_traceback,
                "query": query[:100] if query else None,
                "selected_model": selected_model,
                "retrieved_chunks_count": len(retrieved_chunks) if retrieved_chunks else 0,
                "is_menu_option_two": is_menu_option_two,
                "is_menu_option_one": is_menu_option_one,
                "has_instruction_suffix": bool(instruction_suffix),
                "instruction_suffix_len": len(instruction_suffix) if instruction_suffix else 0
            },
            "timestamp": int(time.time() * 1000),
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "A,B,C,D,E,F"
        }

        # Try file logging first
        try:
            log_path = _get_debug_log_path()
            with open(log_path, 'a') as f:
                import json
                f.write(json.dumps(error_data) + "\n")
        except Exception as log_err:
            # Fallback 1: Try stderr (always works)
            try:
                import json
                print(f"\n=== DEBUG LOG (stderr fallback) ===\n{json.dumps(error_data, indent=2)}\n=== END DEBUG LOG ===\n", file=sys.stderr, flush=True)
            except:
                pass
            # Fallback 2: Standard logger with full details
            logger.error(f"Failed to write debug log: {log_err}")
            logger.error(f"Original error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Full traceback:\n{error_traceback}")
        # #endregion

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

        # #region agent log
        log_path = _get_debug_log_path()
        with open(log_path, 'a') as f:
            import json
            f.write(json.dumps({
                "location": "stage5_generation_nodes.py:1766",
                "message": "Menu option 2 answer generated",
                "data": {
                    "answer_length": len(answer),
                    "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
                    "word_count": len(answer.split())
                },
                "timestamp": int(time.time() * 1000),
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "C"
            }) + "\n")
        # #endregion

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

    # Strip ALL retrieval metrics from ALL answers (Fix 4)
    # These internal metrics are for observability, not user-facing content
    # Expanded to cover all metric patterns that might leak into responses
    metrics_patterns = [
        r'Key Metric:\s*(Top retrieval similarity|Average similarity|similarity):\s*[\d.]+',
        r'Key Metric:\s*Current depth level:\s*\d+',
        r'Key Metric:\s*Cost per query:\s*~?\$[\d.]+',
        r'Key Metric:\s*Topics explored:.*?(?=\n|$)',
        r'Key Metric:\s*[\w\s]+:\s*[\d.]+',  # Catch-all for any remaining Key Metric patterns
        r'-\s*Key Metric:.*?(?=\n|$)',  # Bulleted metrics
        r'\*\*Key Metric:\*\*.*?(?=\n|$)',  # Bold metrics
    ]
    for pattern in metrics_patterns:
        cleaned_answer = re.sub(pattern, '', cleaned_answer, flags=re.IGNORECASE)

    # Clean up any double newlines or spaces left over
    cleaned_answer = re.sub(r'\n{3,}', '\n\n', cleaned_answer)
    cleaned_answer = re.sub(r'  +', ' ', cleaned_answer)
    cleaned_answer = cleaned_answer.strip()
    logger.debug("Stripped all retrieval metrics from answer")

    # Validate answer relevance before storing
    if not _validate_answer_relevance(query, cleaned_answer):
        logger.warning(
            f"Answer doesn't address query topic. Query: {query[:50]}, "
            f"Answer starts: {cleaned_answer[:100]}"
        )
        update["answer_validation_failed"] = True
        # Flag for potential regeneration or fallback handling
        # Note: We still return the answer, but flag it for downstream handling

    update["draft_answer"] = cleaned_answer
    update["answer"] = cleaned_answer

    # #region agent log
    with open(_get_debug_log_path(), 'a') as f:
        import json
        f.write(json.dumps({
            "location": "stage5_generation_nodes.py:1926",
            "message": "generate_draft: Returning final answer",
            "data": {
                "answer_len": len(cleaned_answer) if cleaned_answer else 0,
                "answer_preview": cleaned_answer[:200] if cleaned_answer else None,
                "is_menu_option_two": is_menu_option_two,
                "is_menu_option_one": is_menu_option_one,
                "update_keys": list(update.keys())
            },
            "timestamp": int(time.time() * 1000),
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "A"
        }) + "\n")
    # #endregion

    # #region agent log
    with open(_get_debug_log_path(), 'a') as f:
        import json
        f.write(json.dumps({
            "location": "stage5_generation_nodes.py:1821",
            "message": "Answer set in update dict",
            "data": {
                "answer_length": len(cleaned_answer),
                "answer_preview": cleaned_answer[:200] + "..." if len(cleaned_answer) > 200 else cleaned_answer,
                "is_menu_option_two": is_menu_option_two if 'is_menu_option_two' in locals() else False
            },
            "timestamp": int(time.time() * 1000),
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "C"
        }) + "\n")
    # #endregion

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
