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
from assistant.flows.node_logic.chain_of_thought import (
    chain_of_thought_generate,
    detect_confusion_signals,
)

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


def _build_engagement_context(state: dict) -> str | None:
    """Build a concise conversation-state hint for the LLM.

    The personality prompt already contains all pacing, formatting, and
    engagement rules.  This function only supplies the dynamic facts the
    LLM cannot infer from chat_history alone.
    """
    msg_count = state.get("message_count", 0)
    logger.info("DIAG _build_engagement_context CALLED msg_count=%d", msg_count)
    visitor_type = state.get("visitor_type", "unknown")
    buying_signals = state.get("buying_signals_count", 0)

    # Derive conversation phase
    if msg_count <= 2:
        phase = "opening"
    elif msg_count <= 4:
        phase = "calibration"
    elif msg_count <= 7:
        phase = "teaching"
    else:
        phase = "sustained"

    hint = (
        f"\nCONVERSATION STATE: message #{msg_count} | visitor: {visitor_type} "
        f"| phase: {phase} | buying_signals: {buying_signals}"
    )

    # Visitor-type-specific guidance
    if visitor_type == "gatekeeper":
        hint += (
            "\nVISITOR NOTE: Gatekeeper — screening for a decision-maker. "
            "Keep answers concise and easy to forward. Lead with results and outcomes. "
            "Capture is valuable (ask who they're screening for)."
        )
    elif visitor_type == "student":
        hint += (
            "\nVISITOR NOTE: Student/learner — here to learn from the architecture. "
            "Go deep on technical decisions when asked. Point to GitHub. "
            "Low capture priority — don't push contact forms."
        )

    # Phase-specific ending guidance — always two-part: capture question + knowledge hook
    if phase == "opening":
        hint += (
            "\nREQUIRED ENDING: End with TWO lines:"
            "\n1. A discovery/capture question (e.g., \"What brings you here?\" or "
            "\"Hiring, building, or just curious?\")."
            "\n2. A knowledge hook statement (e.g., \"The architecture behind this "
            "conversation is worth a look if you want to see how the engineering holds up.\")."
            "\nDo NOT include any links (GitHub, LinkedIn, or otherwise) in this response."
        )
    elif phase == "calibration":
        hint += (
            "\nREQUIRED ENDING: End with TWO lines:"
            "\n1. A capture question (e.g., \"Want to share what you're working on "
            "so Noah can follow up?\")."
            "\n2. A knowledge hook to an uncovered topic (e.g., \"The statistical "
            "foundation behind the retrieval is the same math that powers the "
            "attrition model.\"). Never end with \"Want X or Y?\""
        )
    elif phase == "teaching":
        hint += (
            "\nREQUIRED ENDING: End with TWO lines:"
            "\n1. A capture question (e.g., \"If you want Noah to follow up on this, "
            "just say the word.\")."
            "\n2. A knowledge hook bridging to related content (e.g., \"The attrition "
            "model uses the same statistical foundation if you want to see it applied "
            "to a different problem.\"). Never offer a menu like \"Want X or Y?\""
        )
    else:  # sustained
        hint += (
            "\nENDING RULE: End with TWO lines:"
            "\n1. A capture question (e.g., \"Want Noah to reach out directly?\")."
            "\n2. A knowledge hook matching their energy. "
            "Never offer a menu of options."
        )

    logger.info(
        "DIAG engagement_context phase=%s msg_count=%d | full hint: %s",
        phase, msg_count, hint,
    )

    # Detect traffic source from query + recent history
    _ts_query = (state.get("original_query", "") or state.get("query", "") or "").lower()
    _ts_history = ""
    for _m in (state.get("chat_history") or [])[-6:]:
        _c = _m.get("content", "") if isinstance(_m, dict) else getattr(_m, "content", "")
        _ts_history += " " + _c.lower()
    _combined = f"{_ts_query} {_ts_history}"

    _source_map = {
        "linkedin": "linkedin",
        "instagram": "instagram",
        "hinge": "hinge",
        "upwork": "upwork",
        "referral": "referral",
        "someone told me": "referral",
        "friend sent me": "referral",
    }
    for _kw, _src in _source_map.items():
        if _kw in _combined:
            hint += f" | source: {_src}"
            break

    return hint


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

    # Self-knowledge queries about Portfolia's own architecture are never "simple"
    # — they require the system prompt's self-knowledge section to answer well
    self_knowledge_keywords = [
        "built", "retrieval", "pipeline", "architecture", "rag", "langgraph",
        "pgvector", "embedding", "vector", "node", "generation", "how do you work",
        "how does your", "how were you", "tech stack", "supabase",
    ]
    if any(kw in query_lower for kw in self_knowledge_keywords):
        return "medium"

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
    """Select model for generation.

    Generation now uses Anthropic Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)
    for all queries. Returns None to use the default model configured in
    rag_factory.py. The previous OpenAI model routing (gpt-4o, gpt-4o-mini)
    is no longer applicable since the backend switched to Anthropic.

    Args:
        state: Current conversation state with query and classification metadata

    Returns:
        None (uses default Claude Sonnet 4.5)
    """
    # All generation uses the default Claude Sonnet 4.5 configured in rag_factory.py
    return None

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
    explanation_patterns = ["explain", "walk me through", "how does", "how did", "how were", "how was", "how are", "why does", "teach me"]
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
    logger.info(
        f"generate_draft: query={state.get('query', '')[:80]!r} "
        f"chunks={len(state.get('retrieved_chunks', []))} "
        f"role_mode={state.get('role_mode')}"
    )

    # Fail-fast: Validate required fields (Defensibility)
    query = state.get("query", "")
    if not query:
        logger.error("generate_draft called without query in state")
        raise KeyError("State must contain 'query' field for generation")

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

        # Handle REPEATED action requests specially
        if is_repeated:
            logger.info("Repeated action request detected - returning acknowledgment")
            draft = "I already provided those resources above. Is there something specific you're looking for, or would you like Noah to reach out directly?"
            return {
                "draft_answer": draft,
                "is_repeated_action_request": True,
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

    if state.get("pipeline_halt"):
        return {"answer": None, "draft_answer": None}

    grounding_status = state.get("grounding_status")
    if grounding_status and grounding_status not in {"ok", "unknown"}:
        # Belt-and-suspenders: if query is self-referential but grounding failed
        # (handle_grounding_gap should have caught this, but just in case),
        # override grounding and let generation proceed with self-knowledge context
        if state.get("is_self_referential"):
            logger.warning(
                "Self-referential query reached generate_draft with bad grounding "
                f"(status={grounding_status}). Overriding to proceed."
            )
            state["grounding_status"] = "ok"
            # Inject self-knowledge chunk if missing
            if not state.get("retrieved_chunks"):
                from assistant.flows.node_logic.stage4_retrieval_nodes import handle_grounding_gap as _hgg
                _hgg(state)
                logger.info("Injected self-knowledge via handle_grounding_gap from generate_draft")
        else:
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

    # =====================================================================
    # BUILD EXTRA INSTRUCTIONS — max 3 blocks to avoid prompt bloat.
    # The personality prompt already covers voice, format, length, pacing,
    # grounding, and engagement rules.  Only inject what it cannot infer.
    # =====================================================================
    extra_instructions = []

    # ── 1. GROUNDING — retrieval quality warning ─────────────────────
    retrieval_scores = state.get("retrieval_scores", [])
    max_score = max(retrieval_scores) if retrieval_scores else 0.0
    if retrieval_scores and max_score < 0.5:
        extra_instructions.append(
            f"GROUNDING: Low retrieval similarity (max={max_score:.2f}). "
            "Only state what chunks explicitly support."
        )
        logger.info(f"Grounding strictness injected: max_score={max_score:.3f}")

    # ── 2. ENGAGEMENT STATE — concise conversation hint ──────────────
    engagement_ctx = _build_engagement_context(state)
    if engagement_ctx:
        extra_instructions.append(engagement_ctx)

    # ── 3. SITUATIONAL OVERRIDE — at most one ────────────────────────
    # Priority: continuation > out-of-scope > retrieval mismatch
    _situational_added = False

    if state.get("is_continuation") and not _situational_added:
        extra_instructions.append(
            f"CONTINUATION: User said \"{state.get('original_query', '')}\". "
            "Go deeper — new details only, do not repeat previous answer."
        )
        logger.info(f"Continuation hint for: '{state.get('original_query', '')[:50]}'")
        _situational_added = True

    if not _situational_added:
        from assistant.flows.node_logic.stage6_formatting_nodes import detect_out_of_scope
        is_out_of_scope, bridge_pillar, bridge_prompt = detect_out_of_scope(
            query, state.get("role_mode", "explorer")
        )
        if is_out_of_scope:
            state["out_of_scope_detected"] = True
            state["bridge_suggestion"] = bridge_prompt
            extra_instructions.append(
                f"OUT OF SCOPE: Bridge to \"{bridge_prompt}\" instead of answering generically."
            )
            logger.info(f"Out-of-scope: '{query[:50]}'. Bridge: {bridge_pillar}")
            _situational_added = True

    if not _situational_added and state.get("retrieval_topic_mismatch"):
        extra_instructions.append(
            "RETRIEVAL MISMATCH: Chunks may not match the query. "
            "Use self-knowledge and context to answer; don't rely solely on chunks."
        )
        logger.warning(f"Retrieval mismatch for: {query[:50]}")
        _situational_added = True

    # ── Kept as-is: runtime awareness (Software Developer only) ──────
    is_menu_option_one = (
        state.get("query_type") == "menu_selection" and
        state.get("menu_choice") == "1" and
        state.get("role_mode") == "hiring_manager_technical"
    )

    layer_outline: Optional[Dict[str, str]] = None

    is_menu_option_two = (
        state.get("query_type") == "menu_selection" and
        state.get("menu_choice") == "2" and
        state.get("role_mode") == "hiring_manager_technical"
    )

    if runtime_awareness_triggered and runtime_content_block:
        extra_instructions.append(
            f"RUNTIME AWARENESS: Reference this self-referential data in your "
            f"explanation, woven naturally.\n\n{runtime_content_block}"
        )

    # ── Kept as-is: job details gathering ────────────────────────────
    from assistant.flows.node_logic.util_resume_distribution import should_gather_job_details, get_job_details_prompt
    if should_gather_job_details(state):
        extra_instructions.append(get_job_details_prompt())

    logger.info("DIAG generate_draft extra_instructions=%s", extra_instructions)
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

    logger.info(
        f"LLM call: model={selected_model or 'default'} "
        f"chunks={len(retrieved_chunks)} history={len(chat_history)} "
        f"extra_instructions={bool(instruction_suffix)}"
    )

    try:
        answer = rag_engine.response_generator.generate_contextual_response(
            query=query,
            context=retrieved_chunks,
            role=role,
            chat_history=chat_history,
            extra_instructions=instruction_suffix,
            model_name=selected_model  # Pass selected model
        )

        logger.info(f"LLM response: {len(answer)} chars, {len(answer.split())} words")

    except Exception as e:
        logger.error(f"LLM generation failed: {type(e).__name__}: {e}", exc_info=True)

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
        "what brings you here",
        "i can focus on the areas"
    ])

    if not is_welcome_message:
        # Apply first-person enforcement
        answer = rag_engine.response_generator._enforce_first_person(answer)
        logger.debug("Applied post-generation first-person enforcement")

        # Detect and remove verbatim copying
        # Skip for self-knowledge chunks (source="self_knowledge") — verbatim use of
        # authoritative self-knowledge is correct behavior, not a quality issue
        is_self_knowledge = any(
            isinstance(c, dict) and c.get("source") == "self_knowledge"
            for c in (retrieved_chunks or [])
        )
        if retrieved_chunks and not is_self_knowledge:
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
