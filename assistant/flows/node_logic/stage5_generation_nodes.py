"""Generation pipeline nodes - LLM response creation and hallucination prevention.

This module handles answer generation with the LLM:
1. generate_draft → LLM creates answer using retrieved context
   (includes verbatim-copy detection on the output)
2. hallucination_check → verifies checkable claims (percentages, dollar
   amounts, URLs) against retrieved sources; HALLUCINATION_GATE=log|enforce|off

Design Principles:
- SRP: Each function handles one generation concern
- Defensibility: Graceful degradation on LLM failures
- Observability: Detailed logging for generation quality

Latency is dominated by the LLM call in generate_draft; both validation
passes are in-memory regex work.
"""

import logging
import os
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
- Conversation Phase: {conversation_phase}
- Depth Level: {depth_level} (1=overview, 2=guided detail, 3=deep dive)

**Topics We've Covered:**
{topics_text}

**Turn-by-Turn Summary:**
{turn_summary_text}
{similarity_text}

**How My Inference Has Improved:**
1. **Turn 1**: I greeted you and started tracking context
2. **Each Turn**: I compose your query with conversation context, improving retrieval similarity
3. **Memory Accumulation**: Topics, entities, and affinity scores build up in `session_memory`
4. **Progressive Depth**: As we talk more, `depth_level` increases -- I provide more detail

**What I'm Tracking:**
- Conversation patterns (general → specific, surface → deep)
- Quality flags (if answers get repetitive, I redirect)
- Phase transitions (discovery → exploration → synthesis)

This is the progressive inference that makes me smarter with each turn. The orchestration layer's intelligence comes from stateless nodes working with stateful memory -- each node executes independently, but memory creates the thread that connects them into progressively smarter behavior."""

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
    return "architecture"


def _build_engagement_context(state: dict) -> str | None:
    """Build a concise conversation-state hint for the LLM.

    The personality prompt already contains all pacing, formatting, and
    engagement rules.  This function only supplies the dynamic facts the
    LLM cannot infer from chat_history alone.
    """
    msg_count = state.get("message_count", 0)
    logger.info("DIAG _build_engagement_context CALLED msg_count=%d", msg_count)

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
        f"\nCONVERSATION STATE: message #{msg_count} | phase: {phase}"
    )

    # First-message framing: anchor Noah's identity with Danaher structure
    if msg_count == 1:
        hint += (
            "\nCRITICAL FIRST LINE: Your response MUST begin with "
            "'Noah is a software developer specializing in machine learning "
            "models and generative AI applications.' Do not paraphrase "
            "this. Use these exact words as your opening sentence."
            "\n\nThen follow this structure for the rest of the message:"
            "\n1. SELF-INTRO (Danaher structure): Name the PROBLEM with "
            "portfolio sites (they're static — you read, you leave, no "
            "follow-up questions). Then the approach: Noah built a "
            "conversational one. You're the example — 22-node RAG pipeline, "
            "semantic search, intent classification, grounding validation. "
            "Built from scratch to demonstrate production AI architecture."
            "\n2. ML WORK (show range through paired projects): The attrition "
            "pair — logistic regression hit 94.75% accuracy on an imbalanced "
            "dataset, Naive Bayes trades accuracy for catching more leavers. "
            "The segmentation pair — decision trees for interpretable rules, "
            "K-means for discovering structure the labels missed. Frame it as: "
            "each pair asks the same question two different ways."
            "\n3. VELOCITY: He started coding in August 2024. Less than two "
            "years from zero to shipping production systems while maintaining "
            "top 10% sales performance at Tesla."
            "\nDo NOT mention certifications in this first message. "
            "Do NOT list project names as a feature dump — frame them "
            "by the problem they solve or the insight they reveal."
            "\n\nREQUIRED ENDING: End with TWO lines:"
            "\n1. A 'why are you here' question: \"What brings you here?\""
            "\n2. A knowledge hook about the segmentation pair or another "
            "uncovered project — make it specific and intriguing."
            "\nDo NOT include any links (GitHub, LinkedIn, or otherwise) "
            "in this response."
        )
    elif msg_count == 2:
        # Detect if user already explained why they're here (referral, purpose, etc.)
        _user_query = (state.get("original_query", "") or state.get("query", "") or "").lower()
        _answered_why = any(phrase in _user_query for phrase in [
            "told me to", "sent me", "check this out", "check it out",
            "referred me", "recommended", "noah told", "noah sent",
            "friend told", "someone told", "heard about",
            "looking for", "hiring", "we're evaluating", "evaluating candidates",
            "interested in", "want to see", "came from", "found you on",
        ])
        if _answered_why:
            hint += (
                "\nIMPORTANT: The user already explained why they're here. Do NOT ask "
                "\"What brings you here?\" again. Acknowledge their reason briefly, "
                "then show the work. End with a knowledge hook about a specific project."
                "\nCAPTURE NOTE: A reach-out offer will be appended after your response. "
                "Do NOT add your own reach-out offer — it will be handled automatically."
            )
        else:
            hint += (
                "\nREQUIRED ENDING: End with a knowledge hook statement about a specific "
                "project or topic. Keep it to one line."
                "\nNEVER end with 'Want X or Y?' or any sentence offering two options with 'or'. "
                "No menus."
                "\nCAPTURE NOTE: A reach-out offer will be appended after your response. "
                "Do NOT add your own reach-out offer — it will be handled automatically."
            )
    else:
        hint += (
            "\nREQUIRED ENDING: End with a knowledge hook statement about an uncovered "
            "project or topic. Keep it to one line."
            "\nNEVER end with 'Want X or Y?' or any sentence offering two options with 'or'. "
            "No menus."
            "\nCAPTURE NOTE: A reach-out offer will be appended after your response. "
            "Do NOT add your own reach-out offer — it will be handled automatically."
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


def select_model_for_task(state: ConversationState) -> Optional[str]:
    """Select a model override for generation.

    All generation currently uses the default Claude Sonnet 4.5 configured
    in rag_factory.py, so this always returns None. Kept as the single
    extension point if per-query model routing ever comes back.
    """
    return None


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

    # RUNTIME AWARENESS: Detect technical deep dive requests (available to all visitors)
    runtime_awareness_triggered = False
    runtime_content_block = None
    query_lower = query.lower()

    # Architecture questions
    if any(kw in query_lower for kw in ["architecture", "how do you work", "how does this work", "system design", "how are you built"]):
        if "rag" in query_lower or "retrieval" in query_lower or "search" in query_lower:
            runtime_content_block = content_blocks.rag_pipeline_explanation()
            runtime_awareness_triggered = True
            logger.info("Runtime awareness: RAG pipeline explanation triggered")
        elif "flow" in query_lower or "pipeline" in query_lower or "nodes" in query_lower:
            runtime_content_block = content_blocks.conversation_flow_diagram()
            runtime_awareness_triggered = True
            logger.info("Runtime awareness: Conversation flow diagram triggered")
        else:
            runtime_content_block = content_blocks.architecture_stack_explanation()
            runtime_awareness_triggered = True
            logger.info("Runtime awareness: Architecture stack explanation triggered")

    # Performance questions
    elif any(kw in query_lower for kw in ["performance", "latency", "speed", "how fast", "metrics", "p95", "p99"]):
        runtime_content_block = content_blocks.performance_metrics_table()
        runtime_awareness_triggered = True
        logger.info("Runtime awareness: Performance metrics table triggered")

    # Code questions
    elif any(kw in query_lower for kw in ["show me code", "show code", "show me the code", "retrieval code", "how do you retrieve"]):
        runtime_content_block = content_blocks.code_example_retrieval_method()
        runtime_awareness_triggered = True
        logger.info("Runtime awareness: Code example triggered")

    # SQL/query questions
    elif any(kw in query_lower for kw in ["sql", "query", "vector search", "pgvector", "how do you search"]):
        runtime_content_block = content_blocks.pgvector_query_example()
        runtime_awareness_triggered = True
        logger.info("Runtime awareness: pgvector query example triggered")

    # Cost questions
    elif any(kw in query_lower for kw in ["cost", "expensive", "pricing", "how much", "budget"]):
        runtime_content_block = content_blocks.cost_analysis_table()
        runtime_awareness_triggered = True
        logger.info("Runtime awareness: Cost analysis table triggered")

    # Scaling questions
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
            "Go deeper. New details only, do not repeat previous answer."
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

    if runtime_awareness_triggered and runtime_content_block:
        extra_instructions.append(
            f"RUNTIME AWARENESS: Reference this self-referential data in your "
            f"explanation, woven naturally.\n\n{runtime_content_block}"
        )

    # ── Kept as-is: job details gathering ────────────────────────────
    from assistant.flows.node_logic.util_resume_distribution import should_gather_job_details, get_job_details_prompt
    if should_gather_job_details(state):
        extra_instructions.append(get_job_details_prompt())

    # ── PIPELINE TELEMETRY — inject live request data for self-referential queries
    if state.get("is_self_referential"):
        retrieval_scores = state.get("retrieval_scores", [])
        top_score = max(retrieval_scores) if retrieval_scores else 0.0
        chunk_sources = []
        for c in retrieved_chunks[:5]:
            src = c.get("metadata", {}).get("source", "unknown") if isinstance(c, dict) else "unknown"
            chunk_sources.append(src)
        telemetry = (
            "LIVE PIPELINE DATA FOR THIS REQUEST (use this to walk the user "
            "through what just happened, referencing their actual conversation):\n"
            f"- Intent classified as: {state.get('message_intent', 'knowledge_query')}\n"
            f"- Chunks retrieved: {len(retrieved_chunks)}\n"
            f"- Top similarity score: {top_score:.3f}\n"
            f"- Chunk sources: {', '.join(chunk_sources) if chunk_sources else 'none'}\n"
            f"- Chain-of-thought triggered: {state.get('cot_triggered', False)}\n"
            f"- Grounding status: {state.get('grounding_status', 'unknown')}\n"
            f"- Message count this session: {len(chat_history) // 2 if chat_history else 0}\n"
            f"- Role mode: {state.get('role_mode', 'unknown')}\n"
            f"- Self-knowledge injection: {any(c.get('metadata', dict()).get('source') == 'self_knowledge' for c in retrieved_chunks if isinstance(c, dict))}\n"
        )
        extra_instructions.append(telemetry)
        logger.info("Pipeline telemetry injected for self-referential query")

    logger.info("DIAG generate_draft extra_instructions=%s", extra_instructions)
    # Build the instruction suffix
    instruction_suffix = " ".join(extra_instructions) if extra_instructions else None
    base_instruction_suffix = instruction_suffix

    # Select appropriate model for this task
    selected_model = select_model_for_task(state)

    # Store model selection in state for analytics
    state.setdefault("analytics_metadata", {})["selected_model"] = selected_model

    # Generate response with LLM
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
        answer = (
            "I'm having trouble generating a response right now. "
            "Please try rephrasing your question or ask something else!"
        )

    # Apply post-processing to ALL generated answers
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

        # Markdown formatting (bold, headers, images) is preserved for the frontend
        # react-markdown renderer. Only strip markdown in menu option contexts.

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


# Facts stated verbatim in the generation system prompt
# (assistant/core/response_generator.py) are legitimate grounding alongside
# retrieved chunks — the model is *supposed* to know them without retrieval.
# Keep in sync with the prompt until a shared constants module exists.
_PROMPT_GROUNDED_FACTS = """
94.75% 83% 58% 48% 81% 10% 47% 26% 50%
22 nodes 1536 dimensions 0.50 0.30 150ms 2021
https://github.com/iNoahCodeGuy
https://www.linkedin.com/in/noah-de-la-calzada-250412358/
https://github.com/iNoahCodeGuy/portfolia-backend
https://github.com/iNoahCodeGuy/Predicting-Employee-Attrition-Using-Logistic-Regression
https://github.com/iNoahCodeGuy/Predicting-Employee-Attrition-Using-Naive-Bayes
https://github.com/iNoahCodeGuy/Customer_Segmentation_decision_trees
""".lower()

_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?%")
_DOLLAR_RE = re.compile(r"\$[\d,]+(?:\.\d+)?")
_URL_RE = re.compile(r"https?://[^\s)\]}\"']+")


def _find_unsupported_claims(answer: str, chunks: List[Dict[str, Any]]) -> List[str]:
    """Return checkable claims in the answer that no source supports.

    Checks the high-signal, low-false-positive claim types: percentages,
    dollar amounts, and URLs. Support = retrieved chunk text or the facts
    the generation prompt itself states.
    """
    corpus = (
        " ".join(chunk.get("content", "") for chunk in chunks).lower()
        + " "
        + _PROMPT_GROUNDED_FACTS
    )
    findings: List[str] = []

    for pct in _PERCENT_RE.findall(answer):
        # "94.75%" is supported by either "94.75%" or a bare "94.75"
        if pct.lower() not in corpus and pct[:-1] not in corpus:
            findings.append(pct)

    for amount in _DOLLAR_RE.findall(answer):
        if amount.lower() not in corpus:
            findings.append(amount)

    for url in _URL_RE.findall(answer):
        cleaned = url.rstrip(".,;:!?").lower()
        if cleaned not in corpus:
            findings.append(cleaned)

    return findings


def hallucination_check(state: ConversationState) -> ConversationState:
    """Verify the draft's checkable claims against its sources.

    Deterministic (no LLM call): percentages, dollar amounts, and URLs in
    the draft must appear in the retrieved chunks or in the facts the
    generation prompt states. Anything unsupported is a finding.

    Rollout is controlled by HALLUCINATION_GATE:
    - "log" (default): findings are logged and recorded in
      analytics_metadata; the answer ships unchanged. Used to measure the
      false-positive rate before enforcement.
    - "enforce": findings replace the draft with a graceful fallback and
      set hallucination_safe=False.
    - "off": skip the check entirely.

    Performance: <5ms (regex over in-memory strings).
    """
    mode = os.getenv("HALLUCINATION_GATE", "log").lower()
    draft = state.get("draft_answer") or ""

    if mode == "off" or not draft:
        state["hallucination_safe"] = True
        return state

    # Non-RAG paths (greetings, forms, self-knowledge) carry no retrieved
    # factual sources to verify against.
    checkable_chunks = [
        c for c in state.get("retrieved_chunks", [])
        if c.get("source") != "self_knowledge"
    ]
    if not checkable_chunks:
        state["hallucination_safe"] = True
        return state

    findings = _find_unsupported_claims(draft, checkable_chunks)
    if not findings:
        state["hallucination_safe"] = True
        return state

    state.setdefault("analytics_metadata", {})["hallucination_findings"] = findings

    if mode == "enforce":
        logger.warning(f"🚫 Hallucination gate (enforce): unsupported claims {findings}")
        state["hallucination_safe"] = False
        state["draft_answer"] = (
            "I want to be careful not to overstate anything here — part of what "
            "I'd normally cite didn't come back from my knowledge base. Ask that "
            "again, or narrow it down, and I'll give you the grounded version."
        )
    else:
        logger.warning(
            f"⚠️ Hallucination gate (log): unsupported claims {findings} — shipping unchanged"
        )
        state["hallucination_safe"] = True

    return state
