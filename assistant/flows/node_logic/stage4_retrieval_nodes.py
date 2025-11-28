"""Retrieval pipeline nodes - pgvector search with re-ranking and grounding validation.

This module handles the retrieval phase of the conversation pipeline:
1. retrieve_chunks → Supabase pgvector search + MMR diversification in one pass
2. validate_grounding → Quality gate ensuring sufficient context before generation
3. handle_grounding_gap → Graceful fallback when retrieval confidence is low

Merged re_rank_and_dedup logic into retrieve_chunks for streamlined retrieval.

Design Principles:
- SRP: Each function handles one retrieval concern
- Defensibility: Graceful degradation on retrieval failures
- Observability: LangSmith tracing for retrieval performance
- Reliability: Never crashes the pipeline, returns empty results on failure

Performance Characteristics:
- retrieve_chunks: ~310ms typical (embedding + vector search + MMR dedup)
- validate_grounding: <1ms (threshold check)
- handle_grounding_gap: <1ms (template response)

See: docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md for full pipeline flow
"""

import logging
import re
from typing import Dict, Any, List

from assistant.state.conversation_state import ConversationState
from assistant.core.rag_engine import RagEngine
from assistant.observability.langsmith_tracer import create_custom_span
from assistant.flows.node_logic.util_conversational_edge_case_handler import generate_edge_case_response
from assistant.utils.personality_injection import should_include_personality

logger = logging.getLogger(__name__)


def _clean_chunk_qa_format(content: str) -> str:
    """Strip Q&A format from retrieved chunks to prevent LLM from echoing format.

    KB chunks often stored as "Q: ...?\nA: ..." which causes LLM to copy this format
    instead of synthesizing comprehensive answers from multiple chunks.

    This preprocessor extracts just the answer portion and cleans formatting.

    Args:
        content: Raw chunk content (may contain Q:/A: format)

    Returns:
        Cleaned content with Q&A format removed

    Example:
        >>> _clean_chunk_qa_format("Q: What is RAG?\nA: Retrieval-Augmented Generation...")
        "Retrieval-Augmented Generation..."
    """
    # Pattern 1: Extract answer from "Q: ... A: ..." format
    qa_pattern = r'Q:\s*.*?\s*A:\s*(.*)'
    match = re.search(qa_pattern, content, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
        logger.debug(f"Stripped Q&A format from chunk ({len(content)} → {len(cleaned)} chars)")
        return cleaned

    # Pattern 2: If no Q&A format, return as-is
    return content


def _extract_mentioned_files(chat_history: List[Dict]) -> List[str]:
    """Extract code file paths mentioned in previous conversation turns.

    This function identifies files that were discussed in previous turns,
    enabling hierarchical context selection that prioritizes mentioned files.

    Args:
        chat_history: List of message dicts with 'role' and 'content' keys

    Returns:
        List of unique file paths mentioned in conversation

    Example:
        >>> chat_history = [
        ...     {"role": "user", "content": "show me retrieval_nodes.py"},
        ...     {"role": "assistant", "content": "Here's retrieval_nodes.py..."}
        ... ]
        >>> _extract_mentioned_files(chat_history)
        ['retrieval_nodes.py']
    """
    mentioned_files = []
    for msg in chat_history:
        # Extract content from messages (support both dict format and LangGraph message objects)
        if isinstance(msg, dict):
            content = msg.get("content", "").lower()
        elif hasattr(msg, "content"):
            content = msg.content.lower() if hasattr(msg, "content") else ""
        else:
            content = ""
        # Extract file paths from mentions (e.g., "retrieval_nodes.py", "stage4_retrieval")
        file_patterns = [
            r'([a-z_]+\.py)',  # Python files
            r'(stage\d+_[a-z_]+)',  # Stage modules
            r'(assistant/[a-z_/]+\.py)',  # Full paths
            r'([a-z_]+_nodes\.py)',  # Node files
            r'([a-z_]+_retriever\.py)',  # Retriever files
        ]
        for pattern in file_patterns:
            matches = re.findall(pattern, content)
            mentioned_files.extend(matches)
    return list(set(mentioned_files))  # Deduplicate


def _build_retrieval_hints(active_subcategories: List[str]) -> List[str]:
    """Map active technical subcategories to preferred retrieval sources.

    This helper translates user interest signals into concrete KB/code targets:
    - stack_depth → technical_kb.csv (framework/library choices)
    - architecture_depth → architecture_kb.csv + conversation_flow.py
    - data_pipeline_depth → pgvector_retriever.py + RAG docs
    - state_management_depth → conversation_state.py + state patterns

    Args:
        active_subcategories: List of subcategory names (without "_score" suffix)

    Returns:
        List of retrieval source hints (KB names, file patterns)

    Example:
        >>> _build_retrieval_hints(["stack_depth", "architecture_depth"])
        ["technical_kb.csv", "architecture_kb.csv", "conversation_flow.py"]
    """
    hints = []

    if "stack_depth" in active_subcategories:
        hints.extend(["technical_kb.csv", "requirements.txt", "imports_kb.csv"])

    if "architecture_depth" in active_subcategories:
        hints.extend(["architecture_kb.csv", "conversation_flow.py", "system design"])

    if "data_pipeline_depth" in active_subcategories:
        hints.extend(["pgvector_retriever.py", "rag_engine.py", "RAG pipeline", "retrieval"])

    if "state_management_depth" in active_subcategories:
        hints.extend(["conversation_state.py", "conversation_nodes.py", "LangGraph state"])

    return hints


def retrieve_chunks(state: ConversationState, rag_engine: RagEngine, top_k: int = 4) -> Dict[str, Any]:
    """Retrieve relevant KB chunks using RAG engine (pgvector) with subcategory boosting.

    This is the entry point for the retrieval phase. It:
    1. Takes the user's query (or composed query from earlier nodes)
    2. Checks active technical subcategories for retrieval hints
    3. Generates embeddings using OpenAI
    4. Searches Supabase pgvector for similar chunks (with optional source filtering)
    5. Normalizes results and stores similarity scores

    Subcategory-aware retrieval:
    - stack_depth → Prioritize technical_kb.csv (framework comparisons, dependencies)
    - architecture_depth → Include architecture_kb.csv + conversation_flow.py
    - data_pipeline_depth → Boost pgvector_retriever.py, RAG pipeline docs
    - state_management_depth → Include conversation_state.py, state management patterns

    Observability: Logs retrieval performance (latency, chunk count, avg similarity)
    Performance: ~300ms typical (embedding + vector search)

    Design Principles:
    - Reliability (#4): Graceful handling if retrieval fails (returns empty chunks)
    - Observability: Logs retrieval metadata for LangSmith tracing
    - Defensibility: Never raises exceptions, returns empty results on failure
    - Context awareness: Uses subcategory signals to refine retrieval targets

    Args:
        state: ConversationState with query field and analytics_metadata
        rag_engine: RAG engine instance with retriever
        top_k: Number of chunks to retrieve (default 4)

    Returns:
        Updated state with:
        - retrieved_chunks: List of normalized chunk dicts
        - retrieval_scores: Similarity scores for each chunk
        - analytics_metadata: Retrieval performance metrics + source hints

    Example:
        >>> state = ConversationState(
        ...     query="How does RAG work?",
        ...     analytics_metadata={"technical_subcategories": ["data_pipeline_depth"]}
        ... )
        >>> retrieve_chunks(state, rag_engine, top_k=4)
        >>> "pgvector" in str(state["retrieved_chunks"])
        True
    """
    query = state.get("composed_query") or state["query"]
    metadata = state.setdefault("analytics_metadata", {})

    # Extract active technical subcategories for retrieval hints
    active_subcats = metadata.get("technical_subcategories", [])

    # Build retrieval hints for targeted source selection
    retrieval_hints = _build_retrieval_hints(active_subcats)
    if retrieval_hints:
        metadata["retrieval_source_hints"] = retrieval_hints
        logger.info("Retrieval targeting: %s", ", ".join(retrieval_hints))

    # Hierarchical context selection: extract mentioned files from conversation
    chat_history = state.get("chat_history", [])
    mentioned_files = _extract_mentioned_files(chat_history) if chat_history else []
    if mentioned_files:
        metadata["mentioned_files"] = mentioned_files
        logger.debug(f"Files mentioned in conversation: {mentioned_files}")

    # Adaptive top_k based on conversation length and query complexity
    # Later turns need more context to build on previous discussion
    original_top_k = top_k
    if len(chat_history) >= 6:  # 3+ turns
        top_k = min(8, top_k + 2)  # More context for later turns
        logger.debug(f"Increased top_k from {original_top_k} to {top_k} (later turn)")
    elif len(chat_history) <= 2:  # Early turns
        top_k = max(3, top_k - 1)  # Less context for early turns
        logger.debug(f"Decreased top_k from {original_top_k} to {top_k} (early turn)")

    # File-level prioritization for code-related queries
    query_lower = query.lower()
    if "code" in query_lower or "retrieval" in query_lower or "implementation" in query_lower:
        metadata["file_priority"] = True  # Boost codebase chunks
        logger.debug("File priority enabled for code-related query")

    # Determine if personality context would add value
    role = state.get("role", "")
    include_personality = should_include_personality(
        query=query,
        context=[],
        role=role
    )

    # Track personality usage in analytics
    if include_personality:
        metadata["has_personality_context"] = True
        logger.debug("Including personality context in retrieval")

    with create_custom_span("retrieve_chunks", {
        "query": query[:120],
        "top_k": top_k,
        "active_subcategories": active_subcats,
        "source_hints": retrieval_hints,
        "mentioned_files": mentioned_files,
        "include_personality": include_personality
    }):
        try:
            # Use role-aware retrieval with personality if appropriate
            chunks = {}
            if include_personality and rag_engine.pgvector_retriever and role:
                # Directly call pgvector retriever with personality context
                raw_chunks = rag_engine.pgvector_retriever.retrieve_for_role(
                    query=query,
                    role=role,
                    top_k=top_k,
                    include_personality=True
                )
                # Ensure raw_chunks is a list (retrieve_for_role returns List[Dict])
                if not isinstance(raw_chunks, list):
                    raw_chunks = []
                # Build chunks dict for consistency with standard path
                chunks = {
                    "chunks": raw_chunks,
                    "scores": [c.get("similarity", 0.0) for c in raw_chunks if isinstance(c, dict)]
                }
            else:
                # Standard retrieval path
                chunks = rag_engine.retrieve(query, top_k=top_k) or {}

            raw_chunks = chunks.get("chunks", [])
            normalized: List[Dict[str, Any]] = []

            for item in raw_chunks:
                if isinstance(item, dict):
                    # Clean Q&A format from chunk content before storing
                    chunk_copy = item.copy()
                    if "content" in chunk_copy and isinstance(chunk_copy["content"], str):
                        chunk_copy["content"] = _clean_chunk_qa_format(chunk_copy["content"])
                    normalized.append(chunk_copy)
                else:
                    normalized.append({"content": _clean_chunk_qa_format(str(item))})

            state["retrieved_chunks"] = normalized

            raw_scores = chunks.get("scores")
            if isinstance(raw_scores, list) and len(raw_scores) == len(normalized):
                scores = [score if isinstance(score, (int, float)) else 0.0 for score in raw_scores]
            else:
                scores = []
                for chunk in normalized:
                    similarity = chunk.get("similarity", 0.0) if isinstance(chunk, dict) else 0.0
                    scores.append(similarity if isinstance(similarity, (int, float)) else 0.0)

            state["retrieval_scores"] = scores
            metadata["retrieval_count"] = len(normalized)

            if scores:
                avg_similarity = sum(scores) / len(scores)
                metadata["avg_similarity"] = round(avg_similarity, 3)
                logger.info(
                    "Retrieved %s chunks, avg_similarity=%.3f",
                    len(state["retrieved_chunks"]),
                    avg_similarity,
                )

            # Merged: Apply MMR-style diversification (from re_rank_and_dedup)
            # Sort by similarity and deduplicate based on (section, content preview)
            if normalized:
                sorted_chunks = sorted(
                    normalized,
                    key=lambda chunk: chunk.get("similarity", 0.0),
                    reverse=True,
                )

                seen_signatures = set()
                diversified: List[Dict[str, Any]] = []
                for chunk in sorted_chunks:
                    signature = (chunk.get("section"), (chunk.get("content") or "")[:200])
                    if signature in seen_signatures:
                        continue
                    seen_signatures.add(signature)
                    diversified.append(chunk)

                # Conversation-aware code file prioritization
                session_memory = state.get("session_memory", {})
                discussed_files = session_memory.get("discussed_files", [])

                if discussed_files and any(c.get("doc_id") == "codebase" for c in diversified):
                    # Boost chunks from discussed files (user already familiar)
                    for chunk in diversified:
                        file_path = chunk.get("section") or chunk.get("metadata", {}).get("file_path", "")
                        if any(discussed_file in file_path for discussed_file in discussed_files):
                            # Boost similarity score for discussed files
                            current_sim = chunk.get("similarity", 0.0)
                            chunk["similarity"] = min(1.0, current_sim + 0.1)
                            logger.debug(f"Boosted discussed file: {file_path} (similarity: {current_sim:.3f} → {chunk['similarity']:.3f})")

                    # Re-sort by boosted similarity
                    diversified = sorted(diversified, key=lambda c: c.get("similarity", 0.0), reverse=True)

                state["retrieved_chunks"] = diversified
                state["retrieval_scores"] = [c.get("similarity", 0.0) for c in diversified]
                metadata["post_rank_count"] = len(diversified)

                if len(diversified) < len(normalized):
                    logger.info("Deduplicated %d → %d chunks", len(normalized), len(diversified))

        except Exception as e:
            logger.error(
                "Retrieval failed for query '%s...': %s",
                query[:50],
                e,
                exc_info=True,
            )
            state["retrieved_chunks"] = []
            state["retrieval_scores"] = []
            metadata["retrieval_error"] = str(e)

    return state


def re_rank_and_dedup(state: ConversationState) -> ConversationState:
    """DEPRECATED: No-op for backward compatibility.

    Logic merged into retrieve_chunks() for streamlined single-pass retrieval.
    This function now does nothing since MMR diversification happens automatically
    during chunk retrieval.

    Kept for import compatibility only. New code should rely on retrieve_chunks()
    to handle both search and deduplication.
    """
    return state


def validate_grounding(state: ConversationState, threshold: float = 0.45) -> ConversationState:
    """Ensure retrieval produced sufficiently similar chunks before generation.

    This is a quality gate that prevents hallucinations by detecting low-confidence
    retrieval results. If the top similarity score is below threshold, we ask the
    user for clarification instead of generating an answer.

    Threshold tuning:
    - 0.45: Balanced (current) - catches vague queries, allows some flexibility
    - 0.50: Strict - fewer false positives, more clarification requests
    - 0.40: Lenient - fewer clarifications, higher hallucination risk

    Performance: <1ms (simple threshold check)

    Design Principles:
    - Defensibility: Early exit prevents bad LLM generations downstream
    - SRP: Only validates, doesn't modify chunks or generate responses
    - Observability: Logs grounding status for analytics

    Args:
        state: ConversationState with retrieval_scores
        threshold: Minimum similarity score to consider "grounded" (default 0.45)

    Returns:
        Updated state with:
        - grounding_status: "ok" | "no_results" | "insufficient"
        - clarification_needed: bool flag for downstream nodes
        - clarifying_question: Template question if grounding failed
        - analytics_metadata: top_similarity for monitoring

    Example:
        >>> state = {"retrieval_scores": [0.92, 0.85, 0.78]}
        >>> validate_grounding(state, threshold=0.45)
        >>> state["grounding_status"]
        "ok"

        >>> state = {"retrieval_scores": [0.38, 0.32]}
        >>> validate_grounding(state, threshold=0.45)
        >>> state["grounding_status"]
        "insufficient"
    """
    scores = state.get("retrieval_scores", [])
    top_score = max(scores) if scores else 0.0

    status = "ok"
    if not scores:
        status = "no_results"
    elif top_score < threshold:
        status = "insufficient"

    state["grounding_status"] = status
    state.setdefault("analytics_metadata", {})["top_similarity"] = round(top_score, 3)

    if status != "ok":
        state["clarification_needed"] = True
        state["clarifying_question"] = (
            "I want to keep this grounded. Could you share a bit more detail so I can "
            "search the right knowledge?"
        )
    else:
        state["clarification_needed"] = False

    return state


def validate_retrieval_relevance(state: ConversationState) -> ConversationState:
    """Check if retrieved chunks match query intent keywords.

    Quality gate that detects when retrieval results don't match the query intent.
    This helps identify cases where the query was misunderstood or retrieval
    failed to find relevant content.

    Performance: <1ms (keyword matching only)

    Design Principles:
    - Defensibility: Early detection of retrieval mismatches
    - Observability: Logs when chunks don't match intent
    - SRP: Only validates retrieval relevance, doesn't modify chunks

    Args:
        state: ConversationState with query and retrieved_chunks

    Returns:
        Updated state with retrieval_quality_warning if chunks don't match intent

    Example:
        >>> state = {"query": "customer support", "retrieved_chunks": [...]}
        >>> validate_retrieval_relevance(state)
        >>> state.get("retrieval_quality_warning")
        "chunks_dont_match_intent"
    """
    query = state.get("query", "").lower()
    chunks = state.get("retrieved_chunks", [])

    if not query or not chunks:
        return state

    # Extract intent keywords (words > 3 chars, excluding common words)
    common_words = {"what", "how", "when", "where", "why", "who", "this", "that", "with", "from", "have", "does"}
    intent_keywords = [
        w for w in query.split()
        if len(w) > 3 and w not in common_words
    ]

    if not intent_keywords:
        # Query too short to extract intent
        return state

    # Check if any chunk contains intent keywords
    relevant_chunks = 0
    for chunk in chunks:
        content = chunk.get("content", "").lower() if isinstance(chunk, dict) else str(chunk).lower()
        if any(keyword in content for keyword in intent_keywords):
            relevant_chunks += 1

    # If NO chunks match intent, flag as quality issue
    if relevant_chunks == 0:
        state["retrieval_quality_warning"] = "chunks_dont_match_intent"
        logger.warning(
            f"No chunks match query intent. Query: {query[:50]}, "
            f"Intent keywords: {intent_keywords[:5]}"
        )

    return state


def handle_grounding_gap(state: ConversationState) -> ConversationState:
    """Respond gracefully when grounding is insufficient OR edge case detected.

    This node short-circuits the pipeline when validate_grounding detects
    low-confidence retrieval OR when an edge case was detected in classification.
    Instead of generating a potentially hallucinated answer, we provide a helpful
    fallback message and halt the pipeline.

    For edge cases, uses conversational response handler to generate warm,
    inference-rich responses (not templates).

    Performance: <1ms (template response) or ~800ms (LLM-generated edge case response)

    Design Principles:
    - Defensibility: Prevents hallucinations by stopping generation early
    - UX: Provides helpful guidance instead of error message
    - SRP: Only handles grounding gap response, doesn't retry retrieval
    - Conversational: Edge cases get LLM-generated responses, not templates

    Args:
        state: ConversationState with grounding_status and/or edge_case_detected

    Returns:
        Updated state with:
        - answer: Fallback message explaining the issue
        - pipeline_halt: Flag to stop downstream nodes
        - (no changes if grounding_status == "ok" and no edge case)

    Example:
        >>> state = {"grounding_status": "insufficient"}
        >>> handle_grounding_gap(state)
        >>> state["pipeline_halt"]
        True
        >>> "could not find context" in state["answer"]
        True
    """
    if state.get("grounding_status") == "ok" and not state.get("edge_case_detected"):
        return state

    # Check if edge case was detected (priority over normal grounding gap)
    if state.get("edge_case_detected"):
        # Generate conversational edge case response using LLM
        message = generate_edge_case_response(state)
        span_name = "edge_case_response"
        span_data = {
            "edge_case_type": state.get("edge_case_type"),
            "grounding_status": state.get("grounding_status")
        }
    else:
        # Normal grounding gap (insufficient context, still on-topic)
        message = (
            "I could not find context precise enough to stay factual yet. "
            "Tell me a little more about what you want to explore and I will pull "
            "the exact architecture notes or data you need."
        )
        span_name = "grounding_gap_response"
        span_data = {"status": state.get("grounding_status")}

    with create_custom_span(span_name, span_data):
        state["answer"] = message
        state["pipeline_halt"] = True

    return state
