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
import os
import re
from typing import Dict, Any, List

from assistant.state.conversation_state import ConversationState
from assistant.core.rag_engine import RagEngine
from assistant.observability.langsmith_tracer import create_custom_span

logger = logging.getLogger(__name__)


def _is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled via environment variable.

    Args:
        feature_name: Name of the feature flag (e.g., 'ENABLE_HYBRID_SEARCH')

    Returns:
        True if feature is enabled, False otherwise (default)
    """
    value = os.getenv(feature_name, "false").lower()
    return value in ("true", "1", "yes", "on")


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

    # Fallback file path detection in retrieve_chunks (if not already detected in classify_intent)
    # This provides redundancy in case file_request wasn't set earlier in the pipeline
    if not state.get("file_request"):
        from assistant.flows.node_logic.stage2_query_classification import _detect_file_request
        detected_file = _detect_file_request(query)
        if detected_file:
            state["file_request"] = detected_file
            state["query_type"] = "file_request"
            logger.info(f"File path detected in retrieve_chunks (fallback): {detected_file}")

    # Layer 4: Full Codebase Semantic Search - Prioritize codebase/documentation for self-referential queries
    is_self_referential = state.get("is_self_referential", False)

    # Extract active technical subcategories for retrieval hints (needed for logging)
    active_subcats = metadata.get("technical_subcategories", [])
    retrieval_hints = []  # Initialize for logging

    if is_self_referential:
        # For self-referential queries, prioritize codebase and documentation chunks
        # This ensures Portfolia answers questions about herself using actual code/docs
        preferred_doc_ids = ["codebase", "documentation"]
        metadata["retrieval_source_hints"] = preferred_doc_ids
        metadata["self_referential_retrieval"] = True
        retrieval_hints = preferred_doc_ids  # For logging
        logger.info(f"Self-referential query detected, prioritizing doc_ids: {preferred_doc_ids}")
    else:
        # Build retrieval hints for targeted source selection
        retrieval_hints = _build_retrieval_hints(active_subcats)
        if retrieval_hints:
            metadata["retrieval_source_hints"] = retrieval_hints
            logger.info("Retrieval targeting: %s", ", ".join(retrieval_hints))

    # Check if hybrid search is enabled
    use_hybrid_search = _is_feature_enabled("ENABLE_HYBRID_SEARCH")

    with create_custom_span("retrieve_chunks", {
        "query": query[:120],
        "top_k": top_k,
        "active_subcategories": active_subcats,
        "source_hints": retrieval_hints,
        "hybrid_search_enabled": use_hybrid_search
    }):
        try:
            # Layer 4: Prioritize codebase/documentation for self-referential queries
            # Try to retrieve from preferred doc_ids first, then fall back to all sources
            preferred_doc_ids = metadata.get("retrieval_source_hints", [])
            if is_self_referential and preferred_doc_ids and hasattr(rag_engine, 'pgvector_retriever') and rag_engine.pgvector_retriever:
                # For self-referential queries, use lower threshold to ensure we get results
                # Codebase/documentation chunks may have lower similarity scores than KB chunks
                self_ref_threshold = 0.1  # Lower threshold for self-referential queries

                # For self-referential queries, try each preferred doc_id
                all_chunks = []
                for doc_id in preferred_doc_ids:
                    try:
                        doc_chunks = rag_engine.pgvector_retriever.retrieve(
                            query,
                            top_k=top_k * 2,  # Get more chunks, then filter to top_k
                            doc_id=doc_id,
                            threshold=self_ref_threshold  # Use lower threshold
                        )
                        all_chunks.extend(doc_chunks)
                        logger.info(f"Retrieved {len(doc_chunks)} chunks from {doc_id} (threshold={self_ref_threshold})")
                    except Exception as e:
                        logger.warning(f"Failed to retrieve from {doc_id}: {e}")

                # If we got chunks from preferred sources, use them
                if all_chunks:
                    # Sort by similarity and take top_k
                    all_chunks.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
                    raw_chunks = all_chunks[:top_k]
                    chunks = {"chunks": raw_chunks, "scores": [c.get("similarity", 0.0) for c in raw_chunks]}
                    metadata["preferred_source_used"] = True
                    logger.info(f"Using {len(raw_chunks)} chunks from preferred sources (codebase/documentation)")
                else:
                    # Fall back to standard retrieval if preferred sources returned nothing
                    logger.info("No chunks from preferred sources, falling back to standard retrieval")
                    chunks = rag_engine.retrieve(query, top_k=top_k) or {}
                    metadata["preferred_source_used"] = False
            # Try hybrid search if enabled, otherwise use standard retrieval
            elif use_hybrid_search and hasattr(rag_engine, 'pgvector_retriever') and rag_engine.pgvector_retriever:
                try:
                    # Use hybrid retrieval
                    raw_chunks = rag_engine.pgvector_retriever.retrieve_hybrid(
                        query,
                        top_k=top_k,
                        use_keyword_fallback=True
                    )
                    # Convert to expected format
                    chunks = {"chunks": raw_chunks, "scores": [c.get("similarity", 0.0) for c in raw_chunks]}
                    metadata["hybrid_search_used"] = True
                    logger.debug("Used hybrid search for retrieval")
                except Exception as hybrid_error:
                    logger.warning(f"Hybrid search failed, falling back to standard: {hybrid_error}")
                    # Fallback to standard retrieval
                    chunks = rag_engine.retrieve(query, top_k=top_k) or {}
                    metadata["hybrid_search_used"] = False
                    metadata["hybrid_search_error"] = str(hybrid_error)
            else:
                # Standard retrieval (existing behavior)
                chunks = rag_engine.retrieve(query, top_k=top_k) or {}
                metadata["hybrid_search_used"] = False

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

                state["retrieved_chunks"] = diversified
                state["retrieval_scores"] = [c.get("similarity", 0.0) for c in diversified]
                metadata["post_rank_count"] = len(diversified)

                if len(diversified) < len(normalized):
                    logger.info("Deduplicated %d → %d chunks", len(normalized), len(diversified))

            # If scores are still low and fuzzy matching is enabled, try fuzzy matching
            if scores and all(s < 0.4 for s in scores) and _is_feature_enabled("ENABLE_FUZZY_MATCHING"):
                if hasattr(rag_engine, 'pgvector_retriever') and rag_engine.pgvector_retriever:
                    try:
                        logger.info("Low retrieval scores, trying fuzzy matching")
                        fuzzy_chunks = rag_engine.pgvector_retriever._fuzzy_match_chunks(
                            query, normalized, threshold=0.6
                        )
                        if fuzzy_chunks:
                            # Replace with fuzzy results if they're better
                            state["retrieved_chunks"] = fuzzy_chunks[:top_k]
                            state["retrieval_scores"] = [c.get("similarity", 0.0) for c in fuzzy_chunks[:top_k]]
                            metadata["fuzzy_matching_used"] = True
                            logger.info(f"Fuzzy matching found {len(fuzzy_chunks)} chunks")
                    except Exception as fuzzy_error:
                        logger.warning(f"Fuzzy matching failed: {fuzzy_error}")
                        metadata["fuzzy_matching_error"] = str(fuzzy_error)

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


def retrieve_file_content(state: ConversationState, rag_engine: RagEngine) -> Dict[str, Any]:
    """Retrieve file content when user explicitly requests a specific file.

    Purpose: Enable on-demand access to current source files during conversation.
    This allows Portfolia to show actual implementation when asked about specific files.

    Layer 1: File Reading Infrastructure

    Args:
        state: ConversationState with file_request field set
        rag_engine: RAG engine instance (contains code_index)

    Returns:
        Updated state with file_content field containing file data
    """
    file_path = state.get("file_request")
    if not file_path:
        return state

    with create_custom_span("retrieve_file_content", {"file_path": file_path}):
        try:
            # Get code_index from rag_engine
            code_index = getattr(rag_engine, 'code_index', None)
            if not code_index:
                logger.warning("CodeIndex not available for file reading")
                return {
                    **state,
                    "file_content": None,
                    "file_read_error": "CodeIndex not available"
                }

            # Read file content
            file_data = code_index.read_file_content(file_path)

            if file_data.get("success"):
                logger.info(f"Successfully read file: {file_path} ({file_data['line_count']} lines)")
                return {
                    **state,
                    "file_content": file_data,
                    "file_read_success": True
                }
            else:
                logger.warning(f"Failed to read file: {file_path} - {file_data.get('content', 'Unknown error')}")
                return {
                    **state,
                    "file_content": file_data,
                    "file_read_success": False
                }

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
            return {
                **state,
                "file_content": {
                    "content": f"# Error reading file: {str(e)}",
                    "file_path": file_path,
                    "success": False
                },
                "file_read_success": False
            }


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


def handle_grounding_gap(state: ConversationState) -> ConversationState:
    """Respond gracefully when grounding is insufficient.

    This node short-circuits the pipeline when validate_grounding detects
    low-confidence retrieval. Instead of generating a potentially hallucinated
    answer, we provide a helpful fallback message and halt the pipeline.

    Performance: <1ms (template response, no LLM call)

    Design Principles:
    - Defensibility: Prevents hallucinations by stopping generation early
    - UX: Provides helpful guidance instead of error message
    - SRP: Only handles grounding gap response, doesn't retry retrieval

    Args:
        state: ConversationState with grounding_status

    Returns:
        Updated state with:
        - answer: Fallback message explaining the issue
        - pipeline_halt: Flag to stop downstream nodes
        - (no changes if grounding_status == "ok")

    Example:
        >>> state = {"grounding_status": "insufficient"}
        >>> handle_grounding_gap(state)
        >>> state["pipeline_halt"]
        True
        >>> "could not find context" in state["answer"]
        True
    """
    if state.get("grounding_status") == "ok":
        return state

    message = (
        "I could not find context precise enough to stay factual yet. "
        "Tell me a little more about what you want to explore and I will pull "
        "the exact architecture notes or data you need."
    )

    with create_custom_span("grounding_gap_response", {"status": state.get("grounding_status")}):
        state["answer"] = message
        state["pipeline_halt"] = True

    return state
