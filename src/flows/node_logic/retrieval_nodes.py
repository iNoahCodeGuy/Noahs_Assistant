"""Retrieval pipeline nodes - pgvector search, re-ranking, and grounding validation.

This module handles the retrieval phase of the conversation pipeline:
1. retrieve_chunks → Supabase pgvector search with similarity scoring
2. re_rank_and_dedup → MMR-style diversification to avoid redundant chunks
3. validate_grounding → Quality gate ensuring sufficient context before generation
4. handle_grounding_gap → Graceful fallback when retrieval confidence is low

Design Principles:
- SRP: Each function handles one retrieval concern
- Defensibility: Graceful degradation on retrieval failures
- Observability: LangSmith tracing for retrieval performance
- Reliability: Never crashes the pipeline, returns empty results on failure

Performance Characteristics:
- retrieve_chunks: ~300ms typical (embedding + vector search)
- re_rank_and_dedup: <10ms (in-memory sorting)
- validate_grounding: <1ms (threshold check)
- handle_grounding_gap: <1ms (template response)

See: docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md for full pipeline flow
"""

import logging
from typing import Dict, Any, List

from src.state.conversation_state import ConversationState
from src.core.rag_engine import RagEngine
from src.observability.langsmith_tracer import create_custom_span

logger = logging.getLogger(__name__)


def retrieve_chunks(state: ConversationState, rag_engine: RagEngine, top_k: int = 4) -> Dict[str, Any]:
    """Retrieve relevant KB chunks using RAG engine (pgvector).

    This is the entry point for the retrieval phase. It:
    1. Takes the user's query (or composed query from earlier nodes)
    2. Generates embeddings using OpenAI
    3. Searches Supabase pgvector for similar chunks
    4. Normalizes results and stores similarity scores

    Observability: Logs retrieval performance (latency, chunk count, avg similarity)
    Performance: ~300ms typical (embedding + vector search)

    Design Principles:
    - Reliability (#4): Graceful handling if retrieval fails (returns empty chunks)
    - Observability: Logs retrieval metadata for LangSmith tracing
    - Defensibility: Never raises exceptions, returns empty results on failure

    Args:
        state: ConversationState with query field
        rag_engine: RAG engine instance with retriever
        top_k: Number of chunks to retrieve (default 4)

    Returns:
        Updated state with:
        - retrieved_chunks: List of normalized chunk dicts
        - retrieval_scores: Similarity scores for each chunk
        - analytics_metadata: Retrieval performance metrics

    Example:
        >>> state = ConversationState(query="How does RAG work?")
        >>> retrieve_chunks(state, rag_engine, top_k=4)
        >>> len(state["retrieved_chunks"])  # Should be <= 4
        3
    """
    query = state.get("composed_query") or state["query"]
    metadata = state.setdefault("analytics_metadata", {})

    with create_custom_span("retrieve_chunks", {"query": query[:120], "top_k": top_k}):
        try:
            chunks = rag_engine.retrieve(query, top_k=top_k) or {}
            raw_chunks = chunks.get("chunks", [])
            normalized: List[Dict[str, Any]] = []

            for item in raw_chunks:
                if isinstance(item, dict):
                    normalized.append(item)
                else:
                    normalized.append({"content": str(item)})

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
    """Apply lightweight MMR-style diversification to retrieved chunks.

    Problem: Vector search can return near-duplicate chunks with slightly
    different similarity scores. This creates redundant context that wastes
    tokens and degrades answer quality.

    Solution: Sort by similarity (highest first), then deduplicate based on
    a signature of (section, content_preview). This preserves high-scoring
    chunks while removing redundancy.

    Performance: <10ms for typical 4-chunk result set (in-memory only)

    Design Principles:
    - SRP: Only handles diversification, doesn't modify scores or content
    - Simplicity (KISS): Simple signature-based dedup, no fancy MMR math
    - Observability: Logs pre/post chunk counts for monitoring

    Args:
        state: ConversationState with retrieved_chunks and retrieval_scores

    Returns:
        Updated state with:
        - retrieved_chunks: Diversified chunk list (sorted, deduplicated)
        - retrieval_scores: Updated scores matching new chunk order
        - analytics_metadata: post_rank_count for observability

    Example:
        Before: [chunk_A (0.9), chunk_A_dup (0.88), chunk_B (0.7)]
        After:  [chunk_A (0.9), chunk_B (0.7)]
    """
    with create_custom_span(
        "re_rank_and_dedup",
        {"retrieval_count": len(state.get("retrieved_chunks", []))}
    ):
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            return state

        sorted_chunks = sorted(
            chunks,
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
        state.setdefault("analytics_metadata", {})["post_rank_count"] = len(diversified)

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
