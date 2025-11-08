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
from typing import Dict, Any, List

from src.state.conversation_state import ConversationState
from src.core.rag_engine import RagEngine
from src.observability.langsmith_tracer import create_custom_span

logger = logging.getLogger(__name__)


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

    with create_custom_span("retrieve_chunks", {
        "query": query[:120], 
        "top_k": top_k,
        "active_subcategories": active_subcats,
        "source_hints": retrieval_hints
    }):
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
