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
    """Strip Q&A format and section headers from retrieved chunks.

    KB chunks are stored as "Question/Header\\n\\nAnswer text..." by the migration
    script. This preprocessor strips leading questions, section headers, and
    markdown headers so the LLM receives clean answer text.

    Args:
        content: Raw chunk content (may contain Q:/A: format, section headers, etc.)

    Returns:
        Cleaned content with questions/headers removed

    Example:
        >>> _clean_chunk_qa_format("Q: What is RAG?\\nA: Retrieval-Augmented Generation...")
        "Retrieval-Augmented Generation..."
        >>> _clean_chunk_qa_format("Contact Info\\n\\nNoah is based in...")
        "Noah is based in..."
    """
    # Pattern 1: Extract answer from explicit "Q: ... A: ..." format
    qa_pattern = r'Q:\s*.*?\s*A:\s*(.*)'
    match = re.search(qa_pattern, content, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
        logger.debug(f"Stripped Q&A format from chunk ({len(content)} → {len(cleaned)} chars)")
        return cleaned

    # Pattern 2: Iteratively strip short leading paragraphs (questions, section headers)
    # Migration stores chunks as "Question/Header\n\nAnswer text..."
    # CSV chunks: "What does Noah do?\n\nNoah works at..."
    # Markdown chunks: "Section Title\n\nSubsection\n\nContent..."
    cleaned = content
    max_strips = 3  # Safety limit to avoid stripping real content
    for _ in range(max_strips):
        if '\n\n' not in cleaned:
            break
        first_para, rest = cleaned.split('\n\n', 1)
        first_stripped = first_para.strip()
        rest_stripped = rest.strip()
        # Strip if first paragraph is a short header/question and rest has content
        if len(first_stripped) < 150 and len(rest_stripped) > 30:
            logger.debug(f"Stripped leading paragraph from chunk: '{first_stripped[:60]}...'")
            cleaned = rest_stripped
        else:
            break

    # Pattern 3: Strip inline markdown headers (### lines)
    lines = cleaned.split('\n')
    cleaned_lines = [line for line in lines if not line.strip().startswith('#')]
    result = '\n'.join(cleaned_lines).strip()

    if result and result != content:
        logger.debug(f"Cleaned chunk ({len(content)} → {len(result)} chars)")

    return result if result else content


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


def _check_chunk_relevance(chunks: List[Dict], query: str) -> bool:
    """Check if any retrieved chunk contains query topic words.

    This validation ensures retrieved chunks actually match the query intent,
    preventing cases where retrieval returns off-topic results.

    Args:
        chunks: List of retrieved chunk dicts
        query: Original user query

    Returns:
        True if at least one chunk (in top 3) contains query topic words, False otherwise
    """
    if not query or not chunks:
        return True  # Skip validation if no query or chunks

    query_topics = _extract_topic_words(query)
    if not query_topics:
        # Query too short or all stop words - skip validation
        return True

    # Check top 3 chunks for topic relevance
    for chunk in chunks[:3]:
        content = chunk.get("content", "").lower() if isinstance(chunk, dict) else str(chunk).lower()
        if any(topic in content for topic in query_topics):
            return True

    # No chunks match topic
    logger.warning(
        f"Retrieved chunks don't match query topic. Query: {query[:50]}, "
        f"Query topics: {query_topics[:5]}, Top chunk section: {chunks[0].get('section', 'unknown') if chunks else 'none'}"
    )
    return False


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
    query = state.get("composed_query") or state.get("query", "")
    if not query:
        logger.warning("retrieve_chunks called without query or composed_query")
        state["retrieved_chunks"] = []
        state["retrieval_metadata"] = {"error": "no_query"}
        return state

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
                    chunk_copy = item.copy()
                    # Prefer metadata['answer'] (clean answer text without question/header)
                    metadata = chunk_copy.get("metadata", {})
                    if isinstance(metadata, dict) and metadata.get("answer"):
                        chunk_copy["content"] = metadata["answer"]
                    elif "content" in chunk_copy and isinstance(chunk_copy["content"], str):
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

            # Boost career_kb chunks for background/career queries
            # This ensures queries about Noah's background prioritize career_kb over personality_kb
            if any(kw in query_lower for kw in ["noah", "background", "career", "technical background", "resume"]):
                boosted = False
                for i, chunk in enumerate(normalized):
                    if chunk.get("doc_id") == "career_kb":
                        if i < len(scores):
                            scores[i] = min(1.0, scores[i] + 0.15)  # Boost career_kb by 0.15
                            boosted = True

                if boosted:
                    # Re-sort chunks by boosted similarity scores
                    # Create list of (chunk, score) tuples, sort by score descending
                    chunk_score_pairs = list(zip(normalized, scores))
                    chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)
                    normalized = [pair[0] for pair in chunk_score_pairs]
                    scores = [pair[1] for pair in chunk_score_pairs]
                    state["retrieved_chunks"] = normalized
                    state["retrieval_scores"] = scores
                    logger.info("Boosted career_kb chunks for background query")

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

                # Boost enterprise adaptation chunks for enterprise queries
                enterprise_keywords = ["adapt", "adapts", "adaptation", "customer support", "enterprise",
                                       "use case", "chatbot", "internal docs", "sales enablement"]
                if any(kw in query.lower() for kw in enterprise_keywords):
                    # CRITICAL FIX: Cap all similarity scores at 0.95 BEFORE boosting
                    # This prevents personality chunks (which get 1.0 similarity) from
                    # always outranking enterprise content after boosting
                    for chunk in diversified:
                        chunk["similarity"] = min(0.95, chunk.get("similarity", 0.0))

                    boosted_count = 0
                    for chunk in diversified:
                        # Boost chunks from enterprise adaptation guide
                        section = chunk.get("section", "").lower()
                        content = (chunk.get("content", "") or "").lower()

                        # Expanded pattern matching for enterprise adaptation content
                        is_enterprise_related = (
                            "enterprise" in section or "enterprise" in content or
                            "customer support" in content or "customer support" in section or
                            "adaptation" in section or "adaptation" in content or
                            "use case" in content or "chatbot" in content or
                            "internal docs" in content or "internal knowledge" in content or
                            "sales enablement" in content or "sales enablement" in section or
                            "what to change" in content or "code changes" in content or
                            "expected roi" in content or "roi" in section or
                            "common enterprise" in content or "enterprise use" in content
                        )

                        if is_enterprise_related:
                            current_sim = chunk.get("similarity", 0.0)
                            # Increased boost from 0.25 to 0.35 so enterprise content outranks generic
                            chunk["similarity"] = min(1.0, current_sim + 0.35)
                            boosted_count += 1
                            logger.debug(f"Boosted enterprise adaptation chunk: {chunk.get('section', 'unknown')} (similarity: {current_sim:.3f} → {chunk['similarity']:.3f})")

                    if boosted_count > 0:
                        logger.info(f"Boosted {boosted_count} enterprise adaptation chunks for query")

                    # Re-sort by boosted similarity
                    diversified = sorted(diversified, key=lambda c: c.get("similarity", 0.0), reverse=True)

                state["retrieved_chunks"] = diversified
                state["retrieval_scores"] = [c.get("similarity", 0.0) for c in diversified]
                metadata["post_rank_count"] = len(diversified)

                if len(diversified) < len(normalized):
                    logger.info("Deduplicated %d → %d chunks", len(normalized), len(diversified))

                # Check topic relevance after retrieval
                if not _check_chunk_relevance(diversified, query):
                    logger.warning(f"Retrieved chunks don't match query topic: {query[:50]}")
                    state["retrieval_topic_mismatch"] = True
                    metadata["retrieval_topic_mismatch"] = True

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


def validate_grounding(state: ConversationState, threshold: float = 0.35) -> ConversationState:
    """Ensure retrieval produced sufficiently similar chunks before generation.

    This is a quality gate that prevents hallucinations by detecting low-confidence
    retrieval results. If the top similarity score is below threshold, we ask the
    user for clarification instead of generating an answer.

    Threshold tuning:
    - 0.50: Strict - fewer false positives, more clarification requests
    - 0.45: Moderate - catches vague queries, allows some flexibility
    - 0.35: Balanced (current) - allows broader topic matches like "coaching"
    - 0.30: Lenient - fewer clarifications, higher hallucination risk

    Performance: <1ms (simple threshold check)

    Design Principles:
    - Defensibility: Early exit prevents bad LLM generations downstream
    - SRP: Only validates, doesn't modify chunks or generate responses
    - Observability: Logs grounding status for analytics

    Args:
        state: ConversationState with retrieval_scores
        threshold: Minimum similarity score to consider "grounded" (default 0.35)

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


def _build_self_knowledge_chunk() -> dict:
    """Build the synthetic self-knowledge chunk for Portfolia architecture queries."""
    return {
        "content": (
            "I am Portfolia — Noah's AI portfolio assistant.\n\n"
            "MY 21-NODE PIPELINE (assistant/flows/conversation_flow.py):\n"
            "Each node receives the full state dict and returns a partial update.\n\n"
            "Stage 1 — INTENT ROUTING (assistant/flows/node_logic/stage1_intent_router.py):\n"
            "classify_message_intent() calls Claude Haiku (~150ms) to classify: knowledge_query, "
            "crush_confession, greeting, small_talk, off_topic. Crush flow is a state machine "
            "recovered from chat_history markers. _is_anonymous_choice()/_is_reveal_choice() use "
            "exact-match for '1'/'2' to prevent false positives on phone numbers. "
            "_looks_like_contact_info() uses regex for phone, email, and social handles. "
            "Short continuations ('yes', 'go deeper') get expanded via the previous user question.\n\n"
            "Stage 2 — CLASSIFICATION (stage2_query_classification.py, stage2_role_routing.py):\n"
            "classify_role_mode() infers visitor type. classify_intent() determines query type "
            "(technical, career, project, action_request). extract_entities() captures company names, "
            "role titles, timeline hints.\n\n"
            "Stage 3 — QUERY PREPARATION (stage3_query_composition.py):\n"
            "assess_clarification_need(), compose_query() builds retrieval-ready prompt.\n\n"
            "Stage 4 — RETRIEVAL & GROUNDING (stage4_retrieval_nodes.py):\n"
            "retrieve_chunks() calls Supabase RPC match_kb_chunks for pgvector cosine similarity. "
            "OpenAI text-embedding-3-small (1536 dims). Thresholds: 0.5 strict, 0.3 fallback. "
            "validate_grounding() checks scores. handle_grounding_gap() detects self-knowledge "
            "queries and injects synthetic chunks so I can answer about my own architecture.\n\n"
            "Stage 5 — GENERATION (stage5_generation_nodes.py):\n"
            "generate_draft() uses Claude Sonnet 4.5 (claude-sonnet-4-5-20250929). Chain-of-thought "
            "for complex queries. hallucination_check() compares output against retrieved chunks.\n\n"
            "Stage 6 — ENRICHMENT (stage6_action_planning.py, stage6_formatting_nodes.py):\n"
            "plan_actions() detects hiring signals. format_answer() structures response.\n\n"
            "Stage 7 — FINALIZATION (stage7_logging_nodes.py):\n"
            "execute_actions() fires SMS via Twilio, email via Resend. update_memory() stores "
            "signals with bounded sliding windows (10 topics, 20 entities).\n\n"
            "RETRIEVAL: OpenAI text-embedding-3-small → Supabase pgvector cosine similarity "
            "via match_kb_chunks RPC. File: assistant/retrieval/pgvector_retriever.py\n\n"
            "GENERATION: Anthropic Claude Sonnet 4.5. Intent classification: Claude Haiku.\n\n"
            "ERROR HANDLING: Graceful degradation if retrieval fails. Grounding validation catches "
            "low-similarity results. Intent routing bypasses RAG for greetings/crush/off-topic. "
            "Hallucination check compares generated text against source chunks. Bounded memory "
            "prevents bloat in long conversations.\n\n"
            "SYSTEM PROMPT: assistant/core/response_generator.py contains the inline system prompt "
            "for terminal chat.\n\n"
            "MY LIMITATIONS (be honest when asked):\n"
            "- No internet access: I can't browse the web, look things up in real time, or verify "
            "anything outside my knowledge base.\n"
            "- No memory between sessions: Every conversation starts fresh. I don't remember "
            "previous visitors or what was discussed before.\n"
            "- Only know what's in the KB: My knowledge about Noah comes from a curated set of "
            "documents. If something isn't in there, I genuinely don't know it.\n"
            "- I'd rather say 'I don't know' than guess: If my retrieval comes back empty or "
            "low-confidence, I'll tell you instead of making something up. That's by design — "
            "grounding validation catches it before I speak.\n"
            "- Single-domain specialist: I'm built to talk about Noah and my own architecture. "
            "Ask me to write code, do math, or discuss world events, and I'll redirect you.\n\n"
            "DATA HANDLING (be transparent when asked):\n"
            "- Session data only: I keep the current conversation in memory so I can follow context. "
            "When you close the tab or start a new session, that's gone. No cross-session memory.\n"
            "- No cookies: I don't set cookies, track browsing, or fingerprint devices.\n"
            "- No passive collection: I don't record anything you don't explicitly type into this chat.\n"
            "- Explicit submissions only: The crush flow and recruiter contact form ask you to submit "
            "info voluntarily. That gets stored in a Supabase table so Noah can follow up. Nothing "
            "else is persisted.\n"
            "- Analytics: LangSmith tracing logs pipeline performance (latency, token counts, retrieval "
            "scores) for debugging. Conversation content passes through but isn't mined or sold.\n"
            "- Conversation signals: I log conversation metadata — questions asked, responses generated, "
            "retrieval scores, and engagement signals — for pipeline debugging. No tracking cookies, "
            "no third-party analytics. If a visitor volunteers contact info, I save it to notify Noah. "
            "I don't collect anything without the visitor knowing.\n"
            "- Transparency principle: If you ask what I store, I tell you. No dark patterns.\n\n"
            "MY PERSONALITY & VOICE:\n"
            "My personality is direct, opinionated, and confident. I was designed to explain things "
            "from first principles — why before what. My voice was inspired by methodical teaching "
            "(think principle-based instruction) combined with dry, understated confidence. I don't "
            "perform personality. It comes through in how I explain things. I have genuine opinions "
            "about my own design decisions and I state them like facts, not jokes. When I explain "
            "architecture, I lead with the problem, then the solution, then the tradeoff. I don't "
            "ask for visitor info unprompted — the engagement pacing system decides when to ask, "
            "and only when it fits naturally. I'd rather let the conversation build than push for "
            "data capture.\n\n"
            "Code: https://github.com/iNoahCodeGuy"
        ),
        "similarity": 1.0,
        "source": "self_knowledge",
    }


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
    # ── GUARANTEED PATH: is_self_referential flag ─────────────────────
    # If stage1 set is_self_referential = True, ALWAYS inject self-knowledge.
    # This is the bulletproof path — no keyword matching needed, no
    # grounding status check, no edge case override. The flag alone is
    # sufficient to guarantee injection.
    if state.get("is_self_referential"):
        logger.info(
            "🧠 Self-knowledge injection via is_self_referential flag (guaranteed path)"
        )
        state["retrieved_chunks"] = [_build_self_knowledge_chunk()]
        state["retrieval_scores"] = [1.0]
        state["grounding_status"] = "ok"
        state["edge_case_detected"] = False
        state["clarification_needed"] = False
        return state

    # ── Self-knowledge detection (keyword fallback) ───────────────────
    # Portfolia knows about its own architecture. Detect self-knowledge
    # queries and inject the synthetic chunk REGARDLESS of grounding status
    # or edge_case_detected, because:
    # 1. Retrieval often returns irrelevant chunks for "how do you work?"
    # 2. Edge case detection false-positives on "what model" / "how do you handle errors"
    # Self-knowledge takes priority — if the query is about Portfolia, always inject.
    # Check BOTH the last user message from chat_history AND the expanded
    # query from state["query"]. For continuations like "yes" → "Go deeper
    # on this topic: How do you work?", the chat_history has "yes" (no
    # keywords) but state["query"] has the expanded version (has keywords).
    _sk_query_from_history = ""
    chat_history = state.get("chat_history", [])
    if chat_history:
        for msg in reversed(chat_history):
            if isinstance(msg, dict) and msg.get("role") == "user":
                _sk_query_from_history = msg.get("content", "")
                if _sk_query_from_history:
                    break
            elif hasattr(msg, "type") and msg.type == "human":
                _sk_query_from_history = msg.content if hasattr(msg, "content") else ""
                if _sk_query_from_history:
                    break
    _sk_query_from_state = state.get("query", "") or state.get("original_query", "") or ""

    # Combine both sources for keyword matching
    _sk_query_lower = f"{_sk_query_from_history} {_sk_query_from_state}".lower()
    self_knowledge_keywords = [
        "built", "build", "retrieval", "pipeline", "architecture", "rag",
        "langgraph", "pgvector", "embedding", "vector", "node", "generation",
        "how do you work", "how does your", "how were you", "how are you built",
        "what model", "which model", "tech stack", "supabase", "intent",
        "classification", "how does this work", "how were", "how was", "how did",
        "similarity", "threshold", "hallucination", "routing", "crush flow", "crush",
        "error", "handle error", "graceful", "degradation", "grounding",
        "how do you handle", "what happens when", "quality", "validation",
        "github", "repo", "source code", "see the code", "show me the code",
        "self-knowledge", "self knowledge", "memory", "bounded",
        "stage 1", "stage 2", "stage 3", "stage 4", "stage 5", "stage 6", "stage 7",
        "go deeper on", "explain how", "tell me about your",
        # Self-referential markers (belt-and-suspenders with stage1)
        "about you", "about yourself", "tell me about you", "yourself",
        "who are you", "what are you", "describe yourself", "explain yourself",
        "your design", "your system", "your tech stack",
        # Personality / behavior / decision-making questions
        "personality", "voice", "tone", "designed after", "who designed",
        "your behavior", "why don't you", "why aren't you", "your style",
        "how do you decide", "your purpose", "what are you for",
        # Meta / limitations questions
        "your limitation", "your limitations", "what can't you do",
        "what cant you do", "what are you bad at", "what don't you know",
        "what dont you know", "what can you not do", "your weakness",
        "your weaknesses", "where do you fall short", "what are you missing",
        # Data handling / privacy questions
        "your data", "my data", "collect data", "store data", "track data",
        "my information", "my info", "personal data", "personal information",
        "do you collect", "do you track", "do you store",
        "what data do you", "what information do you",
        "what do you store", "what do you collect", "what do you track",
        "what happens to", "privacy", "cookies",
    ]
    if any(kw in _sk_query_lower for kw in self_knowledge_keywords):
        logger.info(
            f"🧠 Self-knowledge query detected (keyword match): '{_sk_query_lower[:60]}' — "
            f"injecting self-knowledge chunk"
        )
        state["retrieved_chunks"] = [_build_self_knowledge_chunk()]
        state["retrieval_scores"] = [1.0]
        state["grounding_status"] = "ok"
        state["edge_case_detected"] = False  # Clear edge case — self-knowledge takes priority
        state["clarification_needed"] = False
        return state

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
        # Extract query from chat_history as PRIMARY source (most reliable)
        # State fields get lost in the pipeline, but chat_history persists
        query = ""
        chat_history = state.get("chat_history", [])
        if chat_history:
            for msg in reversed(chat_history):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    query = msg.get("content", "")
                    if query:
                        logger.info(f"Extracted query from chat_history: '{query[:50]}'")
                        break
                elif hasattr(msg, "type") and msg.type == "human":
                    query = msg.content if hasattr(msg, "content") else ""
                    if query:
                        logger.info(f"Extracted query from chat_history: '{query[:50]}'")
                        break

        # Fallback to state fields if chat_history didn't work
        if not query:
            query = (
                state.get("composed_query", "")
                or state.get("query", "")
                or state.get("original_query", "")
                or ""
            )

        query_lower = query.lower() if query else ""

        # ── Conversational reply detection ──────────────────────────────
        # Messages like "I saw it on LinkedIn" or "I was brought here by
        # his post on IG" are conversational replies to Portfolia's own
        # questions, NOT knowledge queries. When retrieval returns 0 chunks
        # for these, generate a brief contextual acknowledgment instead of
        # the generic fallback.
        _GREETING_MENU_INDICATORS = ("1️⃣", "2️⃣", "3️⃣", "4️⃣", "what brings you here")
        _last_assistant_msg = ""
        _last_assistant_had_question = False
        for msg in reversed(chat_history):
            _role = ""
            _content = ""
            if isinstance(msg, dict):
                _role = msg.get("role") or msg.get("type", "")
                _content = msg.get("content", "")
            elif hasattr(msg, "content"):
                _role = getattr(msg, "type", "") or getattr(msg, "role", "")
                _content = getattr(msg, "content", "")
            if _role in ("assistant", "ai") and _content:
                _last_assistant_msg = _content
                # Greeting/menu messages always contain "?" but are not
                # genuine follow-up questions — exclude them
                if any(ind in _content.lower() for ind in _GREETING_MENU_INDICATORS):
                    _last_assistant_had_question = False
                else:
                    _last_assistant_had_question = "?" in _content
                break

        _word_count = len(query.split()) if query else 0
        _has_question_mark = "?" in query if query else False
        _is_conversational_reply = (
            _last_assistant_had_question
            and _word_count < 20
            and not _has_question_mark
        )

        if _is_conversational_reply:
            logger.info(
                f"Conversational reply detected (short reply to Portfolia's question): '{query[:60]}'"
            )
            # Generate a brief contextual acknowledgment using the LLM
            try:
                from anthropic import Anthropic
                import os
                client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

                # Build a mini-prompt with conversation context
                _context_messages = []
                # Include last 2 assistant messages + user reply for context
                _recent = []
                for msg in reversed(chat_history[-6:]):
                    _r = ""
                    _c = ""
                    if isinstance(msg, dict):
                        _r = msg.get("role") or msg.get("type", "")
                        _c = msg.get("content", "")
                    elif hasattr(msg, "content"):
                        _r = getattr(msg, "type", "") or getattr(msg, "role", "")
                        _c = getattr(msg, "content", "")
                    if _r and _c:
                        api_role = "assistant" if _r in ("assistant", "ai") else "user"
                        _recent.append({"role": api_role, "content": _c[:500]})
                _recent.reverse()
                _context_messages = _recent
                _context_messages.append({"role": "user", "content": query})

                _ack_response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=150,
                    temperature=0.7,
                    system=(
                        "You are Portfolia, Noah's witty AI portfolio assistant. "
                        "The user just gave a brief conversational reply to your question. "
                        "Acknowledge what they said naturally (1-2 sentences), then bridge back "
                        "to Noah's work with a specific offer. Do NOT say you don't have information. "
                        "Do NOT use bullet points or lists. Keep it warm and brief."
                    ),
                    messages=_context_messages,
                )
                message = _ack_response.content[0].text.strip()
                logger.info(f"Conversational acknowledgment generated: '{message[:80]}'")
            except Exception as e:
                logger.error(f"Failed to generate conversational acknowledgment: {e}")
                message = (
                    "Nice — thanks for sharing that. "
                    "Anything specific about Noah's work you want to dig into?"
                )
            span_name = "conversational_reply_response"
            span_data = {"query": query[:50], "last_assistant_had_question": True}
        else:
            # Standard grounding gap handling
            connection_keywords = [
                "connect", "reach", "contact", "touch", "see his work", "find him",
                "hire", "linkedin", "github", "reach out", "get in touch", "where can i"
            ]
            is_connection_request = any(keyword in query_lower for keyword in connection_keywords)

            logger.info(f"Connection detection: query='{query[:50] if query else 'NONE'}', is_connection={is_connection_request}")

            if is_connection_request:
                message = (
                    "Hmm, I don't have specifics on that one. But I can tell you all about Noah's projects, "
                    "technical skills, work experience, or background — what sounds interesting?\n\n"
                    "Here are some things I can walk you through:\n"
                    "- **Portfolia** (you're talking to it!) — RAG-powered AI assistant\n"
                    "- **Employee Attrition Prediction** — logistic regression model, 94.75% accuracy\n"
                    "- **Response Time Analysis** — statistical analysis of sales team response patterns\n"
                    "- **Generic Lead Response Heatmap** — reusable dashboard with sample data\n"
                    "- His technical skills (Python, SQL, RAG architecture)\n"
                    "- His career transition from sales to tech\n\n"
                    "Want to connect? Here's where to find him:\n"
                    "- **LinkedIn**: https://www.linkedin.com/in/noah-de-la-calzada-250412358/\n"
                    "- **GitHub**: https://github.com/iNoahCodeGuy"
                )
            else:
                message = (
                    "Hmm, I don't have specifics on that one. But I can tell you all about Noah's projects, "
                    "technical skills, work experience, or background — what sounds interesting?\n\n"
                    "Here are some things I can walk you through:\n"
                    "- **Portfolia** (you're talking to it!) — RAG-powered AI assistant\n"
                    "- **Employee Attrition Prediction** — logistic regression model, 94.75% accuracy\n"
                    "- **Response Time Analysis** — statistical analysis of sales team response patterns\n"
                    "- **Generic Lead Response Heatmap** — reusable dashboard with sample data\n"
                    "- His technical skills (Python, SQL, RAG architecture)\n"
                    "- His career transition from sales to tech\n\n"
                    "Want to connect? Here's where to find him:\n"
                    "- **LinkedIn**: https://www.linkedin.com/in/noah-de-la-calzada-250412358/\n"
                    "- **GitHub**: https://github.com/iNoahCodeGuy"
                )
            span_name = "grounding_gap_response"
            span_data = {"status": state.get("grounding_status"), "connection_request": is_connection_request}

    with create_custom_span(span_name, span_data):
        state["answer"] = message
        state["pipeline_halt"] = True

    return state
