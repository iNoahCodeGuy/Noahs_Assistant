"""Session memory and analytics logging nodes.

This module handles state persistence and analytics tracking:
1. log_and_notify → Persist interaction to analytics table, trigger notifications
2. update_memory → Store soft signals in session_memory for next turn (includes affinity tracking)

Merged suggest_followups logic into formatting_nodes.format_answer for streamlined formatting.

Design Principles:
- SRP: Each function handles one logging concern
- Idempotency: Safe to call multiple times (updates/inserts)
- Observability: LangSmith tracing for analytics performance
- Reliability: Graceful degradation if Supabase unavailable

Performance Characteristics:
- log_and_notify: ~100ms (Supabase insert + optional Resend/Twilio)
- update_memory: <1ms (in-memory dict update + affinity scoring)

See: docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md for full pipeline flow
"""

import logging
from typing import Dict, Any, List

from assistant.state.conversation_state import ConversationState
from assistant.analytics.supabase_analytics import (
    supabase_analytics,
    UserInteractionData,
    RetrievalLogData,
)

logger = logging.getLogger(__name__)


def _build_subcategory_followups(active_subcats: List[str]) -> List[str]:
    """Generate followup questions based on active technical subcategories.

    Maps subcategory focus to natural next-step questions that keep
    users drilling into the technical lane they've signaled interest in.

    Args:
        active_subcats: List of active subcategory names

    Returns:
        List of 3 contextual followup questions

    Example:
        >>> _build_subcategory_followups(["stack_depth", "architecture_depth"])
        ["Want to compare LangChain vs LlamaIndex trade-offs?", ...]
    """
    suggestions = []

    if "stack_depth" in active_subcats:
        suggestions.append("Want to compare LangChain vs LlamaIndex trade-offs?")
        suggestions.append("Should I break down the requirements.txt dependencies?")

    if "architecture_depth" in active_subcats:
        suggestions.append("Want the LangGraph node flow diagram?")
        suggestions.append("Should I map this architecture to your team's stack?")

    if "data_pipeline_depth" in active_subcats:
        suggestions.append("Curious about the pgvector RPC implementation?")
        suggestions.append("Want to see the embedding generation flow?")

    if "state_management_depth" in active_subcats:
        suggestions.append("Should I show the ConversationState transitions?")
        suggestions.append("Want to trace a query through the pipeline?")

    # If we have multiple categories, prioritize the most recent ones
    # Default fallback
    if not suggestions:
        suggestions = [
            "Want me to walk through the LangGraph node transitions in detail?",
            "Curious how the Supabase pgvector query works under load?",
            "Should we map this architecture to your internal stack?",
        ]

    # Return exactly 3 suggestions
    return suggestions[:3] if len(suggestions) >= 3 else suggestions + [
        "Ask me anything else about the implementation!",
        "Want to see how this adapts to your use case?",
    ][:3 - len(suggestions)]


logger = logging.getLogger(__name__)


def log_and_notify(
    state: ConversationState,
    session_id: str,
    latency_ms: int,
    success: bool = True
) -> Dict[str, Any]:
    """Save analytics to Supabase and trigger notifications.

    This is the last step in the pipeline. It records the conversation
    to the database for evaluation and potential follow-up.

    Analytics tables:
    - user_interactions: Full conversation context (query, answer, role, latency)
    - retrieval_logs: Retrieval performance (chunk IDs, scores, grounding status)

    Metadata captured:
    - session_id: Unique session identifier for multi-turn tracking
    - role_mode: User's selected role (affects analytics segmentation)
    - query_type: Classification result (technical, career, data, etc.)
    - latency_ms: End-to-end pipeline duration
    - success: Whether pipeline completed without errors

    Performance: ~50-100ms (Supabase insert with network latency)

    Design Principles:
    - **SRP**: Only handles analytics logging, doesn't modify answer
    - **Defensibility**: Gracefully handles logging failures
    - **Loose Coupling**: Returns partial update with metadata only
    - **Observability**: Logs failures but doesn't crash pipeline

    Args:
        state: Current conversation state with final answer
        session_id: Unique session identifier
        latency_ms: How long the conversation took
        success: Whether the conversation completed successfully

    Returns:
        Partial state update dict with analytics metadata

    Example:
        >>> state = {
        ...     "query": "How does RAG work?",
        ...     "answer": "RAG works by...",
        ...     "role": "Software Developer",
        ...     "query_type": "technical"
        ... }
        >>> log_and_notify(state, session_id="abc123", latency_ms=1250)
        >>> state["analytics_metadata"]["logged_at"]
        True
    """
    # Access required fields with fail-safe defaults
    role = state.get("role", "Just looking around")
    query = state.get("query", "")
    answer = state.get("answer", "")
    query_type = state.get("query_type", "general")

    # Initialize update dict
    update: Dict[str, Any] = {}

    try:
        interaction = UserInteractionData(
            session_id=session_id,
            role_mode=role,
            query=query,
            answer=answer,
            query_type=query_type,
            latency_ms=latency_ms,
            success=success
        )
        message_id = supabase_analytics.log_interaction(interaction)

        # Store analytics metadata in state (inside analytics_metadata dict)
        if "analytics_metadata" not in state:
            state["analytics_metadata"] = {}
        state["analytics_metadata"]["message_id"] = message_id
        state["analytics_metadata"]["logged_at"] = True

        # Log retrieval performance if we retrieved chunks
        if message_id and state.get("retrieved_chunks"):
            topk_ids = [chunk.get("id") for chunk in state["retrieved_chunks"] if chunk.get("id")]
            scores = state.get("retrieval_scores", [])
            retrieval_log = RetrievalLogData(
                message_id=message_id,
                topk_ids=topk_ids,
                scores=scores,
                grounded=state.get("grounding_status") == "ok",
            )
            supabase_analytics.log_retrieval(retrieval_log)
    except Exception as exc:
        logger.error("Failed logging analytics: %s", exc)
        if "analytics_metadata" not in state:
            state["analytics_metadata"] = {}
        state["analytics_metadata"]["logged_at"] = False

    # Note: update dict no longer needed since we write directly to state
    # When we migrate to LangGraph StateGraph, this will return partial dict only
    return state


def suggest_followups(state: ConversationState) -> ConversationState:
    """DEPRECATED: No-op for backward compatibility.

    Logic merged into formatting_nodes.format_answer() for streamlined formatting.
    Followup generation now happens during the final answer structuring phase
    with full access to active subcategories and presentation context.

    Kept for import compatibility only. New code should rely on format_answer()
    to generate followups automatically.
    """
    return state


def update_memory(state: ConversationState) -> ConversationState:
    """Store soft signals and affinity scores in session memory for future turns.

    This node captures conversation patterns and context that should persist
    across multiple turns in the same session:
    - Topics discussed (for coherence tracking)
    - Entities mentioned (company names, roles, contact info)
    - Last grounding status (for retrieval quality monitoring)
    - Enterprise affinity score (0-4 scale tracking business focus)
    - Technical affinity score (0-4 scale tracking technical depth preference)
    - Technical sub-category scores (stack, architecture, data, state management)

    Memory is stored in `session_memory` dict on state and can be used by
    downstream nodes to:
    - Avoid repeating information
    - Maintain conversation continuity
    - Detect patterns (e.g., persistent hiring signals)
    - Adapt presentation depth based on user's progressive interest

    Performance: <1ms (in-memory dict updates)

    Design Principles:
    - SRP: Only updates memory, doesn't use it (consumption is separate)
    - Immutability: Uses setdefault to avoid overwriting
    - Privacy: No PII stored, only business entities and topics
    - Idempotency: Safe to call multiple times

    Affinity Score Mechanics:
        Enterprise (0-4): +2 for business keywords, -1 for technical → threshold ≥2
        Technical (0-4): +2 for code/implementation, -1 for business → threshold ≥2
        Sub-categories: stack_depth, architecture_depth, data_pipeline_depth, state_management_depth

    Args:
        state: ConversationState with query_intent and entities

    Returns:
        Updated state with:
        - session_memory.topics: List of discussed topics
        - session_memory.entities: Dict of extracted entities
        - session_memory.last_grounding_status: Latest grounding result
        - session_memory.persona_hints.enterprise_relevance_score: Enterprise affinity (0-4)
        - session_memory.persona_hints.technical_relevance_score: Technical affinity (0-4)
        - relate_to_enterprise: Boolean flag (score ≥2)
        - show_technical_depth: Boolean flag (score ≥2)

    Example:
        >>> state = {
        ...     "query": "How does governance work?",
        ...     "query_intent": "business_value",
        ...     "entities": {"company": "Acme Corp"},
        ...     "grounding_status": "ok"
        ... }
        >>> update_memory(state)
        >>> state["session_memory"]["persona_hints"]["enterprise_relevance_score"]
        2  # +2 from "governance" keyword
        >>> state["relate_to_enterprise"]
        True  # score ≥2
    """
    memory = state.setdefault("session_memory", {})

    # Store topics
    topics = memory.setdefault("topics", [])
    intent = state.get("query_intent")
    if intent and intent not in topics:
        topics.append(intent)

    # Extract topics from menu selections for progressive inference
    query_type = state.get("query_type")
    if query_type == "menu_selection":
        entities = state.get("entities", {})
        menu_context = entities.get("menu_context")
        if menu_context:
            # Extract topic from menu context (e.g., "orchestration_layer" → "orchestration")
            topic_map = {
                "orchestration_layer": "orchestration",
                "full_tech_stack": "architecture",
                "enterprise_adaptation": "enterprise",
                "technical_background": "career"
            }
            # Try mapping first, fallback to extracting key word
            topic = topic_map.get(menu_context)
            if not topic:
                # Extract key word (remove "_layer", "_adaptation", etc.)
                topic = menu_context.split("_")[0] if "_" in menu_context else menu_context

            if topic and topic not in topics:
                topics.append(topic)
                logger.info(f"📝 Stored topic from menu selection: {topic} (from menu_context: {menu_context})")

    # Store entities
    entities = state.get("entities", {})
    if entities:
        stored_entities = memory.setdefault("entities", {})
        for key, value in entities.items():
            if value and key not in stored_entities:
                stored_entities[key] = value

    memory["last_grounding_status"] = state.get("grounding_status")

    # Track discussed code files for conversation-aware code index
    discussed_files = memory.setdefault("discussed_files", [])
    retrieved_chunks = state.get("retrieved_chunks", [])

    for chunk in retrieved_chunks:
        # Extract file paths from codebase chunks
        if chunk.get("doc_id") == "codebase":
            file_path = chunk.get("section") or chunk.get("metadata", {}).get("file_path")
            if file_path and file_path not in discussed_files:
                discussed_files.append(file_path)
                logger.debug(f"Tracked discussed file: {file_path}")

    # Limit to last 10 files to prevent memory bloat
    memory["discussed_files"] = discussed_files[-10:]

    # Backup chat_history to session_memory for persistence
    # This ensures conversation context persists even if frontend doesn't send it back
    chat_history = state.get("chat_history", [])
    if chat_history:
        # Store last 6 messages (3 exchanges) for context continuity
        memory["chat_history_backup"] = chat_history[-6:]
        logger.debug(f"Backed up chat_history to session_memory: {len(chat_history)} messages")

    # Restore chat_history from backup if it's empty but backup exists
    if not chat_history and memory.get("chat_history_backup"):
        state["chat_history"] = memory["chat_history_backup"]
        logger.debug(f"Restored chat_history from backup: {len(memory['chat_history_backup'])} messages")

    # Update enterprise affinity (merged from update_enterprise_affinity)
    _update_enterprise_affinity(state, memory)

    # Update technical affinity (merged from update_technical_affinity)
    _update_technical_affinity(state, memory)

    return state


def _update_enterprise_affinity(state: ConversationState, memory: dict) -> None:
    """Adjust enterprise framing based on query focus (internal helper)."""
    query_lower = state.get("query", "").lower()
    persona_hints = memory.setdefault("persona_hints", {})
    current_score = persona_hints.get("enterprise_relevance_score", 0)

    enterprise_keywords = {
        "governance", "scale", "enterprise", "rollout", "value", "roi",
        "team", "compliance", "production", "deployment", "multi-tenant",
        "audit", "business", "customer", "operational", "reliability"
    }

    technical_keywords = {
        "code", "implementation", "trace", "architecture", "pipeline",
        "model", "algorithm", "function", "debug", "error", "bug",
        "test", "how does", "how do", "explain the", "walk me through"
    }

    has_enterprise = any(kw in query_lower for kw in enterprise_keywords)
    has_technical = any(kw in query_lower for kw in technical_keywords)

    if has_enterprise:
        current_score = min(current_score + 2, 4)
    elif has_technical:
        current_score = max(current_score - 1, 0)

    persona_hints["enterprise_relevance_score"] = current_score
    state["relate_to_enterprise"] = current_score >= 2
    state.setdefault("analytics_metadata", {})["enterprise_affinity_score"] = current_score


def _update_technical_affinity(state: ConversationState, memory: dict) -> None:
    """Adjust technical depth preference based on query focus (internal helper)."""
    query_lower = state.get("query", "").lower()
    persona_hints = memory.setdefault("persona_hints", {})
    current_score = persona_hints.get("technical_relevance_score", 0)

    technical_keywords = {
        "code", "implementation", "how does", "how do", "architecture",
        "algorithm", "trace", "debug", "api", "function", "error", "bug",
        "test", "pipeline", "model", "retrieval", "embedding", "vector",
        "sql", "query", "langgraph", "node", "workflow", "state", "async",
        "performance", "latency", "optimization", "refactor", "class", "method"
    }

    business_keywords = {
        "roi", "value", "outcome", "business", "stakeholder", "budget",
        "governance", "compliance", "rollout", "why", "what benefit",
        "customer", "user", "team", "scale", "enterprise", "production",
        "career", "experience", "background", "tell me about", "overview"
    }

    has_technical = any(kw in query_lower for kw in technical_keywords)
    has_business = any(kw in query_lower for kw in business_keywords)

    if has_technical:
        current_score = min(current_score + 2, 4)
    elif has_business:
        current_score = max(current_score - 1, 0)

    persona_hints["technical_relevance_score"] = current_score
    state["show_technical_depth"] = current_score >= 2

    # Update technical sub-categories for specialized presentation
    _update_technical_subcategories(state, query_lower, persona_hints)

    state.setdefault("analytics_metadata", {})["technical_affinity_score"] = current_score


def _update_technical_subcategories(state: ConversationState, query_lower: str, persona_hints: dict) -> None:
    """Update fine-grained technical focus scores for specialized presentation (internal helper)."""
    categories = {
        "stack_depth_score": {
            "stack", "tech stack", "dependencies", "framework", "library",
            "python", "langchain", "openai", "supabase", "streamlit",
            "vercel", "postgres", "pgvector", "requirements", "package"
        },
        "architecture_depth_score": {
            "architecture", "design", "pattern", "component", "module",
            "system", "structure", "diagram", "flow", "orchestration",
            "microservices", "monolith", "separation of concerns", "layer"
        },
        "data_pipeline_depth_score": {
            "data", "pipeline", "etl", "vector", "embedding", "retrieval",
            "pgvector", "rag", "knowledge base", "chunking", "indexing",
            "search", "similarity", "ranking", "grounding", "storage"
        },
        "state_management_depth_score": {
            "state", "node", "langgraph", "workflow", "graph", "edge",
            "conversation state", "memory", "session", "turn", "pipeline",
            "orchestration", "conversation flow", "state machine"
        },
    }

    # Apply decay (-1) to all categories not mentioned this turn
    for score_key in categories.keys():
        current = persona_hints.get(score_key, 0)
        persona_hints[score_key] = max(current - 1, 0)

    # Boost (+2, cap at 4) categories with keyword matches
    for score_key, keywords in categories.items():
        if any(kw in query_lower for kw in keywords):
            current = persona_hints.get(score_key, 0)
            persona_hints[score_key] = min(current + 2, 4)

    # Store active focuses in analytics for retrieval strategy decisions
    active_focuses = [
        category.replace("_score", "")
        for category, score in persona_hints.items()
        if category.endswith("_score") and score >= 2
    ]
    state.setdefault("analytics_metadata", {})["technical_subcategories"] = active_focuses
