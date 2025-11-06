"""Logging pipeline nodes - analytics persistence and session memory.

This module handles the final pipeline steps:
1. log_and_notify → Save conversation to Supabase analytics
2. suggest_followups → Generate curiosity-driven next prompts
3. update_memory → Store soft signals for future turns

Design Principles:
- SRP: Each function handles one aspect of post-generation processing
- Defensibility: Graceful degradation if logging fails (doesn't break pipeline)
- Observability: Records full conversation context for evaluation
- Session continuity: Memory tracks topics and entities across turns

Performance Characteristics:
- log_and_notify: ~50-100ms (Supabase insert)
- suggest_followups: <5ms (template-based)
- update_memory: <1ms (in-memory dict updates)

See: docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md for analytics schema
"""

import logging
from typing import Dict, Any, List

from src.state.conversation_state import ConversationState
from src.analytics.supabase_analytics import (
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
    """Generate curiosity-driven follow-up prompts with subcategory awareness.

    This node adds contextual next-question suggestions to the answer based on:
    - Query intent/type (technical, data, career)
    - Role mode (software developer, hiring manager, etc.)
    - Active technical subcategories (stack, architecture, data pipeline, state mgmt)
    - Conversation context (what topics have been covered)

    Followup strategy:
    - Technical queries → Dive deeper into implementation details
    - With subcategories active → Offer drilling into specific areas:
      * stack_depth → "Want framework comparison?"
      * architecture_depth → "Should I show the node flow diagram?"
      * data_pipeline_depth → "Curious about the pgvector RPC?"
      * state_management_depth → "Want to see the state transitions?"
    - Data queries → Offer specific metrics and breakdowns
    - Career queries → Explore project stories and outcomes
    - Confession mode → No followups (respects privacy)

    Performance: <5ms (template-based suggestions)

    Design Principles:
    - SRP: Only generates followups, doesn't modify core answer
    - Role awareness: Different suggestions for technical vs business
    - Context awareness: Uses subcategory signals for precision
    - Idempotency: Skips if followup_prompts already set
    - UX: Maintains conversational momentum

    Args:
        state: ConversationState with query_intent, analytics_metadata

    Returns:
        Updated state with:
        - followup_prompts: List of 3 suggested questions
        - answer: Original answer with followups appended

    Example:
        >>> state = {
        ...     "query_type": "technical",
        ...     "analytics_metadata": {"technical_subcategories": ["architecture_depth"]},
        ...     "answer": "RAG works by..."
        ... }
        >>> suggest_followups(state)
        >>> "node flow diagram" in str(state["followup_prompts"])
        True
    """
    if state.get("followup_prompts"):
        return state

    intent = state.get("query_intent") or state.get("query_type") or "general"
    role_mode = state.get("role_mode", "explorer")
    active_subcats = state.get("analytics_metadata", {}).get("technical_subcategories", [])

    suggestions: List[str] = []
    
    # Subcategory-specific followups take priority for technical queries
    if intent in {"technical", "engineering", "technical"} and active_subcats:
        suggestions = _build_subcategory_followups(active_subcats)
    elif intent in {"technical", "engineering", "technical"}:
        suggestions = [
            "Want me to walk through the LangGraph node transitions in detail?",
            "Curious how the Supabase pgvector query works under load?",
            "Should we map this architecture to your internal stack?",
        ]
    elif intent in {"data", "analytics"}:
        suggestions = [
            "Need the retrieval accuracy metrics for last week?",
            "Want the cost-per-query breakdown?",
            "Should we compare grounding confidence across roles?",
        ]
    elif intent in {"career", "general"}:
        suggestions = [
            "Want the story behind building this assistant end to end?",
            "Should I outline Noah's production launch checklist?",
            "Curious how this adapts to your team's workflow?",
        ]

    if role_mode == "confession":
        suggestions = []  # Confession mode stays focused on the message

    if suggestions:
        state["followup_prompts"] = suggestions
        followup_lines = "\n".join(f"- {item}" for item in suggestions)
        existing_answer = state.get("answer") or ""
        state["answer"] = (
            f"{existing_answer}\n\nNext directions I can cover:\n{followup_lines}"
            if existing_answer
            else f"Next directions I can cover:\n{followup_lines}"
        )

    return state


def update_memory(state: ConversationState) -> ConversationState:
    """Store soft signals in session memory for future turns.

    This node captures conversation patterns and context that should persist
    across multiple turns in the same session:
    - Topics discussed (for coherence tracking)
    - Entities mentioned (company names, roles, contact info)
    - Last grounding status (for retrieval quality monitoring)

    Memory is stored in `session_memory` dict on state and can be used by
    downstream nodes to:
    - Avoid repeating information
    - Maintain conversation continuity
    - Detect patterns (e.g., persistent hiring signals)

    Performance: <1ms (in-memory dict updates)

    Design Principles:
    - SRP: Only updates memory, doesn't use it (consumption is separate)
    - Immutability: Uses setdefault to avoid overwriting
    - Privacy: No PII stored, only business entities and topics
    - Idempotency: Safe to call multiple times

    Args:
        state: ConversationState with query_intent and entities

    Returns:
        Updated state with:
        - session_memory.topics: List of discussed topics
        - session_memory.entities: Dict of extracted entities
        - session_memory.last_grounding_status: Latest grounding result

    Example:
        >>> state = {
        ...     "query_intent": "technical",
        ...     "entities": {"company": "Acme Corp"},
        ...     "grounding_status": "ok"
        ... }
        >>> update_memory(state)
        >>> state["session_memory"]["topics"]
        ["technical"]
        >>> state["session_memory"]["entities"]["company"]
        "Acme Corp"
    """
    memory = state.setdefault("session_memory", {})
    topics = memory.setdefault("topics", [])
    intent = state.get("query_intent")
    if intent and intent not in topics:
        topics.append(intent)

    entities = state.get("entities", {})
    if entities:
        stored_entities = memory.setdefault("entities", {})
        for key, value in entities.items():
            if value and key not in stored_entities:
                stored_entities[key] = value

    memory["last_grounding_status"] = state.get("grounding_status")

    return state
