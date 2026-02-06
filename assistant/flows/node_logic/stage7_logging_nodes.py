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
import time
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

    # Store topics with pruning for scalability (100+ turns)
    topics = memory.setdefault("topics", [])
    intent = state.get("query_intent")
    if intent and intent not in topics:
        topics.append(intent)

    # Extract topic keywords from query for natural language queries
    # This ensures topics accumulate even for non-menu queries
    query = state.get("query", "").lower()
    topic_keywords = {
        "orchestration": ["orchestration", "langgraph", "pipeline", "flow", "nodes"],
        "architecture": ["architecture", "system", "design", "structure", "how it works"],
        "retrieval": ["retrieval", "rag", "vector", "embedding", "search", "pgvector"],
        "enterprise": ["enterprise", "customer support", "adapt", "production", "use case"],
        "deployment": ["deployment", "vercel", "production", "deploy", "hosting"],
        "data": ["data", "supabase", "database", "storage", "analytics"],
    }

    for topic, keywords in topic_keywords.items():
        if any(kw in query for kw in keywords) and topic not in topics:
            topics.append(topic)
            logger.info(f"📝 Extracted topic from query: {topic}")

    # SCALABILITY: Prune old topics (keep last 10 only)
    # This ensures bounded memory usage for indefinite conversations
    if len(topics) > 10:
        pruned_count = len(topics) - 10
        topics[:] = topics[-10:]
        logger.debug(f"Pruned {pruned_count} old topics for memory efficiency (kept last 10)")

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
                # Validate topic accumulation
                topics_before = len(topics)
                topics.append(topic)
                topics_after = len(topics)
                if topics_after <= topics_before:
                    logger.warning(
                        f"Topic not accumulated: topic={topic}, before={topics_before}, after={topics_after}. "
                        f"This may indicate a duplicate detection issue."
                    )
                logger.info(f"📝 Stored topic from menu selection: {topic} (from menu_context: {menu_context})")

    # Store entities
    entities = state.get("entities", {})
    if entities:
        stored_entities = memory.setdefault("entities", {})
        for key, value in entities.items():
            if value and key not in stored_entities:
                stored_entities[key] = value

    memory["last_grounding_status"] = state.get("grounding_status")

    # =========================================================================
    # PILLAR EXPLORATION TRACKING - Track progress through 4 main knowledge pillars
    # =========================================================================
    from assistant.flows.node_logic.stage6_formatting_nodes import _get_explored_pillars, MAIN_PILLARS

    role_mode = state.get("role_mode", "explorer")
    explored_pillars = _get_explored_pillars(memory, role_mode)

    # Store pillar exploration state
    memory["explored_pillars"] = list(explored_pillars)
    memory["pillars_remaining"] = max(0, 4 - len(explored_pillars))

    # Get pillar config for this role
    pillar_config = MAIN_PILLARS.get(role_mode, {})
    all_pillar_keys = [key for key, _ in pillar_config.get("pillars", [])]
    unexplored = [key for key in all_pillar_keys if key not in explored_pillars]
    memory["unexplored_pillar_keys"] = unexplored

    # Log pillar progress
    if len(explored_pillars) == 1:
        logger.info(f"📊 First pillar explored: {list(explored_pillars)[0]}. {memory['pillars_remaining']} remaining.")
    elif len(explored_pillars) >= 4:
        logger.info("🎯 All 4 main pillars explored - user ready for synthesis/conversion")
        state.setdefault("conversation_guidance_needed", [])
        if "all_pillars_explored" not in state["conversation_guidance_needed"]:
            state["conversation_guidance_needed"].append("all_pillars_explored")
    elif len(explored_pillars) >= 2:
        logger.debug(f"📊 Pillars explored: {list(explored_pillars)}. Unexplored: {unexplored}")

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

    # Append user query and assistant answer to chat_history for conversation continuity
    # This is needed for StateGraph version (LangGraph Studio) which doesn't use run_conversation_flow()
    # Skip only for actual greetings (initial greeting before role selection)
    # Role welcome messages and menu selections are part of conversation and should be preserved
    chat_history = state.get("chat_history", [])

    # #region agent log
    with open('/Users/noahdelacalzada/NoahsAIAssistant/NoahsAIAssistant-/.cursor/debug.log', 'a') as f:
        import json
        f.write(json.dumps({
            "location": "stage7_logging_nodes.py:319",
            "message": "update_memory: Before chat_history append check",
            "data": {
                "has_answer": bool(state.get("answer")),
                "is_greeting": state.get("is_greeting"),
                "answer_preview": (state.get("answer", "")[:100] + "...") if state.get("answer") else None,
                "chat_history_len": len(chat_history),
                "has_query": bool(state.get("query")),
                "query": state.get("query", ""),
                "condition_passes": bool(state.get("answer") and not state.get("is_greeting"))
            },
            "timestamp": int(time.time() * 1000),
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": "D"
        }) + "\n")
    # #endregion

    if state.get("answer") and not state.get("is_greeting"):
        # Validate chat_history accumulation
        previous_len = len(chat_history)

        # Append user query if present and not already in history
        if state.get("query"):
            # Check if this query was already added (avoid duplicates)
            # Support both dict format and LangGraph message objects
            last_user_msg = None
            if chat_history:
                last_msg = chat_history[-1]
                # Check if last message is a user message (both formats)
                if isinstance(last_msg, dict):
                    if last_msg.get("role") == "user" or last_msg.get("type") == "human":
                        last_user_msg = last_msg
                elif hasattr(last_msg, "type"):
                    if last_msg.type == "human" or getattr(last_msg, "role", None) == "user":
                        last_user_msg = last_msg

            # Check if query content matches (avoid duplicates)
            query_matches = False
            if last_user_msg:
                if isinstance(last_user_msg, dict):
                    query_matches = last_user_msg.get("content") == state["query"]
                elif hasattr(last_user_msg, "content"):
                    query_matches = last_user_msg.content == state["query"]

            if not query_matches:
                chat_history.append({"role": "user", "content": state["query"]})
        # Append assistant answer
        chat_history.append({"role": "assistant", "content": state["answer"]})
        state["chat_history"] = chat_history

        # Validate chat_history length increased
        current_len = len(state["chat_history"])
        expected_increase = 1 if state.get("query") else 0  # User message may already be present
        expected_increase += 1  # Assistant answer always added
        if current_len <= previous_len and previous_len > 0:
            logger.warning(
                f"Chat history did not accumulate: previous={previous_len}, current={current_len}. "
                f"This may indicate duplicate detection or append logic failure."
            )

        # #region agent log
        with open('/Users/noahdelacalzada/NoahsAIAssistant/NoahsAIAssistant-/.cursor/debug.log', 'a') as f:
            import json
            # Convert LangGraph message objects to serializable format
            messages_serializable = []
            for msg in chat_history:
                if isinstance(msg, dict):
                    messages_serializable.append(msg)
                elif hasattr(msg, "type") or hasattr(msg, "content"):
                    # LangGraph message object - convert to dict
                    messages_serializable.append({
                        "type": getattr(msg, "type", None) or getattr(msg, "role", None),
                        "content": getattr(msg, "content", str(msg))[:200] if hasattr(msg, "content") else str(msg)[:200]
                    })
                else:
                    messages_serializable.append(str(msg)[:200])

            f.write(json.dumps({
                "location": "stage7_logging_nodes.py:329",
                "message": "update_memory: After chat_history append",
                "data": {
                    "chat_history_len": len(chat_history),
                    "messages": messages_serializable
                },
                "timestamp": int(time.time() * 1000),
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": "D"
            }) + "\n")
        # #endregion

        logger.debug(f"Appended to chat_history: {len(chat_history)} messages total")

    # Backup chat_history to session_memory for persistence
    # This ensures conversation context persists even if frontend doesn't send it back
    # SCALABILITY: Only store last 6 messages (bounded for 100+ turns)
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

    # Track progressive inference metrics
    conversation_turn = state.get("conversation_turn", 0)
    topics = memory.get("topics", [])
    chat_history = state.get("chat_history", [])
    depth_level = state.get("depth_level", 1)

    metrics = memory.setdefault("progressive_inference_metrics", {
        "turn_count": 0,
        "topics_count": 0,
        "chat_history_length": 0,
        "depth_level": 1,
        "retrieval_similarity_history": [],
    })

    # Update metrics
    metrics["turn_count"] = conversation_turn
    metrics["topics_count"] = len(topics)
    metrics["chat_history_length"] = len(chat_history)
    metrics["depth_level"] = depth_level

    # Store retrieval similarity for this turn
    if state.get("retrieval_scores") and state["retrieval_scores"]:
        similarity = state["retrieval_scores"][0]
        metrics["retrieval_similarity_history"].append({
            "turn": conversation_turn,
            "similarity": similarity,
            "timestamp": time.time()
        })
        # Keep only last 10 similarity scores
        metrics["retrieval_similarity_history"] = metrics["retrieval_similarity_history"][-10:]

    # TOPIC EXHAUSTION TRACKING - For indefinite conversations (100+ turns)
    # Track how many times each topic has been discussed to detect exhaustion
    topic_query_counts = memory.setdefault("topic_query_counts", {})
    current_topic = state.get("query_intent") or _extract_current_topic(query)

    if current_topic:
        topic_query_counts[current_topic] = topic_query_counts.get(current_topic, 0) + 1

        # Mark topics as exhausted if discussed 3+ times
        exhausted_topics = memory.setdefault("exhausted_topics", [])
        if topic_query_counts[current_topic] >= 3 and current_topic not in exhausted_topics:
            exhausted_topics.append(current_topic)
            logger.info(f"📚 Topic marked as exhausted: {current_topic} (discussed {topic_query_counts[current_topic]} times)")

    # Track unexplored topics based on conversation arcs
    _update_unexplored_topics(state, memory)

    # Log success criteria status
    _log_success_criteria(state, memory)

    # SCALABILITY: Comprehensive memory pruning for indefinite conversations (100+ turns)
    # This ensures bounded memory usage regardless of conversation length
    _prune_session_memory(memory)

    return state


def _prune_session_memory(memory: Dict) -> None:
    """Prune session memory to prevent unbounded growth in long conversations.

    Implements sliding window approach for all memory structures to ensure
    conversations can continue indefinitely (100+ turns) without memory bloat.

    Pruning strategy:
    - Topics: Keep last 10 (already done inline)
    - Discussed files: Keep last 10 (already done inline)
    - Retrieval similarity history: Keep last 10 (already done inline)
    - Entities: Keep last 20 (prune oldest)
    - Chat history backup: Keep last 6 messages (already done inline)

    This function consolidates all pruning logic for maintainability.

    Args:
        memory: Session memory dict to prune
    """
    # Prune entities (keep last 20)
    entities = memory.get("entities", {})
    if len(entities) > 20:
        # Convert to list of tuples, keep last 20, convert back to dict
        entity_items = list(entities.items())
        pruned_count = len(entity_items) - 20
        memory["entities"] = dict(entity_items[-20:])
        logger.debug(f"Pruned {pruned_count} old entities for memory efficiency (kept last 20)")

    # Note: Topics, discussed_files, retrieval_similarity_history, and chat_history_backup
    # are already pruned inline in update_memory function

    logger.debug(
        f"Memory pruning complete: "
        f"topics={len(memory.get('topics', []))}, "
        f"entities={len(memory.get('entities', {}))}, "
        f"discussed_files={len(memory.get('discussed_files', []))}, "
        f"chat_backup={len(memory.get('chat_history_backup', []))}"
    )


def _log_success_criteria(state: ConversationState, session_memory: Dict) -> None:
    """Log whether success criteria are met for this turn."""
    conversation_turn = state.get("conversation_turn", 0)
    chat_history = state.get("chat_history", [])
    topics = session_memory.get("topics", [])
    depth_level = state.get("depth_level", 1)

    criteria = {
        "turn": conversation_turn,
        "chat_history_length": len(chat_history),
        "topics_count": len(topics),
        "depth_level": depth_level,
        "chat_history_preserved": len(chat_history) >= conversation_turn * 2 if conversation_turn > 0 else True,
        "topics_accumulating": len(topics) >= max(0, conversation_turn - 1) if conversation_turn > 0 else True,
        "depth_progressing": depth_level >= min(2, max(1, conversation_turn)) if conversation_turn >= 2 else True,
    }

    # Log if any criteria fail
    if conversation_turn > 0:
        if not criteria["chat_history_preserved"] and conversation_turn > 1:
            logger.warning(
                f"Success criteria failure at turn {conversation_turn}: "
                f"chat_history not preserved (expected >= {conversation_turn * 2}, got {len(chat_history)})"
            )
        if not criteria["topics_accumulating"] and conversation_turn >= 3:
            logger.warning(
                f"Success criteria failure at turn {conversation_turn}: "
                f"topics not accumulating (expected >= {conversation_turn - 1}, got {len(topics)})"
            )
        if not criteria["depth_progressing"] and conversation_turn >= 2:
            logger.warning(
                f"Success criteria failure at turn {conversation_turn}: "
                f"depth not progressing (expected >= 2, got {depth_level})"
            )

    # Store criteria in state for analytics
    state.setdefault("success_criteria", criteria)


def _extract_current_topic(query: str) -> str:
    """Extract the current topic from a query string.

    Args:
        query: User's query string

    Returns:
        Extracted topic or empty string
    """
    query_lower = query.lower()

    # Topic detection patterns
    topic_patterns = {
        "orchestration": ["orchestration", "langgraph", "pipeline", "flow", "nodes", "graph"],
        "architecture": ["architecture", "system design", "how it works", "structure"],
        "rag": ["rag", "retrieval", "vector", "embedding", "search", "pgvector"],
        "enterprise": ["enterprise", "customer support", "adapt", "production", "use case", "scale"],
        "deployment": ["deployment", "vercel", "deploy", "hosting", "serverless"],
        "data": ["data", "supabase", "database", "storage", "analytics"],
        "cost": ["cost", "pricing", "tokens", "budget", "expensive"],
        "testing": ["testing", "qa", "test", "pytest", "validation"],
        "observability": ["observability", "langsmith", "tracing", "monitoring", "logs"],
        "career": ["noah", "career", "background", "experience", "tesla"],
    }

    for topic, keywords in topic_patterns.items():
        if any(kw in query_lower for kw in keywords):
            return topic

    return ""


def _update_unexplored_topics(state: ConversationState, memory: Dict) -> None:
    """Update the list of unexplored topics based on conversation arcs.

    This enables Portfolia to guide users to new territory when current
    topics are exhausted.

    Args:
        state: Conversation state
        memory: Session memory dict
    """
    role_mode = state.get("role_mode", "explorer")
    explored_topics = set(memory.get("topics", []))
    exhausted_topics = set(memory.get("exhausted_topics", []))

    # Define all available topics per role
    all_topics_by_role = {
        "hiring_manager_technical": [
            "tech_stack", "architecture", "orchestration", "rag_pipeline", "data_layer",
            "langgraph_nodes", "state_management", "retrieval", "generation", "observability",
            "enterprise_adaptation", "customer_support", "internal_docs", "sales_enablement",
            "cost_analysis", "deployment", "scaling", "testing", "monitoring",
            "noahs_background", "projects", "philosophy"
        ],
        "software_developer": [
            "architecture", "code_structure", "module_design", "implementation",
            "langgraph_flow", "pgvector_queries", "debugging", "testing", "performance",
            "deployment", "monitoring"
        ],
        "hiring_manager_nontechnical": [
            "career", "achievements", "business_impact", "team_fit", "resume", "contact"
        ],
        "explorer": [
            "casual", "what_is_this", "architecture", "behind_scenes", "fun_facts", "mma_background"
        ]
    }

    all_topics = set(all_topics_by_role.get(role_mode, []))

    # Calculate unexplored topics (not in explored and not exhausted)
    unexplored = all_topics - explored_topics - exhausted_topics

    # Store in memory for use by followup generation
    memory["unexplored_topics"] = list(unexplored)[:10]  # Keep top 10

    # Log for debugging
    if unexplored:
        logger.debug(f"📍 Unexplored topics for {role_mode}: {list(unexplored)[:5]}...")


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
