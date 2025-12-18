"""Quality validation for answer relevance, novelty, and conversation guidance.

This module provides quality gates to ensure:
1. Answers are relevant to the query (key term overlap)
2. Answers are not repetitive (novelty check)
3. Conversations are guided naturally (progression patterns)

Scalability features for indefinite conversations (100+ turns):
- Novelty check: Only compares with last 4 responses (not entire history)
- Conversation flow: Only analyzes last 6 messages (3 exchanges)
- Topic tracking: Uses sliding window (last 10 topics)
- All checks are O(1) or O(recent_history), not O(full_history)

Design Principles:
- SRP: Each function handles one quality concern
- Defensibility: Graceful degradation on edge cases
- Observability: Logs quality issues for debugging
- Scalability: Bounded memory usage regardless of conversation length
"""

import logging
import re
from typing import Dict, Any, List, Set

from assistant.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)


def _should_use_teaching_style(state: ConversationState) -> bool:
    """Determine if answer should use John Danaher-style teaching structure.

    Import here to avoid circular dependency.
    """
    from assistant.flows.node_logic.stage5_generation_nodes import _should_use_teaching_style
    return _should_use_teaching_style(state)


def validate_answer_quality(state: ConversationState) -> ConversationState:
    """Quality gate: ensure answer addresses query and isn't repetitive.

    Performs checks:
    1. Relevance: Does answer contain key terms from query?
    2. Novelty: Is answer different from recent responses?
    3. Turn references: Does answer reference previous turns when it should?
    4. Teaching structure: Does answer follow teaching structure when required?

    Scalability: Only compares with last 4 responses (bounded for 100+ turns).

    Args:
        state: ConversationState with query, draft_answer, chat_history

    Returns:
        Updated state with answer_quality_warning if issues detected

    Example:
        >>> state = {"query": "customer support", "draft_answer": "orchestration..."}
        >>> validate_answer_quality(state)
        >>> state.get("answer_quality_warning")
        "answer_relevance_low_0.25"
    """
    query = state.get("query", "").lower()
    answer = state.get("draft_answer", "")

    # Extract content from answer (handles both dict and LangChain message objects)
    if isinstance(answer, dict):
        answer_content = answer.get("content", "")
    elif hasattr(answer, "content"):
        answer_content = getattr(answer, "content", "")
    else:
        answer_content = str(answer) if answer else ""

    answer_lower = answer_content.lower() if answer_content else ""
    chat_history = state.get("chat_history", [])

    if not query or not answer_content:
        return state

    quality_issues = []
    current_turn = state.get("conversation_turn", 0)
    depth_level = state.get("depth_level", 1)
    should_teach = _should_use_teaching_style(state)

    # 1. RELEVANCE CHECK: Extract key terms from query
    query_words = [w for w in query.split() if len(w) > 3]
    if query_words:
        answer_words = set(answer_lower.split())
        overlap = sum(1 for word in query_words if word in answer_words)
        relevance = overlap / len(query_words)

        # Threshold: at least 30% of key terms should appear
        if relevance < 0.3:
            quality_issues.append(f"answer_relevance_low_{relevance:.2f}")
            logger.warning(f"Answer relevance low: {relevance:.2f} for query: {query[:50]}")

    # 2. NOVELTY CHECK: Compare with recent responses (scalable)
    # Only check last 8 messages (4 exchanges) for scalability
    recent_messages = chat_history[-8:] if len(chat_history) > 8 else chat_history
    recent_responses = []
    for msg in recent_messages:
        # Support both dict format and LangGraph message objects (HumanMessage/AIMessage)
        if isinstance(msg, dict):
            content = msg.get("content", "")
            msg_type = msg.get("type", "") or msg.get("role", "")
        else:
            # LangGraph message object (Pydantic model)
            content = getattr(msg, "content", "")
            msg_type = getattr(msg, "type", "") or getattr(msg, "role", "")

        if msg_type in ("ai", "assistant"):
            recent_responses.append(content.lower())

    # Limit to 4 most recent responses (bounded memory)
    recent_responses = recent_responses[-4:]

    for prev_response in recent_responses:
        if prev_response and len(prev_response) > 50:  # Skip very short responses
            # Simple word overlap similarity
            prev_words = set(prev_response.split())
            answer_words_set = set(answer_lower.split())
            if prev_words and answer_words_set:
                intersection = len(prev_words & answer_words_set)
                union = len(prev_words | answer_words_set)
                similarity = intersection / union if union > 0 else 0

                # Threshold: 85% similar = likely duplicate
                if similarity > 0.85:
                    quality_issues.append(f"answer_too_similar_{similarity:.2f}")
                    logger.warning(f"Answer too similar to previous response: {similarity:.2f}")
                    break  # Only flag once

    # 3. TURN REFERENCE VALIDATION (conditional)
    if current_turn >= 3 and not state.get("is_greeting", False):
        has_turn_reference = any(
            re.search(pattern, answer_content, re.IGNORECASE)
            for pattern in [r"turn \d+", r"Turn \d+", r"building on", r"following up", r"we discussed", r"earlier"]
        )

        # Only warn if answer should reference but doesn't (not for every answer)
        if not has_turn_reference and depth_level >= 2:
            # Check if query relates to previous topics
            session_memory = state.get("session_memory", {})
            topics = session_memory.get("topics", [])

            # If query relates to previous topics, should reference
            relates_to_previous = any(topic in query for topic in topics[-3:]) if topics else False
            if relates_to_previous:
                quality_issues.append("answer_missing_turn_reference")
                logger.warning(f"Answer at Turn {current_turn} missing turn references when query relates to previous topics")

    # 4. TEACHING STRUCTURE VALIDATION (only when teaching style should be used)
    if should_teach:
        # Check for systematic enumeration
        has_numbering = bool(re.search(r'\b(1|2|3|4|5)[\.\)]', answer_content))
        has_purpose = "purpose" in answer_lower or "why" in answer_lower

        if not has_numbering:
            quality_issues.append("teaching_structure_missing_enumeration")
            logger.warning("Teaching style answer missing systematic enumeration")
        if not has_purpose:
            quality_issues.append("teaching_structure_missing_purpose")
            logger.warning("Teaching style answer missing purpose statements")

    if quality_issues:
        state["answer_quality_warning"] = "; ".join(quality_issues)
        logger.info(f"Quality issues detected: {quality_issues}")

    return state


def validate_conversation_guidance(state: ConversationState) -> ConversationState:
    """Detect when conversation needs guidance based on progression patterns and phase.

    Checks for:
    1. Natural progression opportunities (orchestration → enterprise)
    2. Stuck patterns (repetitive answers)
    3. Depth progression opportunities (depth 1 → 2 → 3)
    4. Topic accumulation (4+ topics → suggest synthesis)
    5. Phase-aware guidance adjustments

    Scalability: Only analyzes last 6 messages and last 10 topics (bounded).

    Args:
        state: ConversationState with chat_history, session_memory, depth_level, conversation_phase

    Returns:
        Updated state with conversation_guidance_needed list if guidance needed

    Example:
        >>> state = {"topics": ["orchestration"], "query": "enterprise adaptation", "conversation_phase": "exploration"}
        >>> validate_conversation_guidance(state)
        >>> state.get("conversation_guidance_needed")
        ["missing_enterprise_guidance"]
    """
    chat_history = state.get("chat_history", [])
    session_memory = state.get("session_memory", {})
    topics = session_memory.get("topics", [])
    depth_level = state.get("depth_level", 1)
    phase = state.get("conversation_phase", "discovery")

    guidance_needed = []

    # 1. CHECK FOR NATURAL PROGRESSION OPPORTUNITIES
    # Use sliding window: only analyze last 10 topics (scalable)
    recent_topics = topics[-10:] if len(topics) > 10 else topics
    topics_text = " ".join(recent_topics).lower()

    # Analyze conversation flow (only last 6 messages for scalability)
    flow_analysis = _analyze_conversation_flow_scalable(chat_history, session_memory)

    if flow_analysis.get("pattern") == "orchestration_to_enterprise":
        # User is naturally progressing - ensure followups guide this
        followup_prompts = state.get("followup_prompts", [])
        has_enterprise_guidance = any(
            "enterprise" in f.lower() or "adapt" in f.lower() or "customer support" in f.lower()
            for f in followup_prompts
        )
        if not has_enterprise_guidance:
            guidance_needed.append("missing_enterprise_guidance")
            logger.info("Detected orchestration → enterprise progression, guidance needed")

    # 2. CHECK FOR STUCK PATTERNS (repetitive answers)
    if state.get("answer_quality_warning") and "answer_too_similar" in str(state.get("answer_quality_warning")):
        # User is stuck - need to guide to new topic
        guidance_needed.append("stuck_need_redirection")
        logger.info("Detected repetitive answers, redirection needed")

    # 3. CHECK DEPTH PROGRESSION
    # Use bounded topic count (last 10 topics)
    if depth_level == 1 and len(recent_topics) >= 2:
        # User has engaged but still at surface level - suggest going deeper
        guidance_needed.append("suggest_depth_increase")
        logger.info(f"User at depth 1 with {len(recent_topics)} topics, suggest depth increase")

    # 4. CHECK TOPIC ACCUMULATION
    # Only check recent topics (bounded for scalability)
    if len(recent_topics) >= 4 and not flow_analysis.get("has_progression"):
        # Many topics but no clear progression - suggest synthesis
        guidance_needed.append("suggest_synthesis")
        logger.info(f"User has {len(recent_topics)} topics but no clear progression, suggest synthesis")

    # 5. PHASE-AWARE GUIDANCE ADJUSTMENTS
    # Discovery phase: Don't suggest depth increase yet (too early)
    if phase == "discovery":
        if "suggest_depth_increase" in guidance_needed:
            guidance_needed.remove("suggest_depth_increase")
            logger.debug("Removed suggest_depth_increase - still in discovery phase")

    # Synthesis phase: Prioritize synthesis suggestions
    if phase == "synthesis" and "suggest_synthesis" not in guidance_needed:
        if len(recent_topics) >= 4:
            guidance_needed.append("suggest_synthesis")
            logger.info("Synthesis phase with 4+ topics - adding synthesis suggestion")

    # Extended phase: Ensure variety, prevent staleness
    if phase == "extended" and not guidance_needed:
        guidance_needed.append("suggest_new_territory")
        logger.info("Extended phase with no other guidance - suggesting new territory")

    if guidance_needed:
        state["conversation_guidance_needed"] = guidance_needed
        logger.info(f"Conversation guidance needed: {guidance_needed}")

    return state


def detect_conversation_phase(state: ConversationState) -> ConversationState:
    """Determine conversation phase based on turn count and topic accumulation.

    Phases help guide conversation progression and influence followup suggestions:
    - discovery: Turns 1-3, user is exploring/getting oriented
    - exploration: Turns 4-8, user diving into specific areas
    - synthesis: Turns 8+ with 4+ topics, ready for connections
    - extended: Turns 15+, long-running conversation

    Args:
        state: ConversationState with conversation_turn and session_memory

    Returns:
        Updated state with conversation_phase set

    Example:
        >>> state = {"conversation_turn": 5, "session_memory": {"topics": ["orchestration", "architecture"]}}
        >>> detect_conversation_phase(state)
        >>> state.get("conversation_phase")
        "exploration"
    """
    conversation_turn = state.get("conversation_turn", 0)
    session_memory = state.get("session_memory", {})
    topics = session_memory.get("topics", [])

    # Determine phase based on turn count and topic accumulation
    if conversation_turn >= 15:
        phase = "extended"
    elif conversation_turn >= 8 and len(topics) >= 4:
        phase = "synthesis"
    elif conversation_turn >= 4:
        phase = "exploration"
    else:
        phase = "discovery"

    state["conversation_phase"] = phase
    logger.debug(f"Conversation phase: {phase} (turn={conversation_turn}, topics={len(topics)})")

    return state


def _analyze_conversation_flow_scalable(chat_history: List[Dict], session_memory: Dict) -> Dict:
    """Analyze conversation pattern to infer progression (scalable version).

    Only analyzes RECENT messages (last 6) for scalability with 100+ turn conversations.

    Detects patterns like:
    - orchestration → enterprise
    - architecture → implementation
    - general → specific

    Args:
        chat_history: List of message dicts
        session_memory: Dict with topics

    Returns:
        Dict with pattern name and has_progression flag
    """
    patterns = []
    topics = session_memory.get("topics", []) if session_memory else []

    # Use sliding window: only last 10 topics (scalable)
    recent_topics = topics[-10:] if len(topics) > 10 else topics
    topics_text = " ".join(recent_topics).lower() if recent_topics else ""

    # Only analyze RECENT messages (last 6 = 3 exchanges) for scalability
    if not chat_history or len(chat_history) < 2:
        return {"pattern": None, "topics": recent_topics, "has_progression": False}

    recent_messages = chat_history[-6:] if len(chat_history) > 6 else chat_history

    # Extract content from recent messages
    recent_contents = []
    for msg in recent_messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
        elif hasattr(msg, "content"):
            content = msg.content if hasattr(msg, "content") else ""
        else:
            content = ""
        recent_contents.append(content.lower())
    recent_content = " ".join(recent_contents)

    # Detect pattern: orchestration → enterprise
    # Expanded keywords to catch "customer support", "adapt", "use case", etc.
    enterprise_keywords = ["enterprise", "customer support", "adapt", "use case", "production", "scales"]
    if "orchestration" in topics_text or "orchestration" in recent_content:
        # Check recent messages for enterprise/adaptation mentions
        for msg in recent_messages[-2:]:  # Only last 2 messages
            if isinstance(msg, dict):
                content = msg.get("content", "").lower()
            elif hasattr(msg, "content"):
                content = msg.content.lower() if hasattr(msg, "content") else ""
            else:
                content = ""
            if any(kw in content for kw in enterprise_keywords):
                patterns.append("orchestration_to_enterprise")
                break

    # Detect pattern: architecture → implementation
    if "architecture" in recent_content or "architecture" in topics_text:
        if any(kw in recent_content for kw in ["code", "implementation", "how is this built", "show me"]):
            patterns.append("architecture_to_implementation")

    # Detect pattern: general → specific (only if enough history)
    if len(chat_history) >= 4:
        # Compare first vs recent (bounded check)
        first_msg = chat_history[0]
        recent_msg = chat_history[-1]

        # Extract content
        if isinstance(first_msg, dict):
            first_content = first_msg.get("content", "")
            first_role = first_msg.get("role", "") or first_msg.get("type", "")
        else:
            first_content = getattr(first_msg, "content", "") if hasattr(first_msg, "content") else ""
            first_role = getattr(first_msg, "role", "") or getattr(first_msg, "type", "") if hasattr(first_msg, "role") or hasattr(first_msg, "type") else ""
        first_query = first_content.lower() if (first_role == "user" or first_role == "human") else ""

        if isinstance(recent_msg, dict):
            recent_content_msg = recent_msg.get("content", "")
            recent_role = recent_msg.get("role", "") or recent_msg.get("type", "")
        else:
            recent_content_msg = getattr(recent_msg, "content", "") if hasattr(recent_msg, "content") else ""
            recent_role = getattr(recent_msg, "role", "") or getattr(recent_msg, "type", "") if hasattr(recent_msg, "role") or hasattr(recent_msg, "type") else ""
        recent_query = recent_content_msg.lower() if (recent_role == "user" or recent_role == "human") else ""

        # Check if progression from general to specific
        if first_query and recent_query:
            first_word_count = len(first_query.split())
            recent_word_count = len(recent_query.split())
            if first_word_count <= 5 and recent_word_count > 5:
                patterns.append("general_to_specific")

    return {
        "pattern": patterns[0] if patterns else None,
        "topics": recent_topics,
        "has_progression": len(patterns) > 0
    }


def _extract_key_terms(text: str, min_length: int = 4) -> List[str]:
    """Extract key terms from text (words longer than min_length).

    Helper function for relevance checking.

    Args:
        text: Text to extract terms from
        min_length: Minimum word length to consider

    Returns:
        List of key terms
    """
    words = text.lower().split()
    return [w for w in words if len(w) >= min_length]
