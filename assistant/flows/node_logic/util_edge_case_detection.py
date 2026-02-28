"""Unified edge case detection for conversational handling.

All edge cases follow the same pattern:
1. Detect early (in classify_intent)
2. Flag in state
3. Generate conversational response (not template)
4. Offer meta-teaching (for technical users)
5. Continue gracefully (don't crash pipeline)

This module provides detection functions for all edge cases that Portfolia
needs to handle gracefully, turning potential failures into teaching moments.
"""

import re
import logging
import time
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter

from assistant.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)


# ============================================================================
# Edge Case Detection Functions
# ============================================================================

def detect_edge_cases(state: ConversationState) -> Dict[str, Any]:
    """Detect all edge cases and return flags.

    This is the main entry point for edge case detection. It checks all
    possible edge cases and returns the first one detected (priority order).

    Args:
        state: Current conversation state with query and chat_history

    Returns:
        Dict with edge case flags and metadata. If no edge case detected,
        returns empty dict. If edge case found, includes:
        - edge_case_type: str identifier for the edge case
        - Additional metadata specific to the edge case type
    """
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])
    session_memory = state.get("session_memory", {})

    edge_cases = {}

    # Priority order: Check most specific first, then general cases

    # 1. Empty/whitespace query (check first - simplest case)
    if _is_empty_or_whitespace(query):
        edge_cases["is_empty_query"] = True
        edge_cases["edge_case_type"] = "empty_query"
        return edge_cases

    # 2. Very long query (check before processing)
    if _is_very_long_query(query):
        edge_cases["is_very_long_query"] = True
        edge_cases["edge_case_type"] = "very_long_query"
        edge_cases["query_length"] = len(query)
        return edge_cases

    # 3. Off-topic detection
    if _is_off_topic(query, chat_history):
        edge_cases["is_off_topic"] = True
        edge_cases["edge_case_type"] = "off_topic"
        return edge_cases

    # 4. Nonsensical/gibberish query
    if _is_nonsensical_query(query):
        edge_cases["is_nonsensical_query"] = True
        edge_cases["edge_case_type"] = "nonsensical_query"
        return edge_cases

    # 5. Emoji-only query
    if _is_emoji_only(query):
        edge_cases["is_emoji_only"] = True
        edge_cases["edge_case_type"] = "emoji_only"
        return edge_cases

    # 7. Hypothetical query
    if _is_hypothetical_query(query):
        edge_cases["is_hypothetical_query"] = True
        edge_cases["edge_case_type"] = "hypothetical_query"
        return edge_cases

    # 8. Temporal confusion
    if _has_temporal_confusion(query):
        edge_cases["has_temporal_confusion"] = True
        edge_cases["edge_case_type"] = "temporal_confusion"
        return edge_cases

    # 9. Meta-question about system
    if _is_meta_question(query):
        edge_cases["is_meta_question"] = True
        edge_cases["edge_case_type"] = "meta_question"
        return edge_cases

    # 10. Negative query
    if _is_negative_query(query):
        edge_cases["is_negative_query"] = True
        edge_cases["edge_case_type"] = "negative_query"
        return edge_cases

    # 11. Rapid-fire queries (requires session memory)
    if _detect_rapid_fire(session_memory):
        edge_cases["is_rapid_fire"] = True
        edge_cases["edge_case_type"] = "rapid_fire"
        return edge_cases

    # 12. Query reformulation loop (requires chat history)
    # Skip for continuation-expanded queries ("tell me more" → "Go deeper on: X")
    # These intentionally repeat the previous topic and should not be flagged as loops
    if not state.get("is_continuation") and _detect_reformulation_loop(chat_history):
        edge_cases["is_reformulation_loop"] = True
        edge_cases["edge_case_type"] = "reformulation_loop"
        return edge_cases

    # 13. Conversation loop (requires chat history)
    if _detect_conversation_loop(chat_history):
        edge_cases["is_conversation_loop"] = True
        edge_cases["edge_case_type"] = "conversation_loop"
        return edge_cases

    # 14. Ambiguous pronouns (requires chat history, check after off-topic)
    if _has_ambiguous_pronouns(query, chat_history):
        edge_cases["has_ambiguous_pronouns"] = True
        edge_cases["edge_case_type"] = "ambiguous_pronouns"
        edge_cases["pronouns"] = _extract_pronouns(query)
        return edge_cases

    # 15. Intent drift (requires chat history)
    if _detect_intent_drift(query, chat_history):
        edge_cases["is_intent_drift"] = True
        edge_cases["edge_case_type"] = "intent_drift"
        edge_cases["previous_topic"] = _get_previous_topic(chat_history)
        edge_cases["current_topic"] = _get_current_topic(query)
        return edge_cases

    # 15. Contradictory information (most complex, check last)
    retrieved_chunks = state.get("retrieved_chunks", [])
    if _has_contradictory_information(query, chat_history, retrieved_chunks):
        edge_cases["has_contradictory_information"] = True
        edge_cases["edge_case_type"] = "contradictory_information"
        return edge_cases

    # No edge case detected
    return edge_cases


def _is_off_topic(query: str, chat_history: list) -> bool:
    """Detect queries completely unrelated to Noah/portfolio/GenAI.

    Uses keyword matching to determine if query is about topics Portfolia
    can help with. If no on-topic keywords found and it's a real question,
    flags as off-topic.

    Note: Queries with ambiguous pronouns are handled by ambiguous_pronouns
    detection, not off-topic detection.

    Args:
        query: User's query text
        chat_history: Conversation history (for context)

    Returns:
        True if query is off-topic, False otherwise
    """
    lowered = query.lower()

    # Skip if query contains pronouns as whole words (let ambiguous_pronouns handle it)
    # Check for whole words, not substrings (e.g., "that" in "what's" would be false positive)
    import re
    pronouns = ["his", "her", "their", "it", "this", "that", "these", "those"]
    words = set(re.findall(r'\b\w+\b', lowered))  # Extract whole words
    if any(pronoun in words for pronoun in pronouns):
        return False

    # Topics Portfolia CAN answer (comprehensive list)
    on_topic_keywords = [
        # Core topics
        "noah", "portfolio", "career", "experience", "background", "resume",
        "project", "work", "job", "hiring", "role", "position",

        # Technical topics
        "code", "architecture", "technical", "implementation", "system",
        "rag", "vector", "embedding", "retrieval", "langgraph", "llm",
        "genai", "ai", "machine learning", "python", "javascript",
        "supabase", "pgvector", "openai", "vercel", "deployment",

        # Portfolio-specific
        "how do you work", "how does this work", "show me", "explain",
        "tell me about", "what is", "how is", "why did",
        "built", "build", "how were you", "how are you",
        "your retrieval", "your pipeline", "your architecture",

        # Fun topics
        "mma", "fight", "confess", "crush", "hobby", "fun fact",

        # Career-specific topics
        "coaching", "coach", "tesla", "tql", "logistics", "real estate",
        "biology", "unlv", "bjj", "xtreme couture", "sales",

        # Future/plans topics
        "next", "future", "plan", "goal", "roadmap", "vision",

        # Meta questions about Portfolia
        "who are you", "what are you", "what can you do", "help me",

        # Enterprise adaptation topics (NEW - these are ON-TOPIC)
        "adapt", "adapts", "adaptation", "customer support", "enterprise",
        "use case", "scales to", "applies to", "works for", "chatbot",
        "internal docs", "sales enablement", "production", "deployment"
    ]

    # Check if query contains any on-topic keywords
    if any(keyword in lowered for keyword in on_topic_keywords):
        return False

    # Check if it's a real question (not just greeting/noise)
    question_indicators = ["what", "how", "when", "where", "why", "who", "can you", "tell me"]
    is_question = any(indicator in lowered for indicator in question_indicators)

    # If it's a question with no on-topic keywords, likely off-topic
    if is_question and len(query.split()) > 2:  # Not just "what" or "how"
        return True

    return False


def _is_empty_or_whitespace(query: str) -> bool:
    """Detect empty or whitespace-only queries.

    Args:
        query: User's query text

    Returns:
        True if query is empty or only whitespace, False otherwise
    """
    return not query or not query.strip()


def _extract_query_topic_words(query: str) -> set:
    """Extract key topic words from a query (excluding stop words).

    Args:
        query: User query text

    Returns:
        Set of important topic words
    """
    import re
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                  "have", "has", "had", "do", "does", "did", "will", "would", "could",
                  "should", "may", "might", "can", "what", "how", "why", "when", "where",
                  "who", "to", "for", "of", "in", "on", "at", "see", "me", "tell", "about",
                  "explain", "you", "please", "i", "want", "need", "know", "it", "this",
                  "that", "program", "programs", "be", "adapted"}

    # Remove punctuation before extracting words
    normalized = re.sub(r'[^\w\s]', '', query.lower())
    words = set(normalized.split())
    return words - stop_words


def _get_answer_after_query(chat_history: list, query_index: int) -> str:
    """Get the assistant answer that followed a user query at a given index.

    Args:
        chat_history: Full conversation history
        query_index: Index of the user message in chat_history

    Returns:
        The assistant's answer content, or empty string if not found
    """
    # Look for the next assistant message after the query
    for i in range(query_index + 1, len(chat_history)):
        msg = chat_history[i]
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("type", "")
            if role in ("assistant", "ai"):
                return msg.get("content", "")
        elif hasattr(msg, "type"):
            if msg.type == "ai" or getattr(msg, "role", None) == "assistant":
                return getattr(msg, "content", "")
    return ""


def _answer_addresses_topic(answer: str, query_topics: set, original_query: str = "") -> bool:
    """Check if an answer actually addresses the topic words from a query.

    STRICTER for enterprise queries: Must contain enterprise-specific content markers,
    not just generic word overlap.

    Args:
        answer: The assistant's answer text
        query_topics: Set of topic words from the user's query
        original_query: The original query text (for enterprise detection)

    Returns:
        True if the answer contains enough topic words AND enterprise-specific content if applicable
    """
    if not answer or not query_topics:
        return False

    answer_lower = answer.lower()

    # Check if this is an enterprise query - requires stricter matching
    if _is_enterprise_query(original_query):
        return _enterprise_answer_is_valid(answer_lower, query_topics)

    # Standard matching for non-enterprise queries
    matching_topics = sum(1 for topic in query_topics if topic in answer_lower)

    # For short queries (1 topic word), require 1 match
    # For longer queries (2+ topic words), require 2 matches
    required_matches = min(2, len(query_topics))
    return matching_topics >= required_matches


def _is_enterprise_query(query: str) -> bool:
    """Check if a query is about enterprise adaptation/use cases.

    Args:
        query: User query text

    Returns:
        True if query is about enterprise topics
    """
    if not query:
        return False

    query_lower = query.lower()
    enterprise_markers = [
        "customer support", "use case", "chatbot", "enterprise",
        "adapt", "adaptation", "internal docs", "sales enablement",
        "what to change", "code changes", "how would", "how could",
        "apply", "applies", "works for", "scales to"
    ]
    return any(marker in query_lower for marker in enterprise_markers)


def _enterprise_answer_is_valid(answer_lower: str, query_topics: set) -> bool:
    """Check if an answer properly addresses an enterprise query.

    Enterprise answers MUST contain specific content markers - not just generic
    overlap with words like "enterprise" that appear in orchestration explanations.

    Args:
        answer_lower: Lowercased answer text
        query_topics: Set of topic words from the query

    Returns:
        True if answer contains enterprise-specific content
    """
    # Enterprise answers must contain at least ONE of these specific content markers
    enterprise_content_markers = [
        # Specific enterprise adaptation content
        "knowledge base", "replace", "data sources", "product docs",
        "roles", "actions", "create ticket", "escalate",
        # ROI/business value
        "roi", "$", "savings", "cost", "reduction",
        # Code changes
        "code changes", "roles.py", "roles =",
        # Specific use cases
        "customer support", "internal documentation", "sales enablement",
        # What changes
        "what to change", "what stays"
    ]

    # Check for enterprise content markers
    has_enterprise_content = any(marker in answer_lower for marker in enterprise_content_markers)

    if not has_enterprise_content:
        # Check if answer is just a reformulation response (not a real answer)
        if "i notice you've asked" in answer_lower:
            return False
        # Check if it's about orchestration (wrong topic for enterprise query)
        if "orchestration" in answer_lower and "adapt" not in answer_lower:
            return False

    # Also require 3+ topic word matches for stricter validation
    matching_topics = sum(1 for topic in query_topics if topic in answer_lower)

    return has_enterprise_content or matching_topics >= 3


def _detect_reformulation_loop(chat_history: list, threshold: float = 0.2) -> bool:
    """Detect if user is repeatedly asking same question differently.

    Compares last 2 user messages using word overlap to detect if user
    is reformulating the same question. Changed from 3 to 2 messages for
    faster detection. Includes typo correction and exact duplicate detection.

    CRITICAL: Only triggers if the previous similar query was ACTUALLY ANSWERED
    with relevant content. If the previous answer didn't address the topic,
    the user is legitimately re-asking.

    Args:
        chat_history: Full conversation history
        threshold: Similarity threshold (0.2 = 20% word overlap, lowered for better detection with typos)

    Returns:
        True if reformulation loop detected, False otherwise
    """
    if len(chat_history) < 4:  # Need at least 2 user messages (changed from 6)
        return False

    # Get last 2-3 user messages with their indices (changed from 3)
    user_messages = []
    user_message_indices = []
    for i, msg in enumerate(chat_history[-6:]):  # Look at recent history
        actual_index = len(chat_history) - 6 + i if len(chat_history) >= 6 else i
        if isinstance(msg, dict):
            if msg.get("role") == "user" or msg.get("type") == "human":
                user_messages.append(msg.get("content", ""))
                user_message_indices.append(actual_index)
        elif hasattr(msg, "type") and (msg.type == "human" or getattr(msg, "role", None) == "user"):
            content = getattr(msg, "content", str(msg))
            user_messages.append(content)
            user_message_indices.append(actual_index)

    # Keep last 2 messages and their indices
    if len(user_messages) > 2:
        user_messages = user_messages[-2:]
        user_message_indices = user_message_indices[-2:]

    if len(user_messages) < 2:  # Changed from 3 to 2
        return False

    # Normalize messages: lowercase, remove punctuation, correct common typos
    import re
    corrections = {
        # Architecture typos
        "archexture": "architecture",
        "archetecture": "architecture",
        "architechture": "architecture",
        "architecure": "architecture",
        "architectxure": "architecture",  # 'x' typo
        "architexture": "architecture",   # 'x' typo variant
        "archtecture": "architecture",    # missing 'i'
        "architeture": "architecture",    # missing 'c'
        "architectre": "architecture",    # missing 'u'
        # Other typos
        "custmer": "customer",
        "cusotmer": "customer",
        "suport": "support",
        "supprot": "support",
        "enterpise": "enterprise",
        "enterprize": "enterprise",
        "adaptaion": "adaptation",
        "adaption": "adaptation",
    }

    normalized_messages = []
    for msg in user_messages:
        if msg:
            # Normalize: lowercase, remove punctuation
            normalized = re.sub(r'[^\w\s]', '', msg.lower())
            # Apply same typo corrections as query composition
            for typo, correct in corrections.items():
                normalized = normalized.replace(typo, correct)
            normalized_messages.append(normalized)

    # Check for exact duplicates (after normalization) - now works with 2 messages
    similar_detected = False
    if len(normalized_messages) >= 2:
        if normalized_messages[-1] == normalized_messages[-2]:
            similar_detected = True
            logger.info("Similar queries detected: exact duplicate")

    # Build word sets from normalized messages
    words_sets = []
    for normalized in normalized_messages:
        if normalized:
            words_sets.append(set(normalized.split()))

    if len(words_sets) < 2 and not similar_detected:  # Changed from 3 to 2
        return False

    # Define stop words to exclude from important word matching
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                  "have", "has", "had", "do", "does", "did", "will", "would", "could",
                  "should", "may", "might", "can", "what", "how", "why", "when", "where",
                  "who", "to", "for", "of", "in", "on", "at", "see", "me", "tell", "about",
                  "explain", "you", "please", "i", "want", "need", "know"}

    if not similar_detected:
        # Check if all messages share at least one important common word
        common_words = words_sets[0]
        for words in words_sets[1:]:
            common_words = common_words & words

        # If all queries share at least one important (non-stop) word, likely reformulation
        if common_words:
            important_common_words = common_words - stop_words
            # Need at least 1 important common word for reformulation detection
            if important_common_words:
                similar_detected = True
                logger.info(f"Similar queries detected: shared words {important_common_words}")

        if not similar_detected:
            # Fallback: check word overlap with threshold (excluding stop words)
            # Create word sets without stop words
            important_word_sets = []
            for words in words_sets:
                important_words = words - stop_words
                important_word_sets.append(important_words)

            # Need at least 1 important word in each message to compare
            if all(ws for ws in important_word_sets):
                overlaps = []
                for i in range(len(important_word_sets) - 1):
                    if important_word_sets[i] and important_word_sets[i+1]:
                        overlap = len(important_word_sets[i] & important_word_sets[i+1]) / max(len(important_word_sets[i] | important_word_sets[i+1]), 1)
                        overlaps.append(overlap)

                # If all overlaps are high (at least 50%), likely reformulation loop
                if overlaps and all(overlap >= 0.5 for overlap in overlaps):
                    similar_detected = True

    # CRITICAL CHECK: If similar queries detected, verify the previous answer addressed the topic
    if similar_detected and len(user_message_indices) >= 2:
        first_similar_query = user_messages[0]
        first_query_index = user_message_indices[0]

        # Extract topic words from the query
        query_topics = _extract_query_topic_words(first_similar_query)

        # Get the answer that followed the first similar query
        previous_answer = _get_answer_after_query(chat_history, first_query_index)

        if previous_answer:
            # Check if the previous answer actually addressed the query topic
            # Pass the original query for enterprise-specific validation
            if not _answer_addresses_topic(previous_answer, query_topics, first_similar_query):
                # The previous answer didn't address the topic - user is legitimately re-asking
                logger.info(
                    f"Similar query detected but previous answer didn't address topic. "
                    f"Query topics: {query_topics}. Not triggering reformulation."
                )
                return False
        else:
            # No previous answer found - don't trigger reformulation
            logger.info("Similar query detected but no previous answer found. Not triggering reformulation.")
            return False

        # Previous answer DID address the topic - this is a genuine reformulation
        logger.info(f"Reformulation loop detected: similar queries and previous answer addressed topic")
        return True

    return similar_detected


def _detect_conversation_loop(chat_history: list) -> bool:
    """Detect if conversation is going in circles.

    Checks for repeated Q&A patterns where the same question-answer pair
    appears multiple times in the conversation.

    Args:
        chat_history: Full conversation history

    Returns:
        True if conversation loop detected, False otherwise
    """
    if len(chat_history) < 6:  # Need at least 3 Q&A pairs
        return False

    # Check for repeated Q&A patterns
    # Extract user and assistant messages in order
    user_messages = []
    assistant_messages = []

    for msg in chat_history:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" or msg.get("type") == "human":
                user_messages.append(content)
            elif role == "assistant" or msg.get("type") == "ai":
                assistant_messages.append(content)
        elif hasattr(msg, "type"):
            content = getattr(msg, "content", str(msg))
            if msg.type == "human":
                user_messages.append(content)
            elif msg.type == "ai":
                assistant_messages.append(content)

    # Create Q&A pairs (assume they alternate)
    qa_pairs = []
    min_length = min(len(user_messages), len(assistant_messages))
    for i in range(min_length):
        user_msg = user_messages[i][:50] if user_messages[i] else ""
        assistant_msg = assistant_messages[i][:50] if assistant_messages[i] else ""
        qa_pairs.append((user_msg, assistant_msg))

    # Check for duplicates - if we have fewer unique pairs than total pairs, there's a loop
    unique_pairs = set(qa_pairs)
    return len(qa_pairs) > 1 and len(unique_pairs) < len(qa_pairs)


def _has_ambiguous_pronouns(query: str, chat_history: list) -> bool:
    """Detect ambiguous pronouns that need resolution.

    Checks if query contains pronouns (his, her, it, that, etc.) and
    verifies if the referent can be resolved from recent chat history.

    Args:
        query: User's current query
        chat_history: Conversation history for context

    Returns:
        True if ambiguous pronouns detected, False otherwise
    """
    pronouns = ["his", "her", "their", "it", "this", "that", "these", "those"]
    lowered = query.lower()

    # Check if query contains pronouns
    has_pronoun = any(pronoun in lowered for pronoun in pronouns)

    if not has_pronoun:
        return False

    # Check if we can resolve from recent context
    recent_context = []
    for msg in chat_history[-4:]:
        if isinstance(msg, dict):
            content = msg.get("content", "")
        elif hasattr(msg, "content"):
            content = msg.content
        else:
            content = str(msg)
        recent_context.append(content)

    recent_text = " ".join(recent_context).lower()

    # Simple heuristic: if pronoun appears but no clear referent in recent context
    # This is a simplified check - could be enhanced with NER
    # For now, flag all pronouns as potentially ambiguous if context is sparse
    if len(recent_text) < 50:  # Very sparse context
        return True

    # Check for common referents that would resolve pronouns
    common_referents = ["noah", "portfolio", "system", "code", "project", "rag", "langgraph"]
    has_referent = any(ref in recent_text for ref in common_referents)

    # If pronoun found but no clear referent, it's ambiguous
    return not has_referent


def _extract_pronouns(query: str) -> List[str]:
    """Extract pronouns from query.

    Args:
        query: User's query text

    Returns:
        List of pronouns found in the query
    """
    pronouns = ["his", "her", "their", "it", "this", "that", "these", "those"]
    found = [p for p in pronouns if p in query.lower()]
    return found


def _detect_intent_drift(query: str, chat_history: list) -> bool:
    """Detect if user is shifting topics mid-conversation.

    Compares topic of current query with topics from previous queries
    to detect significant topic shifts.

    Args:
        query: User's current query
        chat_history: Conversation history

    Returns:
        True if intent drift detected, False otherwise
    """
    if len(chat_history) < 4:
        return False

    # Get topic of current query
    current_topic = _get_current_topic(query)

    # Get topics of previous queries
    previous_topics = []
    for msg in chat_history[-4:]:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            previous_topics.append(_get_current_topic(content))
        elif hasattr(msg, "type") and msg.type == "human":
            content = getattr(msg, "content", "")
            previous_topics.append(_get_current_topic(content))

    # If current topic is very different from previous topics
    if previous_topics and current_topic:
        # Check if current topic doesn't overlap with previous
        if current_topic not in previous_topics and current_topic != "general":
            # Additional check: ensure previous topics weren't all "general"
            non_general_previous = [t for t in previous_topics if t != "general"]
            if non_general_previous:
                return True

    return False


def _get_current_topic(query: str) -> str:
    """Extract main topic from query.

    Uses keyword matching to identify the primary topic of the query.

    Args:
        query: User's query text

    Returns:
        Topic identifier string (e.g., "python", "career", "architecture")
    """
    topics = {
        "python": ["python", "py", "django", "flask"],
        "javascript": ["javascript", "js", "node", "react", "typescript"],
        "career": ["career", "resume", "experience", "background", "job"],
        "architecture": ["architecture", "system design", "pipeline", "orchestration"],
        "rag": ["rag", "retrieval", "vector", "embedding"],
        "mma": ["mma", "fight", "ufc", "martial arts"]
    }

    lowered = query.lower()
    for topic, keywords in topics.items():
        if any(keyword in lowered for keyword in keywords):
            return topic

    return "general"


def _get_previous_topic(chat_history: list) -> str:
    """Get topic from previous conversation.

    Args:
        chat_history: Conversation history

    Returns:
        Topic identifier from last user message
    """
    if not chat_history:
        return "general"

    last_user_msg = None
    for msg in reversed(chat_history):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break
        elif hasattr(msg, "type") and msg.type == "human":
            last_user_msg = getattr(msg, "content", "")
            break

    if last_user_msg:
        return _get_current_topic(last_user_msg)
    return "general"


def _detect_rapid_fire(session_memory: dict) -> bool:
    """Detect rapid-fire queries (multiple queries in short time).

    Checks if user sent multiple queries within a short time window,
    indicating they might be spamming or testing the system.

    Args:
        session_memory: Session memory dict with query timestamps

    Returns:
        True if rapid-fire detected, False otherwise
    """
    query_timestamps = session_memory.get("query_timestamps", [])

    if len(query_timestamps) < 2:
        return False

    # Check if last 2 queries were within 2 seconds
    if len(query_timestamps) >= 2:
        time_diff = query_timestamps[-1] - query_timestamps[-2]
        return time_diff < 2.0

    return False


def _is_negative_query(query: str) -> bool:
    """Detect negative queries (what can't, weaknesses, etc.).

    Detects queries asking about limitations, weaknesses, or things
    that can't be done. Exempts questions directed at Portfolia itself
    (e.g. "what are your limitations") — those are self-knowledge queries
    handled by the retrieval pipeline.

    Args:
        query: User's query text

    Returns:
        True if negative query detected, False otherwise
    """
    lowered = query.lower()

    # Exempt: Questions about Portfolia's own limitations → self-knowledge
    self_directed = ["your limitation", "your weakness", "you bad at",
                     "you fall short", "you missing", "you not do",
                     "can't you do", "cant you do", "don't you know",
                     "dont you know", "can you not"]
    if any(phrase in lowered for phrase in self_directed):
        return False

    negative_patterns = [
        r"what.*can.*t.*do",
        r"what.*can.*not",
        r"weakness",
        r"limitation",
        r"what.*bad",
        r"what.*wrong",
        r"can.*t.*do",
        r"cannot",
        r"doesn.*t.*do",
        r"doesn.*t.*have"
    ]

    return any(re.search(pattern, lowered) for pattern in negative_patterns)


def _is_meta_question(query: str) -> bool:
    """Detect meta-questions about the system itself.

    Detects questions asking about why Portfolia said something or
    how she detected something. Does NOT catch legitimate architecture
    questions like "how does your retrieval work?" — those are knowledge queries.

    Args:
        query: User's query text

    Returns:
        True if meta-question detected, False otherwise
    """
    lowered = query.lower()

    # If query contains technical keywords, it's a knowledge query, not a meta-question
    technical_keywords = [
        "retrieval", "architecture", "pipeline", "embedding", "vector",
        "rag", "langgraph", "pgvector", "supabase", "generation",
        "node", "chunk", "built", "build", "stack", "code",
    ]
    if any(kw in lowered for kw in technical_keywords):
        return False

    meta_patterns = [
        r"why.*did.*you.*say",
        r"what.*made.*you.*think",
        r"how.*did.*you.*know",
        r"why.*did.*you.*answer",
        r"explain.*your.*reasoning",
    ]

    return any(re.search(pattern, lowered) for pattern in meta_patterns)


def _is_emoji_only(query: str) -> bool:
    """Detect emoji-only queries.

    Checks if query contains only emojis without meaningful text.

    Args:
        query: User's query text

    Returns:
        True if emoji-only query detected, False otherwise
    """
    # Remove whitespace and check if only emojis remain
    stripped = query.strip()
    if not stripped:
        return False

    # Simple check: if query has no letters/numbers, likely emoji-only
    has_text = bool(re.search(r'[a-zA-Z0-9]', stripped))
    return not has_text and len(stripped) > 0


def _is_very_long_query(query: str, threshold: int = 2000) -> bool:
    """Detect queries exceeding reasonable length.

    Very long queries can indicate:
    - User pasting large text blocks
    - Attempting to exhaust token limits
    - Multiple questions in one query

    Args:
        query: User's query text
        threshold: Character length threshold (default 2000)

    Returns:
        True if query exceeds threshold, False otherwise
    """
    return len(query) > threshold


def _is_nonsensical_query(query: str) -> bool:
    """Detect gibberish or random character queries.

    Detects obvious test strings, random characters, or queries with
    suspicious patterns that indicate testing or gibberish.

    Args:
        query: User's query text

    Returns:
        True if nonsensical query detected, False otherwise
    """
    lowered = query.lower().strip()

    # Common test strings
    test_strings = ["test", "asdf", "qwerty", "hello world", "12345", "abc", "xyz"]
    if lowered in test_strings:
        return True

    # Very short single words that are likely testing
    if len(lowered.split()) == 1 and len(lowered) <= 3 and lowered.isalpha():
        return True

    # Check for repeated characters (like "aaaaaa" or "111111")
    # Check for both short and long repeated character strings
    query_no_spaces = query.replace(" ", "")
    if len(query_no_spaces) >= 4:  # Check strings of 4+ characters
        unique_chars = len(set(query_no_spaces))
        if unique_chars < 3:  # Very few unique characters (repeated pattern)
            return True

    # Check for suspiciously long "words" (likely random characters)
    words = query.split()
    if len(words) > 0:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len > 15:  # Suspiciously long average word length
            return True

    # Check for patterns like "asdfghjkl" (keyboard mashing)
    if len(lowered) > 5 and lowered.isalpha():
        # Check if it's a sequence of adjacent keyboard keys
        keyboard_rows = [
            "qwertyuiop", "asdfghjkl", "zxcvbnm",
            "1234567890"
        ]
        for row in keyboard_rows:
            if lowered in row or row in lowered:
                return True

    return False


def _is_hypothetical_query(query: str) -> bool:
    """Detect hypothetical questions.

    Detects queries asking "what if" scenarios that the knowledge base
    cannot answer because it focuses on actual facts, not hypotheticals.

    Args:
        query: User's query text

    Returns:
        True if hypothetical query detected, False otherwise
    """
    hypothetical_patterns = [
        r"what if",
        r"imagine if",
        r"suppose",
        r"hypothetically",
        r"if.*worked at",
        r"if.*had",
        r"if.*was",
        r"if.*were",
        r"would.*if",
        r"could.*if"
    ]

    lowered = query.lower()
    return any(re.search(pattern, lowered) for pattern in hypothetical_patterns)


def _has_temporal_confusion(query: str) -> bool:
    """Detect time-sensitive queries that might be outdated.

    Detects queries asking about current status when the knowledge base
    might have outdated information (e.g., "what's Noah doing now?").

    Args:
        query: User's query text

    Returns:
        True if temporal confusion detected, False otherwise
    """
    temporal_words = [
        "now", "currently", "recently", "lately", "these days",
        "right now", "at the moment", "presently", "today"
    ]

    lowered = query.lower()
    return any(word in lowered for word in temporal_words)


def _has_contradictory_information(query: str, chat_history: list, retrieved_chunks: list) -> bool:
    """Detect if user statement contradicts KB facts.

    This is a simplified implementation that checks for basic contradictions.
    A full implementation would require:
    - Named Entity Recognition (NER) to extract entities
    - Fact extraction from retrieved chunks
    - Semantic comparison of facts

    For now, this does basic keyword matching for common contradictions.

    Args:
        query: User's query text
        chat_history: Conversation history
        retrieved_chunks: Retrieved knowledge base chunks

    Returns:
        True if contradiction detected, False otherwise
    """
    # This is a placeholder implementation - full NER/fact extraction would be needed
    # for robust contradiction detection

    if not retrieved_chunks:
        return False

    lowered = query.lower()

    # Extract company names from query (simple pattern matching)
    company_keywords = ["google", "microsoft", "apple", "amazon", "meta", "tesla", "netflix"]
    query_companies = [c for c in company_keywords if c in lowered]

    if not query_companies:
        return False

    # Check retrieved chunks for company mentions
    chunk_text = " ".join([
        chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
        for chunk in retrieved_chunks[:3]  # Check top 3 chunks
    ]).lower()

    # If query mentions a company but chunks mention a different company, possible contradiction
    chunk_companies = [c for c in company_keywords if c in chunk_text]

    if query_companies and chunk_companies:
        # Check if there's overlap - if no overlap, might be contradiction
        if not set(query_companies) & set(chunk_companies):
            # Additional check: ensure query is making a statement, not asking
            statement_indicators = ["works at", "worked at", "is at", "was at", "employed at"]
            if any(indicator in lowered for indicator in statement_indicators):
                return True

    return False
