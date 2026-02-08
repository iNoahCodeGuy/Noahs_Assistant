"""Generate conversational responses for edge cases.

Uses LLM to create warm, inference-rich responses (not templates).
Follows same pattern as off-topic handling - turns edge cases into
teaching moments with natural, helpful responses.
"""

import logging
from typing import Dict, Any

from assistant.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)


def generate_edge_case_response(state: ConversationState) -> str:
    """Generate conversational response for detected edge case.

    Uses LLM to create natural, inference-rich responses that:
    1. Acknowledge the edge case naturally
    2. Show understanding of what user might need
    3. Subtly redirect or help
    4. Offer meta-teaching (for technical users)

    Args:
        state: Conversation state with edge_case_type and query

    Returns:
        Conversational response string
    """
    edge_case_type = state.get("edge_case_type")
    query = state.get("query", "")
    role_mode = state.get("role_mode", "explorer")
    chat_history = state.get("chat_history", [])

    is_technical = role_mode in ["software_developer", "hiring_manager_technical"]

    # Build context-aware prompt
    prompt = _build_edge_case_prompt(edge_case_type, query, role_mode, chat_history, is_technical, state)

    # If prompt is None, skip edge case handling (query is actually on-topic)
    if prompt is None:
        logger.info("Skipping edge case handling - query is actually on-topic")
        return ""  # Return empty string to signal no edge case response needed

    try:
        from assistant.core.rag_factory import RagEngineFactory
        factory = RagEngineFactory()
        llm, _ = factory.create_llm()
        response = llm.invoke(prompt, temperature=0.7)  # Slightly higher temp for naturalness
        return response.strip()
    except Exception as e:
        logger.warning(f"Failed to generate edge case response: {e}")
        return _get_fallback_response(edge_case_type, is_technical)


def _build_edge_case_prompt(
    edge_case_type: str,
    query: str,
    role_mode: str,
    chat_history: list,
    is_technical: bool,
    state: ConversationState
) -> str:
    """Build LLM prompt for edge case response.

    Args:
        edge_case_type: Type of edge case detected
        query: User's query
        role_mode: User's role mode
        chat_history: Conversation history
        is_technical: Whether user is technical
        state: Full conversation state for context

    Returns:
        Formatted prompt string for LLM, or None if edge case should be skipped
    """

    # CHECK: Is this actually an enterprise adaptation query? (on-topic, not edge case)
    if edge_case_type == "off_topic":
        query_lower = query.lower()
        enterprise_adaptation_terms = [
            "adapt", "adapts", "adaptation", "customer support", "enterprise",
            "use case", "scales to", "applies to", "works for", "chatbot",
            "internal docs", "sales enablement"
        ]

        if any(term in query_lower for term in enterprise_adaptation_terms):
            # This is actually ON-TOPIC (enterprise adaptation) - don't treat as edge case
            logger.info(f"Query flagged as off_topic but contains enterprise adaptation terms - skipping edge case handling")
            return None  # Signal to skip edge case handling

    # Base prompt structure
    base_prompt = """You are Portfolia, Noah's AI Assistant. A user just asked: "{query}"

{edge_case_description}

Your task: Generate a warm, conversational response that acknowledges their query naturally (not just "I can't help"), shows you understand what they might need, gently redirects to what you CAN help with, and stays in character.

Keep it conversational and warm, 2-3 sentences max, naturally redirecting without being pushy, and always in first person. Do NOT use bold text, bullet points, or numbered lists. Write in flowing prose.

{meta_teaching_offer}

Generate the response now:"""

    # Edge case specific descriptions
    edge_case_descriptions = {
        "off_topic": f"""This question is completely unrelated to what you can help with (Noah's background, technical projects, career experience, or how you're built).

Redirect the user naturally. Do NOT use bold text, bullet points, or numbered lists. Write in flowing conversational prose. Mention a few things you can talk about — like Noah's background, his projects, your own architecture, or how the data pipeline works — and ask what sounds interesting. Keep it to 2-3 sentences max.""",

        "empty_query": """User sent an empty or whitespace-only query.""",

        "reformulation_loop": """User has asked the same question 3+ times in different ways. They might be frustrated or unclear about what they want.""",

        "conversation_loop": """You and the user seem to be going in circles. The conversation isn't progressing.""",

        "ambiguous_pronouns": f"""This query contains ambiguous pronouns ({state.get('pronouns', [])}) that you can't resolve from context. You need to ask for clarification about what specifically they're referring to.""",

        "intent_drift": f"""User was discussing {state.get('previous_topic', 'one topic')} but now asking about {state.get('current_topic', 'another topic')}. This is a topic shift.""",

        "rapid_fire": """User sent multiple queries very quickly. They might be testing the system or impatient.""",

        "negative_query": """User asked a negative question (what can't Noah do, weaknesses, limitations, etc.). Your knowledge base focuses on strengths and achievements.""",

        "meta_question": """User asked a meta-question about how you work or why you said something. This is actually a good teaching moment!""",

        "emoji_only": """User sent only emojis without any text.""",

        "very_long_query": f"""User sent a very long query ({state.get('query_length', 0)} characters). This might be multiple questions combined, or they might need help breaking it down.""",

        "nonsensical_query": """User sent a query that appears to be gibberish, random characters, or obvious testing (like "asdf", "test", etc.). They might be testing the system or accidentally sent something.""",

        "hypothetical_query": """User asked a hypothetical question (e.g., "what if Noah worked at Google?"). Your knowledge base focuses on actual facts and experience, not hypothetical scenarios.""",

        "temporal_confusion": """User asked about current status (using words like "now", "currently", "recently"). Your knowledge base might have information that's not up-to-date, so you should acknowledge this.""",

        "contradictory_information": """User made a statement that contradicts information in your knowledge base. You should acknowledge the discrepancy and clarify what you have in your knowledge base."""
    }

    edge_case_description = edge_case_descriptions.get(edge_case_type, "An edge case was detected.")

    # Meta-teaching offer for technical users
    meta_teaching_offer = ""
    if is_technical:
        if edge_case_type == "off_topic":
            meta_teaching_offer = """By the way — since you're asking about something outside my scope, want me to explain how I detected that? It's actually a neat example of how production AI systems handle edge cases like off-topic queries."""
        elif edge_case_type == "meta_question":
            # Meta-questions are already about the system, so no need for additional offer
            meta_teaching_offer = ""
        else:
            meta_teaching_offer = f"""By the way — want me to explain how I detected this {edge_case_type.replace('_', ' ')} case? It's a good example of production AI systems handling edge cases gracefully."""

    return base_prompt.format(
        query=query,
        edge_case_description=edge_case_description,
        meta_teaching_offer=meta_teaching_offer
    )


def handle_reformulation_loop(state: ConversationState) -> ConversationState:
    """Generate clarification response when user repeats same query.

    Uses conversational style (not teaching) to help user clarify what they need.

    CRITICAL: If no relevant previous answer exists, skip reformulation handling
    and let normal generation proceed.

    ALSO: If we're already in a reformulation response (draft contains "I notice you've asked"),
    don't double-trigger - this prevents recursive self-reference.

    Args:
        state: Conversation state with query, chat_history, edge_case_type="reformulation_loop"

    Returns:
        Updated state with answer and draft_answer set to clarification response,
        OR state with edge_case_handled=False if no relevant previous answer exists
    """
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])

    # PREVENT RECURSION: Check if we're already in a reformulation response
    # draft_answer may be None (e.g., after pipeline_halt clears it), so coerce to string
    draft = state.get("draft_answer") or ""
    if "I notice you've asked" in draft or "I already provided" in draft:
        # Don't double-trigger reformulation - this causes recursive self-reference
        logger.info("Already in reformulation response - preventing recursive trigger")
        state["edge_case_handled"] = False
        return state

    # SPECIAL CASE: Action requests (resume, linkedin, github)
    # For repeated action requests, always acknowledge - don't let high retrieval skip it
    query_type = state.get("query_type", "")
    if query_type == "action_request":
        logger.info("Repeated action request detected - acknowledging instead of re-providing")
        response = "I already provided those resources above. Is there something specific you're looking for, or would you like Noah to reach out directly?"
        state["answer"] = response
        state["draft_answer"] = response
        state["edge_case_handled"] = True
        return state

    # CHECK RETRIEVAL QUALITY: If we have high-quality retrieved content,
    # use it to answer instead of showing a generic clarification prompt.
    # This fixes the bug where relevant content (e.g., "Enterprise Adaptation Guide"
    # with 0.847 similarity) was ignored in favor of a clarification prompt.
    retrieval_scores = state.get("retrieval_scores", [])
    if retrieval_scores and max(retrieval_scores) >= 0.75:
        logger.info(f"High-quality retrieval found (max similarity: {max(retrieval_scores):.3f}) - "
                   "skipping reformulation handling to use retrieved content")
        state["edge_case_handled"] = False
        # Clear the edge case so generate_draft proceeds with normal generation
        session_memory = state.get("session_memory", {})
        if session_memory.get("last_edge_case") == "reformulation_loop":
            session_memory["last_edge_case"] = None
        return state

    # Find the previous answer to this topic
    previous_answer_summary, found_relevant = _extract_previous_answer_summary(chat_history, query)

    # If no relevant previous answer exists, don't handle as reformulation
    if not found_relevant:
        logger.info("No relevant previous answer found - skipping reformulation handling")
        state["edge_case_handled"] = False
        # Clear the edge case from session_memory so generate_draft doesn't trigger again
        session_memory = state.get("session_memory", {})
        if session_memory.get("last_edge_case") == "reformulation_loop":
            session_memory["last_edge_case"] = None
        return state

    # Use conversational style (not teaching) for clarification
    response = f"""I notice you've asked about this topic a few times. Let me clarify:

**What I explained earlier:**
{previous_answer_summary}

**What might need clarification:**
1. Want more specific code examples?
2. Need clarification on a particular aspect?
3. Should I compare to other use cases?
4. Want me to explain it differently?

Which would be most helpful?"""

    state["answer"] = response
    state["draft_answer"] = response
    state["edge_case_handled"] = True
    return state


def _extract_previous_answer_summary(chat_history: list, query: str) -> tuple:
    """Extract summary of previous answer that actually addressed the query topic.

    CRITICAL: Must find an answer that actually addressed the topic,
    not just the most recent assistant message.

    ALSO: Skip reformulation responses ("I notice you've asked...") - these are
    not real answers and would cause recursive self-reference if quoted.

    Args:
        chat_history: Conversation history
        query: Current query

    Returns:
        Tuple of (summary_text, found_relevant_answer)
        - summary_text: Summary of previous answer (first 200 chars) or fallback message
        - found_relevant_answer: True if we found an answer that addressed the topic
    """
    import re

    # Extract key topic words from the query (excluding stop words)
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                  "have", "has", "had", "do", "does", "did", "will", "would", "could",
                  "should", "may", "might", "can", "what", "how", "why", "when", "where",
                  "who", "to", "for", "of", "in", "on", "at", "see", "me", "tell", "about",
                  "explain", "you", "please", "i", "want", "need", "know", "it", "this",
                  "that", "program", "programs", "be"}

    # Remove punctuation before extracting words
    query_normalized = re.sub(r'[^\w\s]', '', query.lower())
    query_words = set(query_normalized.split()) - stop_words

    # Search backwards for an answer that contains query topic words
    for msg in reversed(chat_history):
        if isinstance(msg, dict):
            content = msg.get("content", "")
            msg_type = msg.get("type", "") or msg.get("role", "")
        else:
            content = getattr(msg, "content", "") if hasattr(msg, "content") else ""
            msg_type = getattr(msg, "type", "") or getattr(msg, "role", "")

        if msg_type in ("ai", "assistant") and content:
            # CRITICAL: Skip reformulation responses - they're not real answers
            # This prevents recursive self-reference in answers
            if "I notice you've asked" in content:
                logger.debug("Skipping reformulation response when extracting previous answer")
                continue

            # Skip very short responses (likely greetings or acknowledgments)
            if len(content) < 100:
                continue

            content_lower = content.lower()
            # Check if this answer contains query topic words
            matching_words = sum(1 for word in query_words if word in content_lower)
            if matching_words >= 2:  # At least 2 topic words must match
                summary = content[:200] + "..." if len(content) > 200 else content
                return (summary, True)

    # No relevant previous answer found
    return ("I haven't fully addressed this topic yet. Let me explain:", False)


def _get_fallback_response(edge_case_type: str, is_technical: bool) -> str:
    """Fallback template responses if LLM fails.

    These are still conversational, not error messages.

    Args:
        edge_case_type: Type of edge case
        is_technical: Whether user is technical

    Returns:
        Fallback response string
    """
    fallbacks = {
        "off_topic": (
            "That's a bit outside my wheelhouse, but I've got plenty to talk about. "
            "I can walk you through Noah's background and career path, his technical projects, "
            "how my own pipeline works under the hood, or the data architecture powering all of this. "
            "What sounds interesting?"
        ),

        "empty_query": "Hmm, I didn't catch that! Drop me a question and let's explore something together.",

        "reformulation_loop": (
            "Déjà vu! We might be going in circles. Let me shake things up — "
            "want to explore a completely different area? I've got orchestration, "
            "enterprise patterns, or Noah's background waiting in the wings."
        ),

        "conversation_loop": (
            "I think we've been here before! How about we venture into new territory? "
            "What haven't we explored yet — the data pipeline, enterprise use cases, or Noah's projects?"
        ),

        "ambiguous_pronouns": (
            "I want to make sure I'm on the same page — what specifically are you referring to? "
            "Are you asking about Noah's Python skills, a specific project, or something else?"
        ),

        "intent_drift": (
            "Ooh, topic shift detected! Should I follow this new thread, "
            "or would you rather go back to what we were diving into?"
        ),

        "rapid_fire": (
            "Whoa, you're full of questions! Love the enthusiasm. "
            "Which one should I tackle first?"
        ),

        "negative_query": (
            "Hmm, my knowledge base is more of a highlight reel — Noah's wins and achievements. "
            "Want to explore what he's built or where he's headed?"
        ),

        "meta_question": (
            "Ooh, you want to peek behind the curtain? I love talking about how I'm built! "
            "What would you like to know — orchestration, retrieval, or something else?"
        ),

        "emoji_only": "I see emojis! 😄 What's on your mind? Ask me anything about Noah or how I work!",

        "very_long_query": (
            "That's a meaty question! Let's break it into bite-sized pieces. "
            "What's the most important part you want me to tackle first?"
        ),

        "nonsensical_query": (
            "Hmm, I'm not quite following! Could you rephrase that? "
            "I'm here to chat about Noah's background, technical projects, or my own architecture."
        ),

        "hypothetical_query": (
            "Fun hypothetical! My knowledge base sticks to Noah's actual experience though. "
            "Want to explore what he's really built or achieved?"
        ),

        "temporal_confusion": (
            "My info is from when the knowledge base was last updated. "
            "What specific aspect are you curious about? I'll share what I know!"
        ),

        "contradictory_information": (
            "Hmm, let me make sure I've got this right — could you clarify what you're referring to? "
            "I'll dig into my knowledge base and share what I find."
        )
    }

    response = fallbacks.get(edge_case_type, "I'm here to help! What would you like to know?")

    # Add meta-teaching offer for technical users if not already included
    if is_technical and edge_case_type != "off_topic" and edge_case_type != "meta_question":
        response += f"\n\nWant me to explain how I detected this {edge_case_type.replace('_', ' ')} case? It's a good example of production AI edge case handling."

    return response
