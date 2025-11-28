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
        response = llm.predict(prompt, temperature=0.7)  # Slightly higher temp for naturalness
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

Your task: Generate a warm, conversational response that:
1. **Acknowledges their query naturally** - Don't just say "I can't help with that"
2. **Shows inference** - Demonstrate you understand what they might be trying to do
3. **Subtly redirect or help** - Gently guide them back to what you CAN help with, or clarify what they need
4. **Stay in character** - Use your natural, warm, teaching-focused personality

Keep it:
- Conversational and warm (not robotic)
- Short (2-3 sentences max for redirect, +1 sentence for meta-offer if technical)
- Naturally redirecting (not pushy)
- In first person when talking about yourself

{meta_teaching_offer}

Generate the response now:"""

    # Edge case specific descriptions
    edge_case_descriptions = {
        "off_topic": f"""This question is completely unrelated to what you can help with (Noah's background, technical projects, career experience, or how you're built).

IMPORTANT: Provide helpful guidance by suggesting specific on-topic areas:
- Technical architecture: How the system is built
- Enterprise adaptation: How this pattern applies to customer support, internal docs
- Career background: Noah's experience and projects
- How I work: The RAG pipeline, vector search, orchestration

Ask which direction sounds interesting.""",

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
        "off_topic": "That's an interesting question! I'm Portfolia, Noah's AI Assistant, and I'm focused on helping people learn about Noah's background, technical projects, and how AI systems like me work. What would you like to explore?",

        "empty_query": "I didn't catch that. Could you ask me a question?",

        "reformulation_loop": "I notice you've asked similar questions a few times. Let me try a different approach — what specific aspect would you like me to clarify?",

        "conversation_loop": "I notice we might be going in circles. Let me try a different angle — what would be most helpful?",

        "ambiguous_pronouns": "I want to make sure I understand — could you clarify what specifically you're referring to? For example, are you asking about Noah's Python skills, or something else?",

        "intent_drift": f"I notice we shifted topics. Should I continue with this new direction, or go back to what we were discussing?",

        "rapid_fire": "I'm processing your questions. Could you give me a moment, or let me know which question to prioritize?",

        "negative_query": "My knowledge base focuses on Noah's strengths and achievements. What would you like to know about those?",

        "meta_question": "Great question! I'd be happy to explain how I work. What specifically would you like to know?",

        "emoji_only": "I see you sent emojis! Could you ask your question in words so I can help better?",

        "very_long_query": "That's a detailed question! Could you break it into smaller parts? I can help with each piece one at a time.",

        "nonsensical_query": "I'm not sure what you're asking. Could you rephrase that? I'm here to help with questions about Noah's background, technical projects, or how I'm built.",

        "hypothetical_query": "That's an interesting hypothetical! My knowledge base focuses on Noah's actual experience and achievements. What would you like to know about his real background?",

        "temporal_confusion": "My knowledge base shows Noah's experience as of when it was last updated. This information may have changed. What specific aspect are you curious about?",

        "contradictory_information": "I want to make sure I have the right information. Could you clarify what you're referring to? I can share what I have in my knowledge base."
    }

    response = fallbacks.get(edge_case_type, "I'm here to help! What would you like to know?")

    # Add meta-teaching offer for technical users if not already included
    if is_technical and edge_case_type != "off_topic" and edge_case_type != "meta_question":
        response += f"\n\nWant me to explain how I detected this {edge_case_type.replace('_', ' ')} case? It's a good example of production AI edge case handling."

    return response
