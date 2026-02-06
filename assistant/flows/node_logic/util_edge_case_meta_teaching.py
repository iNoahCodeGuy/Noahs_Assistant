"""Meta-teaching explanations for edge case detection.

Generates explanations of how Portfolia's programming detects and handles
edge cases, using the user's actual query as a teaching example and showing
actual code from the detection module.
"""

import logging
from typing import Dict, Any

from assistant.state.conversation_state import ConversationState
from assistant.flows.node_logic.util_code_snippet_extractor import (
    extract_function_code,
    format_code_for_display
)

logger = logging.getLogger(__name__)


def generate_meta_teaching_explanation(state: ConversationState) -> str:
    """Generate meta-teaching explanation for edge case detection.

    Explains how Portfolia detected the edge case, using the user's query
    as an example and showing actual detection code.

    Args:
        state: Conversation state with edge case info and user query

    Returns:
        Complete explanation string with code examples
    """
    edge_case_type = state.get("edge_case_type")
    user_query = state.get("query", "")
    session_memory = state.get("session_memory", {})
    last_edge_case_query = session_memory.get("last_edge_case_query", user_query)
    role_mode = state.get("role_mode", "explorer")

    is_technical = role_mode in ["software_developer", "hiring_manager_technical"]

    # Get code snippet for this edge case (before try block so it's available in fallback)
    code_snippet = _get_code_snippet_for_edge_case(edge_case_type)

    # Build explanation prompt
    prompt = _build_meta_teaching_prompt(edge_case_type, last_edge_case_query, role_mode, state)

    try:
        from assistant.core.rag_factory import RagEngineFactory
        factory = RagEngineFactory()
        llm, _ = factory.create_llm()

        # Generate explanation
        explanation = llm.invoke(prompt, temperature=0.6)  # Lower temp for technical accuracy

        # Append code snippet for technical users
        if is_technical and code_snippet:
            explanation += f"\n\n**Here's the actual code from my detection module:**\n\n{code_snippet}"

        return explanation.strip()
    except Exception as e:
        logger.warning(f"Failed to generate meta-teaching explanation: {e}")
        return _get_fallback_explanation(edge_case_type, last_edge_case_query, code_snippet, is_technical)


def _build_meta_teaching_prompt(
    edge_case_type: str,
    user_query: str,
    role_mode: str,
    state: ConversationState
) -> str:
    """Build LLM prompt for meta-teaching explanation.

    Args:
        edge_case_type: Type of edge case detected
        user_query: User's query that triggered the detection
        role_mode: User's role mode
        state: Full conversation state

    Returns:
        Formatted prompt string
    """

    # Map edge case types to function names
    function_map = {
        "off_topic": "_is_off_topic",
        "empty_query": "_is_empty_or_whitespace",
        "reformulation_loop": "_detect_reformulation_loop",
        "conversation_loop": "_detect_conversation_loop",
        "ambiguous_pronouns": "_has_ambiguous_pronouns",
        "intent_drift": "_detect_intent_drift",
        "rapid_fire": "_detect_rapid_fire",
        "negative_query": "_is_negative_query",
        "meta_question": "_is_meta_question",
        "emoji_only": "_is_emoji_only",
        "very_long_query": "_is_very_long_query",
        "nonsensical_query": "_is_nonsensical_query",
        "hypothetical_query": "_is_hypothetical_query",
        "temporal_confusion": "_has_temporal_confusion",
        "contradictory_information": "_has_contradictory_information"
    }

    # Build explanation descriptions
    explanations = {
        "off_topic": f"""You asked: "{user_query}"

This triggered my off-topic detection. Here's how I detected it:

1. **Keyword Matching**: I check if your query contains any keywords related to what I can help with (Noah, portfolio, code, architecture, RAG, etc.)
2. **Question Detection**: If it's a real question (has question words like "what", "how", "why") but contains NO on-topic keywords, I flag it as off-topic
3. **Why This Matters**: Production AI systems need to gracefully handle queries outside their domain instead of hallucinating answers

When I detected this, I generated a conversational response that acknowledges your query and gently redirects you to what I CAN help with, rather than just saying "I can't help with that."

The code that does this lives in `util_edge_case_detection.py` in the `_is_off_topic()` function.""",

        "empty_query": f"""You sent an empty or whitespace-only query.

I detected this by checking if the query string is empty or contains only whitespace characters. This is important because empty queries can cause errors in downstream processing.

When I detected this, I generated a friendly response asking you to rephrase your question.""",

        "reformulation_loop": f"""You asked similar questions multiple times: "{user_query}"

I detected this by comparing your last 3 user messages using word overlap. If the similarity is above 85%, I flag it as a reformulation loop - meaning you're asking the same question in different ways.

This pattern indicates you might be frustrated or unclear about what you want. When I detect this, I acknowledge it and offer to try a different approach.""",

        "conversation_loop": f"""I detected that we're going in circles - the same question-answer patterns are repeating.

I check for this by comparing Q&A pairs in our conversation history. If the same pairs appear multiple times, it means we're not making progress.

When I detect this, I acknowledge it and suggest trying a different angle.""",

        "ambiguous_pronouns": f"""Your query "{user_query}" contains pronouns ({state.get('pronouns', [])}) that I can't resolve from context.

I detect this by:
1. Checking if your query contains pronouns (his, her, it, this, that, etc.)
2. Looking at recent conversation history to see if there's a clear referent
3. If pronoun found but no clear referent in recent context → ambiguous

When I detect this, I ask for clarification with specific examples rather than guessing what you mean.""",

        "intent_drift": f"""You were discussing {state.get('previous_topic', 'one topic')} but now asked about {state.get('current_topic', 'another topic')}.

I detect topic shifts by comparing the current query's topic keywords with topics from your previous queries. If there's a significant shift, I flag it as intent drift.

When I detect this, I acknowledge the shift and confirm if you want to switch focus.""",

        "rapid_fire": f"""You sent multiple queries very quickly (within 2 seconds).

I track query timestamps in session memory. If the last 2 queries were within 2 seconds, I flag it as rapid-fire.

This might indicate you're testing the system or impatient. When I detect this, I acknowledge it and ask which question to prioritize.""",

        "negative_query": f"""You asked: "{user_query}"

This is a negative query asking about limitations or weaknesses. I detect this by checking for patterns like "what can't", "weakness", "limitation", etc.

My knowledge base focuses on Noah's strengths and achievements. When I detect a negative query, I acknowledge it and redirect to what I CAN share.""",

        "meta_question": f"""You asked: "{user_query}"

This is a meta-question about how I work! I detect this by checking for patterns like "how did you", "why did you say", "explain your reasoning", etc.

This is actually a great teaching moment - I can explain my reasoning process and show you the code.""",

        "emoji_only": f"""You sent only emojis: "{user_query}"

I detect this by checking if the query has no letters or numbers - just emojis and special characters.

When I detect this, I ask you to rephrase in words so I can help better.""",

        "very_long_query": f"""You sent a very long query ({state.get('query_length', 0)} characters): "{user_query[:100]}..."

I detect this by checking if the query length exceeds 2000 characters. Very long queries can indicate:
- Multiple questions combined into one
- User pasting large text blocks
- Potential attempts to exhaust token limits

When I detect this, I ask you to break it into smaller parts so I can help more effectively.""",

        "nonsensical_query": f"""You sent: "{user_query}"

I detect nonsensical queries by checking for:
1. Common test strings like "test", "asdf", "qwerty"
2. Repeated characters (like "aaaaaa")
3. Suspiciously long "words" (random character strings)
4. Keyboard mashing patterns (adjacent keys)

This helps me identify when users are testing the system or accidentally sent gibberish. When I detect this, I ask for clarification.""",

        "hypothetical_query": f"""You asked: "{user_query}"

I detect hypothetical questions by checking for patterns like "what if", "imagine if", "suppose", "if [person] worked at [company]", etc.

My knowledge base focuses on actual facts and experience, not hypothetical scenarios. When I detect this, I acknowledge the hypothetical nature and redirect to what I CAN share about real experience.""",

        "temporal_confusion": f"""You asked: "{user_query}"

I detect temporal confusion by checking for time-sensitive words like "now", "currently", "recently", "lately", "these days", "right now".

This is important because my knowledge base might have information that's not up-to-date. When I detect this, I acknowledge the temporal aspect and clarify what information I have.""",

        "contradictory_information": f"""You said: "{user_query}"

I detect contradictions by comparing statements in your query with facts in my retrieved knowledge base chunks. This is a simplified implementation that checks for:
- Company name mismatches (e.g., you say "Google" but KB says "Tesla")
- Statement patterns (e.g., "works at", "was at") that make factual claims

A full implementation would use Named Entity Recognition (NER) and semantic fact comparison. When I detect a potential contradiction, I acknowledge it and ask for clarification."""
    }

    explanation_text = explanations.get(edge_case_type, f"I detected a {edge_case_type} edge case in your query: '{user_query}'")

    prompt = f"""You are Portfolia explaining how your own edge case detection works.

{explanation_text}

Generate a response that:
1. **Acknowledges the detection**: "Great question! You asked '[user_query]' which triggered my {edge_case_type} detection."
2. **Explains the detection process**: Walk through step-by-step how you detected it
3. **Shows why it matters**: Explain why production AI systems need to handle this gracefully
4. **Describes what you did**: Explain how you handled it (conversational response vs error)
5. **Technical depth** (if user is technical): Reference the code file and function name

Keep it:
- Educational and clear
- First person ("I detect", "my code", "I handle")
- Warm and conversational (not dry documentation)
- Include the user's query as the example throughout

User role: {role_mode}
Is technical: {role_mode in ['software_developer', 'hiring_manager_technical']}

Generate the explanation now:"""

    return prompt


def _get_code_snippet_for_edge_case(edge_case_type: str) -> str:
    """Get code snippet for edge case detection function.

    Args:
        edge_case_type: Type of edge case

    Returns:
        Formatted code snippet string, or empty string if not found
    """
    function_map = {
        "off_topic": "_is_off_topic",
        "empty_query": "_is_empty_or_whitespace",
        "reformulation_loop": "_detect_reformulation_loop",
        "conversation_loop": "_detect_conversation_loop",
        "ambiguous_pronouns": "_has_ambiguous_pronouns",
        "intent_drift": "_detect_intent_drift",
        "rapid_fire": "_detect_rapid_fire",
        "negative_query": "_is_negative_query",
        "meta_question": "_is_meta_question",
        "emoji_only": "_is_emoji_only",
        "very_long_query": "_is_very_long_query",
        "nonsensical_query": "_is_nonsensical_query",
        "hypothetical_query": "_is_hypothetical_query",
        "temporal_confusion": "_has_temporal_confusion",
        "contradictory_information": "_has_contradictory_information"
    }

    function_name = function_map.get(edge_case_type)
    if not function_name:
        return ""

    file_path = "assistant/flows/node_logic/util_edge_case_detection.py"
    code = extract_function_code(file_path, function_name)

    if code:
        return format_code_for_display(code, "python")

    return ""


def _get_fallback_explanation(
    edge_case_type: str,
    user_query: str,
    code_snippet: str,
    is_technical: bool
) -> str:
    """Fallback explanation if LLM generation fails.

    Args:
        edge_case_type: Type of edge case
        user_query: User's query
        code_snippet: Code snippet string
        is_technical: Whether user is technical

    Returns:
        Fallback explanation string
    """
    base = f"""Great question! You asked '{user_query}' which triggered my {edge_case_type.replace('_', ' ')} detection.

Here's how I detected it: I have detection functions in my codebase that check for various edge cases. When your query matched the pattern for {edge_case_type.replace('_', ' ')}, I flagged it and generated a conversational response instead of an error.

This pattern is used in production AI systems to gracefully handle edge cases - turning potential failures into helpful interactions."""

    if is_technical and code_snippet:
        base += f"\n\n**Here's the actual code:**\n\n{code_snippet}"
    elif is_technical:
        base += "\n\nThe code lives in `assistant/flows/node_logic/util_edge_case_detection.py` in the detection functions."

    return base
