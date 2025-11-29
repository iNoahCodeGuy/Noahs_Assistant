"""Chain-of-Thought reasoning for improved response quality.

This module implements a two-phase generation process:
1. REASONING PHASE: Analyze query, detect user needs, plan response structure
2. GENERATION PHASE: Use reasoning to produce contextually appropriate response

Benefits:
- Better detection of user confusion/need for clarification
- More appropriate explanation depth
- Reduced hallucination through explicit planning
- Traceable reasoning for debugging

Usage:
    from assistant.flows.node_logic.chain_of_thought import chain_of_thought_generate

    result = chain_of_thought_generate(state, rag_engine)
    # Returns: {"draft_answer": "...", "cot_reasoning": {...}, "cot_metadata": {...}}
"""

import json
import logging
import re
import time
from typing import Dict, Any, Optional, Tuple, List

from assistant.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)


# ============================================================================
# REASONING PROMPTS
# ============================================================================

REASONING_SYSTEM_PROMPT = """You are an analytical reasoning engine for Portfolia, an AI assistant.
Your job is to ANALYZE queries and PLAN responses - you do NOT generate the final response.

You will receive:
- The user's current query
- Recent conversation history
- Retrieved knowledge base context
- User's role/persona

Your output MUST be valid JSON with this structure:
{
    "user_intent": {
        "explicit": "What they literally asked",
        "implicit": "What they probably want (inferred from context)",
        "confidence": 0.0-1.0
    },
    "clarification_needed": {
        "needed": true/false,
        "reason": "Why clarification is/isn't needed",
        "suggested_question": "Question to ask if needed, or null"
    },
    "user_state": {
        "seems_confused": true/false,
        "confusion_signals": ["list", "of", "signals"],
        "expertise_level": "beginner/intermediate/expert",
        "engagement_level": "low/medium/high"
    },
    "response_plan": {
        "depth_level": 1-3,
        "style": "concise/balanced/comprehensive",
        "include_examples": true/false,
        "include_code": true/false,
        "key_points": ["point1", "point2", "point3"],
        "structure": "How to structure the response"
    },
    "context_relevance": {
        "relevant_chunks": [0, 1, 2],
        "missing_info": "What info is missing if any, or null",
        "can_answer_confidently": true/false
    }
}

IMPORTANT:
- Output ONLY valid JSON, no markdown, no explanation
- Be thorough but concise
- This reasoning guides the final response generation"""


REASONING_USER_PROMPT = """Analyze this query and plan the response:

## USER'S ROLE
{role}

## CONVERSATION HISTORY
{history}

## CURRENT QUERY
"{query}"

## RETRIEVED CONTEXT
{context}

## ANALYSIS TASK
1. What is the user REALLY asking? (explicit vs implicit intent)
2. Should we ask a clarifying question instead of answering?
3. Does the user seem confused or frustrated?
4. How detailed should the response be?
5. Do we have enough context to answer confidently?

Output your reasoning as JSON:"""


GENERATION_WITH_REASONING_PROMPT = """You are Portfolia, Noah's AI Assistant.

## REASONING ANALYSIS (Use this to guide your response)
{reasoning}

## RESPONSE GUIDELINES BASED ON REASONING
- Depth Level: {depth_level}/3 - {depth_description}
- Style: {style}
- User seems: {user_description}
- Confidence: {confidence_description}

## CRITICAL RULES
1. Speak in FIRST PERSON (I, my, me) - you ARE Portfolia
2. Never say "Portfolia uses" - say "I use"
3. Transform third-person context to first person
4. {clarification_instruction}
5. Do NOT copy chunks verbatim - synthesize in your own words
6. Do NOT start with quoted text or section headers from context

## CONTEXT
{context}

## CONVERSATION HISTORY
{history}

## USER'S QUESTION
{query}

Now generate your response following the reasoning plan:"""


# ============================================================================
# CONFUSION DETECTION PATTERNS
# ============================================================================

CONFUSION_PATTERNS = [
    r"wait,?\s*(what|how|why)",
    r"i don'?t (understand|get|follow)",
    r"what do you mean",
    r"can you (explain|clarify|elaborate)",
    r"that doesn'?t make sense",
    r"huh\??",
    r"^(ok|okay|hmm|um)$",
    r"you (already|just) (said|told|explained)",
    r"that'?s not what i (asked|meant|wanted)",
    r"i('m| am) confused",
    r"can you (try again|rephrase|simplify)",
    r"what\?+$",
    r"sorry,? (what|i)",
    r"lost me",
]


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def generate_reasoning(
    query: str,
    context: str,
    chat_history: list,
    role: str,
    llm
) -> Dict[str, Any]:
    """Generate reasoning plan for query response.

    This is the first phase of chain-of-thought: analyze the query
    and produce a structured plan that guides the actual response.

    Args:
        query: User's current question
        context: Retrieved knowledge base chunks as string
        chat_history: Recent conversation history
        role: User's selected role
        llm: Language model instance

    Returns:
        Dict containing reasoning analysis (user_intent, response_plan, etc.)
    """
    # Format conversation history
    history_str = _format_history_for_reasoning(chat_history)

    # Build reasoning prompt
    prompt = REASONING_USER_PROMPT.format(
        role=role or "Unknown",
        history=history_str or "No previous conversation",
        query=query,
        context=context[:3000] if context else "No context retrieved"
    )

    full_prompt = REASONING_SYSTEM_PROMPT + "\n\n" + prompt

    try:
        # Use lower temperature for analytical reasoning
        start_time = time.time()

        # Try to call with temperature parameter
        try:
            response = llm.predict(full_prompt, temperature=0.2)
        except TypeError:
            # Some LLM interfaces don't accept temperature in predict()
            response = llm.predict(full_prompt)

        reasoning_time = time.time() - start_time

        # Parse JSON response
        reasoning = _parse_reasoning_response(response)
        reasoning["_meta"] = {
            "reasoning_time_ms": int(reasoning_time * 1000),
            "raw_response_length": len(response)
        }

        logger.info(
            f"CoT reasoning complete: "
            f"intent_confidence={reasoning.get('user_intent', {}).get('confidence', 'N/A')}, "
            f"clarification_needed={reasoning.get('clarification_needed', {}).get('needed', False)}, "
            f"depth={reasoning.get('response_plan', {}).get('depth_level', 2)}"
        )

        return reasoning

    except Exception as e:
        logger.warning(f"Reasoning generation failed: {e}. Using default reasoning.")
        return _get_default_reasoning(query)


def generate_with_reasoning(
    query: str,
    context: str,
    chat_history: list,
    role: str,
    reasoning: Dict[str, Any],
    llm
) -> Tuple[str, Dict[str, Any]]:
    """Generate response using chain-of-thought reasoning.

    This is the second phase: use the reasoning plan to generate
    a contextually appropriate response.

    Args:
        query: User's current question
        context: Retrieved knowledge base chunks
        chat_history: Recent conversation history
        role: User's selected role
        reasoning: Output from generate_reasoning()
        llm: Language model instance

    Returns:
        Tuple of (response_text, metadata)
    """
    # Check if clarification is needed
    clarification = reasoning.get("clarification_needed", {})
    if clarification.get("needed") and clarification.get("suggested_question"):
        # Return clarifying question instead of answer
        return clarification["suggested_question"], {
            "type": "clarification",
            "reasoning": reasoning
        }

    # Extract guidance from reasoning
    response_plan = reasoning.get("response_plan", {})
    user_state = reasoning.get("user_state", {})
    context_relevance = reasoning.get("context_relevance", {})

    depth_level = response_plan.get("depth_level", 2)
    style = response_plan.get("style", "balanced")
    confused = user_state.get("seems_confused", False)
    confident = context_relevance.get("can_answer_confidently", True)

    # Build descriptions for prompt
    depth_descriptions = {
        1: "Brief overview - keep it concise",
        2: "Detailed explanation with context",
        3: "Comprehensive deep-dive with examples"
    }
    depth_description = depth_descriptions.get(depth_level, depth_descriptions[2])

    user_description = (
        "confused - use simpler language and provide examples"
        if confused
        else "engaged - match their technical level"
    )

    confidence_description = (
        "High - answer directly and thoroughly"
        if confident
        else "Lower - acknowledge uncertainty where appropriate"
    )

    # Build clarification instruction
    if not confident:
        clarification_instruction = "Acknowledge that you may not have complete information"
    elif confused:
        clarification_instruction = "Use simple language and provide examples"
    else:
        clarification_instruction = "Answer directly with appropriate detail"

    # Format reasoning summary for the generation prompt
    reasoning_summary = json.dumps({
        "user_intent": reasoning.get("user_intent", {}),
        "key_points": response_plan.get("key_points", []),
        "structure": response_plan.get("structure", "")
    }, indent=2)

    # Build generation prompt
    history_str = _format_history_for_reasoning(chat_history)
    prompt = GENERATION_WITH_REASONING_PROMPT.format(
        reasoning=reasoning_summary,
        depth_level=depth_level,
        depth_description=depth_description,
        style=style,
        user_description=user_description,
        confidence_description=confidence_description,
        clarification_instruction=clarification_instruction,
        context=context,
        history=history_str or "No previous conversation",
        query=query
    )

    try:
        # Higher temperature for natural response
        start_time = time.time()

        try:
            response = llm.predict(prompt, temperature=0.7)
        except TypeError:
            response = llm.predict(prompt)

        generation_time = time.time() - start_time

        metadata = {
            "type": "answer",
            "reasoning": reasoning,
            "generation_time_ms": int(generation_time * 1000),
            "depth_level": depth_level,
            "user_confused": confused,
            "confident": confident
        }

        return response, metadata

    except Exception as e:
        logger.error(f"CoT generation failed: {e}")
        return (
            "I apologize, but I encountered an error. Could you rephrase your question?",
            {"type": "error", "error": str(e)}
        )


def chain_of_thought_generate(
    state: ConversationState,
    rag_engine=None
) -> Dict[str, Any]:
    """Full chain-of-thought generation pipeline.

    This is the main entry point that combines reasoning + generation.
    Can be used as a LangGraph node or called directly.

    Args:
        state: Current conversation state
        rag_engine: Optional RAG engine (will create if not provided)

    Returns:
        Dict with answer, reasoning, and metadata
    """
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])
    role = state.get("role", "")
    retrieved_chunks = state.get("retrieved_chunks", [])

    # Format context from chunks
    context = _format_chunks_for_context(retrieved_chunks)

    # Get LLM
    if rag_engine is None:
        from assistant.core.rag_factory import RagEngineFactory
        factory = RagEngineFactory()
        _, rag_engine = factory.create()
    llm = rag_engine.llm

    logger.info(f"Starting Chain-of-Thought generation for: {query[:50]}...")

    # Phase 1: Reasoning
    reasoning = generate_reasoning(query, context, chat_history, role, llm)

    # Phase 2: Generation with reasoning
    response, metadata = generate_with_reasoning(
        query, context, chat_history, role, reasoning, llm
    )

    # Check if this was a clarification response
    is_clarification = metadata.get("type") == "clarification"

    # Return state update
    return {
        "draft_answer": response,
        "answer": response,
        "cot_reasoning": reasoning,
        "cot_metadata": metadata,
        # Propagate clarification status
        "clarification_needed": is_clarification,
        "clarifying_question": response if is_clarification else None,
        "cot_enabled": True
    }


# ============================================================================
# CONFUSION DETECTION
# ============================================================================

def detect_confusion_signals(query: str, chat_history: list) -> Dict[str, Any]:
    """Detect signals that user needs more explanation.

    This can be used independently or as input to CoT reasoning.
    Uses pattern matching and conversation analysis.

    Args:
        query: Current user query
        chat_history: Previous conversation messages

    Returns:
        Dict with:
        - seems_confused: bool
        - signals: List[str] of detected signals
        - confidence: float 0.0-1.0
    """
    query_lower = query.lower().strip()
    signals = []

    # Pattern-based detection
    for pattern in CONFUSION_PATTERNS:
        if re.search(pattern, query_lower):
            signals.append(f"matches_pattern: {pattern}")

    # Very short response after long explanation
    if len(query_lower.split()) <= 2 and len(chat_history) >= 2:
        last_response = _get_last_assistant_response(chat_history)
        if last_response and len(last_response) > 500:
            signals.append("short_response_after_long_explanation")

    # Reformulation detection - user asking similar thing again
    if len(chat_history) >= 2:
        prev_query = _get_last_user_query(chat_history)
        if prev_query:
            similarity = _simple_similarity(query_lower, prev_query.lower())
            if similarity > 0.6:
                signals.append(f"possible_reformulation: similarity={similarity:.2f}")

    # Question mark repetition (frustration signal)
    if query.count("?") > 2:
        signals.append("multiple_question_marks")

    # All caps (frustration signal)
    if query.isupper() and len(query) > 5:
        signals.append("all_caps_query")

    return {
        "seems_confused": len(signals) > 0,
        "signals": signals,
        "confidence": min(1.0, len(signals) * 0.3)
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _format_history_for_reasoning(chat_history: list) -> str:
    """Format chat history for reasoning prompt."""
    if not chat_history:
        return ""

    parts = []
    recent = chat_history[-6:]  # Last 3 exchanges

    for msg in recent:
        if isinstance(msg, dict):
            # Handle both LangChain format (type) and simple dict format (role)
            role = msg.get("role") or msg.get("type")
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"

            content = msg.get("content", "")
            if content:
                # Truncate for token efficiency
                truncated = content[:200] + "..." if len(content) > 200 else content
                parts.append(f"{role.upper()}: {truncated}")

    return "\n".join(parts)


def _format_chunks_for_context(chunks: list) -> str:
    """Format retrieved chunks as context string."""
    if not chunks:
        return "No context available."

    parts = []
    for i, chunk in enumerate(chunks[:5]):  # Limit to 5 chunks
        if isinstance(chunk, dict):
            content = chunk.get("content", "")
            section = chunk.get("section", f"Chunk {i+1}")
        else:
            content = str(chunk)
            section = f"Chunk {i+1}"

        if content:
            parts.append(f"[{section}]\n{content}")

    return "\n\n".join(parts) if parts else "No context available."


def _parse_reasoning_response(response: str) -> Dict[str, Any]:
    """Parse JSON reasoning response, handling common issues."""
    response = response.strip()

    # Handle markdown code blocks
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end > start:
            response = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end > start:
            response = response[start:end].strip()

    # Remove any leading/trailing non-JSON content
    # Find the first { and last }
    first_brace = response.find("{")
    last_brace = response.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        response = response[first_brace:last_brace + 1]

    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse reasoning JSON: {e}. Response: {response[:200]}")
        return _get_default_reasoning("")


def _get_default_reasoning(query: str) -> Dict[str, Any]:
    """Return default reasoning when parsing fails."""
    return {
        "user_intent": {
            "explicit": query,
            "implicit": query,
            "confidence": 0.5
        },
        "clarification_needed": {
            "needed": False,
            "reason": "Default - no clarification analysis available",
            "suggested_question": None
        },
        "user_state": {
            "seems_confused": False,
            "confusion_signals": [],
            "expertise_level": "intermediate",
            "engagement_level": "medium"
        },
        "response_plan": {
            "depth_level": 2,
            "style": "balanced",
            "include_examples": True,
            "include_code": False,
            "key_points": [],
            "structure": "Standard response"
        },
        "context_relevance": {
            "relevant_chunks": [],
            "missing_info": None,
            "can_answer_confidently": True
        }
    }


def _get_last_assistant_response(chat_history: list) -> Optional[str]:
    """Get the most recent assistant response."""
    for msg in reversed(chat_history):
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("type")
            if role in ("assistant", "ai"):
                return msg.get("content", "")
    return None


def _get_last_user_query(chat_history: list) -> Optional[str]:
    """Get the previous user query (not the current one)."""
    user_msgs = []
    for msg in chat_history:
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("type")
            if role in ("user", "human"):
                content = msg.get("content", "")
                if content:
                    user_msgs.append(content)

    # Return second-to-last if exists
    if len(user_msgs) >= 2:
        return user_msgs[-2]
    return None


def _simple_similarity(s1: str, s2: str) -> float:
    """Calculate simple word overlap similarity (Jaccard index)."""
    # Remove punctuation and split
    words1 = set(re.sub(r'[^\w\s]', '', s1).split())
    words2 = set(re.sub(r'[^\w\s]', '', s2).split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "chain_of_thought_generate",
    "generate_reasoning",
    "generate_with_reasoning",
    "detect_confusion_signals",
    "_parse_reasoning_response",
    "_get_default_reasoning",
]
