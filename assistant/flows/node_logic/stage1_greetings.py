# -*- coding: utf-8 -*-
"""Greeting detection utilities.

This module provides minimal utility functions for detecting if a query
is a simple greeting. Since Portfolia now sends the first message via
prompt_for_role_selection in session_management.py, these functions are
rarely used.

Usage:
    from assistant.flows.greetings import should_show_greeting

    if should_show_greeting(query, chat_history):
        # Rare case: user says "hi" after conversation started
        pass
"""


def is_first_turn(chat_history: list) -> bool:
    """Check if this is the first turn of the conversation.

    Args:
        chat_history: List of conversation messages (can be dicts or LangGraph message objects)

    Returns:
        True if this is the first user query (no assistant messages yet)
    """
    if not chat_history:
        return True

    # Check if there are any assistant messages
    # Handle both dict format ({"role": "assistant"}) and LangGraph message format (type="ai")
    assistant_messages = []
    for msg in chat_history:
        # Try dict format first
        if isinstance(msg, dict):
            if msg.get("role") == "assistant" or msg.get("type") == "ai":
                assistant_messages.append(msg)
        # Handle LangGraph message objects (Pydantic models)
        elif hasattr(msg, "type"):
            if msg.type == "ai" or getattr(msg, "role", None) == "assistant":
                assistant_messages.append(msg)
        # Fallback: check for "role" attribute
        elif hasattr(msg, "role") and msg.role == "assistant":
            assistant_messages.append(msg)

    return len(assistant_messages) == 0


def should_show_greeting(query: str, chat_history: list) -> bool:
    """Determine if we should show a greeting instead of answering the query.

    Show greeting if:
    1. This is the first turn AND
    2. The query is a simple greeting/hello (not a substantive question)

    Args:
        query: User's query text
        chat_history: Conversation history

    Returns:
        True if we should respond with a greeting
    """
    if not is_first_turn(chat_history):
        return False

    # Simple greetings that warrant a warm introduction
    greeting_patterns = [
        "hello", "hi", "hey", "greetings", "good morning",
        "good afternoon", "good evening", "what's up", "sup",
        "how are you", "how do you do"
    ]

    query_lower = query.lower().strip()

    # Check if query is primarily a greeting (≤5 words and IS a greeting)
    words = query_lower.split()
    if len(words) <= 5:
        for pattern in greeting_patterns:
            # For single-word patterns, check word boundaries to avoid
            # substring false positives (e.g., "coaching" matching "hi")
            if ' ' in pattern:
                # Multi-word pattern: check if it appears as a phrase
                if pattern in query_lower:
                    return True
            else:
                # Single-word pattern: must be a standalone word
                if pattern in words:
                    return True

    return False
