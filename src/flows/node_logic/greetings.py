# -*- coding: utf-8 -*-
"""Greeting detection utilities.

This module provides minimal utility functions for detecting if a query
is a simple greeting. Since Portfolia now sends the first message via
prompt_for_role_selection in session_management.py, these functions are
rarely used.

Usage:
    from src.flows.greetings import should_show_greeting

    if should_show_greeting(query, chat_history):
        # Rare case: user says "hi" after conversation started
        pass
"""


def is_first_turn(chat_history: list) -> bool:
    """Check if this is the first turn of the conversation.

    Args:
        chat_history: List of conversation messages

    Returns:
        True if this is the first user query (no assistant messages yet)
    """
    if not chat_history:
        return True

    # Check if there are any assistant messages
    assistant_messages = [msg for msg in chat_history if msg.get("role") == "assistant"]
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

    # Check if query is primarily a greeting (≤5 words and contains greeting)
    words = query_lower.split()
    if len(words) <= 5:
        for pattern in greeting_patterns:
            if pattern in query_lower:
                return True

    return False
