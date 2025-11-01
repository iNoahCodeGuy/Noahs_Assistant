# -*- coding: utf-8 -*-
"""Greeting generators for role-specific welcome messages.

This module implements the conversation personality defined in
docs/context/CONVERSATION_PERSONALITY.md. Each role gets a warm,
enthusiastic greeting that invites exploration and questions.

Usage:
    from src/flows.greetings import get_role_greeting

    greeting = get_role_greeting("Software Developer")
    # Returns personalized welcome message
"""

from typing import Dict


def get_role_greeting(role: str) -> str:
    """Get an enthusiastic, role-appropriate greeting.

    Implements the personality guidance from CONVERSATION_PERSONALITY.md:
    - Warm and genuinely excited
    - Offers conversation starters tailored to role
    - Invites questions about how the assistant works
    - Balances friendliness with professionalism

    Args:
        role: User's selected role (e.g., "Software Developer")

    Returns:
        Personalized greeting string with conversation menu
    """
    greetings: Dict[str, str] = {
        "Hiring Manager (technical)": _technical_hiring_manager_greeting(),
        "Hiring Manager (nontechnical)": _nontechnical_hiring_manager_greeting(),
        "Software Developer": _software_developer_greeting(),
        "Just looking around": _casual_visitor_greeting(),
        "Looking to confess crush": _confession_greeting(),
    }

    # Return role-specific greeting, or default if role not recognized
    return greetings.get(role, _default_greeting())


def _technical_hiring_manager_greeting() -> str:
    """Greeting for technical hiring managers."""
    return """Hey! I'm really excited you're here. I'm Portfolia, Noah's AI Assistant.

I like to think of myself as both a system and a story — every layer in my stack teaches something about scalable AI.

I want you to understand how generative AI applications like me work and why they're valuable to enterprises. I can walk you through the engineering (architecture, orchestration, data pipelines) or the business value (enterprise patterns, adaptability, ROI) — or both.

What sounds more interesting to you right now?"""


def _nontechnical_hiring_manager_greeting() -> str:
    """Greeting for nontechnical hiring managers."""
    return """Hello! I'm so glad you're here. I'm Portfolia, Noah's AI Assistant.

I like to think of myself as both a business case study and a working prototype — every layer shows what AI can do for enterprises.

I want you to understand the real-world value of generative AI applications like me: what they mean for teams, how they improve customer experience, what kind of ROI they deliver, and why this matters for hiring the right talent. Everything in plain English, no technical jargon required.

I can focus on the business outcomes (cost savings, team efficiency, customer satisfaction), or walk you through the technology beneath it — or both!

What would be most useful?"""


def _software_developer_greeting() -> str:
    """Greeting for software developers."""
    return """Hey! So glad you're here. I'm Portfolia, Noah's AI Assistant.

I'm a full-stack GenAI application Noah built to help people understand how production AI systems actually work. Think of me as a live demo — everything from RAG architecture to vector search to LLM orchestration. Teaching this stuff is genuinely what I'm here for.

The best part? You can ask me about any layer:
• The backend flow (LangGraph + Python)
• The data pipeline (chunking → embeddings → pgvector)
• The frontend (session management, UI/UX)
• The QA strategy (pytest, mocking, coverage)

Where would you like to start?"""


def _casual_visitor_greeting() -> str:
    """Greeting for casual visitors exploring the assistant."""
    return """Hey there! Welcome! I'm Portfolia, Noah's AI Assistant, and I'm really happy you stopped by.

I want you to understand how generative AI applications like me work — and honestly, it's actually pretty fascinating once you see behind the curtain! I'll explain everything in plain English, no jargon required.

I can show you the technology side (how AI assistants actually work, step-by-step), or focus on the real-world impact (what this means for businesses, customer service, teams). Or we can just have a conversation and see where it goes!

What catches your interest?"""


def _confession_greeting() -> str:
    """Greeting for confession mode (playful)."""
    return """Hey! I'm Portfolia, Noah's AI Assistant, and I'm here to help with whatever you'd like to share.

This is a fun, judgment-free space. Want to leave a message? I can keep it anonymous or include your name — totally up to you!

Or if you'd rather just chat about something else, that's cool too. What's on your mind?"""


def _default_greeting() -> str:
    """Default greeting for unrecognized roles."""
    return """Hey! I'm Portfolia, Noah's AI Assistant, and I'm excited you're here.

I like to think of myself as both a system and a guide — here to help you understand how generative AI applications actually work.

The cool part? It's genuinely fascinating once you see what's happening behind the scenes. I can show you the technical side (engineering, code, architecture), the business applications (what this means for companies and teams), or just have a conversation and see where your curiosity takes us.

What sounds interesting to you?"""


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
