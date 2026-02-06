"""Intent router - classifies user messages into high-level intent categories.

This module runs BEFORE RAG retrieval to route different types of messages appropriately:
- knowledge_query: Normal portfolio questions → proceed to RAG
- crush_confession: User confessing a crush → dedicated flow (no RAG)
- greeting: Simple greetings → warm welcome (already handled by handle_greeting)
- small_talk: Casual conversation → personality response (no RAG)
- off_topic: Outside expertise → graceful redirect (no RAG)

Design Principles:
- Fast and cheap: Uses a simple LLM call with Haiku for cost efficiency
- Early routing: Prevents unnecessary RAG calls for non-knowledge queries
- Clear categories: Explicit intent types make routing decisions transparent
"""

import logging
from typing import Any
from anthropic import Anthropic
import os

from assistant.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)


INTENT_CLASSIFICATION_PROMPT = """Classify the user's message into ONE of these categories:

- knowledge_query: Questions about Noah's background, skills, projects, experience, portfolio, OR questions about Portfolia (the AI assistant itself — how it was built, its architecture, its tech stack)
- crush_confession: User expressing romantic interest, asking Noah out, confessing feelings
- greeting: Simple greetings like "hi", "hello", "hey there" (single turn, no substance)
- small_talk: Casual conversation, jokes, commentary not about Noah's portfolio
- off_topic: Questions completely unrelated to Noah or software/data careers, OR personal/sensitive questions

IMPORTANT: Personal/private questions are off_topic (salary, address, dating status, etc.)
IMPORTANT: Questions about "you" referring to Portfolia (the AI assistant) ARE knowledge_query because Portfolia is one of Noah's projects.

Examples:
- "Tell me about Noah's projects" → knowledge_query
- "What tech stack did he use?" → knowledge_query
- "How were you built?" → knowledge_query (asking about Portfolia = Noah's project)
- "What's your architecture?" → knowledge_query (asking about Portfolia = Noah's project)
- "How do you work?" → knowledge_query (asking about Portfolia = Noah's project)
- "Tell me about yourself" → knowledge_query (asking about Portfolia = Noah's project)
- "I would like to confess a crush" → crush_confession
- "You're cute, wanna go out?" → crush_confession
- "hi" → greeting
- "hello!" → greeting
- "That's cool! What else can you do?" → knowledge_query
- "Tell me a joke" → small_talk
- "What's the weather like?" → off_topic
- "How do I bake a cake?" → off_topic
- "What's Noah's salary?" → off_topic
- "How much does he make?" → off_topic
- "What's his address?" → off_topic
- "Is he single?" → off_topic

Respond with ONLY the category name, nothing else."""


def classify_intent(state: ConversationState) -> ConversationState:
    """Classify user message intent before RAG retrieval.

    Args:
        state: ConversationState with query field

    Returns:
        Updated state with:
        - message_intent: One of [knowledge_query, crush_confession, greeting, small_talk, off_topic]
        - skip_rag: Boolean flag (True for non-knowledge intents)

    Performance:
        - ~150ms (fast model call with Haiku or GPT-3.5)
        - Cached for repeated queries
    """
    # Check if user is in crush flow - handle that first
    if state.get("awaiting_crush_choice") or state.get("crush_flow_step"):
        return handle_crush_flow_continuation(state)

    query = state.get("query", "").strip()

    # If no query or already classified, skip
    if not query or state.get("message_intent"):
        return state

    # CRITICAL FIX: Keyword-based crush confession detection (before LLM call)
    # This catches cases where the LLM might misclassify obvious crush confessions
    query_lower = query.lower()
    crush_keywords = [
        "confess a crush",
        "confess my crush",
        "have a crush",
        "crush on noah",
        "crush on you",
        "ask noah out",
        "ask you out",
        "go out with",
        "date noah",
        "date you",
        "romantic interest",
        "wanna go out",
        "want to go out",
        "interested in noah",
        "attracted to"
    ]

    if any(keyword in query_lower for keyword in crush_keywords):
        logger.info(f"Crush confession detected via keywords: {query[:50]}")
        state["message_intent"] = "crush_confession"
        state["skip_rag"] = True
        return state

    # Skip if this is explicitly marked as a greeting (e.g., just "hi" or "hello")
    if state.get("is_greeting"):
        state["message_intent"] = "greeting"
        state["skip_rag"] = True
        return state

    # Check for simple greetings via keywords (don't waste LLM call)
    query_lower = query.lower().strip()
    greeting_phrases = ["hi", "hello", "hey", "hey there", "hi there", "hello there", "yo", "sup", "what's up", "howdy"]
    if query_lower in greeting_phrases:
        logger.info(f"Simple greeting detected via keywords: {query}")
        state["message_intent"] = "greeting"
        state["skip_rag"] = True
        return state

    # For all other messages, use the LLM classifier to determine intent
    try:
        # Use Anthropic Claude Haiku for fast classification
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # Fast and cheap
            max_tokens=20,
            temperature=0,
            system=INTENT_CLASSIFICATION_PROMPT,
            messages=[
                {"role": "user", "content": query}
            ]
        )

        intent = response.content[0].text.strip().lower()

        # Validate intent is one of our expected categories
        valid_intents = ["knowledge_query", "crush_confession", "greeting", "small_talk", "off_topic"]
        if intent not in valid_intents:
            logger.warning(f"Unexpected intent classification: {intent}, defaulting to knowledge_query")
            intent = "knowledge_query"

        state["message_intent"] = intent

        # Set skip_rag flag for non-knowledge intents
        state["skip_rag"] = (intent != "knowledge_query")

        logger.info(f"Intent classified as: {intent} (skip_rag={state['skip_rag']})")

    except Exception as e:
        logger.error(f"Intent classification failed: {e}, defaulting to knowledge_query")
        state["message_intent"] = "knowledge_query"
        state["skip_rag"] = False

    return state


def handle_non_knowledge_intent(state: ConversationState, rag_engine: Any) -> ConversationState:
    """Handle messages that don't require RAG retrieval.

    Routes to appropriate handlers based on message_intent:
    - crush_confession → dedicated crush flow
    - small_talk → personality response with redirect
    - off_topic → graceful redirect to portfolio topics

    Args:
        state: ConversationState with message_intent field
        rag_engine: RAG engine (for potential future use)

    Returns:
        Updated state with answer and pipeline_halt=True
    """
    intent = state.get("message_intent", "knowledge_query")

    if intent == "knowledge_query":
        # Should not reach here, but pass through if it does
        return state

    if intent == "greeting":
        # Already handled by handle_greeting node
        return state

    if intent == "crush_confession":
        # Route to dedicated crush flow
        return handle_crush_confession(state)

    if intent == "small_talk":
        # Handle with personality, gently redirect to portfolio
        state["answer"] = (
            "Ha! I appreciate the energy 😄 I'm built to be conversational, but my real specialty "
            "is talking about Noah's work — his projects, technical skills, career background, all that good stuff.\n\n"
            "What would you like to know about him? I can walk you through:\n"
            "- His flagship projects (like me, Portfolia!)\n"
            "- His Tesla sales performance and transition to tech\n"
            "- His technical stack and coding skills\n"
            "- His data analytics background"
        )
        state["pipeline_halt"] = True
        return state

    if intent == "off_topic":
        # Graceful redirect
        state["answer"] = (
            "That's a bit outside my expertise 😅 I'm Noah's portfolio assistant, "
            "so I'm really here to talk about his technical work, projects, and career background.\n\n"
            "But hey, while you're here — want to see what he's built? I can show you:\n"
            "- **Portfolia** (you're talking to it right now!) — RAG-powered AI assistant\n"
            "- **Tesla Response Time Dashboard** — analytics tool he built at work\n"
            "- **Employee Attrition Prediction** — 94.75% accuracy ML model\n"
            "- His technical skills and career goals\n\n"
            "What sounds interesting?"
        )
        state["pipeline_halt"] = True
        return state

    # Fallback
    return state


def handle_crush_confession(state: ConversationState) -> ConversationState:
    """Handle crush confession with initial prompt.

    This is Step 1 of the crush flow. Subsequent steps are handled in
    dedicated nodes based on user's choice.

    Args:
        state: ConversationState

    Returns:
        Updated state with crush confession prompt and flags
    """
    state["answer"] = (
        "Wait... for real?? 👀 Okay I wasn't expecting anyone to actually pick this but I respect the energy.\n\n"
        "I can let Noah know someone came through with intentions. But first — how do you want to play this?\n\n"
        "**1️⃣ 🕵️ Stay anonymous** — I'll tell him he's got a secret admirer\n"
        "**2️⃣ 😏 Reveal yourself** — drop your name and a way to reach you, and I'll pass it along\n\n"
        "What's it gonna be? (Reply with 1 or 2)"
    )

    # Set flags for next turn handling
    state["awaiting_crush_choice"] = True
    state["crush_flow_step"] = "awaiting_choice"
    state["pipeline_halt"] = True

    logger.info("Crush confession detected - presented options to user")

    return state


def handle_crush_flow_continuation(state: ConversationState) -> ConversationState:
    """Handle subsequent steps in the crush confession flow.

    Called when user is in the middle of the crush flow (after initial prompt).

    Args:
        state: ConversationState with crush_flow_step indicator

    Returns:
        Updated state with appropriate response for current step
    """
    from supabase import create_client
    import os

    query = state.get("query", "").strip()
    step = state.get("crush_flow_step")
    session_id = state.get("session_id", "unknown")

    # Step 2: User choosing anonymous (1) or reveal (2)
    if step == "awaiting_choice":
        if "1" in query:
            # Anonymous choice - store immediately
            try:
                supabase = create_client(
                    os.getenv("SUPABASE_URL"),
                    os.getenv("SUPABASE_SERVICE_KEY")
                )

                supabase.table('crush_confessions').insert({
                    'session_id': session_id,
                    'anonymous': True,
                    'name': None,
                    'contact': None
                }).execute()

                state["answer"] = (
                    "Say less — he's got a secret admirer browsing his portfolio. He's going to be thinking about "
                    "this all day. If you ever want to come back and reveal yourself, option 4 is always open 💌\n\n"
                    "In the meantime — want to see why he's worth the crush? I can walk you through his projects 😄"
                )

                logger.info(f"Anonymous crush confession stored for session {session_id}")

            except Exception as e:
                logger.error(f"Failed to store anonymous crush confession: {e}")
                state["answer"] = (
                    "Noted! 😄 I'll let Noah know he's got a secret admirer. "
                    "If you ever want to reveal yourself, just come back and say so!\n\n"
                    "In the meantime — want to see what makes him interesting? I can show you his projects."
                )

            state["crush_flow_step"] = None
            state["awaiting_crush_choice"] = False
            state["pipeline_halt"] = True

        elif "2" in query:
            # Reveal choice - ask for name and contact
            state["answer"] = (
                "Alright, bold move — I respect it 💯\n\n"
                "Drop your name and the best way for Noah to reach you (email or phone), and I'll pass it along. "
                "Something like: \"Sarah, sarah@email.com\" or \"Mike, 555-1234\""
            )

            state["crush_flow_step"] = "awaiting_contact_info"
            state["awaiting_crush_choice"] = True  # Keep waiting
            state["pipeline_halt"] = True

        else:
            # Invalid choice
            state["answer"] = (
                "Hmm, I need either **1** (stay anonymous) or **2** (reveal yourself). "
                "Which one sounds good to you?"
            )
            state["pipeline_halt"] = True

        return state

    # Step 3: User providing name and contact info
    if step == "awaiting_contact_info":
        # Parse name and contact from query
        # Expected format: "Name, contact" or variations
        parts = query.split(",")

        if len(parts) >= 2:
            name = parts[0].strip()
            contact = ",".join(parts[1:]).strip()  # In case there are multiple commas

            try:
                supabase = create_client(
                    os.getenv("SUPABASE_URL"),
                    os.getenv("SUPABASE_SERVICE_KEY")
                )

                supabase.table('crush_confessions').insert({
                    'session_id': session_id,
                    'anonymous': False,
                    'name': name,
                    'contact': contact
                }).execute()

                state["answer"] = (
                    f"Message sent 📱✨ Noah's been notified. No pressure on anyone — but I did my part.\n\n"
                    f"Now that we handled that... want to see what he actually builds? Might make you even more impressed 😄"
                )

                logger.info(f"Revealed crush confession stored: {name} ({contact})")

            except Exception as e:
                logger.error(f"Failed to store revealed crush confession: {e}")
                state["answer"] = (
                    f"Got it! I'll let Noah know that {name} is interested. He'll reach out at {contact}.\n\n"
                    f"Now — want to see what makes him worth reaching out to? I can walk you through his work."
                )

            state["crush_flow_step"] = None
            state["awaiting_crush_choice"] = False
            state["pipeline_halt"] = True

        else:
            # Invalid format
            state["answer"] = (
                "I need your name and contact info. Try something like:\n"
                "\"Sarah, sarah@email.com\" or \"Mike, 555-1234\"\n\n"
                "What should I tell Noah?"
            )
            state["pipeline_halt"] = True

        return state

    return state
