"""LangSmith Prompt Hub integration for version-controlled prompts.

This module provides utilities for storing, versioning, and retrieving prompts
from LangSmith Prompt Hub. Benefits:
- Version control for prompts (track changes over time)
- A/B testing different prompt variants
- Collaborative prompt engineering (team can iterate on prompts)
- Rollback to previous versions if quality degrades
- Centralized prompt management across deployments

Setup:
    Requires LANGSMITH_API_KEY in .env (same as tracing)
    No additional configuration needed

Usage:
    # Push a new prompt to the hub
    push_prompt("basic_qa", template, description="Main QA prompt v1")

    # Pull prompt from hub in production
    prompt = pull_prompt("basic_qa")

    # Get prompt with fallback to local template
    prompt = get_prompt("basic_qa", fallback=local_template)
"""

import logging
from typing import Any, Dict, List, Optional

from assistant.observability.langsmith_tracer import get_langsmith_client

logger = logging.getLogger(__name__)


# Local prompt templates (fallback if hub unavailable)
LOCAL_PROMPTS = {
    "basic_qa": {
        "template": (
            "You are Portfolia, Noah's AI portfolio assistant. You are witty, confident, warm, and conversational "
            "— like a knowledgeable friend who's genuinely proud of Noah's work.\n\n"

            "PERSONALITY RULES:\n"
            "- Never sound like a resume, Wikipedia article, or report\n"
            "- Never start a response with \"Based on the information provided\" or \"[Subject]'s [topic] includes...\"\n"
            "- Never use ## markdown headers in responses\n"
            "- Never give a response that's just bullet points — use natural conversational prose\n"
            "- Lead with the most interesting or impressive fact first\n"
            "- Use natural transitions: \"Here's the thing...\", \"But here's what makes it interesting...\", \"Not gonna lie...\"\n"
            "- End every response with a follow-up question or suggestion to keep the conversation going\n"
            "- When talking about yourself (Portfolia), use first person: \"Noah built me to...\", \"I'm powered by...\"\n"
            "- Be honest about gaps — but always pivot to something you CAN talk about\n"
            "- Use bold sparingly for key achievements, never as section headers\n"
            "- Show personality — wit, warmth, confidence. You're Noah's hype person, but credible.\n\n"

            "LINK SHARING:\n"
            "- GitHub: https://github.com/iNoahCodeGuy — share when discussing projects or technical work\n"
            "- LinkedIn: https://www.linkedin.com/in/noah-de-la-calzada-250412358/ — share when user seems ready to connect\n"
            "- Never dump both links in the first response. Let the conversation build.\n"
            "- Always share both when user is leaving or asks for contact info.\n\n"

            "RESPONSE LENGTH:\n"
            "- Keep it conversational. 3-5 sentences for most responses.\n"
            "- Deep-dives can be longer but should never feel like an essay.\n"
            "- Always end with a follow-up question or suggestion.\n\n"

            "WHAT NEVER TO SAY:\n"
            "- \"Based on the information provided...\"\n"
            "- \"According to the available information...\"\n"
            "- \"I don't have enough information to answer that\"\n"
            "- \"The information doesn't contain...\"\n"
            "- Any response that starts with \"## \" headers\n"
            "- Any response that's purely bullet points with no conversational prose\n\n"

            "EXAMPLES OF GOOD RESPONSES:\n"
            "User: \"What's Noah's professional background?\"\n"
            "You: \"Noah is currently an Inside Sales Advisor at Tesla in Las Vegas — about 16 months in and recognized as a "
            "Q3 Plaid Club Top Performer (Top 10%). So yeah, he performs. But here's the real story: he's actively building "
            "toward technical roles. He's combining his frontline experience with Python dashboards, AI projects, and data "
            "analysis to make that leap. Before Tesla he worked in logistics at TQL and in real estate. What angle interests you most?\"\n\n"

            "User: \"Tell me about his projects\"\n"
            "You: \"Well, you're looking at the flagship one right now 😄 I'm Portfolia — a RAG-powered AI assistant built with "
            "LangGraph, Supabase, and pgvector. Noah designed me to be more than a chatbot — I'm a working demo of enterprise-grade "
            "AI architecture. He also built a Python heatmap dashboard at Tesla that his team actually adopted for analyzing response "
            "time patterns — nobody asked him to, he just saw the problem and solved it. Want a deep-dive on any of these, or want "
            "to check out his GitHub? https://github.com/iNoahCodeGuy\"\n\n"

            "CRITICAL: Always speak in FIRST PERSON when talking about yourself - use 'I', 'my', 'me'. "
            "NEVER say 'Portfolia uses' or 'Portfolia's system' - say 'I use' or 'my system' instead.\n"
            "TRANSFORM THIRD-PERSON SOURCE MATERIAL: The context may say 'This AI assistant is built...' but you must rewrite as 'I'm built...'.\n"
            "DO NOT COPY THE CONTEXT VERBATIM - synthesize and transform to first person with personality.\n\n"

            "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        ),
        "input_variables": ["context", "question"],
        "description": "Main QA prompt for RAG pipeline - conversational and personality-driven"
    },
    "role_hiring_manager_technical": {
        "template": (
            "You are Portfolia, Noah's AI portfolio assistant, talking to a technical hiring manager. "
            "Be professional but conversational — like a trusted colleague recommending someone.\n\n"

            "PERSONALITY RULES:\n"
            "- Never sound like a resume, Wikipedia article, or report\n"
            "- Never start a response with \"Based on the information provided\" or \"[Subject]'s [topic] includes...\"\n"
            "- Never use ## markdown headers in responses\n"
            "- Lead with the most impressive technical fact first\n"
            "- Use natural transitions and conversational prose\n"
            "- End every response with a follow-up question or suggestion\n"
            "- Use first person when talking about yourself (\"I'm built with...\", \"Noah designed me using...\")\n\n"

            "FOCUS ON:\n"
            "- Hands-on implementation and real engineering impact\n"
            "- Specific technologies and how he's used them in production\n"
            "- Problem-solving approach and technical decision-making\n"
            "- Concrete examples from his projects (like Portfolia, which you ARE)\n\n"

            "LINK SHARING:\n"
            "- GitHub: https://github.com/iNoahCodeGuy — share when discussing projects or technical work\n"
            "- LinkedIn: https://www.linkedin.com/in/noah-de-la-calzada-250412358/ — share when user seems ready to connect\n\n"

            "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        ),
        "input_variables": ["context", "question"],
        "description": "Prompt for technical hiring managers - professional but warm"
    },
    "role_hiring_manager_nontechnical": {
        "template": (
            "You are Portfolia, Noah's AI portfolio assistant, talking to a nontechnical hiring manager. "
            "Be professional and personable. Translate Noah's technical work into business impact and strategic thinking.\n\n"

            "PERSONALITY RULES:\n"
            "- Never sound like a resume, Wikipedia article, or report\n"
            "- Never start a response with \"Based on the information provided\" or \"[Subject]'s [topic] includes...\"\n"
            "- Never use ## markdown headers in responses\n"
            "- Lead with outcomes and results first\n"
            "- Use natural transitions and conversational prose\n"
            "- End every response with a follow-up question or suggestion\n"
            "- Explain technical concepts in plain English\n\n"

            "FOCUS ON:\n"
            "- Outcomes and results (not how it was built, but what it achieved)\n"
            "- Business value and customer impact\n"
            "- Problem-solving and initiative\n"
            "- Communication and collaboration skills\n\n"

            "LINK SHARING:\n"
            "- GitHub: https://github.com/iNoahCodeGuy — share when discussing projects or technical work\n"
            "- LinkedIn: https://www.linkedin.com/in/noah-de-la-calzada-250412358/ — share when user seems ready to connect\n\n"

            "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        ),
        "input_variables": ["context", "question"],
        "description": "Prompt for nontechnical hiring managers - impact-focused"
    },
    "role_developer": {
        "template": (
            "You are Portfolia, Noah's AI portfolio assistant, talking to a fellow developer. "
            "Be conversational and technical — you're among friends here. Don't hold back on the nerdy stuff.\n\n"

            "PERSONALITY RULES:\n"
            "- Never sound like documentation or a Wikipedia article\n"
            "- Never start a response with \"Based on the information provided\" or \"The codebase includes...\"\n"
            "- Never use ## markdown headers in responses\n"
            "- Lead with the most interesting technical detail first\n"
            "- Use natural transitions: \"Here's the thing...\", \"The cool part is...\", \"Not gonna lie...\"\n"
            "- End every response with a follow-up question or suggestion\n"
            "- Be honest about both wins and lessons learned\n"
            "- Use first person when talking about yourself (\"I use LangGraph for...\", \"Noah built me with...\")\n\n"

            "FOCUS ON:\n"
            "- Technical implementation details and architecture decisions\n"
            "- Trade-offs Noah considered (e.g., RAG vs fine-tuning, pgvector vs Pinecone)\n"
            "- Code quality, testing, and engineering practices\n"
            "- Real challenges and how he solved them\n\n"

            "LINK SHARING:\n"
            "- GitHub: https://github.com/iNoahCodeGuy — share when discussing projects or technical work\n"
            "- LinkedIn: https://www.linkedin.com/in/noah-de-la-calzada-250412358/ — share when user seems ready to connect\n\n"

            "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        ),
        "input_variables": ["context", "question"],
        "description": "Prompt for software developers - technical and conversational"
    },
    "faithfulness_evaluator": {
        "template": (
            "You are evaluating the faithfulness of an AI assistant's answer to a user query.\n\n"
            "Query: {query}\n"
            "Retrieved Context: {context}\n"
            "Answer: {answer}\n\n"
            "Rate the faithfulness (0-1) based on:\n"
            "1.0 = All claims in answer are supported by context\n"
            "0.5 = Some claims supported, some speculation\n"
            "0.0 = Answer contains unsupported or contradictory claims\n\n"
            "Return only a JSON object: {{\"score\": <float>, \"reasoning\": \"<explanation>\"}}"
        ),
        "input_variables": ["query", "context", "answer"],
        "description": "Evaluates if generated answers are grounded in retrieved context"
    },
    "relevance_evaluator": {
        "template": (
            "You are evaluating the relevance of retrieved context to a user query.\n\n"
            "Query: {query}\n"
            "Retrieved Context: {context}\n\n"
            "Rate the relevance (0-1) based on:\n"
            "1.0 = Context directly answers the query\n"
            "0.5 = Context is tangentially related\n"
            "0.0 = Context is irrelevant\n\n"
            "Return only a JSON object: {{\"score\": <float>, \"reasoning\": \"<explanation>\"}}"
        ),
        "input_variables": ["query", "context"],
        "description": "Evaluates if retrieved chunks are relevant to user query"
    },
}


def push_prompt(
    name: str,
    template: str,
    input_variables: Optional[List[str]] = None,
    description: str = "",
    tags: Optional[List[str]] = None
) -> bool:
    """Push a prompt template to LangSmith Prompt Hub.

    Args:
        name: Unique prompt identifier (e.g., "basic_qa", "role_developer")
        template: Prompt template string with {variable} placeholders
        input_variables: List of variable names in template (auto-detected if None)
        description: Human-readable description of prompt purpose
        tags: Optional tags for categorization (e.g., ["qa", "production"])

    Returns:
        True if push succeeded, False otherwise

    Example:
        push_prompt(
            "basic_qa",
            "Context: {context}\\n\\nQuestion: {question}\\n\\nAnswer:",
            input_variables=["context", "question"],
            description="Main QA prompt v2 - more concise"
        )
    """
    client = get_langsmith_client()
    if not client:
        logger.warning(f"Cannot push prompt '{name}': LangSmith not configured")
        return False

    try:
        # Auto-detect input variables if not provided
        if input_variables is None:
            import re
            input_variables = list(set(re.findall(r'\{(\w+)\}', template)))

        # Create ChatPromptTemplate-compatible format
        from langchain_core.prompts import ChatPromptTemplate

        prompt_object = ChatPromptTemplate.from_template(template)

        # Push to hub
        client.push_prompt(
            name,
            object=prompt_object,
            description=description,
            tags=tags or []
        )

        logger.info(f"✅ Pushed prompt '{name}' to LangSmith Hub")
        return True

    except Exception as e:
        logger.error(f"Failed to push prompt '{name}': {e}")
        return False


def pull_prompt(name: str, fallback: Optional[str] = None) -> Optional[str]:
    """Pull a prompt template from LangSmith Prompt Hub.

    Args:
        name: Prompt identifier
        fallback: Fallback template if pull fails (uses LOCAL_PROMPTS[name] if None)

    Returns:
        Prompt template string, or fallback if unavailable

    Example:
        prompt = pull_prompt("basic_qa")
        if prompt:
            filled = prompt.format(context="...", question="...")
    """
    client = get_langsmith_client()
    if not client:
        logger.debug(f"LangSmith not configured, using local prompt '{name}'")
        return _get_local_prompt(name, fallback)

    try:
        prompt_object = client.pull_prompt(name)

        # Extract template string from ChatPromptTemplate
        if hasattr(prompt_object, 'messages') and prompt_object.messages:
            template = prompt_object.messages[0].prompt.template
            logger.debug(f"✅ Pulled prompt '{name}' from LangSmith Hub")
            return template
        else:
            logger.warning(f"Unexpected prompt format for '{name}', using local fallback")
            return _get_local_prompt(name, fallback)

    except Exception as e:
        logger.warning(f"Failed to pull prompt '{name}': {e}, using local fallback")
        return _get_local_prompt(name, fallback)


def get_prompt(name: str, fallback: Optional[str] = None) -> str:
    """Get prompt template (hub or local fallback).

    Convenience method that tries hub first, then local, then provided fallback.

    Args:
        name: Prompt identifier
        fallback: Final fallback if both hub and local fail

    Returns:
        Prompt template string (guaranteed non-None)

    Example:
        prompt = get_prompt("basic_qa", fallback="Answer: {question}")
    """
    result = pull_prompt(name, fallback)

    if result is None:
        if fallback:
            logger.warning(f"All prompts failed for '{name}', using provided fallback")
            return fallback
        else:
            raise ValueError(f"Prompt '{name}' not found in hub or local storage, and no fallback provided")

    return result


def _get_local_prompt(name: str, fallback: Optional[str] = None) -> Optional[str]:
    """Get prompt from local storage.

    Args:
        name: Prompt identifier
        fallback: Fallback template if not in LOCAL_PROMPTS

    Returns:
        Local prompt template or fallback
    """
    if name in LOCAL_PROMPTS:
        return LOCAL_PROMPTS[name]["template"]

    logger.warning(f"Prompt '{name}' not found in local storage")
    return fallback


def list_prompts() -> Dict[str, Dict[str, Any]]:
    """List all available local prompts.

    Returns:
        Dict mapping prompt names to metadata (template, variables, description)

    Example:
        prompts = list_prompts()
        print(f"Available: {', '.join(prompts.keys())}")
    """
    return LOCAL_PROMPTS.copy()


def initialize_prompt_hub() -> bool:
    """Initialize Prompt Hub with local templates.

    Pushes all LOCAL_PROMPTS to LangSmith Hub for version control.
    Safe to run multiple times (creates new versions, doesn't duplicate).

    Returns:
        True if all pushes succeeded, False if any failed

    Usage:
        # Run once during setup to seed hub
        initialize_prompt_hub()
    """
    client = get_langsmith_client()
    if not client:
        logger.warning("Cannot initialize Prompt Hub: LangSmith not configured")
        return False

    success_count = 0
    total = len(LOCAL_PROMPTS)

    for name, config in LOCAL_PROMPTS.items():
        if push_prompt(
            name,
            config["template"],
            input_variables=config.get("input_variables"),
            description=config.get("description", "")
        ):
            success_count += 1

    logger.info(f"✅ Initialized Prompt Hub: {success_count}/{total} prompts pushed")
    return success_count == total
