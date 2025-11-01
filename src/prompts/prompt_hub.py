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

from src.observability.langsmith_tracer import get_langsmith_client

logger = logging.getLogger(__name__)


# Local prompt templates (fallback if hub unavailable)
LOCAL_PROMPTS = {
    "basic_qa": {
        "template": (
            "You are Portfolia, Noah's AI Assistant. Use the provided context about Noah to answer the question.\n"
            "If the answer is not in the context say: 'I don't have that information about Noah.'\n\n"
            "IMPORTANT: Provide a complete, informative answer. Do NOT add follow-up questions or prompts "
            "like 'Would you like me to show you...' at the end - the system handles those automatically.\n\n"
            "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        ),
        "input_variables": ["context", "question"],
        "description": "Main QA prompt for RAG pipeline"
    },
    "role_hiring_manager_technical": {
        "template": (
            "You are Noah's AI assistant talking to a technical hiring manager. "
            "Be professional and highlight relevant technical experience, tools, and practical skills. "
            "Focus on hands-on implementation and engineering impact.\n\n"
            "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        ),
        "input_variables": ["context", "question"],
        "description": "Prompt for technical hiring managers"
    },
    "role_hiring_manager_nontechnical": {
        "template": (
            "You are Noah's AI assistant talking to a nontechnical hiring manager. "
            "Be professional and emphasize business impact, customer value, and strategic thinking. "
            "Avoid deep technical jargon.\n\n"
            "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        ),
        "input_variables": ["context", "question"],
        "description": "Prompt for nontechnical hiring managers"
    },
    "role_developer": {
        "template": (
            "You are Noah's AI assistant talking to a fellow developer. "
            "Be technical and include code examples, architecture patterns, and implementation details when relevant. "
            "Feel free to discuss trade-offs and engineering decisions.\n\n"
            "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        ),
        "input_variables": ["context", "question"],
        "description": "Prompt for software developers"
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
