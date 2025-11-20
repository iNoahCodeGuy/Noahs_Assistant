"""Query preprocessing for typo correction and normalization.

This module provides query preprocessing capabilities including:
- Query normalization (whitespace, encoding fixes)
- Typo correction using spell checking
- Domain-specific dictionary support

All features are optional and gracefully degrade if dependencies are unavailable.
"""

import re
import os
import logging
from typing import Optional

from assistant.state.conversation_state import ConversationState
from assistant.observability.langsmith_tracer import create_custom_span

logger = logging.getLogger(__name__)

# Try to import spell checker, but don't fail if unavailable
try:
    from spellchecker import SpellChecker
    SPELLCHECK_AVAILABLE = True
except ImportError:
    SPELLCHECK_AVAILABLE = False
    logger.warning("pyspellchecker not installed - typo correction will be disabled")

# Domain-specific dictionary (add common terms from your KB)
DOMAIN_TERMS = {
    # Technical terms
    'rag', 'pgvector', 'langchain', 'langgraph', 'supabase', 'vercel',
    'openai', 'embeddings', 'vector', 'retrieval', 'generation',
    'python', 'typescript', 'react', 'nextjs', 'streamlit',
    'genai', 'llm', 'api', 'rpc', 'mmr', 'ivfflat', 'hnsw',
    # Company/product names
    'tesla', 'mma',
    # Common abbreviations
    'ai', 'ml', 'sql', 'csv', 'json', 'http', 'https', 'url',
    # Framework/library names
    'pandas', 'numpy', 'pydantic', 'httpx'
}


def _is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled via environment variable.

    Args:
        feature_name: Name of the feature flag (e.g., 'ENABLE_TYPO_CORRECTION')

    Returns:
        True if feature is enabled, False otherwise (default)
    """
    value = os.getenv(feature_name, "false").lower()
    return value in ("true", "1", "yes", "on")


def normalize_query(query: str) -> str:
    """Normalize query for better retrieval.

    Performs basic normalization:
    - Removes extra whitespace
    - Fixes common character encoding issues
    - Normalizes punctuation

    Args:
        query: Original user query

    Returns:
        Normalized query string
    """
    if not query:
        return query

    # Remove extra whitespace
    query = ' '.join(query.split())

    # Fix common encoding issues
    query = query.replace('â€™', "'").replace('â€œ', '"').replace('â€', '"')
    query = query.replace('â€"', '—').replace('â€"', '–')

    return query.strip()


def correct_typos(query: str, max_corrections: int = 3) -> tuple[str, dict]:
    """Correct typos in user query using spell checker.

    Args:
        query: Original user query
        max_corrections: Maximum number of words to correct (avoid over-correction)

    Returns:
        Tuple of (corrected_query, metadata_dict) where metadata contains:
        - corrections_made: Number of corrections
        - corrections: List of (original, corrected) tuples
    """
    metadata = {
        "corrections_made": 0,
        "corrections": []
    }

    if not SPELLCHECK_AVAILABLE:
        logger.debug("Spell checker unavailable, skipping typo correction")
        return query, metadata

    if not query or not query.strip():
        return query, metadata

    try:
        spell = SpellChecker()
        # Add domain terms to dictionary
        spell.word_frequency.load_words(DOMAIN_TERMS)

        words = query.split()
        corrected_words = []
        corrections_made = 0

        for word in words:
            # Remove punctuation for checking
            clean_word = re.sub(r'[^\w]', '', word.lower())

            # Skip if it's a domain term, very short, or already correct
            if clean_word in DOMAIN_TERMS or len(clean_word) <= 2:
                corrected_words.append(word)
                continue

            # Check if word is misspelled
            if clean_word in spell.unknown([clean_word]):
                # Get correction
                correction = spell.correction(clean_word)
                if correction and correction != clean_word and corrections_made < max_corrections:
                    # Preserve original capitalization/punctuation
                    if word[0].isupper():
                        correction = correction.capitalize()
                    # Preserve trailing punctuation
                    trailing_punct = re.search(r'[^\w]+$', word)
                    if trailing_punct:
                        correction = correction + trailing_punct.group()

                    corrected_words.append(correction)
                    metadata["corrections"].append((word, correction))
                    corrections_made += 1
                    logger.info(f"Typo corrected: '{word}' → '{correction}'")
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        metadata["corrections_made"] = corrections_made
        corrected_query = ' '.join(corrected_words)

        if corrected_query != query:
            logger.info(f"Query corrected: '{query}' → '{corrected_query}'")

        return corrected_query, metadata

    except Exception as e:
        logger.warning(f"Spell checking failed: {e}")
        return query, metadata


def preprocess_query(state: ConversationState) -> ConversationState:
    """Preprocess query with normalization and typo correction.

    This is the main preprocessing node that should be called in the conversation flow.
    It checks feature flags and applies preprocessing only if enabled.

    State Modifications:
        - original_query: Preserved for display (if query was changed)
        - query: Corrected/normalized version (if changed)
        - typo_corrected: Boolean flag indicating if typos were corrected
        - query_preprocessing_metadata: Dictionary with correction details

    Args:
        state: ConversationState with query field

    Returns:
        Updated state with preprocessing applied (if enabled)
    """
    # Check if feature is enabled
    if not _is_feature_enabled("ENABLE_TYPO_CORRECTION"):
        return state

    query = state.get("query", "")
    if not query:
        return state

    with create_custom_span("preprocess_query", {
        "query": query[:120],
        "spellcheck_available": SPELLCHECK_AVAILABLE
    }):
        try:
            # Step 1: Normalize query (always safe to do)
            normalized_query = normalize_query(query)

            # Step 2: Correct typos (if spell checker available)
            if SPELLCHECK_AVAILABLE:
                corrected_query, correction_metadata = correct_typos(normalized_query)
            else:
                corrected_query = normalized_query
                correction_metadata = {"corrections_made": 0, "corrections": []}

            # Step 3: Update state if query was changed
            if corrected_query != query:
                state["original_query"] = query
                state["query"] = corrected_query
                state["typo_corrected"] = True
                state.setdefault("query_preprocessing_metadata", {}).update({
                    "normalized": True,
                    "corrections_made": correction_metadata["corrections_made"],
                    "corrections": correction_metadata["corrections"]
                })
                logger.info(
                    f"Query preprocessed: {correction_metadata['corrections_made']} corrections made"
                )
            else:
                # Still mark as normalized even if no corrections
                state.setdefault("query_preprocessing_metadata", {}).update({
                    "normalized": True,
                    "corrections_made": 0
                })

            # Update analytics metadata
            metadata = state.setdefault("analytics_metadata", {})
            if correction_metadata["corrections_made"] > 0:
                metadata["typo_corrections_applied"] = correction_metadata["corrections_made"]

        except Exception as e:
            logger.error(f"Query preprocessing failed: {e}", exc_info=True)
            # Graceful degradation: continue with original query
            # Don't modify state, just log the error

        return state
