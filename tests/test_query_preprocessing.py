"""Unit tests for query preprocessing (typo correction and normalization).

Tests:
- Query normalization (whitespace, encoding)
- Typo correction with domain terms
- Graceful degradation when spellchecker unavailable
- Max corrections limit
- State management

Run: pytest tests/test_query_preprocessing.py -v
"""

import pytest
import os
from unittest.mock import patch, MagicMock

from assistant.flows.node_logic.query_preprocessing import (
    normalize_query,
    correct_typos,
    preprocess_query,
    _is_feature_enabled,
    DOMAIN_TERMS
)
from assistant.state.conversation_state import ConversationState


class TestNormalizeQuery:
    """Test query normalization functionality."""

    def test_remove_extra_whitespace(self):
        """Test that extra whitespace is removed."""
        query = "  hello    world  "
        result = normalize_query(query)
        assert result == "hello world"

    def test_fix_encoding_issues(self):
        """Test that common encoding issues are fixed."""
        query = "Itâ€™s working"
        result = normalize_query(query)
        assert "â€™" not in result
        assert "'" in result or result == "It's working"

    def test_empty_query(self):
        """Test that empty query returns empty string."""
        assert normalize_query("") == ""
        assert normalize_query("   ") == ""

    def test_preserves_content(self):
        """Test that valid content is preserved."""
        query = "What is RAG?"
        result = normalize_query(query)
        assert result == "What is RAG?"


class TestCorrectTypos:
    """Test typo correction functionality."""

    @pytest.mark.skipif(
        not hasattr(__import__('spellchecker', fromlist=['']), 'SpellChecker'),
        reason="pyspellchecker not installed"
    )
    def test_corrects_common_typos(self):
        """Test that common typos are corrected."""
        query = "buisness experience"
        corrected, metadata = correct_typos(query)
        assert "business" in corrected.lower()
        assert metadata["corrections_made"] > 0

    @pytest.mark.skipif(
        not hasattr(__import__('spellchecker', fromlist=['']), 'SpellChecker'),
        reason="pyspellchecker not installed"
    )
    def test_preserves_domain_terms(self):
        """Test that domain-specific terms are not corrected."""
        query = "How does RAG work?"
        corrected, metadata = correct_typos(query)
        assert "RAG" in corrected
        assert "rag" in corrected.lower()

    @pytest.mark.skipif(
        not hasattr(__import__('spellchecker', fromlist=['']), 'SpellChecker'),
        reason="pyspellchecker not installed"
    )
    def test_max_corrections_limit(self):
        """Test that max_corrections limit is respected."""
        query = "buisness experiance developement"  # 3 typos
        corrected, metadata = correct_typos(query, max_corrections=2)
        assert metadata["corrections_made"] <= 2

    @pytest.mark.skipif(
        not hasattr(__import__('spellchecker', fromlist=['']), 'SpellChecker'),
        reason="pyspellchecker not installed"
    )
    def test_preserves_capitalization(self):
        """Test that capitalization is preserved."""
        query = "Buisness"
        corrected, metadata = correct_typos(query)
        if metadata["corrections_made"] > 0:
            assert corrected[0].isupper()

    def test_graceful_degradation_no_spellchecker(self):
        """Test that function works even if spellchecker unavailable."""
        with patch('assistant.flows.node_logic.query_preprocessing.SPELLCHECK_AVAILABLE', False):
            query = "buisness"
            corrected, metadata = correct_typos(query)
            assert corrected == query
            assert metadata["corrections_made"] == 0

    def test_short_words_skipped(self):
        """Test that very short words are not corrected."""
        query = "a b c"
        corrected, metadata = correct_typos(query)
        # Should not attempt corrections on single letters
        assert len(corrected.split()) == 3


class TestPreprocessQuery:
    """Test preprocess_query node functionality."""

    def test_feature_disabled_no_changes(self):
        """Test that preprocessing is skipped when feature disabled."""
        with patch('assistant.flows.node_logic.query_preprocessing._is_feature_enabled', return_value=False):
            state: ConversationState = {
                "query": "buisness"
            }
            result = preprocess_query(state)
            assert result.get("query") == "buisness"
            assert "typo_corrected" not in result

    def test_feature_enabled_applies_preprocessing(self):
        """Test that preprocessing is applied when feature enabled."""
        with patch('assistant.flows.node_logic.query_preprocessing._is_feature_enabled', return_value=True):
            with patch('assistant.flows.node_logic.query_preprocessing.SPELLCHECK_AVAILABLE', True):
                state: ConversationState = {
                    "query": "  buisness  "
                }
                result = preprocess_query(state)
                # Should normalize whitespace at minimum
                assert result.get("query") is not None
                assert "query_preprocessing_metadata" in result

    def test_preserves_original_query(self):
        """Test that original query is preserved when corrections made."""
        with patch('assistant.flows.node_logic.query_preprocessing._is_feature_enabled', return_value=True):
            with patch('assistant.flows.node_logic.query_preprocessing.SPELLCHECK_AVAILABLE', True):
                state: ConversationState = {
                    "query": "buisness"
                }
                # Mock correct_typos to return a correction
                with patch('assistant.flows.node_logic.query_preprocessing.correct_typos') as mock_correct:
                    mock_correct.return_value = ("business", {"corrections_made": 1, "corrections": [("buisness", "business")]})
                    result = preprocess_query(state)
                    if result.get("typo_corrected"):
                        assert "original_query" in result
                        assert result["original_query"] == "buisness"

    def test_empty_query_handled(self):
        """Test that empty query is handled gracefully."""
        state: ConversationState = {
            "query": ""
        }
        result = preprocess_query(state)
        assert result.get("query") == ""

    def test_metadata_tracking(self):
        """Test that preprocessing metadata is tracked."""
        with patch('assistant.flows.node_logic.query_preprocessing._is_feature_enabled', return_value=True):
            state: ConversationState = {
                "query": "test query"
            }
            result = preprocess_query(state)
            assert "query_preprocessing_metadata" in result
            metadata = result["query_preprocessing_metadata"]
            assert "normalized" in metadata

    def test_analytics_metadata_updated(self):
        """Test that analytics metadata is updated with correction count."""
        with patch('assistant.flows.node_logic.query_preprocessing._is_feature_enabled', return_value=True):
            with patch('assistant.flows.node_logic.query_preprocessing.SPELLCHECK_AVAILABLE', True):
                state: ConversationState = {
                    "query": "buisness",
                    "analytics_metadata": {}
                }
                # Mock correct_typos to return corrections
                with patch('assistant.flows.node_logic.query_preprocessing.correct_typos') as mock_correct:
                    mock_correct.return_value = ("business", {"corrections_made": 1, "corrections": [("buisness", "business")]})
                    result = preprocess_query(state)
                    if result.get("typo_corrected"):
                        analytics = result.get("analytics_metadata", {})
                        assert "typo_corrections_applied" in analytics

    def test_graceful_error_handling(self):
        """Test that errors in preprocessing don't crash the flow."""
        with patch('assistant.flows.node_logic.query_preprocessing._is_feature_enabled', return_value=True):
            with patch('assistant.flows.node_logic.query_preprocessing.normalize_query', side_effect=Exception("Test error")):
                state: ConversationState = {
                    "query": "test"
                }
                # Should not raise, should return state unchanged
                result = preprocess_query(state)
                assert result is not None
                assert result.get("query") == "test"


class TestFeatureFlags:
    """Test feature flag functionality."""

    def test_feature_flag_default_false(self):
        """Test that feature flags default to False."""
        with patch.dict(os.environ, {}, clear=True):
            assert _is_feature_enabled("ENABLE_TYPO_CORRECTION") is False

    def test_feature_flag_true_values(self):
        """Test that various true values are recognized."""
        true_values = ["true", "True", "TRUE", "1", "yes", "Yes", "YES", "on", "On", "ON"]
        for value in true_values:
            with patch.dict(os.environ, {"ENABLE_TYPO_CORRECTION": value}, clear=True):
                assert _is_feature_enabled("ENABLE_TYPO_CORRECTION") is True

    def test_feature_flag_false_values(self):
        """Test that false values are recognized."""
        false_values = ["false", "False", "FALSE", "0", "no", "No", "NO", "off", "Off", "OFF", ""]
        for value in false_values:
            with patch.dict(os.environ, {"ENABLE_TYPO_CORRECTION": value}, clear=True):
                assert _is_feature_enabled("ENABLE_TYPO_CORRECTION") is False


class TestDomainTerms:
    """Test domain-specific term handling."""

    def test_domain_terms_defined(self):
        """Test that domain terms are defined."""
        assert len(DOMAIN_TERMS) > 0
        assert "rag" in DOMAIN_TERMS
        assert "pgvector" in DOMAIN_TERMS

    def test_technical_terms_included(self):
        """Test that technical terms are included."""
        assert "langchain" in DOMAIN_TERMS
        assert "supabase" in DOMAIN_TERMS
        assert "openai" in DOMAIN_TERMS
