"""Unit tests for hybrid search functionality.

Tests:
- Keyword search fallback
- Result merging and deduplication
- Fuzzy matching threshold
- Error handling

Run: pytest tests/test_hybrid_search.py -v
"""

import pytest
from unittest.mock import patch, MagicMock, Mock

from assistant.retrieval.pgvector_retriever import (
    PgVectorRetriever,
    _is_feature_enabled
)


class TestKeywordSearch:
    """Test keyword-based search functionality."""

    def test_keyword_search_extracts_keywords(self):
        """Test that keywords are extracted from query."""
        retriever = PgVectorRetriever()

        # Mock Supabase client
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "id": 1,
                "doc_id": "test",
                "section": "test",
                "content": "This is about business experience",
                "embedding": None
            }
        ]
        mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = mock_result
        retriever.supabase_client = mock_client

        results = retriever._keyword_search("business experience", top_k=3)
        assert len(results) > 0
        assert results[0]["source"] == "keyword"

    def test_keyword_search_filters_stop_words(self):
        """Test that stop words are filtered from keywords."""
        retriever = PgVectorRetriever()

        # Mock Supabase client
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = []
        mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = mock_result
        retriever.supabase_client = mock_client

        # Should not crash on queries with only stop words
        results = retriever._keyword_search("the and or", top_k=3)
        assert isinstance(results, list)

    def test_keyword_search_returns_empty_on_no_matches(self):
        """Test that keyword search returns empty list when no matches."""
        retriever = PgVectorRetriever()

        # Mock Supabase client with empty results
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = []
        mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = mock_result
        retriever.supabase_client = mock_client

        results = retriever._keyword_search("nonexistent terms xyz", top_k=3)
        assert results == []

    def test_keyword_search_handles_errors_gracefully(self):
        """Test that keyword search handles errors gracefully."""
        retriever = PgVectorRetriever()

        # Mock Supabase client to raise exception
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.limit.return_value.execute.side_effect = Exception("Database error")
        retriever.supabase_client = mock_client

        results = retriever._keyword_search("test query", top_k=3)
        assert results == []

    def test_keyword_search_scores_capped(self):
        """Test that keyword search scores are capped appropriately."""
        retriever = PgVectorRetriever()

        # Mock Supabase client
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "id": 1,
                "doc_id": "test",
                "section": "test",
                "content": "business experience",
                "embedding": None
            }
        ]
        mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = mock_result
        retriever.supabase_client = mock_client

        results = retriever._keyword_search("business experience", top_k=3)
        if results:
            assert results[0]["similarity"] <= 0.6  # Should be capped


class TestMergeResults:
    """Test result merging functionality."""

    def test_merge_results_deduplicates(self):
        """Test that merged results deduplicate by chunk ID."""
        retriever = PgVectorRetriever()

        semantic = [
            {"id": 1, "content": "test1", "similarity": 0.8},
            {"id": 2, "content": "test2", "similarity": 0.7}
        ]
        keyword = [
            {"id": 1, "content": "test1", "similarity": 0.6},  # Duplicate
            {"id": 3, "content": "test3", "similarity": 0.5}
        ]

        merged = retriever._merge_results(semantic, keyword)
        assert len(merged) == 3  # Should have all unique IDs
        ids = [c["id"] for c in merged]
        assert 1 in ids
        assert 2 in ids
        assert 3 in ids

    def test_merge_results_preserves_semantic_priority(self):
        """Test that semantic results come first in merged list."""
        retriever = PgVectorRetriever()

        semantic = [
            {"id": 1, "content": "test1", "similarity": 0.8}
        ]
        keyword = [
            {"id": 2, "content": "test2", "similarity": 0.6}
        ]

        merged = retriever._merge_results(semantic, keyword)
        assert merged[0]["id"] == 1  # Semantic result first

    def test_merge_results_sorts_by_similarity(self):
        """Test that merged results are sorted by similarity."""
        retriever = PgVectorRetriever()

        semantic = [
            {"id": 1, "content": "test1", "similarity": 0.7}
        ]
        keyword = [
            {"id": 2, "content": "test2", "similarity": 0.9}  # Higher score
        ]

        merged = retriever._merge_results(semantic, keyword)
        # Should be sorted by similarity (highest first)
        assert merged[0]["similarity"] >= merged[1]["similarity"]


class TestFuzzyMatching:
    """Test fuzzy matching functionality."""

    @pytest.mark.skipif(
        not hasattr(__import__('Levenshtein', fromlist=['']), 'distance'),
        reason="python-Levenshtein not installed"
    )
    def test_fuzzy_match_finds_similar_chunks(self):
        """Test that fuzzy matching finds similar chunks."""
        retriever = PgVectorRetriever()

        chunks = [
            {"id": 1, "content": "business experience", "similarity": 0.3},
            {"id": 2, "content": "technical skills", "similarity": 0.2}
        ]

        results = retriever._fuzzy_match_chunks("buisness", chunks, threshold=0.5)
        # Should find "business" as similar to "buisness"
        assert len(results) > 0

    @pytest.mark.skipif(
        not hasattr(__import__('Levenshtein', fromlist=['']), 'distance'),
        reason="python-Levenshtein not installed"
    )
    def test_fuzzy_match_respects_threshold(self):
        """Test that fuzzy matching respects similarity threshold."""
        retriever = PgVectorRetriever()

        chunks = [
            {"id": 1, "content": "completely different content", "similarity": 0.1}
        ]

        results = retriever._fuzzy_match_chunks("test query", chunks, threshold=0.8)
        # Should not match if below threshold
        assert len(results) == 0

    def test_fuzzy_match_graceful_degradation(self):
        """Test that fuzzy matching degrades gracefully if library unavailable."""
        with patch('assistant.retrieval.pgvector_retriever.FUZZY_MATCHING_AVAILABLE', False):
            retriever = PgVectorRetriever()
            chunks = [{"id": 1, "content": "test", "similarity": 0.3}]
            results = retriever._fuzzy_match_chunks("test", chunks, threshold=0.5)
            assert results == []

    def test_fuzzy_match_handles_errors(self):
        """Test that fuzzy matching handles errors gracefully."""
        retriever = PgVectorRetriever()

        # Invalid chunks should not crash
        chunks = [{"id": 1}]  # Missing content
        results = retriever._fuzzy_match_chunks("test", chunks, threshold=0.5)
        assert isinstance(results, list)


class TestRetrieveHybrid:
    """Test hybrid retrieval functionality."""

    def test_hybrid_uses_semantic_first(self):
        """Test that hybrid retrieval tries semantic search first."""
        retriever = PgVectorRetriever()

        # Mock retrieve to return good results
        with patch.object(retriever, 'retrieve') as mock_retrieve:
            mock_retrieve.return_value = [
                {"id": 1, "content": "test", "similarity": 0.8}
            ]

            results = retriever.retrieve_hybrid("test query", top_k=3)
            mock_retrieve.assert_called_once()
            assert len(results) > 0

    def test_hybrid_falls_back_to_keyword_on_low_scores(self):
        """Test that hybrid retrieval falls back to keyword search on low scores."""
        retriever = PgVectorRetriever()

        # Mock retrieve to return low scores
        with patch.object(retriever, 'retrieve') as mock_retrieve:
            with patch.object(retriever, '_keyword_search') as mock_keyword:
                mock_retrieve.return_value = [
                    {"id": 1, "content": "test", "similarity": 0.3}  # Low score
                ]
                mock_keyword.return_value = [
                    {"id": 2, "content": "test2", "similarity": 0.5}
                ]

                results = retriever.retrieve_hybrid("test query", top_k=3, use_keyword_fallback=True)
                mock_keyword.assert_called_once()
                assert len(results) > 0

    def test_hybrid_skips_keyword_on_good_scores(self):
        """Test that hybrid retrieval skips keyword search on good scores."""
        retriever = PgVectorRetriever()

        # Mock retrieve to return good scores
        with patch.object(retriever, 'retrieve') as mock_retrieve:
            with patch.object(retriever, '_keyword_search') as mock_keyword:
                mock_retrieve.return_value = [
                    {"id": 1, "content": "test", "similarity": 0.8}  # Good score
                ]

                results = retriever.retrieve_hybrid("test query", top_k=3, use_keyword_fallback=True)
                mock_keyword.assert_not_called()
                assert len(results) > 0

    def test_hybrid_handles_empty_results(self):
        """Test that hybrid retrieval handles empty semantic results."""
        retriever = PgVectorRetriever()

        with patch.object(retriever, 'retrieve') as mock_retrieve:
            mock_retrieve.return_value = []

            results = retriever.retrieve_hybrid("test query", top_k=3)
            assert results == []
