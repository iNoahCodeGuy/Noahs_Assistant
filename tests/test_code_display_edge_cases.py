"""Edge case tests for code display functionality.

Tests error handling, performance limits, and unusual scenarios
that could occur in production environments.
"""
import pytest
import time
from unittest.mock import patch, MagicMock

from assistant.core.rag_engine import RagEngine
from assistant.config.settings import Settings
from assistant.agents.role_router import RoleRouter


class TestCodeDisplayEdgeCases:
    """Test edge cases and error scenarios for code display."""

    def test_malformed_query_handling(self):
        """Test handling of malformed or unusual queries."""
        settings = Settings()
        engine = RagEngine(settings=settings)

        edge_case_queries = [
            "",  # Empty query
            "   ",  # Whitespace only
            "a" * 1000,  # Very long query
            "SELECT * FROM users;",  # SQL injection attempt
            "<script>alert('xss')</script>",  # XSS attempt
            "../../etc/passwd",  # Path traversal attempt
        ]

        for query in edge_case_queries:
            result = engine.retrieve_with_code(query, role="Software Developer", include_code=True)

            # Should not crash and should return valid structure
            assert isinstance(result, dict)
            assert 'code_snippets' in result
            assert 'has_code' in result
            assert isinstance(result['code_snippets'], list)

    def test_no_code_results_scenario(self):
        """Test behavior when no code matches are found."""
        settings = Settings()
        engine = RagEngine(settings=settings)

        # Query for something unlikely to exist
        result = engine.retrieve_with_code(
            "quantum_flux_capacitor_implementation_xyz123",
            role="Software Developer",
            include_code=True
        )

        assert result['has_code'] is False
        assert len(result['code_snippets']) == 0
        assert 'matches' in result  # Career info should still work

    def test_code_index_corruption_recovery(self):
        """Test graceful handling of corrupted code index."""
        with patch('src.retrieval.code_index.CodeIndex.search_code',
                   side_effect=Exception("Index corrupted")):
            settings = Settings()
            engine = RagEngine(settings=settings)

            result = engine.retrieve_with_code(
                "test query",
                role="Software Developer",
                include_code=True
            )

            # Should fall back gracefully
            assert result['has_code'] is False
            assert len(result['code_snippets']) == 0

    def test_large_code_file_handling(self):
        """Test handling of very large code files."""
        settings = Settings()
        engine = RagEngine(settings=settings)

        # Mock a very large code result
        large_content = "def large_function():\n" + "    pass\n" * 1000

        with patch('src.retrieval.code_index.CodeIndex.search_code') as mock_search:
            mock_search.return_value = [{
                'file': 'large_file.py',
                'citation': 'large_file.py:1-1000',
                'content': large_content,
                'type': 'function',
                'name': 'large_function',
                'github_url': 'https://github.com/test/repo#L1',
                'line_start': 1,
                'line_end': 1000
            }]

            result = engine.retrieve_with_code("large function", role="Software Developer", include_code=True)

            # Should handle large content appropriately
            assert result['has_code'] is True
            assert len(result['code_snippets']) == 1
            # Content should be present but may be truncated for display
            assert 'large_function' in result['code_snippets'][0]['content']

    def test_concurrent_code_index_access(self):
        """Test thread safety of code index operations."""
        import threading
        import concurrent.futures

        settings = Settings()
        engine = RagEngine(settings=settings)

        def query_code():
            return engine.retrieve_with_code("test", role="Software Developer", include_code=True)

        # Run multiple queries concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(query_code) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # All should complete successfully
        assert len(results) == 10
        for result in results:
            assert isinstance(result, dict)
            assert 'code_snippets' in result

    def test_response_time_limits(self):
        """Test that code retrieval completes within reasonable time."""
        settings = Settings()
        engine = RagEngine(settings=settings)

        start_time = time.time()
        result = engine.retrieve_with_code("RagEngine", role="Software Developer", include_code=True)
        end_time = time.time()

        # Should complete within 10 seconds (generous limit for CI)
        assert (end_time - start_time) < 10.0
        assert isinstance(result, dict)

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters in queries."""
        settings = Settings()
        engine = RagEngine(settings=settings)

        unicode_queries = [
            "función_python",  # Spanish
            "クラス定義",  # Japanese
            "функция_code",  # Russian
            "emoji_test_🚀_code",  # Emoji
            "test-with-dashes",
            "test_with_underscores",
            "test.with.dots",
        ]

        for query in unicode_queries:
            result = engine.retrieve_with_code(query, role="Software Developer", include_code=True)

            # Should handle gracefully without crashing
            assert isinstance(result, dict)
            assert 'code_snippets' in result


class TestCodeDisplayPerformance:
    """Performance-focused tests for code display."""

    def test_multiple_role_queries_performance(self):
        """Test performance when switching between different roles."""
        settings = Settings()
        engine = RagEngine(settings=settings)
        router = RoleRouter()

        roles = [
            "Software Developer",
            "Hiring Manager (technical)",
            "Hiring Manager (nontechnical)"
        ]

        start_time = time.time()

        for role in roles * 3:  # Test each role 3 times
            result = engine.retrieve_with_code("RagEngine", role=role, include_code=(role in [
                "Software Developer", "Hiring Manager (technical)"]))
            assert isinstance(result, dict)

        end_time = time.time()

        # Should complete 9 queries within reasonable time
        assert (end_time - start_time) < 30.0

    def test_code_index_version_tracking_performance(self):
        """Test performance of version tracking operations."""
        settings = Settings()
        engine = RagEngine(settings=settings)

        start_time = time.time()

        # Test version checking multiple times
        for _ in range(20):
            version = engine.code_index_version()
            assert isinstance(version, str)

        end_time = time.time()

        # Version checks should be very fast
        assert (end_time - start_time) < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
