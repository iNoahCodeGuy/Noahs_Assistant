"""Tests for realistic usage scenarios and comprehensive integration.

This module tests real-world scenarios, edge cases, and end-to-end
integration of the common questions system.
"""

import pytest
from unittest.mock import Mock

# Import shared fixtures
try:
    from .common_questions_fixtures import (
        realistic_analytics, assert_question_structure
    )
except ImportError:
    from common_questions_fixtures import (
        realistic_analytics, assert_question_structure
    )


class TestRealWorldScenarios:
    """Test realistic usage scenarios and question patterns."""
    
    def test_realistic_question_ranking(self, realistic_analytics):
        """Test that question ranking works with realistic data."""
        questions = realistic_analytics.get_most_common_questions(days=30, limit=5)
        
        # Should be ordered by frequency
        assert questions[0]['question'] == "What's Noah's background?"
        assert questions[0]['frequency'] == 15
        
        assert questions[1]['question'] == "Show me Noah's code"
        assert questions[1]['frequency'] == 12
        
        assert questions[2]['question'] == "How does the RAG work?"
        assert questions[2]['frequency'] == 10
        
        # Validate structure of all questions
        for question in questions:
            assert_question_structure(question)
    
    def test_role_specific_patterns(self, realistic_analytics):
        """Test role-specific question patterns."""
        # Technical hiring managers should see mix of career and technical
        hm_tech_questions = realistic_analytics.get_most_common_questions(
            role="Hiring Manager (technical)", days=30, limit=5
        )
        
        assert len(hm_tech_questions) >= 2
        questions_text = [q['question'] for q in hm_tech_questions]
        assert "Show me Noah's code" in questions_text
        assert "What's his Python experience?" in questions_text
        
        # Developers should see technical questions
        dev_questions = realistic_analytics.get_most_common_questions(
            role="Software Developer", days=30, limit=5
        )
        
        dev_question_types = [q['query_type'] for q in dev_questions]
        assert all(qt == "technical" for qt in dev_question_types)
    
    def test_suggestion_quality(self, realistic_analytics):
        """Test that suggestions are high quality (successful queries)."""
        suggestions = realistic_analytics.get_suggested_questions_for_role(
            "Hiring Manager (technical)", limit=3
        )
        
        assert len(suggestions) > 0
        
        # All suggestions should be from successful interactions
        for suggestion in suggestions:
            # Verify this was a successful query
            cursor = realistic_analytics.connection.cursor()
            cursor.execute('''
                SELECT AVG(success) FROM user_interactions 
                WHERE query = ? AND user_role = ?
            ''', (suggestion, "Hiring Manager (technical)"))
            
            avg_success = cursor.fetchone()[0]
            assert avg_success > 0.8  # Should be high success rate
    
    def test_cross_role_question_analysis(self, realistic_analytics):
        """Test analysis of questions across different roles."""
        role_questions = realistic_analytics.get_common_questions_by_role(days=30, limit_per_role=3)
        
        # Verify we have data for all expected roles
        expected_roles = [
            'Software Developer',
            'Hiring Manager (technical)',
            'Hiring Manager (nontechnical)',
            'Just looking around'
        ]
        
        for role in expected_roles:
            assert role in role_questions
            if role in ['Software Developer', 'Hiring Manager (technical)']:
                # These roles should have questions
                assert len(role_questions[role]) > 0
    
    def test_temporal_patterns(self, realistic_analytics):
        """Test that temporal filtering works correctly with realistic data."""
        # Recent questions (7 days)
        recent = realistic_analytics.get_most_common_questions(days=7, limit=10)
        
        # Longer period (30 days)
        extended = realistic_analytics.get_most_common_questions(days=30, limit=10)
        
        # Extended period should have same or more questions
        assert len(extended) >= len(recent)
        
        # Top questions should be similar (but frequencies might differ)
        if recent and extended:
            recent_top = recent[0]['question']
            extended_questions = [q['question'] for q in extended]
            assert recent_top in extended_questions


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_no_questions_for_role(self, realistic_analytics):
        """Test behavior when no questions exist for a specific role."""
        # Create a role that doesn't exist in our realistic data
        questions = realistic_analytics.get_most_common_questions(
            role="Non-existent Role", days=30, limit=5
        )
        
        assert questions == []
    
    def test_limit_edge_cases(self, realistic_analytics):
        """Test various limit values."""
        # Test limit = 0
        questions_zero = realistic_analytics.get_most_common_questions(days=30, limit=0)
        assert len(questions_zero) == 0
        
        # Test limit = 1
        questions_one = realistic_analytics.get_most_common_questions(days=30, limit=1)
        assert len(questions_one) == 1
        
        # Test very large limit
        questions_large = realistic_analytics.get_most_common_questions(days=30, limit=1000)
        # Should return all available questions (not crash)
        assert isinstance(questions_large, list)
    
    def test_extreme_time_ranges(self, realistic_analytics):
        """Test extreme time range values."""
        # Very short time range
        questions_short = realistic_analytics.get_most_common_questions(days=1, limit=5)
        assert isinstance(questions_short, list)
        
        # Very long time range
        questions_long = realistic_analytics.get_most_common_questions(days=365, limit=5)
        assert isinstance(questions_long, list)
        
        # Zero days (should return empty)
        questions_zero_days = realistic_analytics.get_most_common_questions(days=0, limit=5)
        assert questions_zero_days == []


class TestPerformanceIntegration:
    """Test performance characteristics in realistic scenarios."""
    
    def test_concurrent_access_simulation(self, realistic_analytics):
        """Test simulated concurrent access patterns."""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker():
            try:
                start = time.time()
                questions = realistic_analytics.get_most_common_questions(days=30, limit=5)
                duration = time.time() - start
                results.append((len(questions), duration))
            except Exception as e:
                errors.append(str(e))
        
        # Simulate 5 concurrent requests
        threads = []
        for _ in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify no errors and reasonable performance
        assert len(errors) == 0, f"Errors during concurrent access: {errors}"
        assert len(results) == 5
        
        # All requests should complete in reasonable time
        for count, duration in results:
            assert count >= 0  # Should return some results
            assert duration < 2.0  # Should be fast
    
    def test_memory_usage_patterns(self, realistic_analytics):
        """Test memory usage doesn't grow excessively."""
        import gc
        import sys
        
        # Get baseline memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Perform multiple queries
        for _ in range(10):
            questions = realistic_analytics.get_most_common_questions(days=30, limit=10)
            role_questions = realistic_analytics.get_common_questions_by_role(days=30)
            suggestions = realistic_analytics.get_suggested_questions_for_role("Software Developer")
        
        # Check memory usage hasn't grown excessively
        gc.collect()
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects
        
        # Should not create excessive objects (allow some growth for test artifacts)
        assert object_growth < 1000, f"Excessive object growth: {object_growth}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
