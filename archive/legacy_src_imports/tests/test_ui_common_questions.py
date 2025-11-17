"""Tests for CommonQuestionsDisplay UI component.

This module tests the Streamlit UI component for displaying
common questions to users.
"""

import pytest
from unittest.mock import Mock, patch

# Import shared fixtures
try:
    from .common_questions_fixtures import (
        mock_analytics, display_component, assert_question_structure
    )
except ImportError:
    from common_questions_fixtures import (
        mock_analytics, display_component, assert_question_structure
    )

# Import modules under test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.ui.components.common_questions import (
        CommonQuestionsDisplay, display_suggested_questions_sidebar, get_question_suggestions
    )
except ImportError as e:
    # Handle missing dependencies gracefully for testing
    print(f"Warning: Import failed - {e}")
    CommonQuestionsDisplay = Mock
    display_suggested_questions_sidebar = Mock
    get_question_suggestions = Mock


class TestCommonQuestionsDisplayComponent:
    """Test the UI component for displaying common questions."""

    def test_initialization_with_analytics(self, mock_analytics):
        """Test component initialization with analytics system."""
        display = CommonQuestionsDisplay(analytics_system=mock_analytics)
        assert display.analytics == mock_analytics
        assert display.fallback_questions is not None

    def test_initialization_without_analytics(self):
        """Test component initialization without analytics system."""
        display = CommonQuestionsDisplay()
        assert display.analytics is None
        assert display.fallback_questions is not None

    def test_fallback_questions_structure(self, display_component):
        """Test that fallback questions are properly structured."""
        fallback = display_component.fallback_questions

        required_roles = [
            'Hiring Manager (technical)',
            'Hiring Manager (nontechnical)',
            'Software Developer',
            'Just looking around'
        ]

        for role in required_roles:
            assert role in fallback
            assert isinstance(fallback[role], list)
            assert len(fallback[role]) > 0

            # Each question should be a non-empty string
            for question in fallback[role]:
                assert isinstance(question, str)
                assert len(question.strip()) > 0

    @patch('streamlit.subheader')
    @patch('streamlit.caption')
    @patch('streamlit.button')
    @patch('streamlit.columns')
    def test_display_for_role_with_analytics(
        self, mock_columns, mock_button, mock_caption, mock_subheader, display_component
    ):
        """Test displaying questions for a role with analytics data."""
        # Mock Streamlit columns
        mock_col1 = Mock()
        mock_col2 = Mock()
        mock_columns.return_value = [mock_col1, mock_col2]

        # Mock button not clicked
        mock_button.return_value = False

        result = display_component.display_for_role("Software Developer", show_analytics=True)

        # Should call analytics
        display_component.analytics.get_most_common_questions.assert_called_once_with(
            role="Software Developer", days=30, limit=5
        )

        # Should display UI elements
        mock_subheader.assert_called_with("💡 Common Questions")
        mock_caption.assert_called()

        # Should return None when no button clicked
        assert result is None

    @patch('streamlit.subheader')
    @patch('streamlit.caption')
    @patch('streamlit.button')
    def test_display_for_role_fallback(self, mock_button, mock_caption, mock_subheader):
        """Test displaying questions using fallback when analytics unavailable."""
        display = CommonQuestionsDisplay()  # No analytics

        # Mock button not clicked
        mock_button.return_value = False

        result = display.display_for_role("Software Developer", show_analytics=True)

        # Should use fallback questions
        mock_subheader.assert_called_with("💡 Common Questions")
        mock_caption.assert_called_with("Popular questions to get you started:")

        # Should call button for each fallback question
        expected_calls = len(display.fallback_questions["Software Developer"])
        assert mock_button.call_count == expected_calls

        assert result is None

    @patch('streamlit.subheader')
    @patch('streamlit.button')
    def test_display_for_role_button_selection(self, mock_button, mock_subheader):
        """Test question selection when button is clicked."""
        display = CommonQuestionsDisplay()

        # Mock first button clicked
        mock_button.side_effect = [True] + [False] * 10  # First button returns True

        result = display.display_for_role("Software Developer", show_analytics=False)

        # Should return the first question
        expected_question = display.fallback_questions["Software Developer"][0]
        assert result == expected_question


class TestQuestionSuggestions:
    """Test question suggestion functionality."""

    def test_get_question_suggestions_with_analytics(self, mock_analytics):
        """Test getting question suggestions programmatically with analytics."""
        suggestions = get_question_suggestions(
            "Software Developer", analytics_system=mock_analytics, limit=3
        )

        mock_analytics.get_suggested_questions_for_role.assert_called_once_with(
            "Software Developer", limit=3
        )
        assert suggestions == [
            "How does the RAG engine work?",
            "Show me the code architecture",
            "What's the memory system?"
        ]

    def test_get_question_suggestions_fallback(self):
        """Test getting question suggestions with fallback."""
        suggestions = get_question_suggestions("Software Developer", analytics_system=None, limit=3)

        assert isinstance(suggestions, list)
        assert len(suggestions) == 3
        assert all(isinstance(q, str) for q in suggestions)

    def test_get_question_suggestions_invalid_role(self):
        """Test getting suggestions for invalid role."""
        suggestions = get_question_suggestions("Invalid Role", analytics_system=None, limit=3)

        assert suggestions == []

    @patch('streamlit.sidebar')
    @patch('streamlit.subheader')
    @patch('streamlit.caption')
    @patch('streamlit.button')
    def test_display_suggested_questions_sidebar_with_analytics(
        self, mock_button, mock_caption, mock_subheader, mock_sidebar, mock_analytics
    ):
        """Test sidebar suggestion display with analytics."""
        mock_button.return_value = False

        result = display_suggested_questions_sidebar("Software Developer", mock_analytics)

        mock_analytics.get_suggested_questions_for_role.assert_called_once_with(
            "Software Developer", limit=3
        )
        assert result is None

    @patch('streamlit.sidebar')
    @patch('streamlit.subheader')
    @patch('streamlit.button')
    def test_display_suggested_questions_sidebar_fallback(
        self, mock_button, mock_subheader, mock_sidebar
    ):
        """Test sidebar suggestion display with fallback."""
        mock_button.return_value = False

        result = display_suggested_questions_sidebar("Software Developer", analytics_system=None)

        # Should use fallback suggestions
        assert mock_button.call_count > 0
        assert result is None


class TestUIErrorHandling:
    """Test UI component error handling."""

    def test_ui_component_error_handling(self):
        """Test UI component error handling with broken analytics."""
        # Create mock analytics that raises exceptions
        broken_analytics = Mock()
        broken_analytics.get_most_common_questions.side_effect = Exception("Database error")
        broken_analytics.get_suggested_questions_for_role.side_effect = Exception("Connection error")

        display = CommonQuestionsDisplay(analytics_system=broken_analytics)

        # Should not raise exceptions, should fall back gracefully
        suggestions = get_question_suggestions("Software Developer", analytics_system=broken_analytics)
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0  # Should use fallback


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
