"""Tests for classify_query node migration to TypedDict pattern.

This test suite validates that classify_query follows LangGraph patterns:
1. Accepts ConversationState (TypedDict)
2. Returns Dict[str, Any] (partial update)
3. Implements fail-fast validation
4. Doesn't mutate input state
5. Returns only modified fields

This is the POC migration test - once this passes, we can migrate remaining nodes.
"""

import pytest
from src.flows.node_logic.query_classification import classify_query
from src.state.conversation_state import ConversationState


class TestClassifyQueryNodeSignature:
    """Test that classify_query follows LangGraph node signature."""

    def test_accepts_typed_dict_state(self):
        """Should accept ConversationState TypedDict as input."""
        state: ConversationState = {
            "query": "How does RAG work?",
            "role": "Software Developer",
            "session_id": "test-123"
        }

        result = classify_query(state)

        # Should return dict (partial update)
        assert isinstance(result, dict)

    def test_returns_partial_update_not_full_state(self):
        """Should return only modified fields, not full state."""
        state: ConversationState = {
            "query": "Show me code",
            "role": "Software Developer",
            "session_id": "test-123",
            "chat_history": [{"role": "user", "content": "Hello"}]
        }

        result = classify_query(state)

        # Result should NOT contain unchanged fields
        assert "query" not in result  # Wasn't modified
        assert "role" not in result  # Wasn't modified
        assert "session_id" not in result  # Wasn't modified
        assert "chat_history" not in result  # Wasn't modified

        # Result SHOULD contain classification fields
        assert "query_type" in result

    def test_does_not_mutate_input_state(self):
        """Should not modify input state (immutability)."""
        state: ConversationState = {
            "query": "technical question",
            "role": "Developer",
            "session_id": "abc"
        }

        # Make copy to verify immutability
        original = state.copy()

        classify_query(state)

        # Input state should be unchanged
        assert state == original


class TestClassifyQueryFailFast:
    """Test fail-fast validation pattern."""

    def test_handles_missing_query_field(self):
        """Should fail-fast with clear error when query missing."""
        state: ConversationState = {
            "role": "Developer",
            "session_id": "123"
        }
        # Missing "query" field

        result = classify_query(state)

        # Should return error state, not crash
        assert result["error"] == "classification_failed"
        assert "query" in result["error_message"].lower()

    def test_handles_empty_state(self):
        """Should gracefully handle completely empty state."""
        state: ConversationState = {}

        result = classify_query(state)

        assert result["error"] == "classification_failed"
        assert "missing required field" in result["error_message"].lower()


class TestClassifyQueryLogic:
    """Test classification business logic."""

    def test_classifies_technical_query(self):
        """Should detect technical queries."""
        state: ConversationState = {
            "query": "How does the RAG pipeline work?",
            "role": "Developer",
            "session_id": "123"
        }

        result = classify_query(state)

        assert result["query_type"] == "technical"
        assert result.get("teaching_moment") is True  # "how does" triggers teaching

    def test_classifies_code_display_request(self):
        """Should detect explicit code display requests."""
        state: ConversationState = {
            "query": "Show me the code for retrieval",
            "role": "Developer",
            "session_id": "123"
        }

        result = classify_query(state)

        assert result["query_type"] == "technical"
        assert result["code_display_requested"] is True

    def test_classifies_career_query(self):
        """Should detect career-related queries."""
        state: ConversationState = {
            "query": "What's Noah's work experience?",
            "role": "Hiring Manager (technical)",
            "session_id": "123"
        }

        result = classify_query(state)

        assert result["query_type"] == "career"

    def test_classifies_data_display_request(self):
        """Should detect data/analytics display requests."""
        state: ConversationState = {
            "query": "Display analytics data",  # Use keyword that triggers data, not "show me"
            "role": "Developer",
            "session_id": "123"
        }

        result = classify_query(state)

        assert result["query_type"] == "data"
        assert result["data_display_requested"] is True

    def test_classifies_mma_query(self):
        """Should detect MMA-related queries."""
        state: ConversationState = {
            "query": "Tell me about your MMA fight",
            "role": "Just looking around",
            "session_id": "123"
        }

        result = classify_query(state)

        assert result["query_type"] == "mma"


class TestClassifyQueryVagueExpansion:
    """Test vague query expansion feature."""

    def test_expands_vague_query(self):
        """Should expand single-word vague queries."""
        state: ConversationState = {
            "query": "engineering",
            "role": "Developer",
            "session_id": "123"
        }

        result = classify_query(state)

        assert "expanded_query" in result
        assert "vague_query_expanded" in result
        assert result["vague_query_expanded"] is True
        # Should expand to fuller question
        assert len(result["expanded_query"]) > len("engineering")

    def test_does_not_expand_specific_query(self):
        """Should not expand already-specific queries."""
        state: ConversationState = {
            "query": "What are Noah's Python skills and experience?",
            "role": "Developer",
            "session_id": "123"
        }

        result = classify_query(state)

        # Specific queries shouldn't be expanded
        assert "vague_query_expanded" not in result or result.get("vague_query_expanded") is not True


class TestClassifyQueryProactiveDetection:
    """Test proactive code/data detection for technical roles."""

    def test_proactive_code_for_technical_role(self):
        """Should detect when code would help technical users."""
        state: ConversationState = {
            "query": "Tell me about the RAG pipeline architecture",
            "role": "Software Developer",
            "session_id": "123"
        }

        result = classify_query(state)

        # Should proactively suggest code for technical role
        assert result.get("code_would_help") is True

    def test_no_proactive_code_for_nontechnical_role(self):
        """Should not proactively offer code to non-technical users."""
        state: ConversationState = {
            "query": "Tell me about the RAG pipeline architecture",
            "role": "Hiring Manager (nontechnical)",
            "session_id": "123"
        }

        result = classify_query(state)

        # Should NOT proactively offer code for non-technical role
        assert result.get("code_would_help") is not True

    def test_proactive_data_detection(self):
        """Should detect when analytics would clarify answer."""
        state: ConversationState = {
            "query": "How many users have asked about RAG?",
            "role": "Developer",
            "session_id": "123"
        }

        result = classify_query(state)

        assert result.get("data_would_help") is True


class TestClassifyQueryTeachingMoments:
    """Test detection of teaching-focused queries."""

    def test_detects_why_questions_as_teaching(self):
        """Should flag 'why' questions as teaching moments."""
        state: ConversationState = {
            "query": "Why use pgvector instead of FAISS?",
            "role": "Developer",
            "session_id": "123"
        }

        result = classify_query(state)

        assert result.get("teaching_moment") is True
        assert result.get("needs_longer_response") is True

    def test_detects_explain_questions_as_teaching(self):
        """Should flag 'explain' questions as teaching moments."""
        state: ConversationState = {
            "query": "Explain how the LangGraph orchestration works",
            "role": "Hiring Manager (technical)",
            "session_id": "123"
        }

        result = classify_query(state)

        assert result.get("teaching_moment") is True


class TestClassifyQueryIntegration:
    """Integration tests simulating real conversation scenarios."""

    def test_developer_asking_technical_implementation_question(self):
        """Real scenario: Developer wants to see how something works."""
        state: ConversationState = {
            "query": "How did you implement the vector search?",
            "role": "Software Developer",
            "session_id": "session-456",
            "chat_history": []
        }

        result = classify_query(state)

        # Should detect multiple signals
        assert result["query_type"] == "technical"
        assert result.get("teaching_moment") is True  # "how did"
        assert result.get("code_would_help") is True  # "vector search" + technical role

    def test_hiring_manager_asking_about_experience(self):
        """Real scenario: HM wants to know about Noah's background."""
        state: ConversationState = {
            "query": "Tell me about your work experience",  # Changed to trigger "career" not "technical"
            "role": "Hiring Manager (technical)",
            "session_id": "session-789"
        }

        result = classify_query(state)

        assert result["query_type"] == "career"
        assert result.get("teaching_moment") is not True  # Not a teaching question

    def test_casual_visitor_with_vague_query(self):
        """Real scenario: Casual user asks one-word question."""
        state: ConversationState = {
            "query": "python",
            "role": "Just looking around",
            "session_id": "session-casual"
        }

        result = classify_query(state)

        # Should expand vague query
        assert result.get("vague_query_expanded") is True
        assert "expanded_query" in result
        # Should classify as career/technical
        assert result["query_type"] in ["career", "technical", "general"]
