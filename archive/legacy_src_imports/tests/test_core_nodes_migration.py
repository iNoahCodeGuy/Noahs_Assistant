"""Test suite for Batch 1: Core Pipeline Nodes (TypedDict Migration).

This suite validates that retrieve_chunks, generate_answer, apply_role_context,
and log_and_notify nodes follow LangGraph TypedDict patterns correctly.

Tests verify:
1. Node signature compliance (accepts TypedDict, returns Dict[str, Any])
2. Fail-fast validation on required fields
3. Business logic correctness
4. Partial state updates (not full state returns)
5. Immutability (no input state mutation)
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from src.state.conversation_state import ConversationState
from src.flows.node_logic.core_nodes import (
    retrieve_chunks,
    generate_answer,
    apply_role_context,
    log_and_notify
)


class TestRetrieveChunksNode:
    """Test retrieve_chunks node migration to TypedDict pattern."""

    def test_node_signature(self):
        """Node should accept ConversationState TypedDict and return Dict[str, Any]."""
        mock_engine = MagicMock()
        mock_engine.retrieve.return_value = {"chunks": [], "matches": [], "scores": []}

        state: ConversationState = {
            "role": "Software Developer",
            "query": "What are your Python skills?",
            "chat_history": []
        }

        result = retrieve_chunks(state, mock_engine)

        # Should return dict (partial update)
        assert isinstance(result, dict)
        assert "retrieved_chunks" in result
        assert "retrieval_matches" in result
        assert "retrieval_scores" in result

    def test_fail_fast_on_missing_query(self):
        """Should raise KeyError if query field missing."""
        mock_engine = MagicMock()
        state: ConversationState = {"role": "Software Developer"}

        with pytest.raises(KeyError, match="State must contain 'query' field"):
            retrieve_chunks(state, mock_engine)

    def test_retrieves_chunks_successfully(self):
        """Should call RAG engine and return chunks in partial update."""
        mock_engine = MagicMock()
        mock_engine.retrieve.return_value = {
            "chunks": ["chunk1", "chunk2"],
            "matches": ["match1", "match2"],
            "scores": [0.9, 0.8]
        }

        state: ConversationState = {
            "role": "Software Developer",
            "query": "How does the RAG pipeline work?",
            "chat_history": []
        }

        result = retrieve_chunks(state, mock_engine)

        # Verify RAG engine called correctly
        mock_engine.retrieve.assert_called_once_with("How does the RAG pipeline work?", top_k=4)

        # Verify partial update returned
        assert result["retrieved_chunks"] == ["chunk1", "chunk2"]
        assert result["retrieval_matches"] == ["match1", "match2"]
        assert result["retrieval_scores"] == [0.9, 0.8]

    def test_uses_expanded_query_if_available(self):
        """Should use expanded_query over original query when present."""
        mock_engine = MagicMock()
        mock_engine.retrieve.return_value = {"chunks": [], "matches": [], "scores": []}

        state: ConversationState = {
            "role": "Software Developer",
            "query": "engineering",
            "expanded_query": "software engineering skills and experience",
            "chat_history": []
        }

        result = retrieve_chunks(state, mock_engine)

        # Should use expanded query
        mock_engine.retrieve.assert_called_once_with(
            "software engineering skills and experience",
            top_k=4
        )

    def test_graceful_degradation_on_rag_failure(self):
        """Should return empty results if RAG engine fails."""
        mock_engine = MagicMock()
        mock_engine.retrieve.side_effect = Exception("Database connection failed")

        state: ConversationState = {
            "role": "Software Developer",
            "query": "test query",
            "chat_history": []
        }

        result = retrieve_chunks(state, mock_engine)

        # Should return empty results (fail-safe)
        assert result["retrieved_chunks"] == []
        assert result["retrieval_matches"] == []
        assert result["retrieval_scores"] == []

    def test_does_not_mutate_input_state(self):
        """Node should not modify input state dict."""
        mock_engine = MagicMock()
        mock_engine.retrieve.return_value = {"chunks": ["test"], "matches": [], "scores": []}

        state: ConversationState = {
            "role": "Software Developer",
            "query": "test",
            "chat_history": []
        }

        original_state = state.copy()
        result = retrieve_chunks(state, mock_engine)

        # Input state should be unchanged
        assert state == original_state
        # Result should be a different dict
        assert result is not state


class TestGenerateAnswerNode:
    """Test generate_answer node migration to TypedDict pattern."""

    def test_node_signature(self):
        """Node should accept ConversationState TypedDict and return Dict[str, Any]."""
        mock_engine = MagicMock()
        mock_engine.response_generator = MagicMock()
        mock_engine.response_generator.generate_contextual_response.return_value = "Test answer"

        state: ConversationState = {
            "role": "Software Developer",
            "query": "What are your Python skills?",
            "chat_history": [],
            "retrieved_chunks": []
        }

        result = generate_answer(state, mock_engine)

        # Should return dict (partial update)
        assert isinstance(result, dict)
        assert "answer" in result

    def test_fail_fast_on_missing_query(self):
        """Should raise KeyError if query field missing."""
        mock_engine = MagicMock()
        state: ConversationState = {"role": "Software Developer"}

        with pytest.raises(KeyError, match="State must contain 'query' field"):
            generate_answer(state, mock_engine)

    def test_generates_answer_with_rag_context(self):
        """Should call LLM with query + retrieved chunks."""
        mock_engine = MagicMock()
        mock_engine.response_generator = MagicMock()
        mock_engine.response_generator.generate_contextual_response.return_value = "Generated answer"

        state: ConversationState = {
            "role": "Software Developer",
            "query": "How does RAG work?",
            "chat_history": [],
            "retrieved_chunks": ["chunk1", "chunk2"]
        }

        result = generate_answer(state, mock_engine)

        # Verify LLM called with correct context
        mock_engine.response_generator.generate_contextual_response.assert_called_once()
        call_kwargs = mock_engine.response_generator.generate_contextual_response.call_args.kwargs
        assert call_kwargs["query"] == "How does RAG work?"
        assert call_kwargs["context"] == ["chunk1", "chunk2"]
        assert call_kwargs["role"] == "Software Developer"

        # Verify answer returned
        assert result["answer"] == "Generated answer"

    def test_data_display_bypasses_llm(self):
        """Data display requests should return placeholder for later rendering."""
        mock_engine = MagicMock()

        state: ConversationState = {
            "role": "Hiring Manager (technical)",
            "query": "Display analytics data",
            "chat_history": [],
            "data_display_requested": True
        }

        result = generate_answer(state, mock_engine)

        # Should return placeholder, not call LLM
        assert "analytics" in result["answer"].lower()
        assert not mock_engine.response_generator.generate_contextual_response.called

    def test_fallback_for_vague_query_with_no_matches(self):
        """Vague expanded queries with no results should get helpful fallback."""
        mock_engine = MagicMock()

        state: ConversationState = {
            "role": "Software Developer",
            "query": "engineering",
            "chat_history": [],
            "retrieved_chunks": [],
            "vague_query_expanded": True,
            "expanded_query": "software engineering skills"
        }

        result = generate_answer(state, mock_engine)

        # Should return fallback message
        assert result["fallback_used"] is True
        assert "more specific" in result["answer"].lower()
        assert not mock_engine.response_generator.generate_contextual_response.called

    def test_sanitizes_sql_artifacts(self):
        """Generated answer should be sanitized for SQL artifacts."""
        mock_engine = MagicMock()
        mock_engine.response_generator = MagicMock()
        mock_engine.response_generator.generate_contextual_response.return_value = (
            "}\n\nSELECT\n\nClean content"
        )

        state: ConversationState = {
            "role": "Software Developer",
            "query": "test",
            "chat_history": [],
            "retrieved_chunks": []
        }

        result = generate_answer(state, mock_engine)

        # Should strip SQL artifacts (sanitize_generated_answer called)
        assert "SELECT" not in result["answer"]

    def test_does_not_mutate_input_state(self):
        """Node should not modify input state dict."""
        mock_engine = MagicMock()
        mock_engine.response_generator = MagicMock()
        mock_engine.response_generator.generate_contextual_response.return_value = "Answer"

        state: ConversationState = {
            "role": "Software Developer",
            "query": "test",
            "chat_history": []
        }

        original_state = state.copy()
        result = generate_answer(state, mock_engine)

        # Input state should be unchanged
        assert state == original_state
        assert result is not state


class TestApplyRoleContextNode:
    """Test apply_role_context node migration to TypedDict pattern."""

    def test_node_signature(self):
        """Node should accept ConversationState TypedDict and return Dict[str, Any]."""
        mock_engine = MagicMock()

        state: ConversationState = {
            "role": "Software Developer",
            "query": "test",
            "chat_history": [],
            "answer": "Base answer",
            "pending_actions": []
        }

        result = apply_role_context(state, mock_engine)

        # Should return dict (partial update)
        assert isinstance(result, dict)
        assert "answer" in result

    def test_fail_fast_on_missing_answer(self):
        """Should raise KeyError if answer field missing."""
        mock_engine = MagicMock()
        state: ConversationState = {
            "role": "Software Developer",
            "query": "test"
        }

        with pytest.raises(KeyError, match="State must contain 'answer' field"):
            apply_role_context(state, mock_engine)

    def test_adds_purpose_overview_when_action_present(self):
        """Should append purpose block when action planned."""
        mock_engine = MagicMock()

        state: ConversationState = {
            "role": "Hiring Manager (technical)",
            "query": "How does this product work?",
            "chat_history": [],
            "answer": "Base answer",
            "pending_actions": [{"type": "include_purpose_overview"}]
        }

        result = apply_role_context(state, mock_engine)

        # Should include purpose block
        assert "Base answer" in result["answer"]
        assert len(result["answer"]) > len("Base answer")

    def test_adds_code_snippets_for_technical_roles(self):
        """Should fetch and append code snippets when action present."""
        mock_engine = MagicMock()
        mock_engine.retrieve_with_code.return_value = {
            "code_snippets": [{
                "content": "def test():\n    return True",
                "citation": "src/test.py"
            }]
        }

        state: ConversationState = {
            "role": "Software Developer",
            "query": "Show me RAG code",
            "chat_history": [],
            "answer": "Here's the logic",
            "pending_actions": [{"type": "include_code_snippets"}]
        }

        result = apply_role_context(state, mock_engine)

        # Should include code block
        assert "Here's the logic" in result["answer"]
        assert "def test" in result["answer"]

    def test_adds_mma_link_when_mma_query(self):
        """Should append MMA fight link for fight queries."""
        mock_engine = MagicMock()

        state: ConversationState = {
            "role": "Just looking around",
            "query": "Tell me about your MMA fight",
            "chat_history": [],
            "answer": "Noah fought in the UFC",
            "pending_actions": [{"type": "share_mma_link"}],
            "query_type": "mma"
        }

        result = apply_role_context(state, mock_engine)

        # Should include YouTube link
        assert "Noah fought in the UFC" in result["answer"]
        assert "youtube.com" in result["answer"].lower()

    def test_skips_enrichment_for_empty_answer(self):
        """Should return empty update if answer is empty."""
        mock_engine = MagicMock()

        state: ConversationState = {
            "role": "Software Developer",
            "query": "test",
            "answer": "",
            "pending_actions": []
        }

        result = apply_role_context(state, mock_engine)

        # Should return empty dict (no update needed)
        assert result == {}

    def test_does_not_mutate_input_state(self):
        """Node should not modify input state dict."""
        mock_engine = MagicMock()

        state: ConversationState = {
            "role": "Software Developer",
            "query": "test",
            "answer": "Base answer",
            "pending_actions": []
        }

        original_state = state.copy()
        result = apply_role_context(state, mock_engine)

        # Input state should be unchanged (note: answer enriched in result, not state)
        assert state == original_state
        assert result is not state


class TestLogAndNotifyNode:
    """Test log_and_notify node migration to TypedDict pattern."""

    def test_node_signature(self):
        """Node should accept ConversationState TypedDict and return Dict[str, Any]."""
        with patch('src.flows.core_nodes.supabase_analytics') as mock_analytics:
            mock_analytics.log_interaction.return_value = "msg_123"

            state: ConversationState = {
                "role": "Software Developer",
                "query": "test query",
                "answer": "test answer",
                "query_type": "technical",
                "chat_history": []
            }

            result = log_and_notify(state, session_id="session_123", latency_ms=500)

            # Should return dict (partial update)
            assert isinstance(result, dict)
            assert "message_id" in result
            assert "logged_at" in result

    def test_logs_interaction_to_supabase(self):
        """Should call analytics logger with correct data."""
        with patch('src.flows.core_nodes.supabase_analytics') as mock_analytics:
            mock_analytics.log_interaction.return_value = "msg_456"

            state: ConversationState = {
                "role": "Hiring Manager (technical)",
                "query": "What are your skills?",
                "answer": "I have Python experience",
                "query_type": "career",
                "chat_history": []
            }

            result = log_and_notify(state, session_id="sess_789", latency_ms=1200)

            # Verify analytics called
            mock_analytics.log_interaction.assert_called_once()
            call_arg = mock_analytics.log_interaction.call_args.args[0]
            assert call_arg.session_id == "sess_789"
            assert call_arg.role_mode == "Hiring Manager (technical)"
            assert call_arg.query == "What are your skills?"
            assert call_arg.answer == "I have Python experience"
            assert call_arg.query_type == "career"
            assert call_arg.latency_ms == 1200

            # Verify message_id returned
            assert result["message_id"] == "msg_456"
            assert result["logged_at"] is True

    def test_graceful_degradation_on_logging_failure(self):
        """Should not crash if analytics logging fails."""
        with patch('src.flows.core_nodes.supabase_analytics') as mock_analytics:
            mock_analytics.log_interaction.side_effect = Exception("Database error")

            state: ConversationState = {
                "role": "Software Developer",
                "query": "test",
                "answer": "test answer",
                "chat_history": []
            }

            result = log_and_notify(state, session_id="session_123", latency_ms=500)

            # Should return logged_at=False but not crash
            assert result["logged_at"] is False

    def test_uses_defaults_for_missing_optional_fields(self):
        """Should use safe defaults if optional fields missing."""
        with patch('src.flows.core_nodes.supabase_analytics') as mock_analytics:
            mock_analytics.log_interaction.return_value = "msg_789"

            state: ConversationState = {
                # Only required field: chat_history (TypedDict constraint)
                "chat_history": []
            }

            result = log_and_notify(state, session_id="session_123", latency_ms=500)

            # Should use defaults
            call_arg = mock_analytics.log_interaction.call_args.args[0]
            assert call_arg.role_mode == "Just looking around"
            assert call_arg.query == ""
            assert call_arg.answer == ""
            assert call_arg.query_type == "general"

    def test_does_not_mutate_input_state(self):
        """Node should not modify input state dict."""
        with patch('src.flows.core_nodes.supabase_analytics') as mock_analytics:
            mock_analytics.log_interaction.return_value = "msg_999"

            state: ConversationState = {
                "role": "Software Developer",
                "query": "test",
                "answer": "test answer",
                "chat_history": []
            }

            original_state = state.copy()
            result = log_and_notify(state, session_id="session_123", latency_ms=500)

            # Input state should be unchanged
            assert state == original_state
            assert result is not state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
