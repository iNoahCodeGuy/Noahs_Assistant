"""Integration tests for edge case handling pipeline.

Tests the full pipeline: detection → conversational response → meta-teaching.
Ensures edge cases are handled gracefully end-to-end.
"""

import pytest
from unittest.mock import patch, MagicMock
from assistant.flows.node_logic.stage2_query_classification import classify_intent
from assistant.flows.node_logic.stage4_retrieval_nodes import handle_grounding_gap
from assistant.flows.node_logic.stage5_generation_nodes import generate_draft
from assistant.flows.node_logic.util_conversational_edge_case_handler import generate_edge_case_response
from assistant.flows.node_logic.util_edge_case_meta_teaching import generate_meta_teaching_explanation
from assistant.state.conversation_state import ConversationState
from assistant.core.rag_engine import RagEngine


class TestEdgeCaseDetectionIntegration:
    """Test edge case detection in classification pipeline."""

    def test_off_topic_detected_in_classification(self):
        """Test off-topic query is detected and flagged in classify_intent."""
        state: ConversationState = {
            "query": "What's the weather today?",
            "role": "Software Developer",
            "role_mode": "software_developer",
            "chat_history": [],
            "session_memory": {}
        }

        update = classify_intent(state)

        assert update.get("edge_case_detected") is True
        assert update.get("edge_case_type") == "off_topic"
        assert update.get("is_off_topic") is True

    def test_empty_query_detected_in_classification(self):
        """Test empty query is detected and flagged."""
        state: ConversationState = {
            "query": "",
            "role": "Software Developer",
            "role_mode": "software_developer",
            "chat_history": [],
            "session_memory": {}
        }

        update = classify_intent(state)

        assert update.get("edge_case_detected") is True
        assert update.get("edge_case_type") == "empty_query"

    def test_ambiguous_pronoun_detected_in_classification(self):
        """Test ambiguous pronoun is detected and flagged."""
        state: ConversationState = {
            "query": "What about his Python skills?",
            "role": "Software Developer",
            "role_mode": "software_developer",
            "chat_history": [],  # No context to resolve "his"
            "session_memory": {}
        }

        update = classify_intent(state)

        assert update.get("edge_case_detected") is True
        assert update.get("edge_case_type") == "ambiguous_pronouns"
        assert "pronouns" in update


class TestConversationalResponseGeneration:
    """Test conversational response generation for edge cases."""

    @patch('assistant.core.rag_factory.RagEngineFactory')
    def test_off_topic_response_generated(self, mock_factory):
        """Test conversational response is generated for off-topic query."""
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "That's an interesting question! I'm Portfolia..."
        mock_factory_instance = MagicMock()
        mock_factory_instance.create_llm.return_value = (mock_llm, False)
        mock_factory.return_value = mock_factory_instance

        state: ConversationState = {
            "query": "What's the weather?",
            "edge_case_detected": True,
            "edge_case_type": "off_topic",
            "is_off_topic": True,
            "role_mode": "software_developer"
        }

        response = generate_edge_case_response(state)

        assert response is not None
        assert len(response) > 0
        mock_llm.predict.assert_called_once()

    @patch('assistant.core.rag_factory.RagEngineFactory')
    def test_fallback_response_on_llm_failure(self, mock_factory):
        """Test fallback response when LLM generation fails."""
        mock_factory_instance = MagicMock()
        mock_factory_instance.create_llm.side_effect = Exception("LLM error")
        mock_factory.return_value = mock_factory_instance

        state: ConversationState = {
            "query": "What's the weather?",
            "edge_case_detected": True,
            "edge_case_type": "off_topic",
            "role_mode": "software_developer"
        }

        response = generate_edge_case_response(state)

        # Should return fallback response, not crash
        assert response is not None
        assert "Portfolia" in response or "help" in response.lower()


class TestGroundingGapHandler:
    """Test grounding gap handler with edge cases."""

    @patch('assistant.flows.node_logic.stage4_retrieval_nodes.generate_edge_case_response')
    def test_edge_case_triggers_conversational_response(self, mock_generate):
        """Test edge case triggers conversational response in grounding gap handler."""
        mock_generate.return_value = "That's interesting! I'm Portfolia..."

        state: ConversationState = {
            "query": "What's the weather?",
            "edge_case_detected": True,
            "edge_case_type": "off_topic",
            "grounding_status": "insufficient"
        }

        result = handle_grounding_gap(state)

        assert result.get("answer") is not None
        assert result.get("pipeline_halt") is True
        mock_generate.assert_called_once_with(state)

    def test_normal_grounding_gap_unchanged(self):
        """Test normal grounding gap (no edge case) still works."""
        state: ConversationState = {
            "query": "Tell me about RAG",
            "edge_case_detected": False,
            "grounding_status": "insufficient"
        }

        result = handle_grounding_gap(state)

        assert result.get("answer") is not None
        assert "could not find context" in result.get("answer", "").lower()
        assert result.get("pipeline_halt") is True


class TestMetaTeachingExplanation:
    """Test meta-teaching explanation generation."""

    @patch('assistant.core.rag_factory.RagEngineFactory')
    @patch('assistant.flows.node_logic.util_edge_case_meta_teaching.extract_function_code')
    def test_meta_teaching_explanation_generated(self, mock_extract, mock_factory):
        """Test meta-teaching explanation is generated with code snippet."""
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "Great question! You asked 'What's the weather?' which triggered my off-topic detection..."
        mock_factory_instance = MagicMock()
        mock_factory_instance.create_llm.return_value = (mock_llm, False)
        mock_factory.return_value = mock_factory_instance

        mock_extract.return_value = "def _is_off_topic(query: str, chat_history: list) -> bool:\n    ..."

        state: ConversationState = {
            "query": "How did you detect that?",
            "edge_case_type": "off_topic",
            "role_mode": "software_developer",
            "session_memory": {
                "last_edge_case": "off_topic",
                "last_edge_case_query": "What's the weather?"
            }
        }

        explanation = generate_meta_teaching_explanation(state)

        assert explanation is not None
        assert len(explanation) > 0
        mock_llm.predict.assert_called_once()
        # Code snippet should be included for technical users
        assert "def _is_off_topic" in explanation or "code" in explanation.lower()

    @patch('assistant.core.rag_factory.RagEngineFactory')
    def test_meta_teaching_fallback_on_failure(self, mock_factory):
        """Test fallback explanation when LLM fails."""
        mock_factory_instance = MagicMock()
        mock_factory_instance.create_llm.side_effect = Exception("LLM error")
        mock_factory.return_value = mock_factory_instance

        state: ConversationState = {
            "query": "How did you detect that?",
            "edge_case_type": "off_topic",
            "role_mode": "software_developer",
            "session_memory": {
                "last_edge_case": "off_topic",
                "last_edge_case_query": "What's the weather?"
            }
        }

        explanation = generate_meta_teaching_explanation(state)

        # Should return fallback, not crash
        assert explanation is not None
        assert "edge case" in explanation.lower() or "detection" in explanation.lower()


class TestFullPipelineIntegration:
    """Test full pipeline integration with edge cases."""

    @patch('assistant.core.rag_factory.RagEngineFactory')
    def test_off_topic_full_pipeline(self, mock_factory):
        """Test off-topic query through full pipeline."""
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "That's interesting! I'm Portfolia..."
        mock_factory_instance = MagicMock()
        mock_factory_instance.create_llm.return_value = (mock_llm, False)
        mock_factory.return_value = mock_factory_instance

        # Step 1: Classification detects edge case
        state: ConversationState = {
            "query": "What's the weather?",
            "role": "Software Developer",
            "role_mode": "software_developer",
            "chat_history": [],
            "session_memory": {}
        }

        update = classify_intent(state)
        state.update(update)

        assert state.get("edge_case_detected") is True

        # Step 2: Grounding gap handler generates conversational response
        state["grounding_status"] = "insufficient"
        result = handle_grounding_gap(state)

        assert result.get("answer") is not None
        assert result.get("pipeline_halt") is True

    @patch('assistant.core.rag_factory.RagEngineFactory')
    @patch('assistant.flows.node_logic.util_edge_case_meta_teaching.extract_function_code')
    def test_meta_teaching_full_pipeline(self, mock_extract, mock_factory):
        """Test meta-teaching request through full pipeline."""
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "Great question! Here's how I detected it..."
        mock_factory_instance = MagicMock()
        mock_factory_instance.create_llm.return_value = (mock_llm, False)
        mock_factory.return_value = mock_factory_instance

        mock_extract.return_value = "def _is_off_topic(...):\n    ..."

        # User asks about detection after edge case was handled
        state: ConversationState = {
            "query": "How did you detect that was off-topic?",
            "role": "Software Developer",
            "role_mode": "software_developer",
            "chat_history": [],
            "session_memory": {
                "last_edge_case": "off_topic",
                "last_edge_case_query": "What's the weather?"
            },
            "retrieved_chunks": []
        }

        # Generate draft should detect meta-teaching request
        rag_engine = MagicMock(spec=RagEngine)
        # Add response_generator mock to rag_engine
        rag_engine.response_generator = MagicMock()
        rag_engine.response_generator.generate_contextual_response = MagicMock(return_value="Test response")
        rag_engine.response_generator._enforce_first_person = MagicMock(side_effect=lambda x: x)
        update = generate_draft(state, rag_engine)

        # Should generate meta-teaching explanation
        assert update.get("answer") is not None or update.get("draft_answer") is not None
        answer = update.get("answer") or update.get("draft_answer", "")
        assert len(answer) > 0


class TestEdgeCasePriority:
    """Test edge case detection priority order."""

    def test_empty_query_has_priority(self):
        """Test empty query is detected before off-topic."""
        state: ConversationState = {
            "query": "",
            "chat_history": [],
            "session_memory": {}
        }

        edge_cases = classify_intent(state)

        # Empty query should be detected (not off-topic)
        assert edge_cases.get("edge_case_type") == "empty_query"
        assert edge_cases.get("is_empty_query") is True

    def test_off_topic_has_priority_over_ambiguous(self):
        """Test off-topic is detected before ambiguous pronouns."""
        state: ConversationState = {
            "query": "What's the weather?",
            "chat_history": [],
            "session_memory": {}
        }

        edge_cases = classify_intent(state)

        # Should detect as off-topic, not ambiguous pronouns
        assert edge_cases.get("edge_case_type") == "off_topic"

    def test_very_long_query_has_priority(self):
        """Test very long query is detected early."""
        long_query = "a" * 2001
        state: ConversationState = {
            "query": long_query,
            "chat_history": [],
            "session_memory": {}
        }

        edge_cases = classify_intent(state)

        assert edge_cases.get("edge_case_type") == "very_long_query"
        assert edge_cases.get("query_length") == 2001

    def test_nonsensical_query_detected(self):
        """Test nonsensical query is detected."""
        state: ConversationState = {
            "query": "asdf",
            "chat_history": [],
            "session_memory": {}
        }

        edge_cases = classify_intent(state)

        assert edge_cases.get("edge_case_type") == "nonsensical_query"

    def test_hypothetical_query_detected(self):
        """Test hypothetical query is detected."""
        state: ConversationState = {
            "query": "What if Noah worked at Google?",
            "chat_history": [],
            "session_memory": {}
        }

        edge_cases = classify_intent(state)

        assert edge_cases.get("edge_case_type") == "hypothetical_query"

    def test_temporal_confusion_detected(self):
        """Test temporal confusion is detected."""
        state: ConversationState = {
            "query": "What is Noah doing now?",
            "chat_history": [],
            "session_memory": {}
        }

        edge_cases = classify_intent(state)

        assert edge_cases.get("edge_case_type") == "temporal_confusion"

    @patch('assistant.core.rag_factory.RagEngineFactory')
    def test_very_long_query_response(self, mock_factory):
        """Test very long query gets conversational response."""
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "That's a detailed question! Could you break it into smaller parts?"
        mock_factory_instance = MagicMock()
        mock_factory_instance.create_llm.return_value = (mock_llm, False)
        mock_factory.return_value = mock_factory_instance

        state: ConversationState = {
            "query": "a" * 2001,
            "edge_case_type": "very_long_query",
            "query_length": 2001,
            "edge_case_detected": True,
            "role_mode": "software_developer"
        }

        response = generate_edge_case_response(state)

        assert response is not None
        assert len(response) > 0

    @patch('assistant.core.rag_factory.RagEngineFactory')
    def test_nonsensical_query_response(self, mock_factory):
        """Test nonsensical query gets conversational response."""
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "I'm not sure what you're asking. Could you rephrase that?"
        mock_factory_instance = MagicMock()
        mock_factory_instance.create_llm.return_value = (mock_llm, False)
        mock_factory.return_value = mock_factory_instance

        state: ConversationState = {
            "query": "asdf",
            "edge_case_type": "nonsensical_query",
            "edge_case_detected": True,
            "role_mode": "explorer"
        }

        response = generate_edge_case_response(state)

        assert response is not None
        assert len(response) > 0

    @patch('assistant.core.rag_factory.RagEngineFactory')
    def test_hypothetical_query_response(self, mock_factory):
        """Test hypothetical query gets conversational response."""
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "That's an interesting hypothetical! My knowledge base focuses on actual experience."
        mock_factory_instance = MagicMock()
        mock_factory_instance.create_llm.return_value = (mock_llm, False)
        mock_factory.return_value = mock_factory_instance

        state: ConversationState = {
            "query": "What if Noah worked at Google?",
            "edge_case_type": "hypothetical_query",
            "edge_case_detected": True,
            "role_mode": "software_developer"
        }

        response = generate_edge_case_response(state)

        assert response is not None
        assert len(response) > 0

    @patch('assistant.core.rag_factory.RagEngineFactory')
    def test_temporal_confusion_response(self, mock_factory):
        """Test temporal confusion gets conversational response."""
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "My knowledge base shows information as of when it was last updated."
        mock_factory_instance = MagicMock()
        mock_factory_instance.create_llm.return_value = (mock_llm, False)
        mock_factory.return_value = mock_factory_instance

        state: ConversationState = {
            "query": "What is Noah doing now?",
            "edge_case_type": "temporal_confusion",
            "edge_case_detected": True,
            "role_mode": "explorer"
        }

        response = generate_edge_case_response(state)

        assert response is not None
        assert len(response) > 0

    @patch('assistant.core.rag_factory.RagEngineFactory')
    def test_contradictory_information_response(self, mock_factory):
        """Test contradictory information gets conversational response."""
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "I want to make sure I have the right information. Could you clarify?"
        mock_factory_instance = MagicMock()
        mock_factory_instance.create_llm.return_value = (mock_llm, False)
        mock_factory.return_value = mock_factory_instance

        state: ConversationState = {
            "query": "Noah works at Google",
            "edge_case_type": "contradictory_information",
            "edge_case_detected": True,
            "retrieved_chunks": [{"content": "Noah worked at Tesla"}],
            "role_mode": "software_developer"
        }

        response = generate_edge_case_response(state)

        assert response is not None
        assert len(response) > 0
