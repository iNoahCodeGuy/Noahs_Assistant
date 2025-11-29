"""Tests for Chain-of-Thought reasoning module.

This module tests the two-phase generation system:
1. Reasoning Phase - analyzes query and produces structured plan
2. Generation Phase - uses plan to produce contextually appropriate response
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import json

from assistant.flows.node_logic.chain_of_thought import (
    generate_reasoning,
    generate_with_reasoning,
    chain_of_thought_generate,
    detect_confusion_signals,
    _parse_reasoning_response,
    _get_default_reasoning,
    _format_history_for_reasoning,
    _format_chunks_for_context,
    _simple_similarity,
    CONFUSION_PATTERNS,
)


class TestConfusionDetection:
    """Tests for pattern-based confusion detection."""

    def test_detects_explicit_confusion(self):
        """Should detect explicit 'I don't understand' pattern."""
        result = detect_confusion_signals("I don't understand what you mean", [])
        assert result["seems_confused"] is True
        assert len(result["signals"]) > 0

    def test_detects_wait_what_pattern(self):
        """Should detect 'wait, what?' confusion pattern."""
        result = detect_confusion_signals("wait, what do you mean?", [])
        assert result["seems_confused"] is True

    def test_detects_clarification_request(self):
        """Should detect explicit clarification request."""
        result = detect_confusion_signals("can you explain that more clearly?", [])
        assert result["seems_confused"] is True

    def test_no_confusion_for_normal_query(self):
        """Should not flag confusion for normal queries."""
        result = detect_confusion_signals("How does the RAG pipeline work?", [])
        assert result["seems_confused"] is False
        assert len(result["signals"]) == 0

    def test_detects_reformulation(self):
        """Should detect when user asks similar question again."""
        # The detection compares current query with the second-to-last user message
        # So: history[-2] user message should be similar to current query
        history = [
            {"role": "user", "content": "How does retrieval work?"},  # Previous (similar) query
            {"role": "assistant", "content": "The retrieval system uses..."},
            {"role": "user", "content": "Thanks"},  # Most recent user msg (ignored)
            {"role": "assistant", "content": "You're welcome!"}
        ]
        # Current query similar to the previous user query (How does retrieval work?)
        result = detect_confusion_signals("How does the retrieval work?", history)
        # High similarity = possible reformulation
        assert any("reformulation" in str(s).lower() for s in result["signals"])

    def test_detects_short_response_after_long_explanation(self):
        """Should detect when user gives very short response after long explanation."""
        history = [
            {"role": "user", "content": "Explain the architecture"},
            {"role": "assistant", "content": "A" * 600}  # Long explanation
        ]
        result = detect_confusion_signals("ok", history)
        assert result["seems_confused"] is True
        assert "short_response_after_long_explanation" in result["signals"]

    def test_detects_multiple_question_marks(self):
        """Should detect frustration signal from multiple question marks."""
        result = detect_confusion_signals("What??? How does that even work???", [])
        assert result["seems_confused"] is True
        assert "multiple_question_marks" in result["signals"]

    def test_confidence_scales_with_signals(self):
        """Confidence should increase with more confusion signals."""
        # Single signal
        result1 = detect_confusion_signals("huh?", [])
        # Multiple signals (confusion + frustration)
        result2 = detect_confusion_signals("Wait, what??? I don't get it at all", [])
        assert result2["confidence"] > result1["confidence"]


class TestReasoningParsing:
    """Tests for JSON reasoning response parsing."""

    def test_parses_valid_json(self):
        """Should correctly parse valid JSON response."""
        response = json.dumps({
            "user_intent": {"explicit": "test query", "confidence": 0.9},
            "clarification_needed": {"needed": False},
            "response_plan": {"depth_level": 2}
        })
        result = _parse_reasoning_response(response)
        assert result["user_intent"]["confidence"] == 0.9
        assert result["response_plan"]["depth_level"] == 2

    def test_handles_markdown_json_codeblock(self):
        """Should extract JSON from markdown code block."""
        response = '''```json
{
    "user_intent": {"explicit": "test", "confidence": 0.8}
}
```'''
        result = _parse_reasoning_response(response)
        assert "user_intent" in result
        assert result["user_intent"]["confidence"] == 0.8

    def test_handles_plain_codeblock(self):
        """Should extract JSON from plain code block."""
        response = '''```
{"user_intent": {"explicit": "test"}}
```'''
        result = _parse_reasoning_response(response)
        assert "user_intent" in result

    def test_handles_json_with_surrounding_text(self):
        """Should extract JSON even with surrounding text."""
        response = '''Here is the analysis:
{"user_intent": {"explicit": "test", "confidence": 0.7}}
That concludes my analysis.'''
        result = _parse_reasoning_response(response)
        assert result["user_intent"]["confidence"] == 0.7

    def test_returns_default_on_invalid_json(self):
        """Should return default reasoning on parse failure."""
        response = "This is not valid JSON at all"
        result = _parse_reasoning_response(response)
        # Should have default structure
        assert "user_intent" in result
        assert "response_plan" in result
        assert result["user_intent"]["confidence"] == 0.5


class TestDefaultReasoning:
    """Tests for default reasoning fallback."""

    def test_default_has_required_fields(self):
        """Default reasoning should have all required fields."""
        result = _get_default_reasoning("test query")

        assert "user_intent" in result
        assert "clarification_needed" in result
        assert "user_state" in result
        assert "response_plan" in result
        assert "context_relevance" in result

    def test_default_uses_query_as_intent(self):
        """Default should use query as explicit intent."""
        result = _get_default_reasoning("How does X work?")
        assert result["user_intent"]["explicit"] == "How does X work?"

    def test_default_is_balanced(self):
        """Default should use balanced settings."""
        result = _get_default_reasoning("test")
        assert result["response_plan"]["depth_level"] == 2
        assert result["response_plan"]["style"] == "balanced"
        assert result["clarification_needed"]["needed"] is False


class TestHelperFunctions:
    """Tests for CoT helper functions."""

    def test_format_history_truncates(self):
        """Should only include last 6 messages."""
        history = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(10)
        ]
        result = _format_history_for_reasoning(history)
        # Should have last 6 messages (3 exchanges worth)
        assert "Message 9" in result
        assert "Message 4" in result
        assert "Message 3" not in result

    def test_format_history_handles_langchain_format(self):
        """Should handle LangChain message format (type: human/ai)."""
        history = [
            {"type": "human", "content": "User question"},
            {"type": "ai", "content": "AI response"}
        ]
        result = _format_history_for_reasoning(history)
        assert "USER: User question" in result
        assert "ASSISTANT: AI response" in result

    def test_format_chunks_limits_to_5(self):
        """Should limit to 5 chunks."""
        chunks = [{"content": f"Chunk {i}", "section": f"Section {i}"} for i in range(10)]
        result = _format_chunks_for_context(chunks)
        assert "Chunk 4" in result
        assert "Chunk 5" not in result

    def test_format_chunks_handles_empty(self):
        """Should handle empty chunks gracefully."""
        result = _format_chunks_for_context([])
        assert "No context available" in result

    def test_simple_similarity_identical(self):
        """Identical strings should have similarity 1.0."""
        result = _simple_similarity("hello world", "hello world")
        assert result == 1.0

    def test_simple_similarity_no_overlap(self):
        """No word overlap should have similarity 0.0."""
        result = _simple_similarity("hello world", "foo bar")
        assert result == 0.0

    def test_simple_similarity_partial(self):
        """Partial overlap should have intermediate similarity."""
        result = _simple_similarity("hello world", "hello there")
        assert 0.0 < result < 1.0


class TestGenerateReasoning:
    """Tests for the reasoning generation phase."""

    def test_calls_llm_with_prompt(self):
        """Should call LLM with reasoning prompt."""
        mock_llm = MagicMock()
        mock_llm.predict.return_value = json.dumps({
            "user_intent": {"explicit": "test", "confidence": 0.9},
            "clarification_needed": {"needed": False}
        })

        result = generate_reasoning(
            query="How does RAG work?",
            context="RAG is a technique...",
            chat_history=[],
            role="Software Developer",
            llm=mock_llm
        )

        # Should have called LLM
        assert mock_llm.predict.called
        # Result should have parsed structure
        assert result["user_intent"]["confidence"] == 0.9

    def test_adds_meta_timing(self):
        """Should add timing metadata to result."""
        mock_llm = MagicMock()
        mock_llm.predict.return_value = '{"user_intent": {"confidence": 0.8}}'

        result = generate_reasoning("test", "context", [], "role", mock_llm)

        assert "_meta" in result
        assert "reasoning_time_ms" in result["_meta"]

    def test_handles_llm_failure(self):
        """Should return default reasoning on LLM failure."""
        mock_llm = MagicMock()
        mock_llm.predict.side_effect = Exception("LLM error")

        result = generate_reasoning("test query", "context", [], "role", mock_llm)

        # Should have default structure
        assert result["user_intent"]["confidence"] == 0.5
        assert result["response_plan"]["depth_level"] == 2


class TestGenerateWithReasoning:
    """Tests for the generation-with-reasoning phase."""

    def test_returns_clarification_when_needed(self):
        """Should return clarifying question when reasoning says to."""
        mock_llm = MagicMock()
        reasoning = {
            "clarification_needed": {
                "needed": True,
                "suggested_question": "Could you clarify what you mean by X?"
            },
            "response_plan": {"depth_level": 2}
        }

        response, metadata = generate_with_reasoning(
            query="Tell me about X",
            context="",
            chat_history=[],
            role="Developer",
            reasoning=reasoning,
            llm=mock_llm
        )

        assert response == "Could you clarify what you mean by X?"
        assert metadata["type"] == "clarification"
        # Should NOT call LLM for clarification
        assert not mock_llm.predict.called

    def test_generates_answer_when_no_clarification(self):
        """Should generate answer when clarification not needed."""
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "Here is your answer about RAG..."

        reasoning = {
            "clarification_needed": {"needed": False},
            "response_plan": {"depth_level": 2, "style": "balanced"},
            "user_state": {"seems_confused": False},
            "context_relevance": {"can_answer_confidently": True},
            "user_intent": {"explicit": "How does RAG work?"}
        }

        response, metadata = generate_with_reasoning(
            query="How does RAG work?",
            context="RAG context...",
            chat_history=[],
            role="Developer",
            reasoning=reasoning,
            llm=mock_llm
        )

        assert "answer" in response.lower() or "rag" in response.lower()
        assert metadata["type"] == "answer"
        assert mock_llm.predict.called

    def test_includes_metadata(self):
        """Should include timing and reasoning metadata."""
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "Answer text"

        reasoning = {
            "clarification_needed": {"needed": False},
            "response_plan": {"depth_level": 3, "style": "comprehensive"},
            "user_state": {"seems_confused": True},
            "context_relevance": {"can_answer_confidently": True}
        }

        _, metadata = generate_with_reasoning(
            query="test", context="", chat_history=[], role="",
            reasoning=reasoning, llm=mock_llm
        )

        assert metadata["depth_level"] == 3
        assert metadata["user_confused"] is True
        assert "generation_time_ms" in metadata


class TestChainOfThoughtGenerate:
    """Integration tests for the full CoT pipeline."""

    @patch('assistant.core.rag_factory.RagEngineFactory')
    def test_full_pipeline_returns_answer(self, mock_factory):
        """Full pipeline should return answer with reasoning metadata."""
        # Setup mock
        mock_rag_engine = MagicMock()
        mock_llm = MagicMock()
        mock_rag_engine.llm = mock_llm
        mock_factory.return_value.create.return_value = (None, mock_rag_engine)

        # Mock LLM responses
        mock_llm.predict.side_effect = [
            # First call: reasoning
            json.dumps({
                "user_intent": {"explicit": "test", "confidence": 0.9},
                "clarification_needed": {"needed": False},
                "response_plan": {"depth_level": 2, "style": "balanced"},
                "user_state": {"seems_confused": False},
                "context_relevance": {"can_answer_confidently": True}
            }),
            # Second call: generation
            "Here is the answer to your question."
        ]

        state = {
            "query": "How does RAG work?",
            "chat_history": [],
            "role": "Software Developer",
            "retrieved_chunks": [{"content": "RAG info", "section": "Architecture"}]
        }

        result = chain_of_thought_generate(state, mock_rag_engine)

        assert "draft_answer" in result
        assert "cot_reasoning" in result
        assert "cot_metadata" in result
        assert result["cot_enabled"] is True

    @patch('assistant.core.rag_factory.RagEngineFactory')
    def test_returns_clarification_flag(self, mock_factory):
        """Should set clarification_needed flag when reasoning determines so."""
        mock_rag_engine = MagicMock()
        mock_llm = MagicMock()
        mock_rag_engine.llm = mock_llm
        mock_factory.return_value.create.return_value = (None, mock_rag_engine)

        mock_llm.predict.return_value = json.dumps({
            "user_intent": {"explicit": "test"},
            "clarification_needed": {
                "needed": True,
                "suggested_question": "What specifically about X?"
            },
            "response_plan": {"depth_level": 2}
        })

        state = {"query": "Tell me about X", "chat_history": [], "role": "", "retrieved_chunks": []}
        result = chain_of_thought_generate(state, mock_rag_engine)

        assert result["clarification_needed"] is True
        assert result["clarifying_question"] == "What specifically about X?"


class TestShouldUseChainOfThought:
    """Tests for the CoT trigger logic in stage5_generation_nodes."""

    def test_import_works(self):
        """Should be able to import _should_use_chain_of_thought."""
        from assistant.flows.node_logic.stage5_generation_nodes import _should_use_chain_of_thought
        assert callable(_should_use_chain_of_thought)

    def test_triggers_for_enterprise_keywords(self):
        """Should trigger for enterprise-related queries."""
        from assistant.flows.node_logic.stage5_generation_nodes import _should_use_chain_of_thought

        enterprise_queries = [
            "How would I adapt this for customer support?",
            "Can this scale to enterprise use cases?",
            "How do I customize this for production?",
            "What would I need to deploy this?"
        ]

        for query in enterprise_queries:
            state = {"query": query, "chat_history": []}
            assert _should_use_chain_of_thought(state) is True, f"Failed for: {query}"

    def test_triggers_for_explanation_requests(self):
        """Should trigger for explicit explanation requests."""
        from assistant.flows.node_logic.stage5_generation_nodes import _should_use_chain_of_thought

        state = {"query": "Explain how the retrieval system works", "chat_history": []}
        assert _should_use_chain_of_thought(state) is True

    def test_triggers_for_deep_conversations(self):
        """Should trigger for deep conversations (10+ messages)."""
        from assistant.flows.node_logic.stage5_generation_nodes import _should_use_chain_of_thought

        history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
                   for i in range(12)]
        state = {"query": "Simple question", "chat_history": history}
        assert _should_use_chain_of_thought(state) is True

    def test_does_not_trigger_for_simple_queries(self):
        """Should not trigger for simple queries."""
        from assistant.flows.node_logic.stage5_generation_nodes import _should_use_chain_of_thought

        state = {"query": "What is your tech stack?", "chat_history": []}
        assert _should_use_chain_of_thought(state) is False
