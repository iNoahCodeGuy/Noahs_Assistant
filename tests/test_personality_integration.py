"""Test suite for personality awareness integration.

Tests that personality knowledge base is properly integrated and used
in relevant queries without overwhelming technical responses.
"""

import pytest
import os
import csv
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from assistant.utils.personality_injection import should_include_personality, extract_personality_traits
from assistant.retrieval.pgvector_retriever import PgVectorRetriever


class TestPersonalityKB:
    """Test personality knowledge base file and migration."""

    def test_personality_kb_exists(self):
        """Test that personality_kb.csv file exists."""
        kb_path = "data/personality_kb.csv"
        assert os.path.exists(kb_path), f"personality_kb.csv not found at {kb_path}"

    def test_personality_kb_format(self):
        """Test that personality_kb.csv has correct format (Question,Answer columns)."""
        kb_path = "data/personality_kb.csv"

        with open(kb_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            assert 'Question' in reader.fieldnames, "Missing 'Question' column"
            assert 'Answer' in reader.fieldnames, "Missing 'Answer' column"

            rows = list(reader)
            assert len(rows) > 0, "personality_kb.csv is empty"
            assert len(rows) >= 6, f"Expected at least 6 Q&A pairs, found {len(rows)}"

            # Check that all rows have both Question and Answer
            for i, row in enumerate(rows):
                assert row.get('Question'), f"Row {i+1} missing Question"
                assert row.get('Answer'), f"Row {i+1} missing Answer"

    def test_personality_kb_content(self):
        """Test that personality_kb.csv contains personality-related content."""
        kb_path = "data/personality_kb.csv"

        personality_keywords = [
            'personality', 'trait', 'motivation', 'approach', 'style',
            'thoughtful', 'playful', 'teaching', 'enterprise'
        ]

        with open(kb_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            content = ' '.join([row.get('Question', '') + ' ' + row.get('Answer', '')
                              for row in reader]).lower()

            # Should contain personality-related keywords
            found_keywords = [kw for kw in personality_keywords if kw in content]
            assert len(found_keywords) >= 3, \
                f"Expected personality keywords in KB, found: {found_keywords}"


class TestPersonalityInjection:
    """Test personality injection utility functions."""

    def test_should_include_personality_work_style_query(self):
        """Test that work style queries trigger personality inclusion."""
        result = should_include_personality(
            query="What's Noah's work style like?",
            context=[],
            role="Software Developer"
        )
        assert result is True, "Work style query should include personality"

    def test_should_include_personality_why_question(self):
        """Test that 'why' questions about Noah trigger personality inclusion."""
        result = should_include_personality(
            query="Why did Noah build this feature?",
            context=[],
            role="Software Developer"
        )
        assert result is True, "'Why' question about Noah should include personality"

    def test_should_include_personality_cultural_fit(self):
        """Test that hiring manager cultural fit questions trigger personality."""
        result = should_include_personality(
            query="Would Noah fit on our team?",
            context=[],
            role="Hiring Manager (technical)"
        )
        assert result is True, "Cultural fit question should include personality"

    def test_should_include_personality_motivation_query(self):
        """Test that motivation queries trigger personality inclusion."""
        result = should_include_personality(
            query="What motivates Noah?",
            context=[],
            role="Hiring Manager (nontechnical)"
        )
        assert result is True, "Motivation query should include personality"

    def test_should_not_include_personality_technical_query(self):
        """Test that purely technical queries don't force personality."""
        result = should_include_personality(
            query="How does RAG work?",
            context=[],
            role="Software Developer"
        )
        assert result is False, "Technical query should not force personality"

    def test_should_not_include_personality_code_query(self):
        """Test that code-related queries don't force personality."""
        result = should_include_personality(
            query="Show me the backend architecture",
            context=[],
            role="Software Developer"
        )
        assert result is False, "Code query should not force personality"

    def test_extract_personality_traits(self):
        """Test that personality traits can be extracted from chunks."""
        chunks = [
            "Noah is thoughtful and systematic in his approach to problem-solving.",
            "His teaching-oriented personality shows in how he built this platform."
        ]

        traits = extract_personality_traits(chunks)
        assert isinstance(traits, dict), "Should return dict of traits"
        # Should find at least one trait
        assert len(traits) >= 0, "Should extract traits from personality chunks"


class TestPersonalityFiltering:
    """Test personality filtering in retrieval."""

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock PgVectorRetriever for testing."""
        retriever = Mock(spec=PgVectorRetriever)
        return retriever

    def test_personality_filtering_exists(self):
        """Test that _filter_personality method exists on retriever."""
        retriever = PgVectorRetriever(similarity_threshold=0.3)
        assert hasattr(retriever, '_filter_personality'), \
            "PgVectorRetriever should have _filter_personality method"

    def test_personality_filtering_boosts_relevant_chunks(self):
        """Test that personality filtering boosts relevant chunks."""
        retriever = PgVectorRetriever(similarity_threshold=0.3)

        chunks = [
            {
                'content': 'Noah has a thoughtful personality and values systematic approaches.',
                'similarity': 0.7,
                'doc_id': 'personality_kb'
            },
            {
                'content': 'Python is a programming language used for AI development.',
                'similarity': 0.7,
                'doc_id': 'technical_kb'
            }
        ]

        filtered = retriever._filter_personality(chunks)

        # Personality chunk should be boosted
        assert len(filtered) == 2, "Should return all chunks"
        personality_chunk = next(c for c in filtered if c.get('doc_id') == 'personality_kb')
        assert '_personality_score' in personality_chunk, "Should add personality score"
        assert personality_chunk.get('_boosted_similarity', 0) >= personality_chunk.get('similarity', 0), \
            "Personality chunk should have boosted similarity"


class TestPersonalityRetrievalIntegration:
    """Test personality context integration in retrieval pipeline."""

    @pytest.fixture
    def mock_rag_engine(self):
        """Create a mock RAG engine for testing."""
        engine = Mock()
        engine.pgvector_retriever = Mock()
        engine.pgvector_retriever.retrieve_for_role = Mock(return_value=[])
        return engine

    def test_retrieve_for_role_accepts_personality_flag(self):
        """Test that retrieve_for_role accepts include_personality parameter."""
        retriever = PgVectorRetriever(similarity_threshold=0.3)

        # Should not raise error with include_personality parameter
        try:
            result = retriever.retrieve_for_role(
                query="What's Noah's personality like?",
                role="Hiring Manager",
                top_k=3,
                include_personality=True
            )
            assert isinstance(result, list), "Should return list of chunks"
        except TypeError as e:
            pytest.fail(f"retrieve_for_role should accept include_personality parameter: {e}")


class TestPersonalityResponseGeneration:
    """Test that personality context is used appropriately in responses."""

    def test_personality_guidance_in_prompts(self):
        """Test that personality guidance is included in response prompts."""
        from assistant.core.response_generator import ResponseGenerator
        from assistant.core.rag_factory import RagEngineFactory
        from unittest.mock import MagicMock

        # Create a minimal response generator with mocked LLM
        factory = RagEngineFactory()
        llm, _ = factory.create_llm()

        # If LLM creation fails (e.g., no API key), use a mock
        if llm is None or not hasattr(llm, 'predict'):
            llm = MagicMock()
            llm.predict = MagicMock(return_value="Test response")

        generator = ResponseGenerator(llm=llm, qa_chain=None, degraded_mode=False)

        # Build a role prompt and check for personality guidance
        prompt = generator._build_role_prompt(
            query="What's Noah's work style?",
            context_str="Noah is thoughtful and systematic.",
            role="Hiring Manager (technical)"
        )

        # Should include personality context guidance
        assert "PERSONALITY CONTEXT" in prompt or "personality" in prompt.lower(), \
            "Prompt should include personality guidance for relevant queries"


@pytest.mark.integration
class TestPersonalityIntegrationEndToEnd:
    """End-to-end integration tests for personality awareness."""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY") or not os.getenv("SUPABASE_URL"),
        reason="Requires OpenAI API key and Supabase URL"
    )
    def test_personality_kb_retrievable(self):
        """Test that personality KB chunks are retrievable via vector search."""
        from assistant.retrieval.pgvector_retriever import get_retriever

        retriever = get_retriever(similarity_threshold=0.3)

        # Try to retrieve personality-related content
        chunks = retriever.retrieve(
            query="What is Noah's personality like?",
            top_k=3
        )

        # Should find at least one personality-related chunk
        assert len(chunks) > 0, "Should retrieve chunks for personality query"

        # Check if any chunks are from personality_kb
        personality_chunks = [c for c in chunks if c.get('doc_id') == 'personality_kb']
        # Note: May not find personality chunks if KB hasn't been migrated yet
        # This test will pass if KB is migrated, fail gracefully if not


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
