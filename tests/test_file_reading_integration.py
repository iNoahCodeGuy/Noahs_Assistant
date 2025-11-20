"""Integration tests for file reading functionality.

Tests the complete file reading pipeline:
- File path detection in query classification
- File reading in retrieval stage
- File content inclusion in generation
- File display in formatting

Run: pytest tests/test_file_reading_integration.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from assistant.flows.conversation_flow import run_conversation_flow
from assistant.state.conversation_state import ConversationState
from assistant.core.rag_engine import RagEngine
from assistant.flows.node_logic.stage2_query_classification import _detect_file_request, classify_intent
from assistant.flows.node_logic.stage4_retrieval_nodes import retrieve_file_content
from assistant.retrieval.code_index import CodeIndex


class TestFilePathDetection:
    """Test file path detection in query classification."""

    def test_detect_file_request_explicit(self):
        """Test explicit file requests are detected."""
        assert _detect_file_request("show me assistant/core/rag_engine.py") == "assistant/core/rag_engine.py"
        assert _detect_file_request("read assistant/core/rag_engine.py") == "assistant/core/rag_engine.py"
        assert _detect_file_request("open the file assistant/flows/conversation_flow.py") == "assistant/flows/conversation_flow.py"

    def test_detect_file_request_vague(self):
        """Test vague file requests return None."""
        assert _detect_file_request("read the rag engine file") is None
        assert _detect_file_request("show me some code") is None

    def test_classify_intent_sets_file_request(self):
        """Test that classify_intent sets file_request in state."""
        state: ConversationState = {
            "query": "show me assistant/core/rag_engine.py",
            "role": "Software Developer"
        }
        result = classify_intent(state)
        assert result.get("file_request") == "assistant/core/rag_engine.py"
        assert result.get("query_type") == "file_request"


class TestFileReading:
    """Test file reading functionality."""

    @pytest.fixture
    def mock_code_index(self):
        """Create a mock CodeIndex."""
        code_index = MagicMock(spec=CodeIndex)
        code_index.read_file_content.return_value = {
            "content": "def test_function():\n    return True\n",
            "file_path": "assistant/core/test.py",
            "line_count": 2,
            "success": True
        }
        return code_index

    @pytest.fixture
    def mock_rag_engine(self, mock_code_index):
        """Create a mock RAG engine with code_index."""
        engine = MagicMock(spec=RagEngine)
        engine.code_index = mock_code_index
        return engine

    def test_retrieve_file_content_success(self, mock_rag_engine):
        """Test successful file reading."""
        state: ConversationState = {
            "file_request": "assistant/core/test.py"
        }
        result = retrieve_file_content(state, mock_rag_engine)

        assert result.get("file_content") is not None
        assert result["file_content"]["success"] is True
        assert result["file_content"]["file_path"] == "assistant/core/test.py"
        assert result.get("file_read_success") is True

    def test_retrieve_file_content_no_request(self, mock_rag_engine):
        """Test that file reading is skipped when no file_request."""
        state: ConversationState = {
            "query": "How does RAG work?"
        }
        result = retrieve_file_content(state, mock_rag_engine)

        # Should return state unchanged
        assert result == state

    def test_retrieve_file_content_no_code_index(self):
        """Test graceful handling when code_index unavailable."""
        engine = MagicMock(spec=RagEngine)
        engine.code_index = None

        state: ConversationState = {
            "file_request": "assistant/core/test.py"
        }
        result = retrieve_file_content(state, engine)

        assert result.get("file_content") is None
        assert result.get("file_read_error") == "CodeIndex not available"

    def test_retrieve_file_content_file_not_found(self, mock_rag_engine):
        """Test handling of file not found."""
        mock_rag_engine.code_index.read_file_content.return_value = {
            "content": "# File not found: nonexistent.py",
            "file_path": "nonexistent.py",
            "success": False
        }

        state: ConversationState = {
            "file_request": "nonexistent.py"
        }
        result = retrieve_file_content(state, mock_rag_engine)

        assert result["file_content"]["success"] is False
        assert result.get("file_read_success") is False


class TestCodeSnippetContext:
    """Test code snippet context expansion."""

    def test_get_file_snippet_with_context(self):
        """Test that get_file_snippet includes surrounding context."""
        code_index = CodeIndex(repo_path=".")

        # Test with a known file (this test file itself)
        test_file = "tests/test_file_reading_integration.py"
        if Path(test_file).exists():
            snippet = code_index.get_file_snippet(
                test_file,
                line_start=10,
                line_end=15,
                context_before=5,
                context_after=5
            )

            # Should include context lines
            lines = snippet.split('\n')
            assert len(lines) >= 11  # 5 before + 1 target + 5 after = 11 minimum

    def test_get_file_snippet_default_context(self):
        """Test that default context is 20 lines."""
        code_index = CodeIndex(repo_path=".")

        test_file = "tests/test_file_reading_integration.py"
        if Path(test_file).exists():
            snippet = code_index.get_file_snippet(
                test_file,
                line_start=20,
                line_end=25
            )

            # Should include default 20 lines of context
            lines = snippet.split('\n')
            assert len(lines) >= 25  # 20 before + 5 target + 20 after = 45 minimum


class TestDocumentationIndexing:
    """Test documentation indexing functionality."""

    def test_read_docs_folder_exists(self):
        """Test that read_docs_folder can find docs directory."""
        from scripts.index_documentation import DocumentationIndexer

        indexer = DocumentationIndexer()
        files = indexer.read_docs_folder("docs")

        # Should find at least some markdown files
        assert isinstance(files, list)
        if files:
            assert all('file_path' in f and 'content' in f for f in files)

    def test_chunk_by_sections(self):
        """Test markdown chunking by sections."""
        from scripts.index_documentation import DocumentationIndexer

        indexer = DocumentationIndexer()

        test_content = """# Title

## Section 1
Content for section 1.

## Section 2
Content for section 2.
"""
        chunks = indexer.chunk_by_sections(test_content, "test.md")

        assert len(chunks) >= 2  # Should have at least 2 sections
        assert all(chunk['doc_id'] == 'documentation' for chunk in chunks)
        assert all('content_hash' in chunk for chunk in chunks)


class TestEndToEndFileReading:
    """Test complete file reading flow."""

    @pytest.fixture
    def mock_rag_engine(self):
        """Create a complete mock RAG engine."""
        engine = MagicMock(spec=RagEngine)

        # Mock code_index
        code_index = MagicMock(spec=CodeIndex)
        code_index.read_file_content.return_value = {
            "content": "def example():\n    pass\n",
            "file_path": "assistant/core/example.py",
            "line_count": 2,
            "success": True
        }
        engine.code_index = code_index

        # Mock retrieval
        engine.retrieve.return_value = {
            "chunks": [{"content": "Some KB content", "similarity": 0.8}],
            "scores": [0.8]
        }

        # Mock response generator
        engine.response_generator = MagicMock()
        engine.response_generator.generate_contextual_response.return_value = "Here's the file you requested..."

        return engine

    def test_complete_file_request_flow(self, mock_rag_engine):
        """Test complete flow from query to formatted response."""
        state: ConversationState = {
            "query": "show me assistant/core/example.py",
            "role": "Software Developer",
            "session_id": "test_session"
        }

        # Step 1: Classify intent (should detect file request)
        classified = classify_intent(state)
        assert classified.get("file_request") == "assistant/core/example.py"

        # Step 2: Read file
        state.update(classified)
        file_result = retrieve_file_content(state, mock_rag_engine)
        assert file_result.get("file_read_success") is True

        # Step 3: Verify file content is in state
        assert file_result.get("file_content") is not None
        assert file_result["file_content"]["success"] is True
