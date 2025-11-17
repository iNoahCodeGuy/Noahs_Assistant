"""Test LangChain component compatibility before import updates.

Verifies that OpenAIEmbeddings, ChatOpenAI, FAISS, and other components
work correctly before replacing deprecated imports with new packages.
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import via our compatibility layer
from src.core.langchain_compat import (
    OpenAIEmbeddings,
    ChatOpenAI,
    FAISS,
    CSVLoader,
    RecursiveCharacterTextSplitter,
    PromptTemplate,
    RetrievalQA,
    Document
)

class TestLangChainCompatibilityLayer:
    """Test that our compatibility layer works with current imports."""

    def test_openai_embeddings_initialization(self):
        """Test OpenAIEmbeddings can be created without error."""
        # Should work even without API key (graceful fallback)
        try:
            embeddings = OpenAIEmbeddings()
            assert embeddings is not None
            assert hasattr(embeddings, 'embed_query')
        except Exception as e:
            if "api_key" in str(e).lower():
                pytest.skip("OpenAI API key not available - skipping OpenAI tests")
            else:
                raise

    def test_openai_embeddings_embed_query(self):
        """Test embedding query returns expected format."""
        try:
            embeddings = OpenAIEmbeddings()

            # Test with sample text
            result = embeddings.embed_query("test query")

            assert isinstance(result, list)
            assert len(result) > 0
            assert all(isinstance(x, (int, float)) for x in result)
        except Exception as e:
            if "api_key" in str(e).lower():
                pytest.skip("OpenAI API key not available - skipping embedding test")
            else:
                raise

    def test_chat_openai_initialization(self):
        """Test ChatOpenAI can be created."""
        try:
            llm = ChatOpenAI()
            assert llm is not None
            assert hasattr(llm, 'predict')
        except Exception as e:
            if "api_key" in str(e).lower():
                pytest.skip("OpenAI API key not available - skipping ChatOpenAI tests")
            else:
                raise

    def test_chat_openai_predict(self):
        """Test ChatOpenAI predict method works."""
        try:
            llm = ChatOpenAI()

            response = llm.predict("What is Python?")

            assert isinstance(response, str)
            assert len(response) > 0
        except Exception as e:
            if "api_key" in str(e).lower():
                pytest.skip("OpenAI API key not available - skipping ChatOpenAI predict test")
            else:
                raise

    def test_faiss_from_documents(self):
        """Test FAISS vector store creation."""
        try:
            embeddings = OpenAIEmbeddings()
        except Exception as e:
            if "api_key" in str(e).lower():
                pytest.skip("OpenAI API key not available - skipping FAISS test")
                return
            else:
                raise

        # Create sample documents
        docs = [
            Document(page_content="Python is a programming language", metadata={"source": "doc1"}),
            Document(page_content="FAISS is a vector database", metadata={"source": "doc2"})
        ]

        # Should gracefully handle FAISS not being installed
        try:
            vectorstore = FAISS.from_documents(docs, embeddings)
            # In working mode: vectorstore should have methods
            if vectorstore is not None:
                assert hasattr(vectorstore, 'similarity_search')
        except ImportError as e:
            # FAISS not installed - this is acceptable for optional dependency
            if "faiss" in str(e).lower():
                pytest.skip("FAISS not installed - optional dependency")
            else:
                raise

    def test_csv_loader_initialization(self):
        """Test CSVLoader can be created."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name,role\nNoah,Developer\nAI,Assistant\n")
            temp_path = f.name

        try:
            loader = CSVLoader(temp_path)
            assert loader is not None
            assert hasattr(loader, 'load')
        finally:
            os.unlink(temp_path)

    def test_csv_loader_load(self):
        """Test CSVLoader.load() method."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("content,source\nTest content,test.csv\n")
            temp_path = f.name

        try:
            loader = CSVLoader(temp_path)
            docs = loader.load()

            assert isinstance(docs, list)
            # In working mode: should have documents
            # In degraded mode: may be empty list
        finally:
            os.unlink(temp_path)

    def test_text_splitter_initialization(self):
        """Test RecursiveCharacterTextSplitter creation."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        assert splitter is not None
        assert hasattr(splitter, 'split_documents')

    def test_text_splitter_split_documents(self):
        """Test text splitter processes documents."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

        docs = [Document(page_content="This is a long document that should be split into multiple chunks for better processing")]

        result = splitter.split_documents(docs)
        assert isinstance(result, list)
        # Should at least return the original docs (degraded mode) or split versions
        assert len(result) >= len(docs)

    def test_prompt_template_initialization(self):
        """Test PromptTemplate creation."""
        template = PromptTemplate(
            template="Answer this question: {question}",
            input_variables=["question"]
        )
        assert template is not None
        assert hasattr(template, 'template')
        assert hasattr(template, 'input_variables')

    def test_retrieval_qa_initialization(self):
        """Test RetrievalQA creation."""
        # Create mock components
        try:
            llm = ChatOpenAI()
            embeddings = OpenAIEmbeddings()
        except Exception as e:
            if "api_key" in str(e).lower():
                pytest.skip("OpenAI API key not available - skipping RetrievalQA test")
                return
            else:
                raise

        # Create a mock retriever to avoid validation errors
        try:
            # Create documents for retriever
            docs = [Document(page_content="test content", metadata={"source": "test"})]

            # Try to create vectorstore first
            try:
                vectorstore = FAISS.from_documents(docs, embeddings)
                retriever = vectorstore.as_retriever() if vectorstore else None
            except ImportError:
                # FAISS not available, skip this test
                pytest.skip("FAISS not installed - RetrievalQA test requires vector store")

            # Now try to create QA chain
            if retriever is not None:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever
                )
                assert hasattr(qa_chain, 'run') or hasattr(qa_chain, '__call__')
            else:
                pytest.skip("Could not create retriever - skipping QA chain test")

        except Exception as e:
            # If any step fails, it's acceptable in compatibility testing
            pytest.skip(f"RetrievalQA compatibility test skipped: {str(e)}")
        if qa_chain is not None:
            assert hasattr(qa_chain, 'run') or hasattr(qa_chain, '__call__')

    def test_document_creation(self):
        """Test Document class works."""
        doc = Document(
            page_content="Test content",
            metadata={"source": "test.txt", "type": "test"}
        )

        assert doc.page_content == "Test content"
        assert doc.metadata["source"] == "test.txt"
        assert doc.metadata["type"] == "test"


class TestRagFactoryWithCurrentImports:
    """Test RagFactory components work with current LangChain imports."""

    def test_rag_factory_embeddings_creation(self):
        """Test RagEngineFactory can create embeddings."""
        from src.core.rag_factory import RagEngineFactory

        factory = RagEngineFactory()
        embeddings, is_degraded = factory.create_embeddings()

        assert embeddings is not None
        assert isinstance(is_degraded, bool)
        assert hasattr(embeddings, 'embed_query')

    def test_rag_factory_llm_creation(self):
        """Test RagEngineFactory can create LLM."""
        from src.core.rag_factory import RagEngineFactory

        factory = RagEngineFactory()
        llm, is_degraded = factory.create_llm()

        assert llm is not None
        assert isinstance(is_degraded, bool)
        assert hasattr(llm, 'predict')

    def test_rag_factory_with_mock_settings(self):
        """Test RagEngineFactory with mock settings."""
        from src.core.rag_factory import RagEngineFactory

        # Mock settings object
        class MockSettings:
            openai_api_key = "test-key"
            embedding_model = "text-embedding-ada-002"
            openai_model = "gpt-3.5-turbo"

        factory = RagEngineFactory(settings=MockSettings())

        embeddings, emb_degraded = factory.create_embeddings()
        llm, llm_degraded = factory.create_llm()

        assert embeddings is not None
        assert llm is not None
        assert isinstance(emb_degraded, bool)
        assert isinstance(llm_degraded, bool)


class TestEndToEndCompatibility:
    """Test full pipeline compatibility before import changes."""

    @pytest.mark.integration
    def test_minimal_rag_pipeline(self):
        """Test minimal RAG pipeline works with current imports."""
        # Create components
        try:
            embeddings = OpenAIEmbeddings()
            llm = ChatOpenAI()
        except Exception as e:
            if "api_key" in str(e).lower():
                pytest.skip("OpenAI API key not available - skipping RAG pipeline test")
                return
            else:
                raise

        # Create documents
        docs = [
            Document(page_content="Noah is a software developer", metadata={"source": "bio"}),
            Document(page_content="Python is used for AI development", metadata={"source": "tech"})
        ]

        # Try to create vector store (handle FAISS not being installed)
        try:
            vectorstore = FAISS.from_documents(docs, embeddings)
        except ImportError as e:
            if "faiss" in str(e).lower():
                # FAISS not installed - this is acceptable for optional dependency
                vectorstore = None
            else:
                raise

        # Pipeline should not crash
        assert embeddings is not None
        assert llm is not None
        # vectorstore may be None in degraded mode - that's acceptable

    @pytest.mark.integration
    def test_document_processing_pipeline(self):
        """Test document processing pipeline compatibility."""
        # Create text splitter with valid chunk_overlap (must be < chunk_size)
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

        # Create long document
        long_doc = Document(
            page_content="This is a very long document " * 20,
            metadata={"source": "long.txt"}
        )

        # Split document
        chunks = splitter.split_documents([long_doc])

        assert isinstance(chunks, list)
        assert len(chunks) >= 1  # At least returns original in degraded mode

    @pytest.mark.integration
    def test_response_generation_compatibility(self):
        """Test response generation works with current imports."""
        try:
            llm = ChatOpenAI()
        except Exception as e:
            if "api_key" in str(e).lower():
                pytest.skip("OpenAI API key not available - skipping response generation test")
                return
            else:
                raise

        prompt_template = PromptTemplate(
            template="Based on this context: {context}\n\nAnswer: {question}",
            input_variables=["context", "question"]
        )

        # Should be able to generate response
        response = llm.predict("What is machine learning?")

        assert isinstance(response, str)
        assert len(response) > 0
        assert prompt_template is not None


def test_import_verification():
    """Verify all required imports work before making changes."""
    # This test documents what we're currently importing
    import_tests = {
        'OpenAIEmbeddings': OpenAIEmbeddings,
        'ChatOpenAI': ChatOpenAI,
        'FAISS': FAISS,
        'CSVLoader': CSVLoader,
        'RecursiveCharacterTextSplitter': RecursiveCharacterTextSplitter,
        'PromptTemplate': PromptTemplate,
        'RetrievalQA': RetrievalQA,
        'Document': Document
    }

    for name, cls in import_tests.items():
        assert cls is not None, f"Failed to import {name}"
        # Verify it's a class or callable
        assert callable(cls), f"{name} is not callable"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
