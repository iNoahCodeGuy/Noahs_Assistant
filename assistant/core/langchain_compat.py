"""LangChain Compatibility Layer

Provides graceful fallbacks for LangChain imports to handle:
- Missing langchain-community package
- Version differences between langchain releases
- Development environments without full dependencies

This isolates import complexity from core RAG logic.

**Note**: FAISS imports have been removed as the system now uses pgvector exclusively.
FAISS stubs are kept for backwards compatibility with old code/tests.
"""
from __future__ import annotations

import os
from typing import List, Any

# --- Resilient OpenAI Embeddings ---
try:
    from langchain_openai import OpenAIEmbeddings  # type: ignore
except Exception:
    try:
        from langchain.embeddings import OpenAIEmbeddings  # type: ignore
    except Exception:
        try:
            from langchain_community.embeddings import OpenAIEmbeddings  # type: ignore
        except Exception:
            class OpenAIEmbeddings:  # type: ignore
                def __init__(self, *_, **__):
                    pass
                def embed_query(self, text: str) -> List[float]:
                    return [float((hash(text) >> i) & 0xFF) / 255.0 for i in range(0, 32)]

# --- FAISS Stub (Deprecated - kept for backwards compatibility) ---
# FAISS has been removed from production. Use pgvector instead.
# This stub exists only to prevent import errors in legacy code.
class _StubVectorStore:
    """Deprecated: Use pgvector instead."""
    def __init__(self, *_, **__):
        raise RuntimeError("FAISS is no longer supported. Use pgvector retrieval instead.")
    def similarity_search(self, *_, **__):
        raise RuntimeError("FAISS is no longer supported. Use pgvector retrieval instead.")
    def save_local(self, *_, **__):
        raise RuntimeError("FAISS is no longer supported. Use pgvector retrieval instead.")
    def as_retriever(self, *_, **__):
        raise RuntimeError("FAISS is no longer supported. Use pgvector retrieval instead.")

class FAISS:  # type: ignore
    """Deprecated: FAISS support removed. Use pgvector instead."""
    @staticmethod
    def from_documents(_docs, _emb):
        raise RuntimeError("FAISS is no longer supported. Use pgvector retrieval instead.")
    @staticmethod
    def load_local(*_, **__):
        raise RuntimeError("FAISS is no longer supported. Use pgvector retrieval instead.")

# --- Resilient Document Loaders ---
try:
    from langchain_community.document_loaders import CSVLoader  # type: ignore
except Exception:
    try:
        from langchain.document_loaders import CSVLoader  # type: ignore
    except Exception:
        class CSVLoader:  # type: ignore
            def __init__(self, file_path: str, source_column: str = "source"):
                self.file_path = file_path
                self.source_column = source_column
            def load(self):
                return []

# --- Resilient Text Splitter ---
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
except Exception:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
    except Exception:
        class RecursiveCharacterTextSplitter:  # type: ignore
            def __init__(self, chunk_size: int = 600, chunk_overlap: int = 60, **__):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
            def split_documents(self, docs):
                return docs

# --- Resilient QA Chain ---
try:
    from langchain_community.chains import RetrievalQA  # type: ignore
except Exception:
    try:
        from langchain.chains import RetrievalQA  # type: ignore
    except Exception:
        class RetrievalQA:  # type: ignore
            @staticmethod
            def from_chain_type(*_, **__):
                return None

# --- Resilient Prompt Template ---
try:
    from langchain.prompts import PromptTemplate  # type: ignore
except Exception:
    try:
        from langchain_core.prompts import PromptTemplate  # type: ignore
    except Exception:
        class PromptTemplate:  # type: ignore
            def __init__(self, template: str, input_variables: List[str]):
                self.template = template
                self.input_variables = input_variables

# --- Resilient ChatAnthropic ---
try:
    from langchain_anthropic import ChatAnthropic  # type: ignore
except Exception:
    try:
        from langchain.chat_models import ChatAnthropic  # type: ignore
    except Exception:
        try:
            from langchain_community.chat_models import ChatAnthropic  # type: ignore
        except Exception:
            class ChatAnthropic:  # type: ignore
                """Fallback ChatAnthropic that implements both legacy and modern interfaces."""
                def __init__(self, *_, **__):
                    pass

                def predict(self, prompt: str) -> str:
                    words = prompt.strip().split()
                    tail = " ".join(words[-40:])
                    return f"[DEGRADED MODE SYNTHESIS]\n{tail}"

                def invoke(self, input_data, **kwargs):
                    """Modern LangChain interface - invoke method."""
                    try:
                        from langchain_core.messages import AIMessage
                    except ImportError:
                        class AIMessage:
                            def __init__(self, content):
                                self.content = content

                    # Handle different input types
                    if isinstance(input_data, str):
                        prompt = input_data
                    elif isinstance(input_data, list):
                        # List of messages - extract content
                        prompt = " ".join(
                            getattr(m, 'content', str(m)) for m in input_data
                        )
                    else:
                        prompt = str(input_data)

                    result = self.predict(prompt)
                    return AIMessage(content=result)

                def __call__(self, *args, **kwargs):
                    """Make callable for legacy compatibility."""
                    return self.invoke(*args, **kwargs)

# Legacy alias for backwards compatibility
ChatOpenAI = ChatAnthropic

# --- Document Schema ---
try:
    from langchain.schema import Document  # type: ignore
except Exception:
    try:
        from langchain_core.documents import Document  # type: ignore
    except Exception:
        class Document:  # type: ignore
            def __init__(self, page_content: str, metadata: dict = None):
                self.page_content = page_content
                self.metadata = metadata or {}

__all__ = [
    "OpenAIEmbeddings",
    "FAISS",
    "CSVLoader",
    "RecursiveCharacterTextSplitter",
    "RetrievalQA",
    "PromptTemplate",
    "ChatOpenAI",
    "Document"
]
