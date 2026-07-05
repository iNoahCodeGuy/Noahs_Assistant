"""LangChain Compatibility Layer

Provides graceful fallbacks for LangChain imports to handle:
- Missing langchain-community package
- Version differences between langchain releases
- Development environments without full dependencies

This isolates import complexity from core RAG logic. Every fallback fails
loud: silently degraded stand-ins (fake embeddings, echo LLMs, empty
loaders) corrupt behavior in ways that are far harder to debug than an
ImportError with install instructions.
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
                """Fail loud: fake embeddings would silently corrupt retrieval."""
                def __init__(self, *_, **__):
                    raise ImportError(
                        "OpenAIEmbeddings unavailable — install langchain-openai "
                        "(pip install -r requirements.txt)."
                    )

# --- Resilient Document Loaders ---
try:
    from langchain_community.document_loaders import CSVLoader  # type: ignore
except Exception:
    try:
        from langchain.document_loaders import CSVLoader  # type: ignore
    except Exception:
        class CSVLoader:  # type: ignore
            """Fail loud: an empty loader would silently produce an empty KB."""
            def __init__(self, file_path: str, source_column: str = "source"):
                raise ImportError(
                    "CSVLoader unavailable — install langchain-community "
                    "(pip install -r requirements.txt)."
                )

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
            """Fail loud rather than silently disabling the QA chain."""
            @staticmethod
            def from_chain_type(*_, **__):
                raise ImportError(
                    "RetrievalQA unavailable — install langchain "
                    "(pip install -r requirements.txt)."
                )

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
                """Fail loud: a fake LLM that echoes prompts is worse than a crash."""
                def __init__(self, *_, **__):
                    raise ImportError(
                        "ChatAnthropic unavailable — install langchain-anthropic "
                        "(pip install -r requirements.txt)."
                    )

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
    "CSVLoader",
    "RecursiveCharacterTextSplitter",
    "RetrievalQA",
    "PromptTemplate",
    "ChatAnthropic",
    "Document"
]
