"""Document Processing Pipeline

Handles document loading, chunking, and preparation for RAG systems.
Supports multiple document sources and formats with consistent output.
"""
from __future__ import annotations

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .langchain_compat import Document, CSVLoader, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 60):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_from_csv(self, file_path: str, source_column: str = "Question") -> List[Document]:
        """Load documents from CSV file."""
        if not os.path.exists(file_path):
            logger.warning(f"CSV file not found: {file_path}")
            return []

        try:
            loader = CSVLoader(file_path=file_path, source_column=source_column)
            raw_docs = loader.load()
            return self._split_documents(raw_docs)
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return []

    def load_from_career_kb(self, career_kb, chunk: bool = True) -> List[Document]:
        """Load documents from CareerKnowledgeBase object."""
        try:
            rows = career_kb.get_all_entries()
            docs = []
            for row in rows:
                q = row.get("Question") or row.get("question") or ""
                a = row.get("Answer") or row.get("answer") or ""
                content = f"Q: {q}\nA: {a}".strip()
                if content:
                    docs.append(Document(page_content=content, metadata={"source": "career_kb"}))

            return self._split_documents(docs) if chunk else docs
        except Exception as e:
            logger.error(f"Error loading from career KB: {e}")
            return []

    def load_from_text_files(self, directory: str, pattern: str = "*.txt") -> List[Document]:
        """Load documents from text files in directory."""
        docs = []
        try:
            path = Path(directory)
            for file_path in path.glob(pattern):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    docs.append(Document(
                        page_content=content,
                        metadata={"source": str(file_path), "type": "text_file"}
                    ))
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")

            return self._split_documents(docs)
        except Exception as e:
            logger.error(f"Error loading text files from {directory}: {e}")
            return []

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        try:
            return self.splitter.split_documents(docs)
        except Exception as e:
            logger.warning(f"Document splitting failed: {e}")
            return docs

    def create_document(self, content: str, metadata: Dict[str, Any] = None) -> Document:
        """Create a single document with metadata."""
        return Document(page_content=content, metadata=metadata or {})

    def merge_documents(self, doc_lists: List[List[Document]]) -> List[Document]:
        """Merge multiple document lists."""
        merged = []
        for doc_list in doc_lists:
            merged.extend(doc_list)
        return merged

    def filter_documents(self, docs: List[Document], min_length: int = 10) -> List[Document]:
        """Filter documents by minimum content length."""
        return [doc for doc in docs if len(doc.page_content.strip()) >= min_length]

    def get_document_stats(self, docs: List[Document]) -> Dict[str, Any]:
        """Get statistics about document collection."""
        if not docs:
            return {"count": 0, "total_chars": 0, "avg_length": 0, "sources": []}

        total_chars = sum(len(doc.page_content) for doc in docs)
        sources = list(set(doc.metadata.get("source", "unknown") for doc in docs))

        return {
            "count": len(docs),
            "total_chars": total_chars,
            "avg_length": total_chars // len(docs),
            "sources": sources
        }
