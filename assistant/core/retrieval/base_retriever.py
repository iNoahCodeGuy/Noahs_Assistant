"""Base retriever interface for RAG engine.

This module defines the abstract interface that all retriever implementations
must follow, enabling easy swapping between pgvector, FAISS, and future retrieval
systems.

Why this abstraction:
- Consistent API across different retrieval backends
- Easy to test (mock one retriever without affecting others)
- Follows Adapter pattern for clean separation
- Open/Closed Principle: extend by adding new adapters, not modifying existing code
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RetrievalResult:
    """Standardized result from any retriever.

    All retrievers return this format, ensuring consistency across
    pgvector, FAISS, and any future retrieval systems.
    """
    matches: List[str]              # Retrieved text chunks
    scores: List[float]             # Similarity scores (0-1)
    sources: List[str]              # Source identifiers
    metadata: Dict[str, Any]        # Additional metadata
    latency_ms: int                 # Retrieval time
    retriever_type: str             # 'pgvector', 'faiss', etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format for backward compatibility."""
        # Extract skills (simple heuristic)
        skills_fragments = [m for m in self.matches if "skill" in m.lower()]

        return {
            "matches": self.matches,
            "scores": self.scores,
            "skills": skills_fragments if skills_fragments else ["No explicit skills extracted"],
            "raw": self.matches,
            "sources": self.sources,
            "latency_ms": self.latency_ms,
            "retriever": self.retriever_type,
            **self.metadata
        }


class BaseRetriever(ABC):
    """Abstract base class for all retrievers.

    Any retrieval system (pgvector, FAISS, Pinecone, etc.) must implement
    this interface to work with RagEngine.

    Example:
        class MyRetriever(BaseRetriever):
            def retrieve(self, query: str, top_k: int = 4) -> RetrievalResult:
                # ... implementation
                pass

            def embed(self, text: str) -> List[float]:
                # ... implementation
                pass
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 4) -> RetrievalResult:
        """Retrieve semantically similar documents.

        Args:
            query: User query text
            top_k: Number of results to return

        Returns:
            RetrievalResult with matches, scores, and metadata
        """
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector (typically 1536-dim for OpenAI)
        """
        pass

    def health_check(self) -> Dict[str, Any]:
        """Check if retriever is operational.

        Returns:
            Status dict with health indicators
        """
        try:
            # Test embedding
            test_embedding = self.embed("health check")

            # Test retrieval
            test_result = self.retrieve("test", top_k=1)

            return {
                "status": "healthy" if test_embedding and test_result.matches else "degraded",
                "embedding_working": bool(test_embedding),
                "retrieval_working": bool(test_result.matches),
                "retriever_type": self.retriever_type
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "retriever_type": self.retriever_type
            }

    @property
    @abstractmethod
    def retriever_type(self) -> str:
        """Return retriever type identifier.

        Returns:
            Type string like 'pgvector', 'faiss', 'pinecone'
        """
        pass

    @property
    def is_available(self) -> bool:
        """Check if retriever is available and configured.

        Returns:
            True if retriever can be used, False otherwise
        """
        try:
            health = self.health_check()
            return health["status"] in ["healthy", "degraded"]
        except:
            return False


class RoleAwareRetriever(BaseRetriever):
    """Extended interface for role-aware retrieval.

    Some retrievers (like pgvector) support role-based filtering
    to return different results for different user types.
    """

    def retrieve_for_role(
        self,
        query: str,
        role: str,
        top_k: int = 4
    ) -> RetrievalResult:
        """Retrieve with role-based filtering.

        Args:
            query: User query
            role: User role (e.g., "Software Developer", "Hiring Manager")
            top_k: Number of results

        Returns:
            RetrievalResult tailored for the role
        """
        # Default implementation: just call regular retrieve
        # Subclasses can override for role-specific behavior
        return self.retrieve(query, top_k)


class LoggingRetriever(BaseRetriever):
    """Extended interface for retrievers that support analytics logging.

    Some retrievers can log retrieval events for evaluation and monitoring.
    """

    def retrieve_with_logging(
        self,
        query: str,
        message_id: int,
        top_k: int = 4
    ) -> RetrievalResult:
        """Retrieve with analytics logging.

        Args:
            query: User query
            message_id: ID from messages table (for linking logs)
            top_k: Number of results

        Returns:
            RetrievalResult with logged=True in metadata
        """
        # Default implementation: just call regular retrieve
        # Subclasses can override to add logging
        result = self.retrieve(query, top_k)
        result.metadata["logged"] = False
        return result
