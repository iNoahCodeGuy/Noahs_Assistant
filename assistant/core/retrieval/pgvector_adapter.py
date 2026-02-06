"""pgvector retriever adapter for RAG engine.

This adapter wraps the existing pgvector_retriever module to conform
to the BaseRetriever interface, enabling seamless integration with RagEngine.

Why an adapter:
- Existing pgvector_retriever.py has different API
- Don't want to break existing code
- Adapter pattern allows clean integration
- Can add observability and metrics here
"""

import logging
import time
from typing import List, Dict, Any, Optional

from .base_retriever import (
    BaseRetriever,
    RoleAwareRetriever,
    LoggingRetriever,
    RetrievalResult
)

# Import existing pgvector retriever
try:
    from retrieval.pgvector_retriever import PgVectorRetriever, get_retriever
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    PgVectorRetriever = None

logger = logging.getLogger(__name__)


class PgVectorRetrieverAdapter(RoleAwareRetriever, LoggingRetriever):
    """Adapter for pgvector retriever to conform to BaseRetriever interface.

    This wraps the existing PgVectorRetriever class and adds:
    - Standardized RetrievalResult output
    - Role-aware retrieval
    - Analytics logging
    - Health checks
    - Observability metrics

    Example:
        adapter = PgVectorRetrieverAdapter(similarity_threshold=0.7)
        result = adapter.retrieve("What are Noah's skills?")
        print(f"Found {len(result.matches)} chunks")
    """

    def __init__(self, similarity_threshold: float = 0.7):
        """Initialize pgvector retriever adapter.

        Args:
            similarity_threshold: Minimum similarity score (0-1)
        """
        if not PGVECTOR_AVAILABLE:
            raise ImportError("pgvector_retriever not available. Is Supabase configured?")

        self._retriever = get_retriever(similarity_threshold=similarity_threshold)
        self.similarity_threshold = similarity_threshold
        logger.info(f"PgVectorRetrieverAdapter initialized with threshold={similarity_threshold}")

    def retrieve(self, query: str, top_k: int = 4) -> RetrievalResult:
        """Retrieve semantically similar documents from pgvector.

        Args:
            query: User query text
            top_k: Number of results to return

        Returns:
            RetrievalResult with matches, scores, and metadata
        """
        start_time = time.time()

        try:
            # Call underlying pgvector retriever
            chunks = self._retriever.retrieve(query, top_k=top_k)

            # Extract data
            matches = [c['content'] for c in chunks]
            scores = [c.get('similarity', 0.0) for c in chunks]
            sources = [c.get('source', 'career_kb') for c in chunks]

            latency_ms = int((time.time() - start_time) * 1000)

            logger.debug(f"pgvector retrieved {len(matches)} chunks in {latency_ms}ms")

            return RetrievalResult(
                matches=matches,
                scores=scores,
                sources=sources,
                metadata={
                    'threshold': self.similarity_threshold,
                    'avg_similarity': sum(scores) / len(scores) if scores else 0.0
                },
                latency_ms=latency_ms,
                retriever_type='pgvector'
            )

        except Exception as e:
            logger.error(f"pgvector retrieval failed: {e}")
            # Return empty result rather than crash
            return RetrievalResult(
                matches=[],
                scores=[],
                sources=[],
                metadata={'error': str(e)},
                latency_ms=int((time.time() - start_time) * 1000),
                retriever_type='pgvector'
            )

    def embed(self, text: str) -> List[float]:
        """Generate embedding vector using OpenAI.

        Args:
            text: Input text to embed

        Returns:
            1536-dimensional embedding vector
        """
        try:
            return self._retriever.embed(text)
        except Exception as e:
            logger.error(f"pgvector embedding failed: {e}")
            return []

    def retrieve_for_role(
        self,
        query: str,
        role: str,
        top_k: int = 4,
        include_personality: bool = False
    ) -> RetrievalResult:
        """Retrieve with role-based filtering.

        Uses pgvector's role-aware retrieval to return different
        results based on user role.

        Args:
            query: User query
            role: User role (e.g., "Software Developer", "Hiring Manager")
            top_k: Number of results

        Returns:
            RetrievalResult tailored for the role
        """
        start_time = time.time()

        try:
            # Call pgvector's role-aware retrieval
            chunks = self._retriever.retrieve_for_role(
                query=query,
                role=role,
                top_k=top_k,
                include_personality=include_personality
            )

            matches = [c['content'] for c in chunks]
            scores = [c.get('similarity', 0.0) for c in chunks]
            sources = [c.get('source', 'career_kb') for c in chunks]

            latency_ms = int((time.time() - start_time) * 1000)

            logger.debug(f"pgvector role-aware retrieval for {role}: {len(matches)} chunks")

            return RetrievalResult(
                matches=matches,
                scores=scores,
                sources=sources,
                metadata={
                    'role': role,
                    'threshold': self.similarity_threshold,
                    'avg_similarity': sum(scores) / len(scores) if scores else 0.0
                },
                latency_ms=latency_ms,
                retriever_type='pgvector'
            )

        except Exception as e:
            logger.error(f"pgvector role-aware retrieval failed: {e}")
            # Fallback to standard retrieval
            return self.retrieve(query, top_k)

    def retrieve_with_logging(
        self,
        query: str,
        message_id: int,
        top_k: int = 4
    ) -> RetrievalResult:
        """Retrieve with analytics logging.

        Logs retrieval event to Supabase for evaluation and monitoring.

        Args:
            query: User query
            message_id: ID from messages table
            top_k: Number of results

        Returns:
            RetrievalResult with logged=True in metadata
        """
        start_time = time.time()

        try:
            # Call pgvector's logging retrieval
            chunks = self._retriever.retrieve_and_log(
                query=query,
                message_id=message_id,
                top_k=top_k
            )

            matches = [c['content'] for c in chunks]
            scores = [c.get('similarity', 0.0) for c in chunks]
            sources = [c.get('source', 'career_kb') for c in chunks]

            latency_ms = int((time.time() - start_time) * 1000)

            logger.debug(f"pgvector logged retrieval: message_id={message_id}")

            return RetrievalResult(
                matches=matches,
                scores=scores,
                sources=sources,
                metadata={
                    'logged': True,
                    'message_id': message_id,
                    'threshold': self.similarity_threshold,
                    'avg_similarity': sum(scores) / len(scores) if scores else 0.0
                },
                latency_ms=latency_ms,
                retriever_type='pgvector'
            )

        except Exception as e:
            logger.error(f"pgvector logged retrieval failed: {e}")
            # Fallback to standard retrieval
            result = self.retrieve(query, top_k)
            result.metadata['logged'] = False
            result.metadata['log_error'] = str(e)
            return result

    def health_check(self) -> Dict[str, Any]:
        """Check if pgvector is operational.

        Returns:
            Status dict with health indicators
        """
        try:
            # Use pgvector's built-in health check
            health = self._retriever.health_check()

            return {
                "status": health["status"],
                "embedding_working": health.get("embedding_model") is not None,
                "retrieval_working": health["status"] == "healthy",
                "retriever_type": "pgvector",
                "embedding_model": health.get("embedding_model", "unknown"),
                "threshold": self.similarity_threshold
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "retriever_type": "pgvector"
            }

    @property
    def retriever_type(self) -> str:
        """Return retriever type identifier."""
        return "pgvector"

    @property
    def is_available(self) -> bool:
        """Check if pgvector is available and configured."""
        return PGVECTOR_AVAILABLE and self._retriever is not None


def create_pgvector_adapter(similarity_threshold: float = 0.7) -> Optional[PgVectorRetrieverAdapter]:
    """Factory function to create pgvector adapter.

    Args:
        similarity_threshold: Minimum similarity score (0-1)

    Returns:
        PgVectorRetrieverAdapter if available, None otherwise
    """
    if not PGVECTOR_AVAILABLE:
        logger.warning("pgvector not available")
        return None

    try:
        return PgVectorRetrieverAdapter(similarity_threshold=similarity_threshold)
    except Exception as e:
        logger.error(f"Failed to create pgvector adapter: {e}")
        return None
