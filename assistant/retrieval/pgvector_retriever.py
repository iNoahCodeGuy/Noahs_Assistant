"""pgvector-based retrieval service for Noah's AI Assistant.

This module replaces FAISS with Supabase pgvector for production deployment.

Why pgvector over FAISS:
1. **Centralized**: Single source of truth, no file syncing
2. **Scalable**: Works with serverless (Vercel functions)
3. **Observable**: All queries logged to retrieval_logs table
4. **Real-time updates**: No redeployment needed for new data
5. **Multi-instance**: Consistent across all deployments
6. **Cost-efficient**: No bundle size overhead, faster cold starts

Architecture:
    User Query
        ↓
    embed(query) → OpenAI text-embedding-3-small → [1536-dim vector]
        ↓
    Supabase.rpc('search_kb_chunks', {query_embedding, threshold, count})
        ↓
    PostgreSQL: ORDER BY embedding <=> $1 LIMIT 3
        ↓
    Return chunks + similarity scores
        ↓
    Log to retrieval_logs (for evaluation)
"""

import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

from assistant.config.supabase_config import get_supabase_client, supabase_settings
from assistant.analytics.supabase_analytics import supabase_analytics, RetrievalLogData

logger = logging.getLogger(__name__)


class PgVectorRetriever:
    """pgvector-based retrieval service using Supabase.

    This class handles:
    - Embedding generation via OpenAI
    - Similarity search via Supabase pgvector
    - Retrieval logging for observability
    - Role-based filtering

    Example usage:
        retriever = PgVectorRetriever()

        # Basic retrieval
        results = retriever.retrieve("What programming languages does Noah know?")

        # With role filtering
        results = retriever.retrieve_for_role(
            query="Explain Noah's technical skills",
            role="Hiring Manager (technical)",
            top_k=3
        )

        # With logging (for production)
        results = retriever.retrieve_and_log(
            query="Tell me about Noah's AI projects",
            message_id=123,
            top_k=3
        )
    """

    def __init__(self, similarity_threshold: float = 0.60):
        """Initialize retriever with OpenAI and Supabase clients.

        Args:
            similarity_threshold: Minimum cosine similarity (0-1) for results.
                Default 0.60 balances precision and recall for diverse queries.
                Lower (0.5-0.55) for broader results.
                Higher (0.70+) for strict matching.

        Why 0.60:
        - Lowered from 0.7 to 0.60 to improve recall on technical queries
        - "How does this product work?" has embedding variations (0.53-0.70 depending on wording)
        - Captures semantically similar queries without requiring exact phrasing
        - Trade-off: Slight increase in false positives, but better user experience
        """
        self.similarity_threshold = similarity_threshold
        self.openai_client = OpenAI(api_key=supabase_settings.api_key)
        self.supabase_client = get_supabase_client()

        # Embedding model configuration
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimensions = 1536

        logger.info(f"PgVectorRetriever initialized with threshold={similarity_threshold}")

    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text.

        Args:
            text: Input text to embed

        Returns:
            1536-dimensional embedding vector

        Why text-embedding-3-small:
        - Cost: $0.00002 per 1K tokens (5x cheaper than ada-002)
        - Performance: Better on RAG benchmarks than ada-002
        - Dimensions: 1536 (same as ada-002, easy migration)
        - Maintained: Latest OpenAI model (stable API)

        Error handling:
        - Returns empty list on failure (caller should check)
        - Logs error for observability
        - Allows graceful degradation
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        threshold: Optional[float] = None,
        doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve similar chunks from Supabase using pgvector.

        Args:
            query: Search query text
            top_k: Number of results to return (default 3)
            threshold: Override default similarity threshold
            doc_id: Filter by document ID (e.g., 'career_kb', 'code_index')

        Returns:
            List of chunk dicts with keys:
            - id: Chunk ID
            - doc_id: Document source
            - section: Section name/title
            - content: Full text content
            - similarity: Cosine similarity score (0-1)
            - metadata: Additional metadata (source, timestamps, etc.)

        Why this structure:
        - Matches Supabase schema exactly
        - Includes similarity score for evaluation
        - Metadata for debugging and citations
        - Compatible with LangChain Document if needed
        """
        # Use default threshold if not specified
        if threshold is None:
            threshold = self.similarity_threshold

        # Generate query embedding
        embedding = self.embed(query)
        if not embedding:
            logger.warning("Empty embedding, returning no results")
            return []

        # Convert to list of native Python floats
        embedding = [float(x) for x in embedding]

        try:
            # WORKAROUND: PostgREST has issues with the RPC function
            # So we fetch all chunks and compute similarity client-side
            logger.debug(f"Fetching all chunks for client-side similarity calculation")

            # Fetch all chunks with embeddings
            # Increased limit to accommodate all KBs: career_kb (20) + technical_kb (13) + architecture_kb (245) = 278 total
            result = self.supabase_client.table('kb_chunks')\
                .select('id, doc_id, section, content, embedding')\
                .limit(500)\
                .execute()

            if not result.data:
                logger.warning("No chunks found in database")
                return []

            # Calculate cosine similarity client-side
            import numpy as np
            query_vec = np.array(embedding)

            chunks_with_similarity = []
            for chunk in result.data:
                if not chunk.get('embedding'):
                    continue

                # Parse embedding
                chunk_emb = chunk['embedding']
                if isinstance(chunk_emb, str):
                    import json
                    chunk_emb = json.loads(chunk_emb)

                # Calculate cosine similarity
                chunk_vec = np.array(chunk_emb)
                similarity = 1 - np.dot(query_vec, chunk_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)
                )
                similarity = 1 - similarity  # Convert distance to similarity

                chunks_with_similarity.append({
                    'id': chunk['id'],
                    'doc_id': chunk['doc_id'],
                    'section': chunk['section'],
                    'content': chunk['content'],
                    'similarity': float(similarity)
                })

            # Sort by similarity (highest first) and apply threshold
            chunks_with_similarity.sort(key=lambda x: x['similarity'], reverse=True)

            # Log top scores for debugging
            top_5_scores = [f"{c['similarity']:.3f}" for c in chunks_with_similarity[:5]]
            logger.info(f"Top 5 similarity scores: {', '.join(top_5_scores)}")

            chunks = [c for c in chunks_with_similarity if c['similarity'] > threshold][:top_k]

            # Filter by doc_id if specified
            if doc_id:
                chunks = [c for c in chunks if c.get('doc_id') == doc_id]

            logger.info(f"Retrieved {len(chunks)} chunks (threshold={threshold}) for query: '{query[:50]}...'")
            return chunks

        except Exception as e:
            logger.error(f"pgvector retrieval failed: {e}")
            return []

    def retrieve_and_log(
        self,
        query: str,
        message_id: int,
        top_k: int = 3,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve and log to analytics for evaluation.

        This is the **production method** that should be used in the chat API.
        It retrieves chunks AND logs the retrieval event for observability.

        Args:
            query: Search query text
            message_id: ID from messages table (for linking)
            top_k: Number of results
            threshold: Optional threshold override

        Returns:
            List of retrieved chunks

        Side effects:
            - Inserts row into retrieval_logs table
            - Enables evaluation metrics
            - Tracks which chunks are useful

        Why logging matters:
        - **Evaluation**: Calculate precision/recall for RAG pipeline
        - **A/B testing**: Compare different thresholds or models
        - **Debugging**: See what context was used for bad responses
        - **Analytics**: Identify most/least useful chunks
        - **Compliance**: Audit trail for sensitive queries
        """
        chunks = self.retrieve(query, top_k, threshold)

        if not chunks:
            logger.warning(f"No chunks retrieved for message_id={message_id}")
            return []

        # Log retrieval event
        try:
            chunk_ids = [c['id'] for c in chunks]
            scores = [c['similarity'] for c in chunks]

            # Check if response is grounded (will be updated by response generator)
            # For now, mark as True if similarity > 0.8 (high confidence)
            grounded = any(score > 0.8 for score in scores)

            supabase_analytics.log_retrieval(RetrievalLogData(
                message_id=message_id,
                topk_ids=chunk_ids,
                scores=scores,
                grounded=grounded
            ))

            logger.debug(f"Logged retrieval: message_id={message_id}, chunks={len(chunk_ids)}")

        except Exception as e:
            logger.error(f"Failed to log retrieval: {e}")
            # Don't fail the request if logging fails

        return chunks

    def retrieve_for_role(
        self,
        query: str,
        role: str,
        top_k: int = 3,
        threshold: Optional[float] = None,
        include_personality: bool = False
    ) -> List[Dict[str, Any]]:
        """Retrieve with role-specific filtering.

        Different roles need different types of information:
        - **Technical roles**: Code examples, architecture, implementation details
        - **Hiring managers**: Career achievements, project outcomes, skills
        - **Casual users**: Fun facts, personal interests, MMA background

        Strategy:
        1. Retrieve more candidates than needed (top_k * 2)
        2. Filter/score based on role preferences
        3. Return top_k after filtering

        Args:
            query: Search query
            role: User role (e.g., "Software Developer", "Hiring Manager")
            top_k: Final number of results
            threshold: Optional similarity threshold

        Returns:
            Filtered and re-ranked chunks

        Why this approach:
        - Maintains high retrieval quality (start with good candidates)
        - Adds role-specific intelligence
        - Still fast (pgvector does heavy lifting)
        - Extensible (easy to add new role logic)
        """
        # Retrieve more candidates for filtering
        candidates = self.retrieve(query, top_k * 2, threshold)

        if not candidates:
            return []

        # Apply role-specific filtering/scoring
        if "technical" in role.lower() or "developer" in role.lower():
            # Prefer technical content
            filtered = self._filter_technical(candidates)

        elif "hiring" in role.lower() or "manager" in role.lower():
            # Prefer career/achievement content
            filtered = self._filter_career(candidates)

        elif "looking around" in role.lower() or "casual" in role.lower():
            # Prefer fun/personal content
            filtered = self._filter_casual(candidates)

        else:
            # No filtering, use as-is
            filtered = candidates

        # Retrieve personality context if requested
        if include_personality:
            personality_query = "What is Noah's personality like? How does Noah approach problems?"
            personality_chunks = self.retrieve(personality_query, top_k=2, threshold=threshold)
            if personality_chunks:
                personality_filtered = self._filter_personality(personality_chunks)
                # Add personality chunks to results (limit to 2)
                filtered.extend(personality_filtered[:2])
                # Re-sort by similarity if needed
                filtered.sort(key=lambda c: c.get('_boosted_similarity', c.get('similarity', 0)), reverse=True)

        # Return top_k after filtering
        return filtered[:top_k]

    def _filter_technical(self, chunks: List[Dict]) -> List[Dict]:
        """Boost technical content for developer roles.

        Technical indicators:
        - Mentions of code, programming, implementation
        - Technology names (Python, AI, ML, etc.)
        - Architecture or design patterns
        """
        technical_keywords = [
            'code', 'python', 'programming', 'implementation', 'architecture',
            'ai', 'ml', 'api', 'database', 'algorithm', 'data structure',
            'langchain', 'rag', 'vector', 'embedding', 'model'
        ]

        # Score each chunk by technical keyword density
        scored = []
        for chunk in chunks:
            content_lower = chunk['content'].lower()
            tech_score = sum(1 for kw in technical_keywords if kw in content_lower)

            # Boost similarity by technical score
            chunk['_tech_score'] = tech_score
            chunk['_boosted_similarity'] = chunk['similarity'] + (tech_score * 0.02)
            scored.append(chunk)

        # Sort by boosted similarity
        scored.sort(key=lambda c: c['_boosted_similarity'], reverse=True)
        return scored

    def _filter_career(self, chunks: List[Dict]) -> List[Dict]:
        """Boost career/achievement content for hiring manager roles.

        Career indicators:
        - Work experience, projects, achievements
        - Skills, background, education
        - Impact metrics, results
        """
        career_keywords = [
            'tesla', 'sales', 'experience', 'project', 'achievement',
            'skill', 'background', 'education', 'work', 'role',
            'developed', 'built', 'led', 'improved', 'implemented'
        ]

        scored = []
        for chunk in chunks:
            content_lower = chunk['content'].lower()
            career_score = sum(1 for kw in career_keywords if kw in content_lower)

            chunk['_career_score'] = career_score
            chunk['_boosted_similarity'] = chunk['similarity'] + (career_score * 0.02)
            scored.append(chunk)

        scored.sort(key=lambda c: c['_boosted_similarity'], reverse=True)
        return scored

    def _filter_casual(self, chunks: List[Dict]) -> List[Dict]:
        """Boost fun/personal content for casual roles.

        Casual indicators:
        - MMA, fighting, personal interests
        - Fun facts, hobbies
        - Conversational tone
        """
        casual_keywords = [
            'mma', 'fight', 'fighting', 'cage', 'amateur', 'professional',
            'chess', 'hobby', 'interest', 'fun', 'personal'
        ]

        scored = []
        for chunk in chunks:
            content_lower = chunk['content'].lower()
            casual_score = sum(1 for kw in casual_keywords if kw in content_lower)

            chunk['_casual_score'] = casual_score
            chunk['_boosted_similarity'] = chunk['similarity'] + (casual_score * 0.03)
            scored.append(chunk)

        scored.sort(key=lambda c: c['_boosted_similarity'], reverse=True)
        return scored

    def _filter_personality(self, chunks: List[Dict]) -> List[Dict]:
        """Boost personality-related content when relevant.

        Personality indicators:
        - Personality traits, communication style, work style
        - Motivations, values, approach to problems
        - How personality shows in code/work
        """
        personality_keywords = [
            'personality', 'trait', 'characteristic', 'motivation',
            'approach', 'style', 'value', 'preference', 'philosophy',
            'thoughtful', 'playful', 'systematic', 'authentic',
            'teaching', 'enterprise', 'work style', 'cultural fit'
        ]

        scored = []
        for chunk in chunks:
            content_lower = chunk['content'].lower()
            personality_score = sum(1 for kw in personality_keywords if kw in content_lower)

            chunk['_personality_score'] = personality_score
            chunk['_boosted_similarity'] = chunk['similarity'] + (personality_score * 0.02)
            scored.append(chunk)

        scored.sort(key=lambda c: c['_boosted_similarity'], reverse=True)
        return scored

    def health_check(self) -> Dict[str, Any]:
        """Check if retrieval service is operational.

        Tests:
        1. Supabase connection
        2. Embedding generation
        3. Simple similarity search

        Returns:
            Status dict with health indicators
        """
        try:
            # Test embedding generation
            test_embedding = self.embed("health check test")
            if not test_embedding:
                return {"status": "unhealthy", "reason": "Embedding generation failed"}

            # Test Supabase query
            result = self.supabase_client.table('kb_chunks').select('id').limit(1).execute()
            if not result:
                return {"status": "unhealthy", "reason": "Supabase query failed"}

            # Test similarity search
            chunks = self.retrieve("test query", top_k=1)

            return {
                "status": "healthy",
                "embedding_model": self.embedding_model,
                "embedding_dimensions": self.embedding_dimensions,
                "similarity_threshold": self.similarity_threshold,
                "test_retrieval_count": len(chunks)
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "reason": str(e)
            }


# Global retriever instance (for backward compatibility)
_retriever = None

def get_retriever(similarity_threshold: float = 0.7) -> PgVectorRetriever:
    """Get or create global retriever instance.

    Why global instance:
    - Reuse OpenAI and Supabase clients (connection pooling)
    - Avoid re-initialization overhead
    - Thread-safe (clients are thread-safe)
    """
    global _retriever
    if _retriever is None:
        _retriever = PgVectorRetriever(similarity_threshold)
    return _retriever
