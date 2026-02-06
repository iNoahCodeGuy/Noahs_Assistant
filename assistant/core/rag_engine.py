"""Retrieval Augmented Generation engine with Supabase pgvector.

**PRODUCTION ARCHITECTURE**: Uses Supabase pgvector exclusively for all vector operations.
- Centralized: Single source of truth for embeddings
- Scalable: Works with Vercel serverless functions
- Observable: Built-in retrieval logging and tracing with LangSmith
- Real-time: Updates without redeployment
- Cost-efficient: Smaller bundle size, faster cold starts

**OBSERVABILITY**: Integrated with LangSmith for tracing and evaluation.
- All retrieval and generation calls are traced
- Metrics logged to Supabase and LangSmith
- Optional LLM-as-judge evaluation (sampling-based to reduce costs)

**INITIALIZATION**:
- RagEngine(settings=Settings()) -> loads from Supabase pgvector
- RagEngine(career_kb, code_index) -> uses provided objects (for tests with mocks)

**KEY METHODS**:
- embed(text) -> embedding vector via OpenAI
- retrieve(query) -> dict with 'matches', 'skills', 'scores' keys
- generate_response(query) -> string answer with role-aware context
- retrieve_with_logging(query, message_id) -> retrieval with analytics

**ARCHITECTURE BENEFITS**:
- No file syncing across deployments
- Horizontal scaling on serverless
- Built-in similarity search with PostgreSQL
- Automatic replication and backups
- Centralized embeddings in Supabase
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Union
import os
import logging
from dataclasses import dataclass  # added for CodeDisplayMetrics
from datetime import datetime      # added for CodeDisplayMetrics
import time  # for latency tracking

# Clean imports using compatibility layer
from .langchain_compat import (
    OpenAIEmbeddings, CSVLoader, RecursiveCharacterTextSplitter,
    RetrievalQA, PromptTemplate, ChatOpenAI, Document
)

# Import Supabase configuration
from assistant.config.supabase_config import supabase_settings

# Import observability (gracefully handle if not available)
try:
    from observability import (
        trace_rag_call,
        trace_retrieval,
        trace_generation,
        calculate_retrieval_metrics,
        RetrievalMetrics
    )
    OBSERVABILITY_ENABLED = True
except ImportError:
    OBSERVABILITY_ENABLED = False
    # Create no-op decorators
    def trace_rag_call(f): return f
    def trace_retrieval(f): return f
    def trace_generation(f): return f

logger = logging.getLogger(__name__)

class RagEngine:
    """Complete RAG implementation using Supabase pgvector exclusively.
    ...existing docstring...
    """

    # ========== INITIALIZATION ==========
    def __init__(self, *args, **kwargs):
        """Flexible initializer using factory pattern."""
        self.settings = kwargs.get("settings", supabase_settings)
        self._provided_career_kb = None
        self._provided_code_index = None

        # pgvector mode flag (defaults to True, requires Supabase)
        self.use_pgvector = kwargs.get("use_pgvector", True)

        # Initialize pgvector retriever
        self.pgvector_retriever = None
        if self.use_pgvector:
            try:
                supabase_settings.validate_supabase()
                from assistant.retrieval.pgvector_retriever import get_retriever
                self.pgvector_retriever = get_retriever(similarity_threshold=0.3)  # Very low threshold for better recall
                logger.info("pgvector retriever initialized successfully")
            except Exception as e:
                logger.error(f"pgvector initialization failed: {e}")
                raise RuntimeError(
                    f"Failed to initialize pgvector retriever: {e}. "
                    "Ensure Supabase is configured with SUPABASE_URL and SUPABASE_KEY."
                ) from e

        # Parse initialization arguments
        if len(args) == 1 and not hasattr(self.settings, "validate_configuration"):
            # Assume it's supabase_settings if it has validate_configuration
            possible_settings = args[0]
            if hasattr(possible_settings, "validate_configuration"):
                self.settings = possible_settings
        elif len(args) == 2:  # (career_kb, code_index) test path
            self._provided_career_kb, self._provided_code_index = args

        # Validate configuration
        try:
            self.settings.validate_configuration()
        except Exception as e:
            logger.warning(f"Configuration validation warning: {e}")

        # Initialize using factory
        from .rag_factory import RagEngineFactory
        factory = RagEngineFactory(self.settings)

        # Create core components
        self.embeddings, emb_degraded = factory.create_embeddings()
        self.llm, llm_degraded = factory.create_llm()
        self.degraded_mode = emb_degraded or llm_degraded

        # Create knowledge bases
        self.career_kb = factory.create_career_kb(self._provided_career_kb)
        self.code_index = factory.create_code_index(self._provided_code_index)

        # Initialize code service
        from assistant.retrieval.code_service import CodeIndexService
        self.code_service = CodeIndexService(settings=self.settings, code_index=self.code_index)
        self._code_index_snapshot = self.code_service._snapshot  # compatibility

        # Load and process documents (for compatibility)
        self._career_docs = factory.load_documents(self.career_kb, self._provided_career_kb)

        # Create response generator
        from .response_generator import ResponseGenerator
        self.response_generator = ResponseGenerator(
            llm=self.llm,
            qa_chain=None,
            degraded_mode=self.degraded_mode
        )

    # ========== CORE RETRIEVAL ==========
    @trace_retrieval
    def retrieve(self, query: str, top_k: int = 4):
        """Retrieve semantically relevant docs using pgvector.

        **Architecture**:
        - Uses Supabase pgvector for vector similarity search
        - Centralized embeddings, no local files
        - Retrieval logging for observability
        - Scales horizontally on Vercel

        **Observability**: Traced with LangSmith, metrics logged

        Returns dict with keys: 'matches', 'skills', 'raw', 'scores', 'chunks'
        """
        start_time = time.time()
        matches: List[str] = []
        scores: List[float] = []
        chunks: List[Dict] = []  # ← PRESERVE full chunk data

        # Use pgvector for retrieval
        if self.pgvector_retriever:
            try:
                chunks = self.pgvector_retriever.retrieve(query, top_k)
                matches = [c['content'] for c in chunks]
                scores = [c.get('similarity', 0.0) for c in chunks]
                logger.debug(f"pgvector retrieved {len(matches)} chunks")
            except Exception as e:
                logger.error(f"pgvector retrieval failed: {e}")
                raise RuntimeError(f"Retrieval failed: {e}. Ensure Supabase is configured.")

        latency_ms = int((time.time() - start_time) * 1000)

        # Log retrieval metrics if observability enabled
        if OBSERVABILITY_ENABLED and matches:
            try:
                metrics = calculate_retrieval_metrics(
                    query=query,
                    chunks=[{'content': m, 'score': s} for m, s in zip(matches, scores)],
                    latency_ms=latency_ms
                )
                logger.debug(f"Retrieval metrics: {metrics.num_chunks} chunks, {metrics.avg_similarity:.3f} avg similarity")
            except Exception as e:
                logger.warning(f"Failed to calculate retrieval metrics: {e}")

        # Build skills extraction (simple heuristic)
        skills_fragments: List[str] = [m for m in matches if "skill" in m.lower()]
        return {
            "matches": matches,
            "skills": skills_fragments if skills_fragments else ["No explicit skills extracted"],
            "raw": matches,
            "scores": scores,
            "chunks": chunks  # ← INCLUDE full chunks with metadata for source citations
        }

    @trace_retrieval
    def retrieve_with_logging(self, query: str, message_id: int):
        """Retrieve with analytics logging (production method).

        This method should be used in production `/api/chat` endpoint.
        It retrieves chunks AND logs the event for evaluation.

        Args:
            query: User query
            message_id: ID from messages table (for linking retrieval logs)
            top_k: Number of chunks to retrieve

        Returns:
            Same format as retrieve() but with logging

        Why this matters:
        - Enables RAG evaluation metrics
        - Tracks which chunks are useful
        - A/B testing different thresholds
        - Debugging bad responses
        """
        if self.use_pgvector and self.pgvector_retriever:
            try:
                # Use pgvector's built-in logging
                chunks = self.pgvector_retriever.retrieve_and_log(
                    query=query,
                    message_id=message_id,
                    top_k=3
                )
                matches = [c['content'] for c in chunks]
                skills_fragments = [m for m in matches if "skill" in m.lower()]

                return {
                    "matches": matches,
                    "skills": skills_fragments if skills_fragments else ["No explicit skills extracted"],
                    "raw": matches,
                    "similarity_scores": [c['similarity'] for c in chunks],
                    "logged": True
                }
            except Exception as e:
                logger.error(f"Logged retrieval failed: {e}")
                # Fall back to regular retrieve

        # Fallback: regular retrieve without logging
        result = self.retrieve(query, top_k=3)
        result["logged"] = False
        return result

    # ========== CORE GENERATION ==========
    @trace_generation
    def generate_response(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate an answer using RetrievalQA chain if available.

        **Observability**: Full RAG pipeline traced with LangSmith
        - Retrieval metrics
        - Generation tokens and latency
        - Optional evaluation metrics

        Args:
            query: The user's question
            chat_history: Previous conversation messages for context
        """
        start_time = time.time()

        # Retrieve context
        retrieved = self.retrieve(query)

        # Generate response with context and chat history
        response = self.response_generator.generate_basic_response(
            query,
            fallback_docs=retrieved.get("matches", []),
            chat_history=chat_history
        )

        latency_ms = int((time.time() - start_time) * 1000)
        logger.debug(f"Full RAG pipeline completed in {latency_ms}ms")

        return response

    # ========== ADVANCED RETRIEVAL ==========
    @trace_retrieval
    def retrieve_with_code(self, query: str, role: str):
        """Enhanced retrieval that can include code snippets when allowed.

        **NEW**: Uses pgvector's role-aware retrieval when available.

        DEPRECATION: passing only `role` to trigger code inclusion will be removed in a future version.
        Callers should pass include_code=bool explicitly (RoleRouter now handles this).
        """
        include_code = None
        if include_code is None and role is not None:
            logger.debug("DEPRECATION: implicit role-based code inclusion – supply include_code explicitly.")

        # Ensure latest code before searching (unless disabled)
        self.ensure_code_index_current()

        # Use pgvector role-aware retrieval if available
        if self.use_pgvector and self.pgvector_retriever and role:
            try:
                chunks = self.pgvector_retriever.retrieve_for_role(
                    query=query,
                    role=role,
                    top_k=5
                )
                matches = [c['content'] for c in chunks]
                skills_fragments = [m for m in matches if "skill" in m.lower()]
                career_results = {
                    "matches": matches,
                    "skills": skills_fragments if skills_fragments else ["No explicit skills extracted"],
                    "raw": matches
                }
                logger.debug(f"pgvector role-aware retrieval: role={role}, chunks={len(chunks)}")
            except Exception as e:
                logger.error(f"pgvector role retrieval failed, using standard: {e}")
                career_results = self.retrieve(query, top_k=5)
        else:
            # Standard retrieval
            career_results = self.retrieve(query, top_k=5)

        # Decide if code should be included
        if include_code is None:
            try:
                from assistant.agents.roles import role_include_code  # lightweight, no circular dependency
                include_code = role_include_code(role)
            except Exception:
                include_code = False

        code_snippets: List[Dict[str, Any]] = []
        if include_code and self.code_index:
            try:
                query_keywords = [w.strip().lower() for w in query.split()
                                   if len(w) > 3 and w.lower() not in {'what','how','the','this','that'}]
                code_results = self.code_index.search_code(query, max_results=3)
                if not code_results and query_keywords:
                    code_results = self.code_index.search_by_keywords(query_keywords, max_results=3)
                for r in code_results:
                    code_snippets.append({
                        "file": r["file"],
                        "citation": r["citation"],
                        "content": r["content"],
                        "type": r["type"],
                        "name": r["name"],
                        "github_url": r["github_url"],
                        "line_start": r["line_start"],
                        "line_end": r["line_end"],
                    })
            except Exception as e:
                logger.warning(f"Code retrieval failed: {e}")

        return {**career_results, "code_snippets": code_snippets, "has_code": bool(code_snippets), "code_index_version": self.code_index_version()}

    # ========== BACKWARD COMPATIBILITY ==========
    def ensure_code_index_current(self):
        if getattr(self, 'code_service', None):
            self.code_service.ensure_current()
            self._code_index_snapshot = self.code_service._snapshot

    def code_index_version(self) -> str:
        """Return code index version hash for tracking changes."""
        if getattr(self, 'code_service', None):
            return getattr(self.code_service, '_snapshot', 'none')
        return "none"

    # ========== HEALTH & MONITORING ==========
    def health_check(self) -> Dict[str, Any]:
        """Check if RAG engine is operational.

        Returns:
            Status dict with health indicators
        """
        status = {
            "status": "unknown",
            "mode": "pgvector",
            "checks": {}
        }

        try:
            # Check embeddings
            test_embedding = self.embed("health check")
            status["checks"]["embeddings"] = "ok" if test_embedding else "failed"

            # Check retrieval
            test_retrieval = self.retrieve("test", top_k=1)
            status["checks"]["retrieval"] = "ok" if test_retrieval.get("matches") else "failed"

            # Check pgvector
            if self.use_pgvector and self.pgvector_retriever:
                pgv_health = self.pgvector_retriever.health_check()
                status["checks"]["pgvector"] = pgv_health["status"]

            # Overall status
            all_ok = all(v == "ok" or v == "healthy" for v in status["checks"].values())
            status["status"] = "healthy" if all_ok else "degraded"

        except Exception as e:
            status["status"] = "unhealthy"
            status["error"] = str(e)

        return status

@dataclass
class CodeDisplayMetrics:
    """Metrics for code display operations."""
    timestamp: datetime
    query_time: float
    # ... all fields documented
