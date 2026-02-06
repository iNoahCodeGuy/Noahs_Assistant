"""LangSmith integration for tracing and monitoring.

This module provides decorators and utilities for tracing RAG operations
using LangSmith. All traces are automatically sent to the LangSmith platform
for visualization and analysis.

Features:
- Automatic tracing of OpenAI API calls
- Custom span creation for retrieval/generation steps
- Token usage tracking
- Latency monitoring
- Error tracking and debugging

Setup:
    Add to .env:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=lsv2_pt_...
        LANGCHAIN_PROJECT=noahs-ai-assistant

Why LangSmith:
- Visual traces of RAG pipeline
- Token usage analysis
- Performance bottleneck identification
- Error debugging
- A/B testing support
- Free tier: 5k traces/month
"""

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import LangSmith (graceful degradation if not installed)
try:
    from langsmith import Client, traceable
    from langsmith.run_helpers import trace
    from langsmith.wrappers import wrap_openai
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    logger.warning("LangSmith not available. Install with: pip install langsmith")

    # Create no-op wrapper if not available
    def wrap_openai(client):
        return client

    # Create no-op types and functions if LangSmith not available
    Client = None

    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])

    def trace(*args, **kwargs):
        """No-op context manager"""
        class NoOpContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return NoOpContext()


def initialize_langsmith() -> bool:
    """Initialize LangSmith tracing.

    Returns:
        True if LangSmith is configured and ready, False otherwise

    Checks:
    1. LANGCHAIN_TRACING_V2 is set to 'true'
    2. LANGCHAIN_API_KEY is set
    3. LangSmith package is installed
    """
    if not LANGSMITH_AVAILABLE:
        logger.warning("LangSmith not installed. Tracing disabled.")
        return False

    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    api_key = os.getenv("LANGCHAIN_API_KEY")

    if not tracing_enabled:
        logger.info("LangSmith tracing disabled (LANGCHAIN_TRACING_V2 not set)")
        return False

    if not api_key:
        logger.warning("LANGCHAIN_API_KEY not set. Tracing disabled.")
        return False

    project = os.getenv("LANGCHAIN_PROJECT", "noahs-ai-assistant")
    logger.info(f"LangSmith initialized. Project: {project}")
    return True


def get_langsmith_client():
    """Get LangSmith client if available.

    Returns:
        LangSmith client or None if not configured
    """
    if not LANGSMITH_AVAILABLE:
        return None

    if not initialize_langsmith():
        return None

    try:
        return Client()
    except Exception as e:
        logger.error(f"Failed to create LangSmith client: {e}")
        return None


def trace_rag_call(func: Callable) -> Callable:
    """Decorator to trace an entire RAG call (retrieve + generate).

    Usage:
        @trace_rag_call
        def generate_answer(query: str) -> str:
            # ... RAG logic
            pass

    Captures:
    - Query text
    - Retrieved chunks
    - Generated answer
    - Total latency
    - Token usage (if available)
    - Any errors
    """
    @wraps(func)
    @traceable(name="rag_pipeline", run_type="chain")
    def wrapper(*args, **kwargs):
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            latency_ms = int((time.time() - start_time) * 1000)

            # Log success metrics
            logger.debug(f"RAG call completed in {latency_ms}ms")

            return result

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"RAG call failed after {latency_ms}ms: {e}")
            raise

    return wrapper


def trace_retrieval(func: Callable) -> Callable:
    """Decorator to trace retrieval operations.

    Usage:
        @trace_retrieval
        def retrieve(query: str, top_k: int = 3) -> List[Dict]:
            # ... retrieval logic
            pass

    Captures:
    - Query text
    - Number of chunks retrieved
    - Similarity scores
    - Retrieval latency
    - Chunk sources
    """
    @wraps(func)
    @traceable(name="retrieval", run_type="retriever")
    def wrapper(*args, **kwargs):
        start_time = time.time()

        # Extract query from args
        query = args[0] if args else kwargs.get('query', 'unknown')
        top_k = args[1] if len(args) > 1 else kwargs.get('top_k', 3)

        try:
            result = func(*args, **kwargs)
            latency_ms = int((time.time() - start_time) * 1000)

            # Extract metrics from result
            if isinstance(result, dict):
                num_chunks = len(result.get('matches', []))
                scores = result.get('scores', [])
            elif isinstance(result, list):
                num_chunks = len(result)
                scores = [r.get('score', 0) for r in result if isinstance(r, dict)]
            else:
                num_chunks = 0
                scores = []

            # Log retrieval metrics
            logger.debug(
                f"Retrieved {num_chunks} chunks in {latency_ms}ms. "
                f"Avg similarity: {sum(scores)/len(scores) if scores else 0:.3f}"
            )

            return result

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Retrieval failed after {latency_ms}ms: {e}")
            raise

    return wrapper


def trace_generation(func: Callable) -> Callable:
    """Decorator to trace LLM generation operations.

    Usage:
        @trace_generation
        def generate(prompt: str) -> str:
            # ... OpenAI call
            pass

    Captures:
    - Prompt text
    - Generated response
    - Token usage (prompt + completion)
    - Generation latency
    - Model name
    - Any errors
    """
    @wraps(func)
    @traceable(name="generation", run_type="llm")
    def wrapper(*args, **kwargs):
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            latency_ms = int((time.time() - start_time) * 1000)

            # Extract token usage if available
            tokens_prompt = None
            tokens_completion = None

            if hasattr(result, 'usage'):
                tokens_prompt = result.usage.prompt_tokens
                tokens_completion = result.usage.completion_tokens
            elif isinstance(result, dict) and 'usage' in result:
                tokens_prompt = result['usage'].get('prompt_tokens')
                tokens_completion = result['usage'].get('completion_tokens')

            # Log generation metrics
            logger.debug(
                f"Generated response in {latency_ms}ms. "
                f"Tokens: {tokens_prompt or '?'} prompt + {tokens_completion or '?'} completion"
            )

            return result

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Generation failed after {latency_ms}ms: {e}")
            raise

    return wrapper


def log_trace_metadata(
    run_id: str,
    metadata: Dict[str, Any]
) -> None:
    """Add metadata to an existing trace.

    Useful for adding evaluation metrics after the trace completes.

    Args:
        run_id: LangSmith run ID
        metadata: Additional metadata to attach
    """
    client = get_langsmith_client()
    if not client:
        return

    try:
        client.update_run(run_id, extra=metadata)
        logger.debug(f"Added metadata to trace {run_id}")
    except Exception as e:
        logger.error(f"Failed to add trace metadata: {e}")


def create_custom_span(
    name: str,
    inputs: Dict[str, Any],
    run_type: str = "chain"
):
    """Create a custom trace span.

    Usage:
        with create_custom_span("my_operation", {"input": "data"}):
            # ... operation
            pass

    Args:
        name: Span name
        inputs: Input data to log
        run_type: Type of operation (chain, tool, retriever, llm)
    """
    if not LANGSMITH_AVAILABLE:
        # Return no-op context manager
        class NoOpContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return NoOpContext()

    return trace(
        name=name,
        inputs=inputs,
        run_type=run_type
    )


# Initialize on module load
LANGSMITH_ENABLED = initialize_langsmith()

if LANGSMITH_ENABLED:
    logger.info("✅ LangSmith tracing enabled")
else:
    logger.info("❌ LangSmith tracing disabled (install langsmith and set LANGCHAIN_TRACING_V2=true)")
