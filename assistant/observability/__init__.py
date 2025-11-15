# -*- coding: utf-8 -*-
"""Observability module for Noah's AI Assistant.

This module provides comprehensive monitoring and tracing capabilities:
- LangSmith integration for OpenAI call tracing
- Custom metrics (faithfulness, relevance, answer quality)
- LLM-based evaluation functions
- Performance monitoring (latency, token usage)
- Retrieval quality metrics

Architecture:
    User Query
        ↓
    [LangSmith Trace Start]
        ↓
    Retrieve (pgvector) → [Log similarity scores]
        ↓
    Generate (OpenAI) → [Log tokens, latency]
        ↓
    Evaluate → [Faithfulness, Relevance, Quality]
        ↓
    [LangSmith Trace End + Store Metrics]

Usage:
    from observability import trace_rag_call, evaluate_response

    # Wrap RAG operations
    @trace_rag_call
    def retrieve_and_generate(query):
        # ... your code
        pass

    # Evaluate responses
    metrics = evaluate_response(
        query="What are Noah's skills?",
        context=["Noah has experience with Python, JavaScript..."],
        answer="Noah is proficient in Python and JavaScript..."
    )
"""

from .langsmith_tracer import (
    trace_rag_call,
    trace_retrieval,
    trace_generation,
    get_langsmith_client,
    initialize_langsmith
)

from .metrics import (
    RetrievalMetrics,
    GenerationMetrics,
    EvaluationMetrics,
    calculate_retrieval_metrics,
    calculate_generation_metrics
)

from .evaluators import (
    evaluate_faithfulness,
    evaluate_relevance,
    evaluate_answer_quality,
    evaluate_response
)

__all__ = [
    # Tracing
    'trace_rag_call',
    'trace_retrieval',
    'trace_generation',
    'get_langsmith_client',
    'initialize_langsmith',

    # Metrics
    'RetrievalMetrics',
    'GenerationMetrics',
    'EvaluationMetrics',
    'calculate_retrieval_metrics',
    'calculate_generation_metrics',

    # Evaluation
    'evaluate_faithfulness',
    'evaluate_relevance',
    'evaluate_answer_quality',
    'evaluate_response',
]
