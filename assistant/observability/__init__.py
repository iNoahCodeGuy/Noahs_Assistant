# -*- coding: utf-8 -*-
"""Observability module for Noah's AI Assistant.

Provides LangSmith integration for tracing and monitoring.
"""

from .langsmith_tracer import (
    trace_rag_call,
    trace_retrieval,
    trace_generation,
    get_langsmith_client,
    initialize_langsmith,
    create_custom_span
)

__all__ = [
    'trace_rag_call',
    'trace_retrieval',
    'trace_generation',
    'get_langsmith_client',
    'initialize_langsmith',
    'create_custom_span',
]
