"""Metrics collection for RAG evaluation.

This module defines data structures and calculation functions for
monitoring retrieval and generation quality.

Metrics tracked:
- Retrieval: similarity scores, chunk count, latency, relevance
- Generation: token usage, latency, faithfulness
- Evaluation: answer quality, groundedness, citation accuracy

Why these metrics:
- Similarity scores: Indicate retrieval quality
- Token usage: Track costs and optimize prompts
- Latency: Identify performance bottlenecks
- Faithfulness: Ensure responses cite sources
- Relevance: Verify retrieved chunks match query
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import statistics

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval operation.

    Attributes:
        query: User query text
        num_chunks: Number of chunks retrieved
        similarity_scores: Cosine similarity scores (0-1)
        avg_similarity: Average similarity score
        latency_ms: Retrieval time in milliseconds
        chunk_sources: Source identifiers for retrieved chunks
        timestamp: When retrieval occurred
    """
    query: str
    num_chunks: int
    similarity_scores: List[float]
    avg_similarity: float
    latency_ms: int
    chunk_sources: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_retrieval_result(
        cls,
        query: str,
        result: Dict[str, Any],
        latency_ms: int
    ) -> 'RetrievalMetrics':
        """Create metrics from retrieval result.

        Args:
            query: User query
            result: Retrieval result dict with 'matches' and optional 'scores'
            latency_ms: Retrieval latency

        Returns:
            RetrievalMetrics instance
        """
        chunks = result.get('matches', [])
        scores = result.get('scores', [0.0] * len(chunks))

        return cls(
            query=query,
            num_chunks=len(chunks),
            similarity_scores=scores,
            avg_similarity=statistics.mean(scores) if scores else 0.0,
            latency_ms=latency_ms,
            chunk_sources=[c.get('source', 'unknown') for c in chunks] if isinstance(chunks, list) and chunks and isinstance(chunks[0], dict) else []
        )


@dataclass
class GenerationMetrics:
    """Metrics for a single LLM generation.

    Attributes:
        prompt: Input prompt text
        response: Generated response text
        tokens_prompt: Number of prompt tokens
        tokens_completion: Number of completion tokens
        total_tokens: Total tokens used
        latency_ms: Generation time in milliseconds
        model: Model name (e.g., 'gpt-4')
        cost_usd: Estimated cost in USD
        timestamp: When generation occurred
    """
    prompt: str
    response: str
    tokens_prompt: int
    tokens_completion: int
    total_tokens: int
    latency_ms: int
    model: str
    cost_usd: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_openai_response(
        cls,
        prompt: str,
        response: Any,
        latency_ms: int,
        model: str = "gpt-4"
    ) -> 'GenerationMetrics':
        """Create metrics from OpenAI API response.

        Args:
            prompt: Input prompt
            response: OpenAI response object
            latency_ms: Generation latency
            model: Model name

        Returns:
            GenerationMetrics instance
        """
        # Extract response text
        if hasattr(response, 'choices') and response.choices:
            response_text = response.choices[0].message.content
        elif isinstance(response, dict) and 'choices' in response:
            response_text = response['choices'][0]['message']['content']
        else:
            response_text = str(response)

        # Extract token usage
        if hasattr(response, 'usage'):
            tokens_prompt = response.usage.prompt_tokens
            tokens_completion = response.usage.completion_tokens
        elif isinstance(response, dict) and 'usage' in response:
            tokens_prompt = response['usage']['prompt_tokens']
            tokens_completion = response['usage']['completion_tokens']
        else:
            tokens_prompt = 0
            tokens_completion = 0

        total_tokens = tokens_prompt + tokens_completion

        # Estimate cost (approximate rates)
        cost_usd = calculate_openai_cost(model, tokens_prompt, tokens_completion)

        return cls(
            prompt=prompt,
            response=response_text,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            model=model,
            cost_usd=cost_usd
        )


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for response quality.

    These are calculated post-generation using LLM-as-judge.

    Attributes:
        faithfulness_score: Does response cite retrieved context? (0-1)
        relevance_score: Are retrieved chunks relevant to query? (0-1)
        answer_quality_score: Overall answer quality (0-1)
        groundedness: Are claims supported by context? (0-1)
        citation_accuracy: Are citations correct? (0-1)
        explanation: Human-readable explanation of scores
        timestamp: When evaluation occurred
    """
    faithfulness_score: float
    relevance_score: float
    answer_quality_score: float
    groundedness: float = 0.0
    citation_accuracy: float = 0.0
    explanation: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def overall_score(self) -> float:
        """Calculate overall quality score (0-1).

        Weighted average of all metrics:
        - 30% faithfulness
        - 25% relevance
        - 25% answer quality
        - 10% groundedness
        - 10% citation accuracy
        """
        return (
            0.30 * self.faithfulness_score +
            0.25 * self.relevance_score +
            0.25 * self.answer_quality_score +
            0.10 * self.groundedness +
            0.10 * self.citation_accuracy
        )


def calculate_retrieval_metrics(
    query: str,
    chunks: List[Dict[str, Any]],
    latency_ms: int
) -> RetrievalMetrics:
    """Calculate retrieval metrics from raw retrieval data.

    Args:
        query: User query
        chunks: Retrieved chunks with 'content', 'score', 'source' keys
        latency_ms: Retrieval latency

    Returns:
        RetrievalMetrics instance
    """
    scores = [c.get('score', 0.0) for c in chunks]
    sources = [c.get('source', 'unknown') for c in chunks]

    return RetrievalMetrics(
        query=query,
        num_chunks=len(chunks),
        similarity_scores=scores,
        avg_similarity=statistics.mean(scores) if scores else 0.0,
        latency_ms=latency_ms,
        chunk_sources=sources
    )


def calculate_generation_metrics(
    prompt: str,
    response: str,
    tokens_prompt: int,
    tokens_completion: int,
    latency_ms: int,
    model: str = "gpt-4"
) -> GenerationMetrics:
    """Calculate generation metrics from raw generation data.

    Args:
        prompt: Input prompt
        response: Generated response
        tokens_prompt: Prompt token count
        tokens_completion: Completion token count
        latency_ms: Generation latency
        model: Model name

    Returns:
        GenerationMetrics instance
    """
    total_tokens = tokens_prompt + tokens_completion
    cost_usd = calculate_openai_cost(model, tokens_prompt, tokens_completion)

    return GenerationMetrics(
        prompt=prompt,
        response=response,
        tokens_prompt=tokens_prompt,
        tokens_completion=tokens_completion,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        model=model,
        cost_usd=cost_usd
    )


def calculate_openai_cost(
    model: str,
    tokens_prompt: int,
    tokens_completion: int
) -> float:
    """Estimate OpenAI API cost in USD.

    Pricing (as of 2024):
    - GPT-4: $0.03/1K prompt, $0.06/1K completion
    - GPT-4-turbo: $0.01/1K prompt, $0.03/1K completion
    - GPT-3.5-turbo: $0.0015/1K prompt, $0.002/1K completion

    Args:
        model: Model name
        tokens_prompt: Prompt tokens
        tokens_completion: Completion tokens

    Returns:
        Estimated cost in USD
    """
    pricing = {
        'gpt-4': (0.03, 0.06),
        'gpt-4-turbo': (0.01, 0.03),
        'gpt-4-1106-preview': (0.01, 0.03),
        'gpt-3.5-turbo': (0.0015, 0.002),
    }

    # Default to GPT-4 pricing if model not found
    prompt_price, completion_price = pricing.get(model, (0.03, 0.06))

    cost = (
        (tokens_prompt / 1000.0) * prompt_price +
        (tokens_completion / 1000.0) * completion_price
    )

    return round(cost, 6)


def log_metrics_to_supabase(
    retrieval_metrics: Optional[RetrievalMetrics] = None,
    generation_metrics: Optional[GenerationMetrics] = None,
    evaluation_metrics: Optional[EvaluationMetrics] = None,
    message_id: Optional[int] = None
) -> bool:
    """Log metrics to Supabase for analysis.

    Args:
        retrieval_metrics: Retrieval metrics to log
        generation_metrics: Generation metrics to log
        evaluation_metrics: Evaluation metrics to log
        message_id: Message ID to link metrics to

    Returns:
        True if successful, False otherwise
    """
    try:
        from analytics.supabase_analytics import supabase_analytics

        # Log retrieval metrics
        if retrieval_metrics and message_id:
            supabase_analytics.client.table('retrieval_metrics').insert({
                'message_id': message_id,
                'num_chunks': retrieval_metrics.num_chunks,
                'avg_similarity': retrieval_metrics.avg_similarity,
                'latency_ms': retrieval_metrics.latency_ms,
                'similarity_scores': retrieval_metrics.similarity_scores,
                'chunk_sources': retrieval_metrics.chunk_sources,
            }).execute()

        # Log generation metrics
        if generation_metrics and message_id:
            supabase_analytics.client.table('generation_metrics').insert({
                'message_id': message_id,
                'tokens_prompt': generation_metrics.tokens_prompt,
                'tokens_completion': generation_metrics.tokens_completion,
                'total_tokens': generation_metrics.total_tokens,
                'latency_ms': generation_metrics.latency_ms,
                'model': generation_metrics.model,
                'cost_usd': generation_metrics.cost_usd,
            }).execute()

        # Log evaluation metrics
        if evaluation_metrics and message_id:
            supabase_analytics.client.table('evaluation_metrics').insert({
                'message_id': message_id,
                'faithfulness_score': evaluation_metrics.faithfulness_score,
                'relevance_score': evaluation_metrics.relevance_score,
                'answer_quality_score': evaluation_metrics.answer_quality_score,
                'groundedness': evaluation_metrics.groundedness,
                'citation_accuracy': evaluation_metrics.citation_accuracy,
                'overall_score': evaluation_metrics.overall_score(),
                'explanation': evaluation_metrics.explanation,
            }).execute()

        return True

    except Exception as e:
        logger.error(f"Failed to log metrics to Supabase: {e}")
        return False
