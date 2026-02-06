"""LLM-based evaluation functions for RAG quality assessment.

This module implements "LLM-as-judge" patterns to evaluate:
- Faithfulness: Does the response cite the retrieved context?
- Relevance: Are the retrieved chunks relevant to the query?
- Answer Quality: Is the response helpful, accurate, and complete?

Why LLM-as-judge:
- More nuanced than keyword matching
- Captures semantic meaning
- Scales better than human evaluation
- Provides explanations for scores

Cost considerations:
- Uses GPT-3.5-turbo for evaluation (~$0.002/1K tokens)
- Only evaluates sampled responses (not all queries)
- Can be disabled to save costs
"""

import logging
from typing import Dict, Any, List, Tuple
from openai import OpenAI
import os

from .metrics import EvaluationMetrics

logger = logging.getLogger(__name__)


def get_evaluation_client() -> OpenAI:
    """Get OpenAI client for evaluation.

    Uses same API key as main application.
    """
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def evaluate_faithfulness(
    query: str,
    context: List[str],
    answer: str,
    model: str = "gpt-3.5-turbo"
) -> Tuple[float, str]:
    """Evaluate if the answer is faithful to the retrieved context.

    Faithfulness means:
    - Claims in the answer are supported by the context
    - No hallucinations or fabricated information
    - Proper attribution to sources

    Args:
        query: User query
        context: Retrieved chunks
        answer: Generated answer
        model: Model to use for evaluation

    Returns:
        (score, explanation) where score is 0-1 and explanation is reasoning
    """
    prompt = f"""You are evaluating the faithfulness of an AI assistant's answer to a user query.

**User Query:**
{query}

**Retrieved Context:**
{chr(10).join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context))}

**AI Answer:**
{answer}

**Task:**
Evaluate if the answer is faithful to the retrieved context.
- Score 1.0: All claims are supported by context, no hallucinations
- Score 0.7-0.9: Mostly faithful, minor unsupported details
- Score 0.4-0.6: Some claims not supported by context
- Score 0.0-0.3: Major hallucinations or unsupported claims

Provide:
1. A faithfulness score (0.0-1.0)
2. A brief explanation (1-2 sentences)

Format your response as:
SCORE: <score>
EXPLANATION: <explanation>
"""

    try:
        client = get_evaluation_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Deterministic evaluation
            max_tokens=200
        )

        content = response.choices[0].message.content
        score, explanation = _parse_evaluation_response(content)

        logger.debug(f"Faithfulness score: {score:.2f}")
        return score, explanation

    except Exception as e:
        logger.error(f"Faithfulness evaluation failed: {e}")
        return 0.5, f"Evaluation failed: {str(e)}"


def evaluate_relevance(
    query: str,
    context: List[str],
    model: str = "gpt-3.5-turbo"
) -> Tuple[float, str]:
    """Evaluate if the retrieved context is relevant to the query.

    Relevance means:
    - Retrieved chunks contain information to answer the query
    - Chunks are semantically related to the query
    - Minimal irrelevant information

    Args:
        query: User query
        context: Retrieved chunks
        model: Model to use for evaluation

    Returns:
        (score, explanation) where score is 0-1 and explanation is reasoning
    """
    prompt = f"""You are evaluating the relevance of retrieved context to a user query.

**User Query:**
{query}

**Retrieved Context:**
{chr(10).join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context))}

**Task:**
Evaluate if the retrieved context is relevant to answering the query.
- Score 1.0: All chunks highly relevant, directly answer query
- Score 0.7-0.9: Most chunks relevant, some tangential
- Score 0.4-0.6: Mixed relevance, some useful information
- Score 0.0-0.3: Mostly irrelevant, cannot answer query

Provide:
1. A relevance score (0.0-1.0)
2. A brief explanation (1-2 sentences)

Format your response as:
SCORE: <score>
EXPLANATION: <explanation>
"""

    try:
        client = get_evaluation_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200
        )

        content = response.choices[0].message.content
        score, explanation = _parse_evaluation_response(content)

        logger.debug(f"Relevance score: {score:.2f}")
        return score, explanation

    except Exception as e:
        logger.error(f"Relevance evaluation failed: {e}")
        return 0.5, f"Evaluation failed: {str(e)}"


def evaluate_answer_quality(
    query: str,
    answer: str,
    model: str = "gpt-3.5-turbo"
) -> Tuple[float, str]:
    """Evaluate overall answer quality.

    Quality metrics:
    - Helpfulness: Does it answer the user's question?
    - Clarity: Is it well-written and easy to understand?
    - Completeness: Does it cover all aspects of the query?
    - Accuracy: Is the information correct?

    Args:
        query: User query
        answer: Generated answer
        model: Model to use for evaluation

    Returns:
        (score, explanation) where score is 0-1 and explanation is reasoning
    """
    prompt = f"""You are evaluating the quality of an AI assistant's answer.

**User Query:**
{query}

**AI Answer:**
{answer}

**Task:**
Evaluate the overall quality of the answer considering:
- Helpfulness: Does it answer the question?
- Clarity: Is it well-written?
- Completeness: Does it cover all aspects?
- Accuracy: Is the information correct?

Score:
- 1.0: Excellent answer, helpful and complete
- 0.7-0.9: Good answer, minor improvements possible
- 0.4-0.6: Adequate answer, lacks detail or clarity
- 0.0-0.3: Poor answer, unhelpful or incorrect

Provide:
1. A quality score (0.0-1.0)
2. A brief explanation (1-2 sentences)

Format your response as:
SCORE: <score>
EXPLANATION: <explanation>
"""

    try:
        client = get_evaluation_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200
        )

        content = response.choices[0].message.content
        score, explanation = _parse_evaluation_response(content)

        logger.debug(f"Answer quality score: {score:.2f}")
        return score, explanation

    except Exception as e:
        logger.error(f"Answer quality evaluation failed: {e}")
        return 0.5, f"Evaluation failed: {str(e)}"


def evaluate_response(
    query: str,
    context: List[str],
    answer: str,
    model: str = "gpt-3.5-turbo"
) -> EvaluationMetrics:
    """Comprehensive evaluation of a RAG response.

    Combines all evaluation metrics into a single result.

    Args:
        query: User query
        context: Retrieved chunks
        answer: Generated answer
        model: Model to use for evaluation

    Returns:
        EvaluationMetrics with all scores
    """
    # Evaluate all aspects
    faithfulness, faith_exp = evaluate_faithfulness(query, context, answer, model)
    relevance, rel_exp = evaluate_relevance(query, context, model)
    quality, qual_exp = evaluate_answer_quality(query, answer, model)

    # Combine explanations
    explanation = (
        f"Faithfulness: {faith_exp} | "
        f"Relevance: {rel_exp} | "
        f"Quality: {qual_exp}"
    )

    return EvaluationMetrics(
        faithfulness_score=faithfulness,
        relevance_score=relevance,
        answer_quality_score=quality,
        groundedness=faithfulness,  # Same as faithfulness for now
        citation_accuracy=1.0 if faithfulness > 0.8 else 0.5,  # Heuristic
        explanation=explanation
    )


def _parse_evaluation_response(content: str) -> Tuple[float, str]:
    """Parse evaluation response from LLM.

    Expected format:
        SCORE: 0.85
        EXPLANATION: The answer is mostly faithful...

    Args:
        content: Raw LLM response

    Returns:
        (score, explanation) tuple
    """
    lines = content.strip().split('\n')
    score = 0.5
    explanation = "Could not parse evaluation"

    for line in lines:
        if line.startswith('SCORE:'):
            try:
                score = float(line.split('SCORE:')[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith('EXPLANATION:'):
            explanation = line.split('EXPLANATION:')[1].strip()

    # Clamp score to [0, 1]
    score = max(0.0, min(1.0, score))

    return score, explanation


def should_evaluate_sample(
    sample_rate: float = 0.1
) -> bool:
    """Determine if this response should be evaluated.

    To save costs, we only evaluate a sample of responses.

    Args:
        sample_rate: Fraction of responses to evaluate (0.0-1.0)

    Returns:
        True if should evaluate, False otherwise
    """
    import random
    return random.random() < sample_rate


def batch_evaluate_responses(
    responses: List[Dict[str, Any]],
    sample_rate: float = 0.1,
    model: str = "gpt-3.5-turbo"
) -> List[EvaluationMetrics]:
    """Evaluate multiple responses in batch.

    Args:
        responses: List of dicts with 'query', 'context', 'answer' keys
        sample_rate: Fraction of responses to evaluate
        model: Model to use for evaluation

    Returns:
        List of EvaluationMetrics (only for sampled responses)
    """
    results = []

    for resp in responses:
        if not should_evaluate_sample(sample_rate):
            continue

        try:
            metrics = evaluate_response(
                query=resp['query'],
                context=resp.get('context', []),
                answer=resp['answer'],
                model=model
            )
            results.append(metrics)
        except Exception as e:
            logger.error(f"Batch evaluation error: {e}")

    return results
