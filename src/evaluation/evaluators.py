"""Custom evaluators for RAG quality assessment.

This module provides evaluators for measuring:
1. Accuracy - does the answer correctly reflect Noah's background?
2. Tone - is the response appropriate for the role (technical vs business)?
3. Response time - is latency acceptable?
4. Grounding - are claims supported by retrieved context?
5. Relevance - is retrieved context actually useful?

Usage with LangSmith evaluate():
    from langsmith import evaluate
    from src.evaluation.evaluators import (
        accuracy_evaluator,
        tone_evaluator,
        response_time_evaluator
    )

    results = evaluate(
        lambda inputs: rag_engine.generate_response(inputs["query"]),
        data="golden_dataset",
        evaluators=[
            accuracy_evaluator,
            tone_evaluator,
            response_time_evaluator
        ]
    )
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.config.supabase_config import supabase_settings

logger = logging.getLogger(__name__)


def accuracy_evaluator(run: Any, example: Any) -> Dict[str, Any]:
    """Evaluate if answer correctly reflects Noah's background.

    Uses LLM-as-judge to score accuracy against expected output.

    Args:
        run: LangSmith run object with outputs
        example: Dataset example with expected_output

    Returns:
        Dict with score (0-1) and reasoning

    Example:
        score = accuracy_evaluator(run, example)
        # {"score": 0.9, "key": "accuracy", "reasoning": "Mentions key points..."}
    """
    try:
        actual_output = run.outputs.get("answer", "")
        expected_output = example.outputs.get("expected_output", "")
        query = example.inputs.get("query", "")

        if not actual_output or not expected_output:
            return {"score": 0.0, "key": "accuracy", "reasoning": "Missing output"}

        # LLM-as-judge prompt
        client = OpenAI(api_key=supabase_settings.openai_api_key)

        prompt = f"""You are evaluating the accuracy of an AI assistant's response.

Query: {query}
Expected Output: {expected_output}
Actual Output: {actual_output}

Score the accuracy (0.0-1.0) based on:
- 1.0 = Actual output covers all key points from expected output accurately
- 0.7 = Most key points covered, minor omissions
- 0.5 = Some key points covered, significant gaps
- 0.3 = Few key points covered, mostly inaccurate
- 0.0 = Completely inaccurate or irrelevant

Return ONLY a JSON object: {{"score": <float>, "reasoning": "<explanation>"}}
"""

        response = client.chat.completions.create(
            model=supabase_settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        result = json.loads(response.choices[0].message.content)
        result["key"] = "accuracy"
        return result

    except Exception as e:
        logger.error(f"Accuracy evaluation failed: {e}")
        return {"score": 0.0, "key": "accuracy", "reasoning": f"Evaluation error: {str(e)}"}


def tone_evaluator(run: Any, example: Any) -> Dict[str, Any]:
    """Evaluate if tone matches role requirements.

    Checks:
    - Technical roles: includes code/architecture details
    - Nontechnical roles: business-focused, minimal jargon
    - Developer roles: technical depth, examples
    - Casual roles: warm, conversational

    Args:
        run: LangSmith run object with outputs
        example: Dataset example with role and evaluation_criteria

    Returns:
        Dict with score (0-1) and reasoning
    """
    try:
        actual_output = run.outputs.get("answer", "")
        role = example.inputs.get("role", "")
        criteria = example.outputs.get("evaluation_criteria", "")

        if not actual_output:
            return {"score": 0.0, "key": "tone", "reasoning": "Missing output"}

        client = OpenAI(api_key=supabase_settings.openai_api_key)

        prompt = f"""You are evaluating the tone appropriateness of an AI assistant's response.

Role: {role}
Evaluation Criteria: {criteria}
Actual Output: {actual_output}

Score the tone (0.0-1.0) based on:
- 1.0 = Tone perfectly matches role expectations and criteria
- 0.7 = Good tone match, minor misalignment
- 0.5 = Acceptable tone, noticeable issues
- 0.3 = Poor tone match, significant misalignment
- 0.0 = Completely inappropriate tone

Consider:
- Technical depth appropriate for role
- Language complexity (jargon vs plain)
- Professional vs conversational balance
- Emphasis on business value vs technical details

Return ONLY a JSON object: {{"score": <float>, "reasoning": "<explanation>"}}
"""

        response = client.chat.completions.create(
            model=supabase_settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        result = json.loads(response.choices[0].message.content)
        result["key"] = "tone"
        return result

    except Exception as e:
        logger.error(f"Tone evaluation failed: {e}")
        return {"score": 0.0, "key": "tone", "reasoning": f"Evaluation error: {str(e)}"}


def response_time_evaluator(run: Any, example: Any) -> Dict[str, Any]:
    """Evaluate if response latency is acceptable.

    Thresholds:
    - <1s: Excellent (1.0)
    - 1-2s: Good (0.8)
    - 2-3s: Acceptable (0.6)
    - 3-5s: Slow (0.4)
    - >5s: Too slow (0.2)

    Args:
        run: LangSmith run object with timing info
        example: Dataset example (unused)

    Returns:
        Dict with score (0-1) and reasoning
    """
    try:
        # Extract latency from run metadata
        latency_ms = run.end_time - run.start_time if hasattr(run, 'end_time') else 0
        latency_s = latency_ms / 1000.0

        # Score based on thresholds
        if latency_s < 1.0:
            score = 1.0
            grade = "Excellent"
        elif latency_s < 2.0:
            score = 0.8
            grade = "Good"
        elif latency_s < 3.0:
            score = 0.6
            grade = "Acceptable"
        elif latency_s < 5.0:
            score = 0.4
            grade = "Slow"
        else:
            score = 0.2
            grade = "Too slow"

        return {
            "score": score,
            "key": "response_time",
            "reasoning": f"{grade}: {latency_s:.2f}s"
        }

    except Exception as e:
        logger.error(f"Response time evaluation failed: {e}")
        return {"score": 0.0, "key": "response_time", "reasoning": f"Evaluation error: {str(e)}"}


def grounding_evaluator(run: Any, example: Any) -> Dict[str, Any]:
    """Evaluate if answer is grounded in retrieved context.

    Checks for hallucinations by comparing answer to retrieved chunks.

    Args:
        run: LangSmith run object with retrieved_chunks in outputs
        example: Dataset example with query

    Returns:
        Dict with score (0-1) and reasoning
    """
    try:
        answer = run.outputs.get("answer", "")
        chunks = run.outputs.get("retrieved_chunks", [])
        query = example.inputs.get("query", "")

        if not answer:
            return {"score": 0.0, "key": "grounding", "reasoning": "Missing answer"}

        if not chunks:
            return {"score": 0.5, "key": "grounding", "reasoning": "No retrieved context (may be valid for greetings)"}

        # Combine chunks into context
        context = "\n\n".join([
            chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            for chunk in chunks
        ])

        client = OpenAI(api_key=supabase_settings.openai_api_key)

        prompt = f"""You are evaluating if an AI answer is grounded in retrieved context.

Query: {query}
Retrieved Context: {context}
Answer: {answer}

Score grounding (0.0-1.0) based on:
- 1.0 = All claims in answer are supported by context
- 0.7 = Most claims supported, minor speculation
- 0.5 = Some claims supported, noticeable gaps
- 0.3 = Few claims supported, significant hallucination
- 0.0 = Answer contradicts or ignores context

Return ONLY a JSON object: {{"score": <float>, "reasoning": "<explanation>"}}
"""

        response = client.chat.completions.create(
            model=supabase_settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        result = json.loads(response.choices[0].message.content)
        result["key"] = "grounding"
        return result

    except Exception as e:
        logger.error(f"Grounding evaluation failed: {e}")
        return {"score": 0.0, "key": "grounding", "reasoning": f"Evaluation error: {str(e)}"}


def relevance_evaluator(run: Any, example: Any) -> Dict[str, Any]:
    """Evaluate if retrieved context is relevant to query.

    Args:
        run: LangSmith run object with retrieved_chunks in outputs
        example: Dataset example with query

    Returns:
        Dict with score (0-1) and reasoning
    """
    try:
        chunks = run.outputs.get("retrieved_chunks", [])
        query = example.inputs.get("query", "")

        if not chunks:
            return {"score": 0.0, "key": "relevance", "reasoning": "No chunks retrieved"}

        # Combine chunks
        context = "\n\n".join([
            chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            for chunk in chunks
        ])

        client = OpenAI(api_key=supabase_settings.openai_api_key)

        prompt = f"""You are evaluating if retrieved context is relevant to a user query.

Query: {query}
Retrieved Context: {context}

Score relevance (0.0-1.0) based on:
- 1.0 = Context directly answers the query
- 0.7 = Context is highly relevant with minor gaps
- 0.5 = Context is somewhat relevant
- 0.3 = Context is tangentially related
- 0.0 = Context is irrelevant

Return ONLY a JSON object: {{"score": <float>, "reasoning": "<explanation>"}}
"""

        response = client.chat.completions.create(
            model=supabase_settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        result = json.loads(response.choices[0].message.content)
        result["key"] = "relevance"
        return result

    except Exception as e:
        logger.error(f"Relevance evaluation failed: {e}")
        return {"score": 0.0, "key": "relevance", "reasoning": f"Evaluation error: {str(e)}"}


def conciseness_evaluator(run: Any, example: Any) -> Dict[str, Any]:
    """Evaluate if answer is concise without losing key information.

    Penalizes both too-short (incomplete) and too-long (verbose) answers.

    Args:
        run: LangSmith run object with answer in outputs
        example: Dataset example with expected_output for comparison

    Returns:
        Dict with score (0-1) and reasoning
    """
    try:
        answer = run.outputs.get("answer", "")
        expected = example.outputs.get("expected_output", "")

        if not answer:
            return {"score": 0.0, "key": "conciseness", "reasoning": "Missing answer"}

        # Simple heuristic: compare word counts
        answer_words = len(answer.split())
        expected_words = len(expected.split())

        # Ideal: within 30% of expected length
        ratio = answer_words / expected_words if expected_words > 0 else 1.0

        if 0.7 <= ratio <= 1.3:
            score = 1.0
            grade = "Ideal length"
        elif 0.5 <= ratio <= 1.5:
            score = 0.7
            grade = "Good length"
        elif 0.3 <= ratio <= 2.0:
            score = 0.5
            grade = "Acceptable length"
        else:
            score = 0.3
            grade = "Too verbose" if ratio > 2.0 else "Too brief"

        return {
            "score": score,
            "key": "conciseness",
            "reasoning": f"{grade}: {answer_words} words (expected ~{expected_words})"
        }

    except Exception as e:
        logger.error(f"Conciseness evaluation failed: {e}")
        return {"score": 0.0, "key": "conciseness", "reasoning": f"Evaluation error: {str(e)}"}
