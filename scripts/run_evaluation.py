#!/usr/bin/env python3
"""Run RAG evaluation pipeline with LangSmith.

This script evaluates the RAG system against a golden dataset using multiple metrics:
- Accuracy: Does answer correctly reflect Noah's background?
- Tone: Is response appropriate for role?
- Response time: Is latency acceptable?
- Grounding: Are claims supported by retrieved context?
- Relevance: Is retrieved context useful?
- Conciseness: Is answer appropriately detailed?

Usage:
    # Run full evaluation suite
    python scripts/run_evaluation.py

    # Run specific evaluators only
    python scripts/run_evaluation.py --evaluators accuracy tone

    # Use custom dataset
    python scripts/run_evaluation.py --dataset my_custom_dataset

    # Run with experiment name for A/B testing
    python scripts/run_evaluation.py --experiment "prompt-v2-test"

Output:
    Results are logged to LangSmith and printed as summary table.
    View detailed traces at: https://smith.langchain.com/
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Dict, Any
import pandas as pd
from langsmith import Client

from src.core.rag_engine import RagEngine
from src.state.conversation_state import ConversationState
from src.flows.conversation_flow import run_conversation_flow
from src.evaluation.evaluators import (
    accuracy_evaluator,
    tone_evaluator,
    response_time_evaluator,
    grounding_evaluator,
    relevance_evaluator,
    conciseness_evaluator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


EVALUATOR_MAP = {
    "accuracy": accuracy_evaluator,
    "tone": tone_evaluator,
    "response_time": response_time_evaluator,
    "grounding": grounding_evaluator,
    "relevance": relevance_evaluator,
    "conciseness": conciseness_evaluator,
}


def load_dataset(dataset_path: str) -> str:
    """Load evaluation dataset to LangSmith.

    Args:
        dataset_path: Path to CSV file with columns: role, query, expected_output, evaluation_criteria

    Returns:
        Dataset name in LangSmith
    """
    client = Client()
    dataset_name = Path(dataset_path).stem

    # Check if dataset already exists
    try:
        existing = client.read_dataset(dataset_name=dataset_name)
        logger.info(f"✅ Found existing dataset: {dataset_name}")
        return dataset_name
    except Exception:
        logger.info(f"Creating new dataset: {dataset_name}")

    # Load CSV
    df = pd.read_csv(dataset_path)

    # Create dataset in LangSmith
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=f"Golden dataset for RAG evaluation (loaded from {dataset_path})"
    )

    # Add examples
    for _, row in df.iterrows():
        client.create_example(
            dataset_id=dataset.id,
            inputs={
                "role": row["role"],
                "query": row["query"]
            },
            outputs={
                "expected_output": row["expected_output"],
                "evaluation_criteria": row.get("evaluation_criteria", "")
            }
        )

    logger.info(f"✅ Created dataset with {len(df)} examples")
    return dataset_name


def predict_answer(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Run RAG pipeline for evaluation.

    Args:
        inputs: Dict with 'role' and 'query' keys

    Returns:
        Dict with 'answer' and 'retrieved_chunks' keys
    """
    role = inputs["role"]
    query = inputs["query"]

    # Initialize RAG engine
    rag_engine = RagEngine()

    # Run conversation flow
    state = ConversationState(
        role=role,
        query=query,
        session_id="evaluation-session"
    )

    result = run_conversation_flow(
        state=state,
        rag_engine=rag_engine,
        session_id="evaluation"
    )

    return {
        "answer": result.get("answer", ""),
        "retrieved_chunks": result.get("retrieved_chunks", [])
    }


def run_evaluation(
    dataset_name: str,
    evaluators: list,
    experiment_name: str = None
) -> Dict[str, Any]:
    """Run evaluation pipeline.

    Args:
        dataset_name: Name of dataset in LangSmith
        evaluators: List of evaluator functions
        experiment_name: Optional experiment name for A/B testing

    Returns:
        Evaluation results dictionary
    """
    from langsmith import evaluate

    logger.info(f"Running evaluation with {len(evaluators)} evaluators")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Evaluators: {[e.__name__ for e in evaluators]}")

    # Run evaluation
    results = evaluate(
        predict_answer,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_name or "rag-eval"
    )

    return results


def print_summary(results: Any):
    """Print evaluation results summary.

    Args:
        results: LangSmith evaluation results
    """
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)

    # Extract aggregate metrics
    if hasattr(results, 'results'):
        all_scores = {}

        for run_result in results.results:
            for feedback in run_result.get('feedback', []):
                key = feedback.get('key')
                score = feedback.get('score')

                if key and score is not None:
                    if key not in all_scores:
                        all_scores[key] = []
                    all_scores[key].append(score)

        # Calculate averages
        logger.info("\nMetric Averages:")
        for metric, scores in sorted(all_scores.items()):
            avg = sum(scores) / len(scores) if scores else 0
            logger.info(f"  {metric:20s}: {avg:.3f} (n={len(scores)})")

        # Overall score
        all_values = [s for scores in all_scores.values() for s in scores]
        overall = sum(all_values) / len(all_values) if all_values else 0
        logger.info(f"\n  {'OVERALL':20s}: {overall:.3f}")

    logger.info("\n" + "="*60)
    logger.info("View detailed results at: https://smith.langchain.com/")
    logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation pipeline")
    parser.add_argument(
        "--dataset",
        default="data/evaluation/golden_dataset.csv",
        help="Path to evaluation dataset CSV"
    )
    parser.add_argument(
        "--evaluators",
        nargs="+",
        choices=list(EVALUATOR_MAP.keys()),
        default=["accuracy", "tone", "grounding", "conciseness"],
        help="Evaluators to run"
    )
    parser.add_argument(
        "--experiment",
        help="Experiment name for A/B testing"
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Only upload dataset to LangSmith, don't run evaluation"
    )

    args = parser.parse_args()

    # Check environment
    if not os.getenv("LANGSMITH_API_KEY"):
        logger.error("❌ LANGSMITH_API_KEY not set. Cannot run evaluation.")
        logger.info("Set in .env: LANGSMITH_API_KEY=lsv2_pt_...")
        sys.exit(1)

    # Load dataset
    dataset_name = load_dataset(args.dataset)

    if args.upload_only:
        logger.info("✅ Dataset uploaded. Skipping evaluation (--upload-only)")
        return

    # Select evaluators
    evaluators = [EVALUATOR_MAP[name] for name in args.evaluators]

    # Run evaluation
    try:
        results = run_evaluation(
            dataset_name=dataset_name,
            evaluators=evaluators,
            experiment_name=args.experiment
        )

        # Print summary
        print_summary(results)

        logger.info("✅ Evaluation complete!")

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
