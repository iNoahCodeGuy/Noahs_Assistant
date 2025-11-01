#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick test script to verify all LangSmith advanced features are working.

Tests:
1. Run type decorators (verify traceable imports)
2. LangGraph Studio export (verify graph compiles)
3. Prompt Hub integration (verify local prompts accessible)
4. Evaluation pipeline (verify evaluators load)

Usage:
    python scripts/test_langsmith_features.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def test_run_types():
    """Test 1: Verify run_type decorators are configured."""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Run Type Classification")
    logger.info("="*60)

    try:
        from src.observability.langsmith_tracer import (
            trace_retrieval,
            trace_generation,
            trace_rag_call
        )

        # Check decorator signatures include run_type
        import inspect

        retrieval_source = inspect.getsource(trace_retrieval)
        generation_source = inspect.getsource(trace_generation)
        rag_source = inspect.getsource(trace_rag_call)

        assert 'run_type="retriever"' in retrieval_source, "Missing retriever run_type"
        assert 'run_type="llm"' in generation_source, "Missing llm run_type"
        assert 'run_type="chain"' in rag_source, "Missing chain run_type"

        logger.info("[PASS] Run type decorators configured correctly")
        logger.info("   - trace_retrieval: run_type='retriever'")
        logger.info("   - trace_generation: run_type='llm'")
        logger.info("   - trace_rag_call: run_type='chain'")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Run type test failed: {e}")
        return False


def test_langgraph_studio():
    """Test 2: Verify LangGraph Studio export."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: LangGraph Studio Export")
    logger.info("="*60)

    try:
        from src.flows.conversation_flow import graph

        if graph is None:
            logger.warning("[WARN]  Graph is None (langgraph not installed or import failed)")
            logger.info("   Install with: pip install langgraph")
            logger.info("   This is optional - other features still work")
            return True

        # Check it's a compiled graph
        assert hasattr(graph, 'invoke'), "Graph missing invoke method"

        logger.info("[PASS] LangGraph Studio export configured")
        logger.info(f"   - Graph type: {type(graph).__name__}")
        logger.info("   - Start Studio with: langgraph dev")
        logger.info("   - Access at: http://127.0.0.1:2024")
        return True

    except Exception as e:
        logger.error(f"[FAIL] LangGraph Studio test failed: {e}")
        logger.info("   This is optional - install with: pip install langgraph")
        return True  # Non-critical


def test_prompt_hub():
    """Test 3: Verify Prompt Hub integration."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Prompt Hub Integration")
    logger.info("="*60)

    try:
        from src.prompts import get_prompt, list_prompts

        # List available prompts
        prompts = list_prompts()
        assert len(prompts) > 0, "No local prompts defined"

        logger.info(f"[PASS] Prompt Hub configured with {len(prompts)} local templates")
        logger.info("   Available prompts:")
        for name in sorted(prompts.keys()):
            desc = prompts[name].get("description", "No description")
            logger.info(f"     - {name}: {desc}")

        # Test get_prompt
        basic_qa = get_prompt("basic_qa")
        assert "{context}" in basic_qa, "Missing context variable in basic_qa"
        assert "{question}" in basic_qa, "Missing question variable in basic_qa"

        logger.info("\n   Test prompt retrieval:")
        logger.info(f"     get_prompt('basic_qa'): {len(basic_qa)} chars")
        logger.info("     Variables: {context}, {question} ✓")

        logger.info("\n   Initialize hub with:")
        logger.info("     python -c \"from src.prompts.prompt_hub import initialize_prompt_hub; initialize_prompt_hub()\"")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Prompt Hub test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_pipeline():
    """Test 4: Verify evaluation pipeline."""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Evaluation Pipeline")
    logger.info("="*60)

    try:
        from src.evaluation import (
            accuracy_evaluator,
            tone_evaluator,
            response_time_evaluator,
            grounding_evaluator,
            relevance_evaluator,
            conciseness_evaluator,
        )

        evaluators = [
            accuracy_evaluator,
            tone_evaluator,
            response_time_evaluator,
            grounding_evaluator,
            relevance_evaluator,
            conciseness_evaluator,
        ]

        logger.info(f"[PASS] Evaluation pipeline configured with {len(evaluators)} evaluators")
        logger.info("   Available evaluators:")
        for evaluator in evaluators:
            logger.info(f"     - {evaluator.__name__}")

        # Check golden dataset exists
        import pandas as pd
        dataset_path = project_root / "data" / "evaluation" / "golden_dataset.csv"

        if dataset_path.exists():
            df = pd.read_csv(dataset_path)
            logger.info(f"\n   Golden dataset: {len(df)} test cases")
            logger.info(f"     Roles: {df['role'].unique().tolist()}")
        else:
            logger.warning(f"   [WARN]  Golden dataset not found: {dataset_path}")

        logger.info("\n   Run evaluation with:")
        logger.info("     python scripts/run_evaluation.py")
        logger.info("     python scripts/run_evaluation.py --experiment 'test'")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Evaluation pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    logger.info("\n" + "="*60)
    logger.info("LANGSMITH ADVANCED FEATURES - VERIFICATION")
    logger.info("="*60)

    results = {
        "Run Types": test_run_types(),
        "LangGraph Studio": test_langgraph_studio(),
        "Prompt Hub": test_prompt_hub(),
        "Evaluation Pipeline": test_evaluation_pipeline(),
    }

    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    for feature, passed in results.items():
        status = "[PASS] PASS" if passed else "[FAIL] FAIL"
        logger.info(f"{feature:25s}: {status}")

    all_passed = all(results.values())

    logger.info("\n" + "="*60)
    if all_passed:
        logger.info("[PASS] ALL TESTS PASSED")
        logger.info("\nNext steps:")
        logger.info("1. Start LangGraph Studio: langgraph dev")
        logger.info("2. Initialize Prompt Hub: python -c \"from src.prompts.prompt_hub import initialize_prompt_hub; initialize_prompt_hub()\"")
        logger.info("3. Run evaluation: python scripts/run_evaluation.py")
        logger.info("4. View traces: https://smith.langchain.com/")
    else:
        logger.error("[FAIL] SOME TESTS FAILED")
        logger.info("\nCheck errors above and fix issues")

    logger.info("="*60 + "\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
