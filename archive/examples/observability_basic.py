"""Example: Basic Observability Usage

This example demonstrates basic observability features:
- Tracing RAG calls
- Collecting retrieval metrics
- Monitoring token usage
- Viewing traces in LangSmith

Setup:
    1. Set LANGCHAIN_TRACING_V2=true in .env
    2. Set LANGCHAIN_API_KEY in .env
    3. Run this script
    4. View traces at https://smith.langchain.com/
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import observability components
from observability import (
    initialize_langsmith,
    calculate_retrieval_metrics,
    calculate_generation_metrics
)

# Import RAG engine
from core.rag_engine import RagEngine


def basic_traced_query():
    """Example 1: Basic query with automatic tracing."""
    print("\n=== Example 1: Basic Traced Query ===")

    # Check if LangSmith is enabled
    if initialize_langsmith():
        print("✅ LangSmith tracing enabled")
    else:
        print("❌ LangSmith not configured (see docs/LANGSMITH_SETUP.md)")

    # Create RAG engine (automatically traced)
    engine = RagEngine()

    # Execute query (trace appears in LangSmith dashboard)
    query = "What programming languages does Noah know?"
    print(f"\nQuery: {query}")

    response = engine.generate_response(query)
    print(f"\nResponse: {response}\n")

    print("✅ Check LangSmith dashboard for trace!")


def manual_metrics_collection():
    """Example 2: Manual metrics collection."""
    print("\n=== Example 2: Manual Metrics Collection ===")

    # Simulate retrieval
    query = "What are Noah's AI projects?"
    chunks = [
        {'content': 'Noah built a RAG system using pgvector...', 'score': 0.85},
        {'content': 'He created an AI assistant with LangChain...', 'score': 0.78},
        {'content': 'His work includes LangSmith integration...', 'score': 0.72}
    ]
    latency_ms = 150

    # Calculate metrics
    metrics = calculate_retrieval_metrics(
        query=query,
        chunks=chunks,
        latency_ms=latency_ms
    )

    # Display metrics
    print(f"Query: {query}")
    print(f"Chunks retrieved: {metrics.num_chunks}")
    print(f"Avg similarity: {metrics.avg_similarity:.3f}")
    print(f"Latency: {metrics.latency_ms}ms")
    print(f"Similarity scores: {[f'{s:.3f}' for s in metrics.similarity_scores]}")


def token_usage_tracking():
    """Example 3: Track token usage and costs."""
    print("\n=== Example 3: Token Usage Tracking ===")

    # Simulate generation
    prompt = "Explain Noah's technical background"
    response = "Noah is a software engineer with expertise in Python..."
    tokens_prompt = 150
    tokens_completion = 200
    latency_ms = 800
    model = "gpt-4"

    # Calculate metrics
    metrics = calculate_generation_metrics(
        prompt=prompt,
        response=response,
        tokens_prompt=tokens_prompt,
        tokens_completion=tokens_completion,
        latency_ms=latency_ms,
        model=model
    )

    # Display metrics
    print(f"Model: {metrics.model}")
    print(f"Prompt tokens: {metrics.tokens_prompt}")
    print(f"Completion tokens: {metrics.tokens_completion}")
    print(f"Total tokens: {metrics.total_tokens}")
    print(f"Cost: ${metrics.cost_usd:.4f}")
    print(f"Latency: {metrics.latency_ms}ms")


def compare_retrieval_quality():
    """Example 4: Compare retrieval quality across queries."""
    print("\n=== Example 4: Compare Retrieval Quality ===")

    engine = RagEngine()

    queries = [
        "What are Noah's technical skills?",
        "Tell me about Noah's education",
        "What is Noah's favorite color?"  # Likely low quality
    ]

    print("\nQuery Quality Comparison:")
    print("-" * 60)

    for query in queries:
        result = engine.retrieve(query, top_k=3)
        scores = result.get('scores', [])
        avg_score = sum(scores) / len(scores) if scores else 0.0

        quality = "🟢 Good" if avg_score > 0.75 else "🟡 Medium" if avg_score > 0.5 else "🔴 Poor"

        print(f"{quality} | Avg: {avg_score:.3f} | Query: {query[:40]}...")


def end_to_end_example():
    """Example 5: End-to-end RAG with full observability."""
    print("\n=== Example 5: End-to-End RAG Pipeline ===")

    # Initialize
    engine = RagEngine()

    # Query
    query = "What makes Noah's RAG system unique?"
    print(f"\nQuery: {query}\n")

    # Step 1: Retrieve
    print("Step 1: Retrieving context...")
    import time
    start_time = time.time()

    retrieval_result = engine.retrieve(query, top_k=3)
    retrieval_latency = int((time.time() - start_time) * 1000)

    print(f"  ✓ Retrieved {len(retrieval_result['matches'])} chunks in {retrieval_latency}ms")

    # Step 2: Generate
    print("\nStep 2: Generating response...")
    start_time = time.time()

    response = engine.generate_response(query)
    generation_latency = int((time.time() - start_time) * 1000)

    print(f"  ✓ Generated response in {generation_latency}ms")

    # Display result
    print(f"\nResponse:\n{response}\n")

    # Summary
    print(f"Total latency: {retrieval_latency + generation_latency}ms")
    print(f"Chunks used: {len(retrieval_result['matches'])}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Observability Examples - Noah's AI Assistant")
    print("=" * 60)

    try:
        # Example 1: Basic traced query
        basic_traced_query()

        # Example 2: Manual metrics
        manual_metrics_collection()

        # Example 3: Token usage
        token_usage_tracking()

        # Example 4: Quality comparison
        compare_retrieval_quality()

        # Example 5: End-to-end
        end_to_end_example()

        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. View traces: https://smith.langchain.com/")
        print("2. Try docs/OBSERVABILITY_GUIDE.md")
        print("3. Experiment with evaluation (see observability_evaluation.py)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check .env configuration")
        print("2. Ensure dependencies installed: pip install langsmith")
        print("3. See docs/LANGSMITH_SETUP.md")


if __name__ == "__main__":
    main()
