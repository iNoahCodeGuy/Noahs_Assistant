"""Example: LLM-as-Judge Evaluation

This example demonstrates quality evaluation using LLM-as-judge:
- Faithfulness scoring
- Relevance evaluation
- Answer quality assessment
- Batch evaluation with sampling

Setup:
    1. Ensure OPENAI_API_KEY is set in .env
    2. Optional: Set LANGCHAIN_TRACING_V2=true for tracing
    3. Run this script

Cost: ~$0.0004 per evaluation (GPT-3.5-turbo)
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from observability import (
    evaluate_faithfulness,
    evaluate_relevance,
    evaluate_answer_quality,
    evaluate_response,
    batch_evaluate_responses
)

from core.rag_engine import RagEngine


def example_faithfulness_evaluation():
    """Example 1: Evaluate if response is faithful to context."""
    print("\n=== Example 1: Faithfulness Evaluation ===")

    query = "What programming languages does Noah know?"

    context = [
        "Noah is proficient in Python, JavaScript, and TypeScript.",
        "He has experience with React, FastAPI, and LangChain.",
        "Noah specializes in full-stack development and AI systems."
    ]

    # Good answer (faithful to context)
    good_answer = "Noah is proficient in Python, JavaScript, and TypeScript. He has experience with frameworks like React and FastAPI."

    # Bad answer (hallucination)
    bad_answer = "Noah is an expert in Ruby, PHP, and C++."

    print("\n📊 Evaluating Good Answer:")
    print(f"Answer: {good_answer}")
    score, explanation = evaluate_faithfulness(query, context, good_answer)
    print(f"Faithfulness Score: {score:.2f}")
    print(f"Explanation: {explanation}")

    print("\n📊 Evaluating Bad Answer (Hallucination):")
    print(f"Answer: {bad_answer}")
    score, explanation = evaluate_faithfulness(query, context, bad_answer)
    print(f"Faithfulness Score: {score:.2f}")
    print(f"Explanation: {explanation}")


def example_relevance_evaluation():
    """Example 2: Evaluate if retrieved context is relevant."""
    print("\n=== Example 2: Relevance Evaluation ===")

    query = "What AI technologies has Noah worked with?"

    # Relevant context
    relevant_context = [
        "Noah built a RAG system using pgvector and LangChain.",
        "He integrated LangSmith for tracing and evaluation.",
        "His AI assistant uses OpenAI GPT-4 for generation."
    ]

    # Irrelevant context
    irrelevant_context = [
        "Noah enjoys playing guitar in his free time.",
        "He has a degree in computer science.",
        "Noah lives in California."
    ]

    print("\n📊 Evaluating Relevant Context:")
    score, explanation = evaluate_relevance(query, relevant_context)
    print(f"Relevance Score: {score:.2f}")
    print(f"Explanation: {explanation}")

    print("\n📊 Evaluating Irrelevant Context:")
    score, explanation = evaluate_relevance(query, irrelevant_context)
    print(f"Relevance Score: {score:.2f}")
    print(f"Explanation: {explanation}")


def example_answer_quality_evaluation():
    """Example 3: Evaluate overall answer quality."""
    print("\n=== Example 3: Answer Quality Evaluation ===")

    query = "Explain Noah's RAG architecture"

    # High-quality answer
    good_answer = """Noah's RAG architecture uses Supabase pgvector for vector storage,
    providing centralized and scalable retrieval. The system embeds queries using OpenAI's
    text-embedding-3-small model, performs similarity search, and generates responses with GPT-4.
    It includes observability through LangSmith tracing and evaluation metrics."""

    # Low-quality answer
    poor_answer = "Noah uses RAG. It works with AI."

    print("\n📊 Evaluating High-Quality Answer:")
    score, explanation = evaluate_answer_quality(query, good_answer)
    print(f"Quality Score: {score:.2f}")
    print(f"Explanation: {explanation}")

    print("\n📊 Evaluating Low-Quality Answer:")
    score, explanation = evaluate_answer_quality(query, poor_answer)
    print(f"Quality Score: {score:.2f}")
    print(f"Explanation: {explanation}")


def example_comprehensive_evaluation():
    """Example 4: Comprehensive evaluation (all metrics)."""
    print("\n=== Example 4: Comprehensive Evaluation ===")

    query = "What are Noah's technical strengths?"

    context = [
        "Noah has 5+ years of experience in Python and JavaScript.",
        "He specializes in RAG systems and AI applications.",
        "Noah is proficient in FastAPI, React, and LangChain."
    ]

    answer = """Noah's technical strengths include extensive experience with Python and
    JavaScript, specializing in building RAG systems and AI applications. He's proficient
    in modern frameworks like FastAPI for backend development and React for frontend work,
    and has deep expertise in LangChain for AI orchestration."""

    print(f"\nQuery: {query}")
    print(f"\nAnswer: {answer}")
    print("\nEvaluating...")

    # Comprehensive evaluation
    metrics = evaluate_response(query, context, answer)

    print("\n📊 Evaluation Results:")
    print(f"Faithfulness:     {metrics.faithfulness_score:.2f} / 1.00")
    print(f"Relevance:        {metrics.relevance_score:.2f} / 1.00")
    print(f"Answer Quality:   {metrics.answer_quality_score:.2f} / 1.00")
    print(f"Groundedness:     {metrics.groundedness:.2f} / 1.00")
    print(f"Overall Score:    {metrics.overall_score():.2f} / 1.00")
    print(f"\nExplanation: {metrics.explanation}")


def example_real_world_evaluation():
    """Example 5: Evaluate real RAG responses."""
    print("\n=== Example 5: Real-World RAG Evaluation ===")

    engine = RagEngine()

    queries = [
        "What programming languages does Noah know?",
        "Tell me about Noah's AI projects",
        "What frameworks has Noah used?"
    ]

    print("\nEvaluating real RAG responses...\n")
    print("-" * 80)

    for query in queries:
        print(f"\n🔍 Query: {query}")

        # Retrieve and generate
        retrieval_result = engine.retrieve(query, top_k=3)
        context = retrieval_result['matches']
        answer = engine.generate_response(query)

        print(f"✅ Answer: {answer[:100]}...")

        # Evaluate (only show scores, not full explanation for brevity)
        metrics = evaluate_response(query, context, answer)

        print(f"📊 Scores:")
        print(f"   Faithfulness: {metrics.faithfulness_score:.2f}")
        print(f"   Relevance:    {metrics.relevance_score:.2f}")
        print(f"   Quality:      {metrics.answer_quality_score:.2f}")
        print(f"   Overall:      {metrics.overall_score():.2f}")


def example_batch_evaluation_with_sampling():
    """Example 6: Batch evaluation with cost-saving sampling."""
    print("\n=== Example 6: Batch Evaluation with Sampling ===")

    # Simulate multiple responses
    responses = [
        {
            'query': f"Query {i}: What is Noah's experience?",
            'context': ["Noah has 5+ years experience...", "..."],
            'answer': f"Response {i}: Noah has extensive experience..."
        }
        for i in range(1, 21)  # 20 responses
    ]

    print(f"\nTotal responses: {len(responses)}")
    print("Sampling rate: 10% (to save costs)")

    # Evaluate with sampling
    results = batch_evaluate_responses(
        responses=responses,
        sample_rate=0.1  # Only evaluate 10%
    )

    print(f"Evaluated: {len(results)} / {len(responses)} responses")

    if results:
        avg_faithfulness = sum(r.faithfulness_score for r in results) / len(results)
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        avg_quality = sum(r.answer_quality_score for r in results) / len(results)

        print(f"\n📊 Aggregate Metrics:")
        print(f"Avg Faithfulness: {avg_faithfulness:.2f}")
        print(f"Avg Relevance:    {avg_relevance:.2f}")
        print(f"Avg Quality:      {avg_quality:.2f}")

        # Cost estimation
        cost_per_eval = 0.0004  # ~$0.0004 per evaluation with GPT-3.5-turbo
        total_cost = len(results) * cost_per_eval
        print(f"\n💰 Evaluation Cost: ${total_cost:.4f}")


def example_production_sampling():
    """Example 7: Production-ready evaluation with sampling."""
    print("\n=== Example 7: Production Sampling Pattern ===")

    from observability.evaluators import should_evaluate_sample

    # Simulate 100 queries
    print("\nSimulating 100 queries with 10% sampling...")

    evaluated_count = 0
    scores = []

    for i in range(100):
        # Check if should evaluate
        if should_evaluate_sample(sample_rate=0.1):
            # Evaluate this query
            evaluated_count += 1
            scores.append(0.85)  # Simulated score

    print(f"Evaluated: {evaluated_count} / 100 queries")
    print(f"Sampling rate: ~{evaluated_count}%")

    # Cost analysis
    cost_per_eval = 0.0004
    daily_queries = 1000
    monthly_queries = daily_queries * 30

    print(f"\n💰 Cost Projection (10% sampling):")
    print(f"Daily queries:     {daily_queries}")
    print(f"Daily evaluations: {int(daily_queries * 0.1)}")
    print(f"Daily cost:        ${daily_queries * 0.1 * cost_per_eval:.2f}")
    print(f"Monthly cost:      ${monthly_queries * 0.1 * cost_per_eval:.2f}")


def main():
    """Run all evaluation examples."""
    print("=" * 80)
    print("LLM-as-Judge Evaluation Examples - Noah's AI Assistant")
    print("=" * 80)

    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY not set")
        print("Please add OPENAI_API_KEY to your .env file")
        return

    try:
        # Example 1: Faithfulness
        example_faithfulness_evaluation()

        # Example 2: Relevance
        example_relevance_evaluation()

        # Example 3: Answer quality
        example_answer_quality_evaluation()

        # Example 4: Comprehensive
        example_comprehensive_evaluation()

        # Example 5: Real-world
        # example_real_world_evaluation()  # Uncomment to test with actual RAG

        # Example 6: Batch with sampling
        # example_batch_evaluation_with_sampling()  # Uncomment to test (costs ~$0.01)

        # Example 7: Production pattern
        example_production_sampling()

        print("\n" + "=" * 80)
        print("✅ All examples completed!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("1. Use sampling (10-20%) in production to save costs")
        print("2. Faithfulness detects hallucinations")
        print("3. Relevance identifies poor retrieval")
        print("4. Quality assesses overall helpfulness")
        print("5. Cost: ~$1-5/month for typical production usage")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
