#!/usr/bin/env python3
"""
Verify pgvector IVFFLAT index setup and test retrieval performance.

This script assumes you've already run the migration SQL in Supabase.
It verifies the setup and benchmarks performance.

Usage:
    python scripts/verify_pgvector_setup.py

Requirements:
    - Migration SQL already executed in Supabase SQL Editor
    - SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env
"""

import sys
import logging
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant.config.supabase_config import get_supabase_client, supabase_settings
from assistant.retrieval.pgvector_retriever import PgVectorRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_function_exists(supabase) -> bool:
    """Verify the match_kb_chunks RPC function exists."""
    logger.info("\n1. Verifying match_kb_chunks function exists...")

    try:
        # Try calling the function with a dummy embedding
        dummy_embedding = [0.0] * 1536
        result = supabase.rpc(
            'match_kb_chunks',
            {
                'query_embedding': dummy_embedding,
                'match_threshold': 0.99,  # High threshold so we get 0 results
                'match_count': 1
            }
        ).execute()

        logger.info("   ✅ match_kb_chunks function exists and is callable")
        return True

    except Exception as e:
        error_msg = str(e).lower()
        if 'function' in error_msg or 'does not exist' in error_msg or 'not found' in error_msg:
            logger.error("   ❌ match_kb_chunks function not found")
            logger.error("\n" + "="*80)
            logger.error("MIGRATION NOT RUN YET")
            logger.error("="*80)
            logger.error("\nPlease run the migration SQL first:")
            logger.error("1. Open Supabase Dashboard → SQL Editor")
            logger.error("2. Copy contents of: supabase/migrations/20250205_native_pgvector_search.sql")
            logger.error("3. Paste and click 'RUN'")
            logger.error("4. Run this script again to verify")
            logger.error("="*80 + "\n")
            return False
        else:
            # Function exists but had a different error
            logger.warning(f"   ⚠️  Function callable but returned unexpected error: {e}")
            return True


def check_kb_chunks_count(supabase) -> int:
    """Check how many KB chunks exist."""
    logger.info("\n2. Checking KB chunks count...")

    try:
        result = supabase.table('kb_chunks').select('id', count='exact').execute()
        count = result.count
        logger.info(f"   ✅ Found {count} KB chunks in database")
        return count
    except Exception as e:
        logger.error(f"   ❌ Failed to query kb_chunks table: {e}")
        return 0


def benchmark_retrieval(retriever: PgVectorRetriever) -> None:
    """Benchmark retrieval performance with IVFFLAT index."""
    logger.info("\n3. Benchmarking retrieval performance...")

    test_queries = [
        "What programming languages does Noah know?",
        "Tell me about Noah's RAG pipeline architecture",
        "What is Noah's MMA background?",
        "How does the vector search work?",
        "What are Noah's technical skills?"
    ]

    latencies = []
    results_summary = []

    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n   Test {i}/{len(test_queries)}: '{query[:60]}...'")

        try:
            start = time.time()
            chunks = retriever.retrieve(query, top_k=3, threshold=0.60)
            latency = (time.time() - start) * 1000  # Convert to ms
            latencies.append(latency)

            if chunks:
                logger.info(f"   ✅ Retrieved {len(chunks)} chunks in {latency:.0f}ms")
                logger.info(f"      Top similarity: {chunks[0]['similarity']:.3f}")
                logger.info(f"      Preview: {chunks[0]['content'][:80]}...")

                results_summary.append({
                    'query': query,
                    'latency_ms': latency,
                    'num_chunks': len(chunks),
                    'top_similarity': chunks[0]['similarity']
                })
            else:
                logger.warning(f"   ⚠️  No chunks retrieved (similarity below threshold)")

        except Exception as e:
            logger.error(f"   ❌ Retrieval failed: {e}")
            raise

    # Print summary
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)

        logger.info("\n" + "="*80)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*80)
        logger.info(f"Total queries: {len(latencies)}")
        logger.info(f"Average latency: {avg_latency:.0f}ms")
        logger.info(f"Min latency: {min_latency:.0f}ms")
        logger.info(f"Max latency: {max_latency:.0f}ms")
        logger.info("\nExpected with IVFFLAT index: 50-150ms")
        logger.info("Without index: 200-500ms")

        if avg_latency < 150:
            logger.info("\n✅ Performance is excellent! IVFFLAT index is working.")
        elif avg_latency < 300:
            logger.info("\n⚠️  Performance is OK but could be better.")
            logger.info("   Index might not be optimally configured or not being used.")
        else:
            logger.info("\n❌ Performance is slow. Index may not exist or not being used.")
            logger.info("   Check if the IVFFLAT index was created successfully.")

        logger.info("="*80)


def test_with_and_without_threshold(retriever: PgVectorRetriever) -> None:
    """Test retrieval with different thresholds."""
    logger.info("\n4. Testing different similarity thresholds...")

    query = "Tell me about Noah's technical projects"
    thresholds = [0.50, 0.60, 0.70, 0.80]

    logger.info(f"\n   Query: '{query}'")

    for threshold in thresholds:
        try:
            chunks = retriever.retrieve(query, top_k=5, threshold=threshold)
            if chunks:
                avg_sim = sum(c['similarity'] for c in chunks) / len(chunks)
                logger.info(f"   Threshold {threshold:.2f}: {len(chunks)} chunks (avg sim: {avg_sim:.3f})")
            else:
                logger.info(f"   Threshold {threshold:.2f}: No results")
        except Exception as e:
            logger.error(f"   Threshold {threshold:.2f}: Error - {e}")


def main():
    """Main verification flow."""
    logger.info("="*80)
    logger.info("PGVECTOR SETUP VERIFICATION & PERFORMANCE TEST")
    logger.info("="*80)

    try:
        # Validate configuration
        logger.info("\nValidating Supabase configuration...")
        supabase_settings.validate_supabase()
        logger.info("✅ Configuration valid\n")

        # Get Supabase client
        supabase = get_supabase_client()

        # Verify function exists
        function_exists = verify_function_exists(supabase)
        if not function_exists:
            logger.error("\n❌ Setup incomplete. Please run the migration first.")
            sys.exit(1)

        # Check KB chunks
        chunk_count = check_kb_chunks_count(supabase)
        if chunk_count == 0:
            logger.warning("\n⚠️  No KB chunks found. You may need to populate the database.")
            logger.info("   Run: python scripts/migrate_all_kb_to_supabase.py")
            sys.exit(1)

        # Create retriever and benchmark
        retriever = PgVectorRetriever(similarity_threshold=0.60)

        # Run benchmarks
        benchmark_retrieval(retriever)
        test_with_and_without_threshold(retriever)

        # Success!
        logger.info("\n" + "="*80)
        logger.info("✅ VERIFICATION COMPLETE!")
        logger.info("="*80)
        logger.info("\nYour pgvector search is working with IVFFLAT indexing.")
        logger.info("The assistant is ready to use!")
        logger.info("="*80 + "\n")

    except Exception as e:
        logger.error(f"\n❌ Verification failed: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Check .env has SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
        logger.error("2. Run the migration SQL in Supabase SQL Editor")
        logger.error("3. Verify pgvector extension is enabled")
        sys.exit(1)


if __name__ == "__main__":
    main()
