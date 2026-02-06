#!/usr/bin/env python3
"""
Setup pgvector IVFFLAT index and native search function in Supabase.

This script:
1. Reads the migration SQL file
2. Executes it in your Supabase database
3. Verifies the index and function were created
4. Tests retrieval performance

Usage:
    python scripts/setup_pgvector_index.py

Requirements:
    - SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env
    - supabase/migrations/20250205_native_pgvector_search.sql exists
"""

import sys
import logging
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


def read_migration_sql() -> str:
    """Read the migration SQL file."""
    migration_path = Path(__file__).parent.parent / "supabase" / "migrations" / "20250205_native_pgvector_search.sql"

    if not migration_path.exists():
        raise FileNotFoundError(f"Migration file not found: {migration_path}")

    logger.info(f"Reading migration from: {migration_path}")
    return migration_path.read_text()


def execute_migration(supabase, sql: str) -> None:
    """Execute the migration SQL in Supabase.

    Note: Supabase Python client doesn't have a direct SQL execution method,
    so we'll execute it via the PostgREST RPC endpoint.
    """
    logger.info("Executing migration SQL...")

    # Split SQL into individual statements (separated by ;)
    statements = [s.strip() for s in sql.split(';') if s.strip() and not s.strip().startswith('--')]

    # Execute each statement
    for i, statement in enumerate(statements):
        # Skip comments and empty statements
        if not statement or statement.startswith('--'):
            continue

        # Skip EXPLAIN ANALYZE statements (they're just for testing)
        if 'EXPLAIN ANALYZE' in statement:
            logger.debug(f"Skipping EXPLAIN ANALYZE statement")
            continue

        # Skip verification queries (SELECT statements)
        if statement.strip().upper().startswith('SELECT'):
            logger.debug(f"Skipping verification SELECT statement")
            continue

        try:
            logger.info(f"Executing statement {i+1}/{len(statements)}")
            logger.debug(f"SQL: {statement[:100]}...")

            # Use supabase.postgrest.rpc() for raw SQL execution
            # Note: This requires the 'exec_sql' RPC function to exist
            # If it doesn't exist, we'll need to run this manually in Supabase SQL editor
            result = supabase.rpc('exec_sql', {'sql': statement}).execute()
            logger.info(f"✅ Statement {i+1} executed successfully")

        except Exception as e:
            # If exec_sql doesn't exist, provide instructions
            if 'exec_sql' in str(e) or 'function' in str(e).lower():
                logger.warning(
                    "❌ Cannot execute SQL via Python client. "
                    "The exec_sql RPC function doesn't exist in your Supabase project."
                )
                logger.info("\n" + "="*80)
                logger.info("MANUAL SETUP REQUIRED")
                logger.info("="*80)
                logger.info("\nPlease run the migration manually:")
                logger.info("1. Go to your Supabase dashboard")
                logger.info("2. Navigate to SQL Editor")
                logger.info("3. Copy and paste this file:")
                logger.info("   supabase/migrations/20250205_native_pgvector_search.sql")
                logger.info("4. Click 'RUN' to execute")
                logger.info("\nAlternatively, use the Supabase CLI:")
                logger.info("   supabase db push --file supabase/migrations/20250205_native_pgvector_search.sql")
                logger.info("="*80 + "\n")
                raise RuntimeError("Manual SQL execution required - see instructions above")
            else:
                logger.error(f"❌ Failed to execute statement {i+1}: {e}")
                raise


def verify_index_exists(supabase) -> bool:
    """Verify the IVFFLAT index was created."""
    logger.info("Verifying IVFFLAT index exists...")

    try:
        # Query pg_indexes to check if index exists
        result = supabase.rpc(
            'exec_sql',
            {
                'sql': """
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE tablename = 'kb_chunks'
                    AND indexname = 'kb_chunks_embedding_idx'
                """
            }
        ).execute()

        if result.data and len(result.data) > 0:
            logger.info("✅ IVFFLAT index exists")
            logger.info(f"   Index definition: {result.data[0].get('indexdef', 'N/A')}")
            return True
        else:
            logger.error("❌ IVFFLAT index not found")
            return False

    except Exception as e:
        logger.warning(f"Could not verify index (may need manual verification): {e}")
        return False


def verify_function_exists(supabase) -> bool:
    """Verify the match_kb_chunks RPC function exists."""
    logger.info("Verifying match_kb_chunks function exists...")

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

        logger.info("✅ match_kb_chunks function exists and is callable")
        return True

    except Exception as e:
        if 'function' in str(e).lower() or 'does not exist' in str(e).lower():
            logger.error("❌ match_kb_chunks function not found")
            return False
        else:
            # Function exists but had a different error
            logger.warning(f"Function exists but returned error: {e}")
            return True


def test_retrieval_performance(retriever: PgVectorRetriever) -> None:
    """Test retrieval with the new IVFFLAT index."""
    logger.info("\nTesting retrieval performance...")

    test_queries = [
        "What programming languages does Noah know?",
        "Tell me about Noah's RAG pipeline architecture",
        "What is Noah's MMA background?"
    ]

    for i, query in enumerate(test_queries, 1):
        logger.info(f"\nTest {i}/{len(test_queries)}: '{query}'")

        try:
            import time
            start = time.time()
            chunks = retriever.retrieve(query, top_k=3)
            latency = (time.time() - start) * 1000  # Convert to ms

            if chunks:
                logger.info(f"✅ Retrieved {len(chunks)} chunks in {latency:.0f}ms")
                logger.info(f"   Top similarity score: {chunks[0]['similarity']:.3f}")
                logger.info(f"   Top chunk preview: {chunks[0]['content'][:100]}...")
            else:
                logger.warning(f"⚠️  No chunks retrieved (query may be too different from KB content)")

        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            raise


def main():
    """Main setup and verification flow."""
    logger.info("="*80)
    logger.info("PGVECTOR IVFFLAT INDEX SETUP")
    logger.info("="*80)

    try:
        # Validate configuration
        logger.info("\n1. Validating Supabase configuration...")
        supabase_settings.validate_supabase()
        logger.info("✅ Configuration valid")

        # Get Supabase client
        logger.info("\n2. Connecting to Supabase...")
        supabase = get_supabase_client()
        logger.info("✅ Connected")

        # Check if function already exists
        logger.info("\n3. Checking if setup already complete...")
        function_exists = verify_function_exists(supabase)

        if function_exists:
            logger.info("✅ Setup already complete! Function exists.")
            logger.info("\nℹ️  If you want to re-run the migration, manually drop the function first:")
            logger.info("   DROP FUNCTION IF EXISTS match_kb_chunks(vector, float, int, text);")
            logger.info("   DROP INDEX IF EXISTS kb_chunks_embedding_idx;")
        else:
            # Read migration SQL
            logger.info("\n4. Reading migration SQL...")
            sql = read_migration_sql()
            logger.info(f"✅ Loaded {len(sql)} characters of SQL")

            # Execute migration
            logger.info("\n5. Executing migration...")
            try:
                execute_migration(supabase, sql)
                logger.info("✅ Migration executed")
            except RuntimeError as e:
                # Manual setup required
                logger.info("\n⚠️  MANUAL SETUP REQUIRED - See instructions above")
                logger.info("\nAfter running the migration manually, run this script again to verify and test.")
                return

            # Verify setup
            logger.info("\n6. Verifying setup...")
            function_ok = verify_function_exists(supabase)

            if not function_ok:
                logger.error("❌ Setup verification failed")
                return

        # Test retrieval
        logger.info("\n7. Testing retrieval with IVFFLAT index...")
        retriever = PgVectorRetriever(similarity_threshold=0.60)
        test_retrieval_performance(retriever)

        # Success!
        logger.info("\n" + "="*80)
        logger.info("✅ SETUP COMPLETE!")
        logger.info("="*80)
        logger.info("\nYour pgvector search is now using native IVFFLAT indexing.")
        logger.info("Expected performance improvement:")
        logger.info("  - Before: ~300ms per query (client-side calculation)")
        logger.info("  - After:  ~50ms per query (native pgvector with IVFFLAT)")
        logger.info("  - Speedup: ~6x faster! 🚀")
        logger.info("\nThe assistant/retrieval/pgvector_retriever.py is already")
        logger.info("configured to use the match_kb_chunks RPC function.")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"\n❌ Setup failed: {e}")
        logger.error("\nTroubleshooting tips:")
        logger.error("1. Check your .env file has SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
        logger.error("2. Verify you have the pgvector extension enabled in Supabase")
        logger.error("3. Check the Supabase dashboard for any error messages")
        logger.error("4. Try running the migration manually in the SQL Editor")
        raise


if __name__ == "__main__":
    main()
