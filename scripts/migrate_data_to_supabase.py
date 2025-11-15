"""Data migration script for Noah's AI Assistant.

This script reads the career knowledge base CSV, generates OpenAI embeddings,
and populates the Supabase kb_chunks table with pgvector data.

Key features:
- Batch embedding generation (up to 100 texts per API call)
- Idempotent inserts (checks for existing records)
- Exponential backoff retry logic for API failures
- Progress tracking with cost estimates
- Structured logging for observability

Usage:
    python scripts/migrate_data_to_supabase.py

Requirements:
    - OPENAI_API_KEY in environment
    - SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in environment
    - data/career_kb.csv file present
"""

import sys
import os
import csv
import time
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openai import OpenAI
from assistant.config.supabase_config import get_supabase_client, supabase_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
BATCH_SIZE = 100  # OpenAI allows up to 2048, but 100 is safer
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


class DataMigration:
    """Handles migration of career KB data to Supabase with embeddings.

    Why this class structure:
    - Encapsulates state (OpenAI client, Supabase client, stats)
    - Easy to test with dependency injection
    - Clear separation of concerns (read, embed, insert)
    """

    def __init__(self):
        """Initialize migration with OpenAI and Supabase clients."""
        self.openai_client = OpenAI(api_key=supabase_settings.api_key)
        self.supabase_client = get_supabase_client()

        # Track migration stats
        self.stats = {
            'rows_read': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'chunks_inserted': 0,
            'chunks_skipped': 0,
            'api_calls': 0,
            'failures': 0,
            'total_cost': 0.0,
            'start_time': time.time()
        }

    def read_career_kb(self, csv_path: str) -> List[Dict[str, str]]:
        """Read career knowledge base from CSV.

        Args:
            csv_path: Path to career_kb.csv

        Returns:
            List of dicts with 'question' and 'answer' keys

        Why CSV:
        - Simple, human-editable format
        - Easy for non-technical team members to update
        - Can be version-controlled in Git
        """
        logger.info(f"📄 Reading {csv_path}...")

        rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    'question': row['Question'].strip(),
                    'answer': row['Answer'].strip()
                })

        self.stats['rows_read'] = len(rows)
        logger.info(f"   Found {len(rows)} rows")
        return rows

    def create_chunks(self, rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Convert CSV rows into chunks for embedding.

        Why this structure:
        - Each Q&A pair becomes one chunk
        - Combines question + answer for better context
        - Adds metadata for filtering and debugging

        Args:
            rows: List of question/answer dicts

        Returns:
            List of chunk dicts with doc_id, section, content, metadata
        """
        logger.info("🔢 Creating chunks...")

        chunks = []
        for idx, row in enumerate(rows):
            # Combine question and answer for richer context
            content = f"Q: {row['question']}\nA: {row['answer']}"

            chunk = {
                'doc_id': 'career_kb',
                'section': row['question'][:100],  # Use question as section name
                'content': content,
                'metadata': {
                    'source': 'career_kb.csv',
                    'row_index': idx,
                    'question': row['question'],
                    'migrated_at': datetime.utcnow().isoformat()
                }
            }
            chunks.append(chunk)

        self.stats['chunks_created'] = len(chunks)
        logger.info(f"   Created {len(chunks)} chunks")
        return chunks

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts with retry logic.

        Why batching:
        - OpenAI allows up to 2048 texts per call (we use 100 for safety)
        - Reduces API calls by 100x
        - Faster overall migration
        - Lower cost (fewer network round trips)

        Why retry logic:
        - API can fail due to rate limits (429)
        - Network issues (timeouts, 5xx errors)
        - Exponential backoff prevents hammering the API

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is list of 1536 floats)
        """
        for attempt in range(MAX_RETRIES):
            try:
                self.stats['api_calls'] += 1

                response = self.openai_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=texts,
                    encoding_format="float"
                )

                # Extract embeddings from response
                embeddings = [item.embedding for item in response.data]

                # Calculate cost (text-embedding-3-small: $0.00002 per 1K tokens)
                # Rough estimate: 1 token ≈ 4 chars
                total_chars = sum(len(text) for text in texts)
                tokens = total_chars / 4
                cost = (tokens / 1000) * 0.00002
                self.stats['total_cost'] += cost

                return embeddings

            except Exception as e:
                logger.warning(f"   Embedding attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")

                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    logger.info(f"   Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error("   All retry attempts exhausted")
                    self.stats['failures'] += 1
                    raise

    def generate_all_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for all chunks in batches.

        Args:
            chunks: List of chunk dicts

        Returns:
            Same chunks with 'embedding' key added
        """
        logger.info("🧠 Generating embeddings...")

        # Extract texts for embedding
        texts = [chunk['content'] for chunk in chunks]

        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

            logger.info(f"   Batch {batch_num}/{total_batches}: Processing {len(batch_texts)} texts...")

            batch_embeddings = self.generate_embeddings_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)

            # Progress indicator
            progress = min(i + BATCH_SIZE, len(texts))
            pct = (progress / len(texts)) * 100
            logger.info(f"   Progress: {progress}/{len(texts)} ({pct:.1f}%)")

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk['embedding'] = embedding

        self.stats['embeddings_generated'] = len(all_embeddings)
        logger.info(f"   ✅ Generated {len(all_embeddings)} embeddings")
        logger.info(f"   💰 Estimated cost: ${self.stats['total_cost']:.4f}")

        return chunks

    def check_existing_chunks(self, doc_id: str) -> int:
        """Check if chunks already exist for this doc_id.

        Why idempotency matters:
        - Script can be re-run safely after failures
        - Prevents duplicate data
        - Allows incremental updates

        Args:
            doc_id: Document identifier (e.g., 'career_kb')

        Returns:
            Count of existing chunks
        """
        try:
            result = self.supabase_client.table('kb_chunks')\
                .select('id', count='exact')\
                .eq('doc_id', doc_id)\
                .execute()

            count = result.count if hasattr(result, 'count') else len(result.data)
            return count
        except Exception as e:
            logger.warning(f"Failed to check existing chunks: {e}")
            return 0

    def insert_chunks_batch(self, chunks: List[Dict[str, Any]]) -> int:
        """Insert a batch of chunks into Supabase.

        Args:
            chunks: List of chunk dicts with embeddings

        Returns:
            Number of chunks successfully inserted
        """
        try:
            # Prepare data for insertion
            insert_data = []
            for chunk in chunks:
                insert_data.append({
                    'doc_id': chunk['doc_id'],
                    'section': chunk['section'],
                    'content': chunk['content'],
                    'embedding': chunk['embedding'],
                    'metadata': chunk['metadata']
                })

            result = self.supabase_client.table('kb_chunks').insert(insert_data).execute()
            return len(result.data) if result.data else 0

        except Exception as e:
            logger.error(f"Failed to insert batch: {e}")
            self.stats['failures'] += 1
            return 0

    def insert_all_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 50):
        """Insert all chunks into Supabase in batches.

        Why batch inserts:
        - Supabase has payload size limits
        - Better error isolation (one bad chunk doesn't fail all)
        - Progress tracking

        Args:
            chunks: List of chunks with embeddings
            batch_size: Number of chunks per insert (50 is safe for pgvector)
        """
        logger.info("💾 Inserting chunks to Supabase...")

        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1

            logger.info(f"   Batch {batch_num}/{total_batches}: Inserting {len(batch)} chunks...")

            inserted = self.insert_chunks_batch(batch)
            self.stats['chunks_inserted'] += inserted

            # Progress indicator
            progress = min(i + batch_size, len(chunks))
            pct = (progress / len(chunks)) * 100
            logger.info(f"   Progress: {progress}/{len(chunks)} ({pct:.1f}%)")

        logger.info(f"   ✅ Inserted {self.stats['chunks_inserted']} chunks")

    def run(self, csv_path: str, force: bool = False):
        """Execute the full migration pipeline.

        Args:
            csv_path: Path to career_kb.csv
            force: If True, delete existing chunks and re-import
        """
        logger.info("🚀 Starting data migration to Supabase...")
        logger.info(f"   Embedding model: {EMBEDDING_MODEL}")
        logger.info(f"   Dimensions: {EMBEDDING_DIMENSIONS}")
        logger.info("")

        # Check for existing data
        existing_count = self.check_existing_chunks('career_kb')
        if existing_count > 0 and not force:
            logger.warning(f"⚠️  Found {existing_count} existing chunks for 'career_kb'")
            logger.warning("   Use --force flag to delete and re-import")
            logger.warning("   Skipping migration...")
            return

        if existing_count > 0 and force:
            logger.info(f"🗑️  Deleting {existing_count} existing chunks...")
            self.supabase_client.table('kb_chunks').delete().eq('doc_id', 'career_kb').execute()
            logger.info("   ✅ Deleted existing chunks")

        try:
            # Step 1: Read CSV
            rows = self.read_career_kb(csv_path)

            # Step 2: Create chunks
            chunks = self.create_chunks(rows)

            # Step 3: Generate embeddings
            chunks = self.generate_all_embeddings(chunks)

            # Step 4: Insert to Supabase
            self.insert_all_chunks(chunks)

            # Print summary
            duration = time.time() - self.stats['start_time']
            logger.info("")
            logger.info("✨ Migration complete!")
            logger.info(f"   📊 Summary:")
            logger.info(f"      CSV rows read: {self.stats['rows_read']}")
            logger.info(f"      Chunks created: {self.stats['chunks_created']}")
            logger.info(f"      Embeddings generated: {self.stats['embeddings_generated']}")
            logger.info(f"      Chunks inserted: {self.stats['chunks_inserted']}")
            logger.info(f"      API calls: {self.stats['api_calls']}")
            logger.info(f"      Failed operations: {self.stats['failures']}")
            logger.info(f"      Total cost: ${self.stats['total_cost']:.4f}")
            logger.info(f"      Duration: {duration:.1f}s")

        except KeyboardInterrupt:
            logger.warning("\n⚠️  Migration interrupted by user")
            logger.info(f"   Processed {self.stats['chunks_inserted']} chunks before interruption")
            sys.exit(1)
        except Exception as e:
            logger.error(f"\n❌ Migration failed: {e}")
            logger.error(f"   Stats at failure: {json.dumps(self.stats, indent=2)}")
            sys.exit(1)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Migrate career KB data to Supabase with embeddings'
    )
    parser.add_argument(
        '--csv',
        default='data/career_kb.csv',
        help='Path to career_kb.csv (default: data/career_kb.csv)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Delete existing chunks and re-import'
    )

    args = parser.parse_args()

    # Validate environment
    if not supabase_settings.api_key:
        logger.error("❌ OPENAI_API_KEY not found in environment")
        sys.exit(1)

    try:
        supabase_settings.validate_supabase()
    except Exception as e:
        logger.error(f"❌ Supabase configuration invalid: {e}")
        sys.exit(1)

    # Check CSV exists
    if not os.path.exists(args.csv):
        logger.error(f"❌ CSV file not found: {args.csv}")
        sys.exit(1)

    # Run migration
    migration = DataMigration()
    migration.run(args.csv, force=args.force)


if __name__ == '__main__':
    main()
