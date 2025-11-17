"""Enhanced data migration script for all Noah's AI Assistant knowledge bases.

This script migrates ALL knowledge bases to Supabase:
- career_kb.csv → Career history, achievements, experience
- technical_kb.csv → Technical implementations, RAG details, system design
- architecture_kb.csv → System architecture diagrams, code examples

Usage:
    python scripts/migrate_all_kb_to_supabase.py
    python scripts/migrate_all_kb_to_supabase.py --force  # Re-import all
    python scripts/migrate_all_kb_to_supabase.py --kb technical_kb  # Just one KB
"""

import sys
import os
import csv
import time
import logging
from typing import List, Dict, Any
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openai import OpenAI
from src.config.supabase_config import get_supabase_client, supabase_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 2

# Knowledge base definitions
KNOWLEDGE_BASES = {
    'career_kb': {
        'path': 'data/career_kb.csv',
        'description': 'Career history, achievements, experience',
        'doc_id': 'career_kb'
    },
    'technical_kb': {
        'path': 'data/technical_kb.csv',
        'description': 'Technical implementations, RAG system details',
        'doc_id': 'technical_kb'
    },
    'architecture_kb': {
        'path': 'data/architecture_kb.csv',
        'description': 'System architecture, diagrams, code examples',
        'doc_id': 'architecture_kb'
    }
}


class EnhancedMigration:
    """Handles migration of multiple knowledge bases to Supabase."""

    def __init__(self):
        self.openai_client = OpenAI(api_key=supabase_settings.api_key)
        self.supabase_client = get_supabase_client()
        self.total_stats = {
            'kbs_migrated': 0,
            'total_chunks': 0,
            'total_embeddings': 0,
            'total_cost': 0.0,
            'start_time': time.time()
        }

    def read_kb_csv(self, csv_path: str) -> List[Dict[str, str]]:
        """Read knowledge base CSV."""
        logger.info(f"📄 Reading {csv_path}...")

        rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle different CSV formats
                question = row.get('Question') or row.get('question') or ''
                answer = row.get('Answer') or row.get('answer') or row.get('content') or ''

                rows.append({
                    'question': question.strip(),
                    'answer': answer.strip()
                })

        logger.info(f"   ✅ Read {len(rows)} rows")
        return rows

    def create_chunks(self, rows: List[Dict], doc_id: str) -> List[Dict[str, Any]]:
        """Create chunks from KB rows."""
        chunks = []

        for i, row in enumerate(rows):
            # Combine question and answer for embedding
            combined_text = f"{row['question']}\n\n{row['answer']}"

            chunk = {
                'doc_id': doc_id,
                'section': f"entry_{i+1}",
                'content': combined_text,
                'metadata': {
                    'question': row['question'],
                    'answer': row['answer'],
                    'index': i
                }
            }
            chunks.append(chunk)

        logger.info(f"   ✅ Created {len(chunks)} chunks")
        return chunks

    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for all chunks."""
        logger.info(f"🧮 Generating embeddings for {len(chunks)} chunks...")

        texts = [chunk['content'] for chunk in chunks]
        embeddings = []

        # Batch process
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]

            for attempt in range(MAX_RETRIES):
                try:
                    response = self.openai_client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=batch
                    )

                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)

                    # Cost tracking (text-embedding-3-small: $0.00002 per 1K tokens)
                    tokens = response.usage.total_tokens
                    cost = (tokens / 1000) * 0.00002
                    self.total_stats['total_cost'] += cost

                    logger.info(f"   Batch {i//BATCH_SIZE + 1}: {len(batch)} embeddings, {tokens} tokens, ${cost:.6f}")
                    break

                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"   Retry {attempt + 1}/{MAX_RETRIES}: {e}")
                        time.sleep(RETRY_DELAY * (2 ** attempt))
                    else:
                        logger.error(f"   Failed after {MAX_RETRIES} attempts: {e}")
                        raise

        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding

        logger.info(f"   ✅ Generated {len(embeddings)} embeddings")
        self.total_stats['total_embeddings'] += len(embeddings)
        return chunks

    def insert_chunks(self, chunks: List[Dict], doc_id: str):
        """Insert chunks into Supabase."""
        logger.info(f"💾 Inserting {len(chunks)} chunks for {doc_id}...")

        # Prepare rows for insertion
        rows = []
        for chunk in chunks:
            rows.append({
                'doc_id': chunk['doc_id'],
                'section': chunk['section'],
                'content': chunk['content'],
                'embedding': chunk['embedding'],
                'metadata': chunk.get('metadata', {})
            })

        # Insert in batches (Supabase limit)
        batch_size = 100
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]

            try:
                self.supabase_client.table('kb_chunks').insert(batch).execute()
                logger.info(f"   Inserted batch {i//batch_size + 1}: {len(batch)} rows")
            except Exception as e:
                logger.error(f"   Failed to insert batch: {e}")
                raise

        logger.info(f"   ✅ Inserted {len(rows)} chunks")
        self.total_stats['total_chunks'] += len(rows)

    def check_existing(self, doc_id: str) -> int:
        """Check if chunks already exist."""
        try:
            result = self.supabase_client.table('kb_chunks').select('id', count='exact').eq('doc_id', doc_id).execute()
            return result.count or 0
        except:
            return 0

    def delete_existing(self, doc_id: str):
        """Delete existing chunks."""
        self.supabase_client.table('kb_chunks').delete().eq('doc_id', doc_id).execute()
        logger.info(f"   🗑️  Deleted existing chunks for {doc_id}")

    def migrate_kb(self, kb_name: str, force: bool = False):
        """Migrate a single knowledge base."""
        kb_config = KNOWLEDGE_BASES[kb_name]
        csv_path = kb_config['path']
        doc_id = kb_config['doc_id']

        logger.info(f"\n{'='*60}")
        logger.info(f"📚 Migrating: {kb_name}")
        logger.info(f"   Description: {kb_config['description']}")
        logger.info(f"   Path: {csv_path}")
        logger.info(f"{'='*60}\n")

        # Check if file exists
        if not os.path.exists(csv_path):
            logger.warning(f"⚠️  File not found: {csv_path}, skipping...")
            return

        # Check for existing data
        existing_count = self.check_existing(doc_id)
        if existing_count > 0:
            if not force:
                logger.warning(f"⚠️  Found {existing_count} existing chunks, skipping (use --force to re-import)")
                return
            else:
                self.delete_existing(doc_id)

        # Migration pipeline
        rows = self.read_kb_csv(csv_path)
        chunks = self.create_chunks(rows, doc_id)
        chunks = self.generate_embeddings(chunks)
        self.insert_chunks(chunks, doc_id)

        self.total_stats['kbs_migrated'] += 1
        logger.info(f"✅ {kb_name} migration complete!\n")

    def migrate_all(self, force: bool = False, specific_kb: str = None):
        """Migrate all knowledge bases."""
        logger.info("\n" + "="*60)
        logger.info("🚀 Noah's AI Assistant - Complete KB Migration")
        logger.info("="*60 + "\n")

        # Determine which KBs to migrate
        if specific_kb:
            if specific_kb not in KNOWLEDGE_BASES:
                logger.error(f"❌ Unknown KB: {specific_kb}")
                logger.info(f"   Available: {', '.join(KNOWLEDGE_BASES.keys())}")
                return
            kbs_to_migrate = [specific_kb]
        else:
            kbs_to_migrate = list(KNOWLEDGE_BASES.keys())

        # Migrate each KB
        for kb_name in kbs_to_migrate:
            try:
                self.migrate_kb(kb_name, force)
            except Exception as e:
                logger.error(f"❌ Failed to migrate {kb_name}: {e}")
                continue

        # Summary
        elapsed = time.time() - self.total_stats['start_time']
        logger.info("\n" + "="*60)
        logger.info("📊 Migration Summary")
        logger.info("="*60)
        logger.info(f"   KBs migrated: {self.total_stats['kbs_migrated']}")
        logger.info(f"   Total chunks: {self.total_stats['total_chunks']}")
        logger.info(f"   Total embeddings: {self.total_stats['total_embeddings']}")
        logger.info(f"   Total cost: ${self.total_stats['total_cost']:.4f}")
        logger.info(f"   Time elapsed: {elapsed:.1f}s")
        logger.info("="*60 + "\n")
        logger.info("✅ All migrations complete!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Migrate all KB data to Supabase')
    parser.add_argument('--force', action='store_true', help='Delete existing and re-import')
    parser.add_argument('--kb', type=str, help='Migrate specific KB only (career_kb, technical_kb, architecture_kb)')

    args = parser.parse_args()

    # Validate environment
    if not supabase_settings.api_key:
        logger.error("❌ OPENAI_API_KEY not found")
        sys.exit(1)

    try:
        supabase_settings.validate_supabase()
    except Exception as e:
        logger.error(f"❌ Supabase config invalid: {e}")
        sys.exit(1)

    # Run migration
    migration = EnhancedMigration()
    migration.migrate_all(force=args.force, specific_kb=args.kb)


if __name__ == '__main__':
    main()
