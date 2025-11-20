"""Enhanced data migration script for all Noah's AI Assistant knowledge bases.

This script migrates ALL knowledge bases to Supabase:
- career_kb.csv → Career history, achievements, experience
- technical_kb.csv → Technical implementations, RAG details, system design
- architecture_kb.csv → System architecture diagrams, code examples
- docs/ folder → Documentation files (markdown) with intelligent chunking

Usage:
    python scripts/migrate_all_kb_to_supabase.py
    python scripts/migrate_all_kb_to_supabase.py --force  # Re-import all
    python scripts/migrate_all_kb_to_supabase.py --kb technical_kb  # Just one KB
    python scripts/migrate_all_kb_to_supabase.py --kb documentation  # Just docs folder
"""

import sys
import os
import csv
import time
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openai import OpenAI
from assistant.config.supabase_config import get_supabase_client, supabase_settings

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
        'doc_id': 'career_kb',
        'type': 'csv'
    },
    'technical_kb': {
        'path': 'data/technical_kb.csv',
        'description': 'Technical implementations, RAG system details',
        'doc_id': 'technical_kb',
        'type': 'csv'
    },
    'architecture_kb': {
        'path': 'data/architecture_kb.csv',
        'description': 'System architecture, diagrams, code examples',
        'doc_id': 'architecture_kb',
        'type': 'csv'
    },
    'documentation': {
        'path': 'docs',
        'description': 'Documentation files (markdown) with intelligent chunking',
        'doc_id': 'documentation',
        'type': 'docs_folder'
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

    def read_docs_folder(self, docs_path: str = "docs") -> List[Dict[str, str]]:
        """Recursively read all markdown files from docs/ directory.

        Purpose: Collect all documentation files for indexing.

        Args:
            docs_path: Path to docs directory (default "docs")

        Returns:
            List of dicts with 'file_path' and 'content' keys
        """
        logger.info(f"📄 Reading documentation from {docs_path}...")

        docs_dir = Path(docs_path)
        if not docs_dir.exists():
            logger.error(f"Documentation directory not found: {docs_path}")
            return []

        files = []
        for md_file in docs_dir.rglob("*.md"):
            # Skip archive and certain patterns
            if "archive" in str(md_file) or "__pycache__" in str(md_file):
                continue

            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Get relative path from project root
                rel_path = str(md_file.relative_to(Path('.')))

                files.append({
                    'file_path': rel_path,
                    'content': content
                })
            except Exception as e:
                logger.warning(f"Failed to read {md_file}: {e}")

        logger.info(f"   ✅ Found {len(files)} markdown files")
        return files

    def chunk_by_sections(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Chunk markdown content by section headers with overlap.

        Purpose: Intelligent chunking preserves document structure while
        maintaining context across section boundaries.

        Strategy:
        - Split on ## and ### headers (major sections)
        - Each chunk includes section title + content
        - Preserve structure for better semantic search

        Args:
            content: Full markdown content
            file_path: Source file path for metadata

        Returns:
            List of chunk dicts with doc_id, section, content, metadata
        """
        chunks = []
        lines = content.split('\n')

        current_section = "Introduction"
        current_content = []
        section_start = 0

        for i, line in enumerate(lines):
            # Detect section headers (## or ###)
            if line.startswith('##'):
                # Save previous section if it has content
                if current_content:
                    section_text = '\n'.join(current_content)
                    if len(section_text.strip()) > 50:  # Minimum chunk size
                        chunks.append({
                            'section': current_section,
                            'content': section_text,
                            'line_start': section_start,
                            'line_end': i - 1
                        })

                # Start new section
                current_section = line.lstrip('#').strip()
                current_content = [line]  # Include header in chunk
                section_start = i
            else:
                current_content.append(line)

        # Add final section
        if current_content:
            section_text = '\n'.join(current_content)
            if len(section_text.strip()) > 50:
                chunks.append({
                    'section': current_section,
                    'content': section_text,
                    'line_start': section_start,
                    'line_end': len(lines) - 1
                })

        # If no sections found, create single chunk
        if not chunks:
            chunks.append({
                'section': os.path.basename(file_path),
                'content': content,
                'line_start': 0,
                'line_end': len(lines) - 1
            })

        # Add metadata to each chunk
        for idx, chunk in enumerate(chunks):
            chunk['doc_id'] = 'documentation'
            chunk['metadata'] = {
                'file_path': file_path,
                'section_title': chunk['section'],
                'chunk_index': idx,
                'indexed_at': datetime.utcnow().isoformat()
            }
            # Generate content hash for idempotency
            chunk['content_hash'] = hashlib.sha256(chunk['content'].encode('utf-8')).hexdigest()[:16]

        return chunks

    def create_chunks(self, rows: List[Dict], doc_id: str, kb_type: str = 'csv') -> List[Dict[str, Any]]:
        """Create chunks from KB rows or documentation files.

        Args:
            rows: List of rows (CSV rows or file dicts)
            doc_id: Document identifier
            kb_type: Type of KB ('csv' or 'docs_folder')
        """
        chunks = []

        if kb_type == 'csv':
            # CSV-based chunking
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
        elif kb_type == 'docs_folder':
            # Documentation-based chunking
            for file_data in rows:
                file_chunks = self.chunk_by_sections(file_data['content'], file_data['file_path'])
                chunks.extend(file_chunks)

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
        path = kb_config['path']
        doc_id = kb_config['doc_id']
        kb_type = kb_config.get('type', 'csv')

        logger.info(f"\n{'='*60}")
        logger.info(f"📚 Migrating: {kb_name}")
        logger.info(f"   Description: {kb_config['description']}")
        logger.info(f"   Path: {path}")
        logger.info(f"   Type: {kb_type}")
        logger.info(f"{'='*60}\n")

        # Check if path exists
        if not os.path.exists(path):
            logger.warning(f"⚠️  Path not found: {path}, skipping...")
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
        if kb_type == 'docs_folder':
            rows = self.read_docs_folder(path)
        else:
            rows = self.read_kb_csv(path)

        chunks = self.create_chunks(rows, doc_id, kb_type)
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
    parser.add_argument('--kb', type=str, help='Migrate specific KB only (career_kb, technical_kb, architecture_kb, documentation)')

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
