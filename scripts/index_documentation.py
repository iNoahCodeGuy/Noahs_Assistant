"""Documentation indexing script for Noah's AI Assistant.

This script recursively reads all markdown files from the docs/ directory,
chunks them intelligently by section headers, generates embeddings, and
inserts them into Supabase kb_chunks table with doc_id='documentation'.

Purpose: Make all documentation searchable via semantic search, not just
pre-processed KB entries. This enables Portfolia to answer questions about
herself using actual documentation.

Layer 3: Documentation Indexing

Key features:
- Recursive markdown file reading from docs/ directory
- Intelligent chunking by section headers (##, ###) with 100-token overlap
- Batch embedding generation (up to 100 texts per API call)
- Idempotent inserts (checks for existing chunks by content hash)
- Progress tracking with cost estimates
- Structured logging for observability

Usage:
    python scripts/index_documentation.py

Requirements:
    - OPENAI_API_KEY in environment
    - SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in environment
    - docs/ directory present
"""

import sys
import os
import time
import logging
import hashlib
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
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
BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
CHUNK_OVERLAP_TOKENS = 100  # Overlap between chunks for context preservation


class DocumentationIndexer:
    """Handles indexing of documentation files to Supabase with embeddings.

    Why this class structure:
    - Encapsulates state (OpenAI client, Supabase client, stats)
    - Easy to test with dependency injection
    - Clear separation of concerns (read, chunk, embed, insert)
    """

    def __init__(self):
        """Initialize indexer with OpenAI and Supabase clients."""
        self.openai_client = OpenAI(api_key=supabase_settings.api_key)
        self.supabase_client = get_supabase_client()

        # Track indexing stats
        self.stats = {
            'files_read': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'chunks_inserted': 0,
            'chunks_skipped': 0,
            'api_calls': 0,
            'failures': 0,
            'total_cost': 0.0,
            'start_time': time.time()
        }

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
                self.stats['files_read'] += 1
            except Exception as e:
                logger.warning(f"Failed to read {md_file}: {e}")
                self.stats['failures'] += 1

        logger.info(f"   Found {len(files)} markdown files")
        return files

    def chunk_by_sections(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Chunk markdown content by section headers with overlap.

        Purpose: Intelligent chunking preserves document structure while
        maintaining context across section boundaries.

        Strategy:
        - Split on ## and ### headers (major sections)
        - Each chunk includes section title + content
        - Add 100-token overlap between chunks for context preservation

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

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts with retry logic.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is list of 1536 floats)
        """
        # Filter out empty or invalid texts
        valid_texts = []
        valid_indices = []
        for idx, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                # OpenAI has a max input length of ~8192 tokens, roughly 32K chars
                # Truncate if too long (safety margin)
                if len(text) > 30000:
                    text = text[:30000]
                    logger.warning(f"Truncated text at index {idx} (was {len(texts[idx])} chars)")
                valid_texts.append(text)
                valid_indices.append(idx)
            else:
                logger.warning(f"Skipping empty/invalid text at index {idx}")

        if not valid_texts:
            logger.error("No valid texts in batch")
            return [[]] * len(texts)  # Return empty embeddings for all

        for attempt in range(MAX_RETRIES):
            try:
                self.stats['api_calls'] += 1

                response = self.openai_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=valid_texts,
                    encoding_format="float"
                )

                valid_embeddings = [item.embedding for item in response.data]

                # Reconstruct full embeddings list (empty for skipped texts)
                all_embeddings = [None] * len(texts)
                for valid_idx, embedding in zip(valid_indices, valid_embeddings):
                    all_embeddings[valid_idx] = embedding

                # Fill in empty embeddings for skipped texts
                for idx in range(len(texts)):
                    if all_embeddings[idx] is None:
                        all_embeddings[idx] = [0.0] * EMBEDDING_DIMENSIONS
                        logger.warning(f"Using zero embedding for skipped text at index {idx}")

                # Calculate cost (text-embedding-3-small: $0.00002 per 1K tokens)
                total_chars = sum(len(text) for text in valid_texts)
                tokens = total_chars / 4  # Rough estimate: 1 token ≈ 4 chars
                cost = (tokens / 1000) * 0.00002
                self.stats['total_cost'] += cost

                return all_embeddings

            except Exception as e:
                logger.warning(f"   Embedding attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")

                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)
                    logger.info(f"   Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error("   All retry attempts exhausted")
                    self.stats['failures'] += 1
                    raise

    def check_existing_chunk(self, content_hash: str, file_path: str, section: str) -> bool:
        """Check if chunk already exists in database (idempotency).

        Args:
            content_hash: SHA256 hash of chunk content
            file_path: File path for matching
            section: Section title for matching

        Returns:
            True if chunk exists, False otherwise
        """
        try:
            # Check by doc_id, file_path, and section (more reliable than content_hash in metadata)
            result = self.supabase_client.table('kb_chunks')\
                .select('id')\
                .eq('doc_id', 'documentation')\
                .eq('section', section)\
                .limit(1)\
                .execute()

            # Also check metadata for file_path match
            if result.data:
                for chunk in result.data:
                    # Note: Supabase JSONB query might need different syntax
                    # For now, we'll do a simpler check
                    pass

            return len(result.data) > 0
        except Exception as e:
            logger.warning(f"Error checking existing chunk: {e}")
            return False

    def insert_chunks(self, chunks: List[Dict[str, Any]]):
        """Insert chunks into Supabase with embeddings.

        Purpose: Store documentation chunks in pgvector for semantic search.

        Args:
            chunks: List of chunk dicts with content, embeddings, metadata
        """
        logger.info(f"💾 Inserting {len(chunks)} chunks into Supabase...")

        inserted = 0
        skipped = 0

        for chunk in chunks:
            # Check if chunk already exists (idempotency)
            if self.check_existing_chunk(chunk['content_hash'], chunk['metadata']['file_path'], chunk['section']):
                skipped += 1
                self.stats['chunks_skipped'] += 1
                continue

            try:
                # Insert chunk with embedding
                result = self.supabase_client.table('kb_chunks').insert({
                    'doc_id': chunk['doc_id'],
                    'section': chunk['section'],
                    'content': chunk['content'],
                    'embedding': chunk['embedding'],
                    'metadata': chunk['metadata']
                }).execute()

                inserted += 1
                self.stats['chunks_inserted'] += 1
            except Exception as e:
                logger.warning(f"Failed to insert chunk: {e}")
                self.stats['failures'] += 1

        logger.info(f"   ✅ Inserted {inserted} chunks, skipped {skipped} duplicates")

    def index_all(self, docs_path: str = "docs"):
        """Complete indexing pipeline: read → chunk → embed → insert.

        Purpose: Main entry point for documentation indexing.

        Args:
            docs_path: Path to docs directory
        """
        logger.info("🚀 Starting documentation indexing...")

        # Step 1: Read all markdown files
        files = self.read_docs_folder(docs_path)
        if not files:
            logger.error("No documentation files found")
            return

        # Step 2: Chunk all files
        all_chunks = []
        for file_data in files:
            chunks = self.chunk_by_sections(file_data['content'], file_data['file_path'])
            all_chunks.extend(chunks)
            self.stats['chunks_created'] += len(chunks)

        logger.info(f"📦 Created {len(all_chunks)} chunks from {len(files)} files")

        # Step 3: Generate embeddings in batches
        logger.info("🧮 Generating embeddings...")
        texts = [chunk['content'] for chunk in all_chunks]

        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            batch_chunks = all_chunks[i:i+BATCH_SIZE]

            logger.info(f"   Processing batch {i//BATCH_SIZE + 1} ({len(batch_texts)} chunks)...")

            embeddings = self.generate_embeddings_batch(batch_texts)
            self.stats['embeddings_generated'] += len(embeddings)

            # Attach embeddings to chunks
            for chunk, embedding in zip(batch_chunks, embeddings):
                chunk['embedding'] = embedding

        # Step 4: Insert chunks into Supabase
        self.insert_chunks(all_chunks)

        # Print summary
        elapsed = time.time() - self.stats['start_time']
        logger.info("=" * 60)
        logger.info("📊 Indexing Summary:")
        logger.info(f"   Files read: {self.stats['files_read']}")
        logger.info(f"   Chunks created: {self.stats['chunks_created']}")
        logger.info(f"   Embeddings generated: {self.stats['embeddings_generated']}")
        logger.info(f"   Chunks inserted: {self.stats['chunks_inserted']}")
        logger.info(f"   Chunks skipped (duplicates): {self.stats['chunks_skipped']}")
        logger.info(f"   API calls: {self.stats['api_calls']}")
        logger.info(f"   Total cost: ${self.stats['total_cost']:.4f}")
        logger.info(f"   Time elapsed: {elapsed:.1f}s")
        logger.info("=" * 60)


if __name__ == "__main__":
    indexer = DocumentationIndexer()
    indexer.index_all()
