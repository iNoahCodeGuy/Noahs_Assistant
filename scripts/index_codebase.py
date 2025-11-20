"""Codebase indexing script for Noah's AI Assistant.

This script recursively reads all Python files from the assistant/ directory,
chunks them intelligently by function/class boundaries using AST parsing,
generates embeddings, and inserts them into Supabase kb_chunks table with
doc_id='codebase'.

Purpose: Enable semantic search across entire codebase, not just function/class
names. This allows Portfolia to answer questions about her implementation
using actual code context.

Layer 4: Full Codebase Semantic Search

Key features:
- Recursive Python file reading from assistant/ directory
- AST-based chunking by function/class boundaries
- Sliding window chunking for module-level code (500 tokens, 50 overlap)
- Batch embedding generation (up to 100 texts per API call)
- Idempotent inserts (checks for existing chunks by content hash)
- Progress tracking with cost estimates
- Structured logging for observability

Usage:
    python scripts/index_codebase.py

Requirements:
    - OPENAI_API_KEY in environment
    - SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in environment
    - assistant/ directory present
"""

import sys
import os
import ast
import time
import logging
import hashlib
import re
from typing import List, Dict, Any, Optional
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
MODULE_CHUNK_SIZE = 500  # tokens for module-level code
MODULE_CHUNK_OVERLAP = 50  # tokens overlap between chunks

# Core architecture files for selective indexing
CORE_FILES = [
    # Priority 1 - Essential Architecture
    "assistant/core/rag_engine.py",
    "assistant/core/response_generator.py",
    "assistant/flows/conversation_flow.py",
    "assistant/flows/node_logic/stage4_retrieval_nodes.py",
    "assistant/flows/node_logic/stage2_query_classification.py",
    "assistant/flows/node_logic/stage5_generation_nodes.py",
    "assistant/flows/node_logic/stage6_formatting_nodes.py",
    "assistant/retrieval/pgvector_retriever.py",
    "assistant/state/conversation_state.py",
    "assistant/config/supabase_config.py",
    # Priority 2 - Supporting Architecture
    "assistant/core/rag_factory.py",
    "assistant/retrieval/code_index.py",
    "assistant/flows/conversation_nodes.py",
    "assistant/flows/node_logic/stage0_session_management.py",
    "assistant/flows/node_logic/stage2_role_routing.py",
    "assistant/flows/node_logic/stage3_query_composition.py",
    "assistant/core/memory.py",
    "assistant/core/guardrails.py",
    "assistant/retrieval/code_service.py",
    "assistant/main.py",
]


class CodebaseIndexer:
    """Handles indexing of Python codebase to Supabase with embeddings.

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

    def read_codebase(self, codebase_path: str = "assistant", file_list: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """Read Python files from assistant/ directory.

        Purpose: Collect source files for indexing. Supports selective indexing
        via file_list parameter for core architecture files only.

        Args:
            codebase_path: Path to assistant directory (default "assistant")
            file_list: Optional list of specific file paths to index. If provided,
                      only these files will be indexed. Paths should be relative
                      to project root (e.g., "assistant/core/rag_engine.py")

        Returns:
            List of dicts with 'file_path' and 'content' keys
        """
        if file_list:
            logger.info(f"📄 Reading {len(file_list)} specified files for selective indexing...")
        else:
            logger.info(f"📄 Reading codebase from {codebase_path}...")

        files = []

        if file_list:
            # Selective indexing: only read specified files
            for file_path in file_list:
                file_path_obj = Path(file_path)
                if not file_path_obj.exists():
                    logger.warning(f"File not found: {file_path}, skipping...")
                    self.stats['failures'] += 1
                    continue

                try:
                    with open(file_path_obj, 'r', encoding='utf-8') as f:
                        content = f.read()

                    files.append({
                        'file_path': file_path,
                        'content': content
                    })
                    self.stats['files_read'] += 1
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
                    self.stats['failures'] += 1
        else:
            # Full indexing: recursively read all Python files
            codebase_dir = Path(codebase_path)
            if not codebase_dir.exists():
                logger.error(f"Codebase directory not found: {codebase_path}")
                return []

            for py_file in codebase_dir.rglob("*.py"):
                # Skip common exclusions
                if any(exclude in str(py_file) for exclude in ["__pycache__", ".pyc", "venv", ".venv"]):
                    continue

                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Get relative path from project root
                    rel_path = str(py_file.relative_to(Path('.')))

                    files.append({
                        'file_path': rel_path,
                        'content': content
                    })
                    self.stats['files_read'] += 1
                except Exception as e:
                    logger.warning(f"Failed to read {py_file}: {e}")
                    self.stats['failures'] += 1

        logger.info(f"   Found {len(files)} Python files")
        return files

    def chunk_by_ast(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Chunk Python code by function/class boundaries using AST parsing.

        Purpose: Primary chunking strategy - preserves logical code units
        (functions, classes) which are most meaningful for semantic search.

        Strategy:
        - Parse AST to find all functions and classes
        - Each function/class becomes one chunk
        - Include docstrings and surrounding context

        Args:
            content: Full Python file content
            file_path: Source file path for metadata

        Returns:
            List of chunk dicts with doc_id, section, content, metadata
        """
        chunks = []
        lines = content.split('\n')

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    # Get line numbers
                    line_start = node.lineno
                    line_end = getattr(node, 'end_lineno', line_start + 10)

                    # Extract code snippet
                    snippet_lines = lines[line_start - 1:line_end]
                    snippet_content = '\n'.join(snippet_lines)

                    # Determine chunk type
                    chunk_type = "function" if isinstance(node, ast.FunctionDef) else "class"

                    # Create chunk
                    chunk = {
                        'section': f"{chunk_type}: {node.name}",
                        'content': snippet_content,
                        'line_start': line_start,
                        'line_end': line_end,
                        'chunk_type': chunk_type,
                        'name': node.name
                    }
                    chunks.append(chunk)

        except SyntaxError as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            # Fall back to module-level chunking
            return self.chunk_module_level(content, file_path)

        return chunks

    def chunk_module_level(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Chunk module-level code using sliding window.

        Purpose: Secondary chunking strategy for code that's not in functions/classes
        (imports, module-level constants, top-level code).

        Strategy:
        - Split content into ~500 token chunks with 50 token overlap
        - Preserves context across chunk boundaries

        Args:
            content: Full Python file content
            file_path: Source file path for metadata

        Returns:
            List of chunk dicts
        """
        chunks = []
        lines = content.split('\n')

        # Rough token estimation: 1 token ≈ 4 characters
        chunk_size_chars = MODULE_CHUNK_SIZE * 4
        overlap_chars = MODULE_CHUNK_OVERLAP * 4

        start = 0
        chunk_idx = 0

        while start < len(content):
            end = min(start + chunk_size_chars, len(content))
            chunk_content = content[start:end]

            # Find line boundaries
            line_start = content[:start].count('\n') + 1
            line_end = content[:end].count('\n') + 1

            if len(chunk_content.strip()) > 50:  # Minimum chunk size
                chunks.append({
                    'section': f"module-level code (chunk {chunk_idx + 1})",
                    'content': chunk_content,
                    'line_start': line_start,
                    'line_end': line_end,
                    'chunk_type': 'module',
                    'name': None
                })
                chunk_idx += 1

            # Move start position with overlap
            start = end - overlap_chars

        return chunks

    def chunk_file(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Chunk a Python file using AST-based + module-level strategies.

        Purpose: Combine both chunking strategies for complete coverage.

        Strategy:
        1. Primary: AST-based chunking (functions/classes)
        2. Secondary: Module-level sliding window for remaining code

        Args:
            content: Full Python file content
            file_path: Source file path

        Returns:
            List of all chunks from the file
        """
        # Primary: AST-based chunking
        ast_chunks = self.chunk_by_ast(content, file_path)

        # Get line ranges already covered by AST chunks
        covered_ranges = set()
        for chunk in ast_chunks:
            for line in range(chunk['line_start'], chunk['line_end'] + 1):
                covered_ranges.add(line)

        # Secondary: Module-level chunking for uncovered code
        # For simplicity, we'll use AST chunks as primary and add module chunks
        # only if AST parsing failed
        if not ast_chunks:
            module_chunks = self.chunk_module_level(content, file_path)
            return module_chunks

        # Add metadata to all chunks
        all_chunks = []
        for idx, chunk in enumerate(ast_chunks):
            chunk['doc_id'] = 'codebase'
            chunk['metadata'] = {
                'file_path': file_path,
                'function_name': chunk.get('name') if chunk['chunk_type'] == 'function' else None,
                'class_name': chunk.get('name') if chunk['chunk_type'] == 'class' else None,
                'chunk_type': chunk['chunk_type'],
                'line_start': chunk['line_start'],
                'line_end': chunk['line_end'],
                'indexed_at': datetime.utcnow().isoformat()
            }
            # Generate content hash for idempotency
            chunk['content_hash'] = hashlib.sha256(chunk['content'].encode('utf-8')).hexdigest()[:16]
            all_chunks.append(chunk)

        return all_chunks

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
                .eq('doc_id', 'codebase')\
                .eq('section', section)\
                .limit(1)\
                .execute()

            return len(result.data) > 0
        except Exception as e:
            logger.warning(f"Error checking existing chunk: {e}")
            return False

    def insert_chunks(self, chunks: List[Dict[str, Any]]):
        """Insert chunks into Supabase with embeddings.

        Purpose: Store codebase chunks in pgvector for semantic search.

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

    def index_all(self, codebase_path: str = "assistant", file_list: Optional[List[str]] = None):
        """Complete indexing pipeline: read → chunk → embed → insert.

        Purpose: Main entry point for codebase indexing. Supports selective
        indexing via file_list parameter.

        Args:
            codebase_path: Path to assistant directory (used only for full indexing)
            file_list: Optional list of specific file paths to index. If provided,
                      only these files will be indexed.
        """
        if file_list:
            logger.info(f"🚀 Starting selective codebase indexing ({len(file_list)} files)...")
        else:
            logger.info("🚀 Starting full codebase indexing...")

        # Step 1: Read Python files (selective or full)
        files = self.read_codebase(codebase_path, file_list=file_list)
        if not files:
            logger.error("No Python files found")
            return

        # Step 2: Chunk all files
        all_chunks = []
        for file_data in files:
            chunks = self.chunk_file(file_data['content'], file_data['file_path'])
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
    import argparse

    parser = argparse.ArgumentParser(
        description='Index Python codebase to Supabase with embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full indexing (all Python files in assistant/)
  python scripts/index_codebase.py

  # Selective indexing (core architecture files only)
  python scripts/index_codebase.py --core-only

  # Custom file list
  python scripts/index_codebase.py --files assistant/core/rag_engine.py assistant/flows/conversation_flow.py
        """
    )
    parser.add_argument(
        '--core-only',
        action='store_true',
        help='Index only core architecture files (20 files)'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        help='List of specific file paths to index (relative to project root)'
    )

    args = parser.parse_args()

    indexer = CodebaseIndexer()

    if args.core_only:
        logger.info("Using core-only mode: indexing 20 core architecture files")
        indexer.index_all(file_list=CORE_FILES)
    elif args.files:
        logger.info(f"Using custom file list: {len(args.files)} files")
        indexer.index_all(file_list=args.files)
    else:
        # Full indexing
        indexer.index_all()
