# -*- coding: utf-8 -*-
"""Check if codebase chunks exist and have embeddings."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from assistant.config.supabase_config import get_supabase_client

def check_chunks():
    """Check codebase chunks in database."""
    supabase = get_supabase_client()

    # Get sample codebase chunks
    result = supabase.table('kb_chunks')\
        .select('id, doc_id, section, content, embedding')\
        .eq('doc_id', 'codebase')\
        .limit(5)\
        .execute()

    print("=" * 60)
    print("Codebase Chunks Check")
    print("=" * 60)
    print(f"\nFound {len(result.data)} codebase chunks (sample)")

    if result.data:
        for i, chunk in enumerate(result.data):
            has_embedding = chunk.get('embedding') is not None
            content_len = len(chunk.get('content', ''))
            section = chunk.get('section', 'N/A')

            print(f"\nChunk {i+1}:")
            print(f"  Section: {section[:60]}")
            print(f"  Content length: {content_len} chars")
            print(f"  Has embedding: {has_embedding}")
            if has_embedding:
                emb = chunk.get('embedding')
                if isinstance(emb, list):
                    print(f"  Embedding dims: {len(emb)}")
                else:
                    print(f"  Embedding type: {type(emb)}")
    else:
        print("\n⚠️  No codebase chunks found in database!")

    # Check all codebase chunks count
    count_result = supabase.table('kb_chunks')\
        .select('id', count='exact')\
        .eq('doc_id', 'codebase')\
        .execute()

    print(f"\n\nTotal codebase chunks: {count_result.count or 0}")

if __name__ == '__main__':
    check_chunks()
