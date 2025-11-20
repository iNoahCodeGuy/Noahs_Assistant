# -*- coding: utf-8 -*-
"""Quick script to verify codebase chunks in Supabase."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from assistant.config.supabase_config import get_supabase_client

def verify_chunks():
    """Verify codebase and documentation chunks exist."""
    supabase = get_supabase_client()

    # Check codebase chunks
    codebase_result = supabase.table('kb_chunks')\
        .select('id', count='exact')\
        .eq('doc_id', 'codebase')\
        .execute()

    codebase_count = codebase_result.count or 0

    # Get unique files
    codebase_files_result = supabase.table('kb_chunks')\
        .select('metadata')\
        .eq('doc_id', 'codebase')\
        .execute()

    unique_files = set()
    for row in codebase_files_result.data:
        if 'metadata' in row and 'file_path' in row['metadata']:
            unique_files.add(row['metadata']['file_path'])

    # Check documentation chunks
    docs_result = supabase.table('kb_chunks')\
        .select('id', count='exact')\
        .eq('doc_id', 'documentation')\
        .execute()

    docs_count = docs_result.count or 0

    print("=" * 60)
    print("Database Verification Results")
    print("=" * 60)
    print(f"Codebase chunks: {codebase_count}")
    print(f"Codebase unique files: {len(unique_files)}")
    print(f"Documentation chunks: {docs_count}")
    print("=" * 60)

    if codebase_count > 0:
        print("\n✅ Codebase chunks found!")
        print(f"   Sample files: {list(unique_files)[:5]}")
    else:
        print("\n❌ No codebase chunks found")

    if docs_count > 0:
        print("\n✅ Documentation chunks found!")
    else:
        print("\n❌ No documentation chunks found")

    return codebase_count > 0 and docs_count > 0

if __name__ == '__main__':
    success = verify_chunks()
    sys.exit(0 if success else 1)
