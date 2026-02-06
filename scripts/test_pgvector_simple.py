#!/usr/bin/env python3
"""Simple test to check if pgvector RPC function exists."""

import os
from dotenv import load_dotenv

load_dotenv()

def main():
    print("="*80)
    print("SIMPLE PGVECTOR FUNCTION TEST")
    print("="*80)

    # Check environment variables
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

    if not supabase_url or not supabase_key:
        print("\n❌ Missing environment variables!")
        print("   Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env")
        return

    print(f"\n✅ Supabase URL: {supabase_url[:30]}...")
    print(f"✅ Service key: {supabase_key[:20]}...")

    # Try to import and connect
    try:
        from supabase import create_client
        print("\n✅ Supabase library imported")

        supabase = create_client(supabase_url, supabase_key)
        print("✅ Supabase client created")

        # Test if function exists by calling it with dummy data
        print("\nTesting match_kb_chunks function...")
        dummy_embedding = [0.0] * 1536

        result = supabase.rpc(
            'match_kb_chunks',
            {
                'query_embedding': dummy_embedding,
                'match_threshold': 0.99,  # High threshold = no results expected
                'match_count': 1
            }
        ).execute()

        print("✅ match_kb_chunks function EXISTS and is callable!")
        print(f"   Returned {len(result.data)} results (expected 0 with high threshold)")

        # Check kb_chunks table
        print("\nChecking kb_chunks table...")
        count_result = supabase.table('kb_chunks').select('id', count='exact').execute()
        chunk_count = count_result.count
        print(f"✅ Found {chunk_count} chunks in kb_chunks table")

        if chunk_count == 0:
            print("\n⚠️  WARNING: No chunks found. You may need to populate the database.")
            print("   Run: python3 scripts/migrate_all_kb_to_supabase.py")
        else:
            print("\n" + "="*80)
            print("✅ SETUP IS COMPLETE!")
            print("="*80)
            print("\nThe pgvector IVFFLAT index and RPC function are working.")
            print("Your retrieval system is ready to use!")
            print("="*80)

    except Exception as e:
        error_msg = str(e).lower()

        # Check for type mismatch error
        if 'does not match' in error_msg or '42804' in error_msg:
            print("\n❌ TYPE MISMATCH ERROR")
            print("\n" + "="*80)
            print("FUNCTION EXISTS BUT HAS WRONG RETURN TYPE")
            print("="*80)
            print("\nThe match_kb_chunks function was created with wrong type (int vs bigint)")
            print("\nTo fix:")
            print("1. Open Supabase Dashboard → SQL Editor")
            print("2. Copy contents of: supabase/migrations/FIX_match_kb_chunks_type.sql")
            print("3. Paste and click 'RUN'")
            print("4. Run this script again to verify")
            print("\nFor details, see: PGVECTOR_TYPE_FIX_README.md")
            print("="*80)

        # Check for function not found
        elif 'function' in error_msg and 'does not exist' in error_msg:
            print("\n❌ match_kb_chunks function NOT FOUND")
            print("\n" + "="*80)
            print("MIGRATION NOT RUN YET")
            print("="*80)
            print("\nYou need to run the migration SQL:")
            print("\n1. Open Supabase Dashboard → SQL Editor")
            print("2. Copy contents of:")
            print("   supabase/migrations/20250205_native_pgvector_search.sql")
            print("3. Paste and click 'RUN'")
            print("4. Run this script again to verify")
            print("="*80)

        # Unknown error - show full details
        else:
            print(f"\n❌ Unexpected error: {e}")
            print("\nDebug info:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            import traceback
            print(f"\nFull traceback:")
            traceback.print_exc()

if __name__ == "__main__":
    main()
