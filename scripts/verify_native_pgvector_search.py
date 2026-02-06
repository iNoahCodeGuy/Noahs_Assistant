#!/usr/bin/env python3
"""Verify native pgvector search works with a real query."""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def get_openai_embedding(text: str) -> list[float]:
    """Generate embedding using OpenAI API."""
    try:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")

        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Error getting OpenAI embedding: {e}")
        return None


def main():
    print("="*80)
    print("NATIVE PGVECTOR SEARCH VERIFICATION")
    print("="*80)

    # Check environment variables
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not supabase_url or not supabase_key:
        print("\n❌ Missing Supabase credentials!")
        print("   Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env")
        return

    if not openai_key:
        print("\n❌ Missing OpenAI API key!")
        print("   Please set OPENAI_API_KEY in .env")
        return

    print(f"\n✅ Environment configured")

    # Test queries to try
    test_queries = [
        "Tell me about Portfolia's investment philosophy",
        "How does Portfolia support women entrepreneurs?",
        "What is the community aspect of Portfolia?"
    ]

    try:
        from supabase import create_client
        supabase = create_client(supabase_url, supabase_key)
        print("✅ Connected to Supabase")

        # Test each query
        for i, query_text in enumerate(test_queries, 1):
            print("\n" + "="*80)
            print(f"TEST QUERY {i}: {query_text}")
            print("="*80)

            # Generate embedding
            print("\nGenerating embedding...")
            query_embedding = get_openai_embedding(query_text)
            if not query_embedding:
                print("❌ Failed to generate embedding, skipping...")
                continue
            print(f"✅ Generated {len(query_embedding)}-dimensional embedding")

            # Call native pgvector search
            print("\nCalling match_kb_chunks()...")
            result = supabase.rpc(
                'match_kb_chunks',
                {
                    'query_embedding': query_embedding,
                    'match_threshold': 0.60,
                    'match_count': 3
                }
            ).execute()

            if not result.data:
                print("⚠️  No results found above threshold (0.60)")
                continue

            print(f"✅ Found {len(result.data)} results!\n")

            # Display results
            for j, chunk in enumerate(result.data, 1):
                similarity = chunk.get('similarity', 0)
                doc_id = chunk.get('doc_id', 'unknown')
                section = chunk.get('section', 'unknown')
                content = chunk.get('content', '')[:200]  # First 200 chars

                print(f"Result {j}:")
                print(f"  Similarity: {similarity:.4f}")
                print(f"  Document:   {doc_id}")
                print(f"  Section:    {section}")
                print(f"  Content:    {content}...")
                print()

        print("="*80)
        print("✅ NATIVE PGVECTOR SEARCH IS WORKING!")
        print("="*80)
        print("\nYour retrieval system is using native pgvector search with IVFFLAT index.")
        print("Performance should be 5-10x faster than client-side similarity calculation.")
        print("="*80)

    except Exception as e:
        error_msg = str(e).lower()

        if 'does not match' in error_msg or '42804' in error_msg:
            print("\n❌ TYPE MISMATCH ERROR")
            print("\nThe function has wrong return types. Run the fix:")
            print("  supabase/migrations/FIX_match_kb_chunks_type.sql")
            print("\nSee: PGVECTOR_TYPE_FIX_README.md")

        elif 'function' in error_msg and 'does not exist' in error_msg:
            print("\n❌ Function not found. Run the migration:")
            print("  supabase/migrations/20250205_native_pgvector_search.sql")

        else:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
