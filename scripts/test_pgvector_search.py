"""Test script to verify pgvector embeddings are searchable.

This script:
1. Generates an embedding for a test query
2. Performs similarity search using Supabase's search_kb_chunks function
3. Displays results to verify migration worked

Usage:
    python scripts/test_pgvector_search.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openai import OpenAI
from config.supabase_config import get_supabase_client, supabase_settings

def test_search(query: str, top_k: int = 3):
    """Test pgvector similarity search.
    
    Args:
        query: Test query string
        top_k: Number of results to return
    """
    print(f"🔍 Testing query: '{query}'\n")
    
    # Generate query embedding
    print("🧠 Generating query embedding...")
    openai_client = OpenAI(api_key=supabase_settings.api_key)
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response.data[0].embedding
    print(f"   ✅ Generated {len(query_embedding)}-dimensional vector\n")
    
    # Search Supabase
    print(f"🔎 Searching Supabase for top {top_k} results...")
    supabase = get_supabase_client()
    
    result = supabase.rpc('search_kb_chunks', {
        'query_embedding': query_embedding,
        'match_threshold': 0.5,  # Lower threshold for testing
        'match_count': top_k
    }).execute()
    
    if not result.data:
        print("   ❌ No results found!")
        return
    
    print(f"   ✅ Found {len(result.data)} results\n")
    
    # Display results
    for idx, chunk in enumerate(result.data, 1):
        print(f"📄 Result {idx}:")
        print(f"   Section: {chunk['section'][:80]}...")
        print(f"   Content: {chunk['content'][:150]}...")
        print(f"   Similarity: {chunk['similarity']:.4f}")
        print()


def main():
    """Run test queries."""
    print("=" * 60)
    print("pgvector Search Test")
    print("=" * 60)
    print()
    
    # Test queries covering different topics
    test_queries = [
        "What programming languages does Noah know?",
        "Tell me about Noah's Tesla experience",
        "What is Noah's MMA background?",
        "What AI projects has Noah built?"
    ]
    
    for query in test_queries:
        test_search(query, top_k=2)
        print("-" * 60)
        print()


if __name__ == '__main__':
    main()
