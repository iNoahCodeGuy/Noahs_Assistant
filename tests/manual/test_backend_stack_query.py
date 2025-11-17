"""Test backend stack query retrieval locally."""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from assistant.retrieval.pgvector_retriever import PgVectorRetriever
from assistant.config.supabase_config import supabase_settings

def test_backend_stack_query():
    """Test if 'how does this product work' retrieves technical content."""

    print("🧪 Testing backend stack query retrieval...\n")

    # Initialize retriever
    retriever = PgVectorRetriever()

    # Test queries
    test_queries = [
        "how does this product work?",
        "what is in the backend stack?",
        "what tech stack does Noah use?",
        "tell me about the backend"
    ]

    for query in test_queries:
        print(f"📝 Query: '{query}'")
        print("-" * 60)

        # Retrieve chunks with lower threshold for testing
        chunks = retriever.retrieve(query, top_k=3, threshold=0.5)

        if chunks:
            print(f"✅ Retrieved {len(chunks)} chunks:\n")
            for i, chunk in enumerate(chunks, 1):
                doc_id = chunk.get('doc_id', 'unknown')
                similarity = chunk.get('similarity', 0)
                content_preview = chunk.get('content', '')[:150]

                print(f"  {i}. doc_id: {doc_id} (similarity: {similarity:.3f})")
                print(f"     Preview: {content_preview}...\n")
        else:
            print("❌ No chunks retrieved (retrieval failed or below threshold)\n")

        print()

if __name__ == '__main__':
    # Validate environment
    if not supabase_settings.api_key:
        print("❌ OPENAI_API_KEY not set")
        sys.exit(1)

    try:
        supabase_settings.validate_supabase()
    except Exception as e:
        print(f"❌ Supabase config invalid: {e}")
        sys.exit(1)

    test_backend_stack_query()
