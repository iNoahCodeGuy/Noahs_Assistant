"""Test architecture KB retrieval and source formatting"""

from assistant.config.supabase_config import get_supabase_client, supabase_settings
from openai import OpenAI

# Initialize
openai_client = OpenAI(api_key=supabase_settings.api_key)
supabase = get_supabase_client()

# Test queries
test_queries = [
    "show me the system architecture",
    "can you display or reference code?",
    "what is the tech stack?",
    "display the database schema"
]

print("=" * 80)
print("ARCHITECTURE RETRIEVAL TEST")
print("=" * 80)

for query in test_queries:
    print(f"\n🔍 Query: {query}")
    print("-" * 80)

    # Generate embedding
    response = openai_client.embeddings.create(
        model='text-embedding-3-small',
        input=query
    )
    embedding = response.data[0].embedding

    # Search using the correct function name
    results = supabase.rpc(
        'search_kb_chunks',  # ← Correct function name
        {
            'query_embedding': embedding,
            'match_threshold': 0.5,
            'match_count': 3
        }
    ).execute()

    print(f"\n📊 Found {len(results.data)} results:\n")

    for i, row in enumerate(results.data, 1):
        print(f"  {i}. 📄 **{row['doc_id']}** (similarity: {row.get('similarity', 0):.3f})")
        print(f"     Section: {row['section'][:70]}...")
        print(f"     Content: {row['content'][:120]}...\n")

print("=" * 80)
print("✅ Test complete!")
print("=" * 80)
