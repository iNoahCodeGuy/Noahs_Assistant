"""Test retrieval with exact new question."""
from assistant.retrieval.pgvector_retriever import PgVectorRetriever

r = PgVectorRetriever()
print(f"Default threshold: {r.similarity_threshold}\n")

# Test with user's actual query
queries = [
    'How does this product work?',
    'How does this chatbot product work?',
    'what is the backend stack?'
]

for query in queries:
    chunks = r.retrieve(query, top_k=5, threshold=0.4)
    print(f"Query: '{query}'")
    print(f"Matches: {len(chunks)}\n")

    for i, c in enumerate(chunks[:3], 1):
        print(f"{i}. {c['doc_id']} (similarity: {c['similarity']:.3f})")
        print(f"   Preview: {c['content'][:100]}...\n")
    print("-" * 80 + "\n")
