# -*- coding: utf-8 -*-
"""Test codebase chunk retrieval directly."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from assistant.retrieval.pgvector_retriever import PgVectorRetriever

def test_codebase_retrieval():
    """Test retrieving codebase chunks directly."""
    retriever = PgVectorRetriever(similarity_threshold=0.1)  # Lower threshold for testing

    print("Testing codebase chunk retrieval...")
    print("=" * 60)

    # Test query that should match codebase
    query = "retrieval pipeline"

    print(f"\nQuery: '{query}'")
    print(f"Threshold: 0.1 (lowered for testing)")

    # Try without doc_id filter - check all doc_ids
    print("\n1. Retrieving without doc_id filter (top 10):")
    chunks_all = retriever.retrieve(query, top_k=10, threshold=0.1)
    print(f"   Retrieved {len(chunks_all)} chunks")
    if chunks_all:
        doc_ids = {}
        for chunk in chunks_all:
            doc_id = chunk.get('doc_id', 'unknown')
            if doc_id not in doc_ids:
                doc_ids[doc_id] = []
            doc_ids[doc_id].append(chunk.get('similarity', 0))

        print(f"   doc_ids found: {list(doc_ids.keys())}")
        for doc_id, similarities in doc_ids.items():
            avg_sim = sum(similarities) / len(similarities) if similarities else 0
            print(f"   {doc_id}: {len(similarities)} chunks, avg similarity={avg_sim:.3f}")

    # Try with codebase filter - no threshold
    print("\n2. Retrieving with doc_id='codebase' (no threshold):")
    chunks_codebase = retriever.retrieve(query, top_k=10, doc_id="codebase", threshold=0.0)
    print(f"   Retrieved {len(chunks_codebase)} codebase chunks")
    if chunks_codebase:
        print(f"   Top 3 codebase chunks:")
        for i, chunk in enumerate(chunks_codebase[:3]):
            print(f"   Chunk {i+1}: similarity={chunk.get('similarity', 0):.3f}")
            print(f"      Section: {chunk.get('section', 'N/A')[:60]}")
            print(f"      Content preview: {chunk.get('content', '')[:80]}...")
    else:
        print("   ⚠️  No codebase chunks found even with threshold=0.0!")
        print("   This indicates codebase chunks may not be indexed or query doesn't match")

    # Try with documentation filter
    print("\n3. Retrieving with doc_id='documentation' (no threshold):")
    chunks_docs = retriever.retrieve(query, top_k=5, doc_id="documentation", threshold=0.0)
    print(f"   Retrieved {len(chunks_docs)} documentation chunks")
    if chunks_docs:
        for i, chunk in enumerate(chunks_docs[:3]):
            print(f"   Chunk {i+1}: similarity={chunk.get('similarity', 0):.3f}")

if __name__ == '__main__':
    test_codebase_retrieval()
