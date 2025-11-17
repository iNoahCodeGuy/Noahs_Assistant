#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from assistant.retrieval.pgvector_retriever import PgVectorRetriever

print('Testing pgvector retriever with float conversion fix...')
retriever = PgVectorRetriever(similarity_threshold=0.3)

chunks = retriever.retrieve('What is Noah professional background?', top_k=3)
print(f'\n✅ Retrieved {len(chunks)} chunks')

if chunks:
    print('\n🎉 SUCCESS! The retrieval is now working!')
    for i, chunk in enumerate(chunks[:2], 1):
        print(f'\n[{i}] Similarity: {chunk.get("similarity", 0):.4f}')
        print(f'    Section: {chunk.get("section", "N/A")}')
        print(f'    Content: {chunk["content"][:120]}...')
    print('\n✅ The Streamlit app should now work correctly!')
else:
    print('\n❌ Still no results - there may be a deeper PostgREST/Supabase issue')
