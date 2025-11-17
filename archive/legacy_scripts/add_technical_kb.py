#!/usr/bin/env python3
"""
Add technical knowledge base to existing Supabase database.
This adds product/stack information alongside the career KB.
"""

import os
import sys
import csv
import time
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, 'src')

from openai import OpenAI
from config.supabase_config import get_supabase_client, supabase_settings

load_dotenv()

print("=" * 70)
print("🔧 ADDING TECHNICAL KNOWLEDGE BASE")
print("=" * 70)

# Initialize clients
openai_client = OpenAI(api_key=supabase_settings.api_key)
supabase = get_supabase_client()

# Read technical KB
print("\n📄 Reading data/technical_kb.csv...")
rows = []
with open('data/technical_kb.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            'question': row['Question'].strip(),
            'answer': row['Answer'].strip()
        })

print(f"   ✅ Found {len(rows)} technical Q&A pairs")

# Create chunks
print("\n🔢 Creating chunks...")
chunks = []
for idx, row in enumerate(rows):
    content = f"Q: {row['question']}\nA: {row['answer']}"
    chunks.append({
        'doc_id': 'technical_kb',
        'section': row['question'][:100],
        'content': content,
        'metadata': {
            'source': 'technical_kb.csv',
            'row_index': idx,
            'question': row['question'],
            'category': 'technical',
            'migrated_at': datetime.utcnow().isoformat()
        }
    })

print(f"   ✅ Created {len(chunks)} chunks")

# Generate embeddings
print("\n🧠 Generating embeddings...")
texts = [chunk['content'] for chunk in chunks]

# Batch processing (100 at a time)
all_embeddings = []
batch_size = 100

for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    batch_num = (i // batch_size) + 1
    total_batches = (len(texts) + batch_size - 1) // batch_size

    print(f"   Batch {batch_num}/{total_batches}: Processing {len(batch)} texts...")

    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=batch
    )

    batch_embeddings = [item.embedding for item in response.data]
    all_embeddings.extend(batch_embeddings)

    print(f"   Progress: {min(i + batch_size, len(texts))}/{len(texts)}")

# Add embeddings to chunks
for chunk, embedding in zip(chunks, all_embeddings):
    chunk['embedding'] = embedding

print(f"   ✅ Generated {len(all_embeddings)} embeddings")

# Insert into Supabase
print("\n💾 Inserting into Supabase...")

# Delete existing technical_kb chunks if any
try:
    existing = supabase.table('kb_chunks').select('id').eq('doc_id', 'technical_kb').execute()
    if existing.data:
        print(f"   🗑️  Removing {len(existing.data)} existing technical_kb chunks...")
        supabase.table('kb_chunks').delete().eq('doc_id', 'technical_kb').execute()
except:
    pass  # Table might be empty

# Insert new chunks (batch of 20 for safety with large embeddings)
insert_batch_size = 20
total_inserted = 0

for i in range(0, len(chunks), insert_batch_size):
    batch = chunks[i:i + insert_batch_size]

    insert_data = [{
        'doc_id': chunk['doc_id'],
        'section': chunk['section'],
        'content': chunk['content'],
        'embedding': chunk['embedding'],
        'metadata': chunk['metadata']
    } for chunk in batch]

    result = supabase.table('kb_chunks').insert(insert_data).execute()
    total_inserted += len(result.data) if result.data else 0

    print(f"   Progress: {min(i + insert_batch_size, len(chunks))}/{len(chunks)}")

print(f"   ✅ Inserted {total_inserted} chunks")

print("\n" + "=" * 70)
print("🎉 TECHNICAL KB ADDED SUCCESSFULLY!")
print("=" * 70)
print(f"\n📊 Summary:")
print(f"   Technical Q&A pairs: {len(rows)}")
print(f"   Chunks created: {len(chunks)}")
print(f"   Embeddings generated: {len(all_embeddings)}")
print(f"   Chunks inserted: {total_inserted}")
print(f"\n✅ Your assistant can now answer technical questions about:")
print(f"   - Tech stack and architecture")
print(f"   - Files and codebase structure")
print(f"   - Features and capabilities")
print(f"   - Setup and deployment")
print(f"   - Security and scalability")
print("=" * 70)
