"""
Add architecture diagrams and code examples to knowledge base.
Includes Mermaid diagrams, ASCII diagrams, and implementation code.
"""

import json
from openai import OpenAI
from config.supabase_config import get_supabase_client, supabase_settings

def main():
    # Initialize clients
    openai_client = OpenAI(api_key=supabase_settings.api_key)
    supabase = get_supabase_client()

    print("🚀 Adding Architecture & Code Documentation to Knowledge Base")
    print("=" * 70)

    # Step 1: Load architecture KB from JSON
    print("\n📖 Reading architecture_kb.json...")
    with open('data/architecture_kb.json', 'r', encoding='utf-8') as f:
        architecture_data = json.load(f)
    print(f"✅ Found {len(architecture_data)} architecture items")

    # Step 2: Delete existing architecture_kb entries
    print("\n🗑️  Removing old architecture_kb entries...")
    result = supabase.table('kb_chunks').delete().eq('doc_id', 'architecture_kb').execute()
    print(f"✅ Cleared existing architecture_kb")

    # Step 3: Prepare chunks
    print("\n📦 Creating chunks...")
    chunks = []
    for item in architecture_data:
        chunk = {
            'doc_id': 'architecture_kb',
            'section': item['question'],
            'content': f"Question: {item['question']}\n\nAnswer: {item['answer']}",
            'metadata': {
                'source': 'architecture_kb',
                'category': 'architecture',
                'type': 'diagram' if 'diagram' in item['question'].lower() or 'structure' in item['question'].lower() else 'code',
                'question': item['question']
            }
        }
        chunks.append(chunk)

    print(f"✅ Created {len(chunks)} chunks")

    # Step 4: Generate embeddings (batch all at once)
    print("\n🧠 Generating embeddings with OpenAI...")
    texts = [chunk['content'] for chunk in chunks]

    response = openai_client.embeddings.create(
        model='text-embedding-3-small',
        input=texts
    )

    embeddings = [item.embedding for item in response.data]
    print(f"✅ Generated {len(embeddings)} embeddings")

    # Step 5: Add embeddings to chunks
    for chunk, embedding in zip(chunks, embeddings):
        chunk['embedding'] = embedding

    # Step 6: Insert into Supabase
    print("\n💾 Inserting into Supabase...")
    result = supabase.table('kb_chunks').insert(chunks).execute()
    print(f"✅ Inserted {len(chunks)} architecture chunks")

    print("\n" + "=" * 70)
    print("✨ Architecture & Code Documentation Added Successfully!")
    print(f"   Total Chunks: {len(chunks)}")
    print(f"   Coverage: System architecture, data flow, database schema, RAG code, response generation, role routing")
    print("\n💡 Users can now ask:")
    print("   • 'Show me the system architecture'")
    print("   • 'Explain the data flow'")
    print("   • 'Show me the database schema'")
    print("   • 'How does the RAG retrieval work?'")
    print("   • 'Show me the response generation code'")
    print("   • 'Explain the role routing logic'")

if __name__ == '__main__':
    main()
