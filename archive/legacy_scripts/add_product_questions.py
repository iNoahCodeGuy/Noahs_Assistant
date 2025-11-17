"""Add better matching questions to technical_kb for common queries."""
import csv
import os

# Read existing technical_kb
kb_path = 'data/technical_kb.csv'
rows = []

with open(kb_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Add new questions that better match user queries
new_entries = [
    {
        "Question": "How does this chatbot product work?",
        "Answer": "This AI chatbot works by combining **retrieval-augmented generation (RAG)** with **role-aware context**. When you ask a question, the system: (1) Classifies your query type (technical, career, personal), (2) Generates a semantic embedding of your question using OpenAI, (3) Searches Noah's knowledge base (stored in Supabase pgvector) for relevant information, (4) Retrieves the top 3-5 most similar chunks with context about Noah, (5) Feeds this context to GPT-4o-mini which generates a personalized response about Noah, (6) Adds role-specific enhancements (code snippets for developers, career highlights for hiring managers), (7) Logs the interaction for analytics and improvement. The backend is **Python + LangGraph** orchestrating the conversation flow, **Supabase** for database/vector storage, **OpenAI** for LLM and embeddings, and **Next.js** for the frontend UI. Everything runs as **Vercel serverless functions** for scalability. The key innovation is the **role router** that adapts responses based on who you are (hiring manager sees different content than a software developer)."
    },
    {
        "Question": "What backend technologies power this assistant?",
        "Answer": "The backend stack is: **Python 3.11+** for core logic and orchestration, **LangGraph** for multi-step conversation flows (classify → retrieve → generate → execute → log pipeline), **LangChain** for LLM abstraction and prompt management, **OpenAI GPT-4o-mini** for natural language generation, **OpenAI text-embedding-3-small** for 1536-dimensional vector embeddings, **Supabase Postgres** with **pgvector extension** for vector similarity search and relational data storage, **Supabase Storage** for file management (resume PDFs), **Vercel serverless functions** for API deployment (auto-scaling, zero-downtime), **LangSmith** for observability (tracing LLM calls, cost monitoring), **Resend** for transactional emails, **Twilio** for SMS notifications. The architecture is **stateless** - each request is independent, with session state stored in Supabase not memory. This enables horizontal scaling and serverless deployment."
    },
    {
        "Question": "How was this AI assistant built?",
        "Answer": "Noah built this AI assistant through iterative development: **Phase 1 (Foundation)**: Started with a simple Streamlit app and FAISS local vector store for knowledge retrieval. Built the RAG pipeline using LangChain to combine OpenAI embeddings with retrieval. **Phase 2 (Role Intelligence)**: Added the role router system to personalize responses for different user types (hiring managers, developers, casual visitors). Implemented session management and analytics logging. **Phase 3 (Production Infrastructure)**: Migrated from FAISS to Supabase pgvector for cloud-native vector search. Added LangGraph for workflow orchestration (replaced linear pipeline with node-based flow). Built Next.js frontend to replace Streamlit for better performance. Deployed to Vercel serverless for auto-scaling. **Phase 4 (Refinements)**: Implemented third-person language enforcement (responses say 'Noah' not 'I'). Added intelligent follow-up question suggestions for technical roles. Expanded knowledge base from career-only to include technical_kb (system details) and architecture_kb (diagrams, code examples). The development process emphasized **rapid prototyping** (get it working first), **observability** (LangSmith tracing for debugging), and **incremental complexity** (add features one at a time, validate each works)."
    }
]

# Append new entries
rows.extend(new_entries)

# Write back to CSV
with open(kb_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['Question', 'Answer'])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Added {len(new_entries)} new entries to {kb_path}")
print(f"   Total entries: {len(rows)}")
print("\n📝 New questions added:")
for i, entry in enumerate(new_entries, 1):
    print(f"   {i}. {entry['Question']}")

print("\n🔄 Next step: Re-run migration to update embeddings in Supabase:")
print("   python scripts/migrate_all_kb_to_supabase.py --kb technical_kb --force")
