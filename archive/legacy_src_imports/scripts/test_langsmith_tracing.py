# -*- coding: utf-8 -*-
"""
Interactive script to test LangSmith tracing with conversation nodes.
Run this instead of the notebook for easier debugging.

Usage:
    python scripts/test_langsmith_tracing.py
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

print("="*60)
print("LangSmith Node Tracing Test")
print("="*60)

# Verify environment
print("\nEnvironment Configuration:")
print(f"  LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"  LANGCHAIN_API_KEY: {'SET' if os.getenv('LANGCHAIN_API_KEY') else 'MISSING'}")
print(f"  LANGCHAIN_PROJECT: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
print(f"  OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'MISSING'}")
print(f"  SUPABASE_URL: {'SET' if os.getenv('SUPABASE_URL') else 'MISSING'}")

from src.observability.langsmith_tracer import LANGSMITH_ENABLED, get_langsmith_client

print(f"\nLangSmith Status: {'ENABLED' if LANGSMITH_ENABLED else 'DISABLED'}")

if LANGSMITH_ENABLED:
    client = get_langsmith_client()
    print(f"   Client initialized: {client is not None}")
    print(f"\nView traces at: https://smith.langchain.com/")
else:
    print("   LangSmith disabled - set LANGCHAIN_TRACING_V2=true")
    sys.exit(1)

# Initialize RAG engine
print("\n" + "="*60)
print("Initializing RAG Engine...")
print("="*60)

from src.core.rag_engine import RagEngine
from src.state.conversation_state import ConversationState

rag_engine = RagEngine()
print(f"[OK] RAG Engine ready (degraded mode: {rag_engine.degraded_mode})")

# Create conversation state (initialize as dict, then convert)
state = {
    "role": "hiring_manager_technical",
    "query": "What Python frameworks has Noah worked with?",
    "session_id": "script-test-001",
    "chat_history": [],
    "analytics_metadata": {}
}

print(f"\n[OK] ConversationState initialized:")
print(f"   Role: {state['role']}")
print(f"   Query: {state['query']}")
print(f"   Session: {state['session_id']}")

# Test 1: Query Classification
print("\n" + "="*60)
print("TEST 1: Query Classification")
print("="*60)

from src.flows.node_logic.stage2_query_classification import classify_query

classification_update = classify_query(state)
state.update(classification_update)  # Merge the update into original state
print(f"[OK] Classification complete:")
print(f"   Query Type: {state.get('query_type', 'unknown')}")
print(f"   Intent: {state.get('intent', 'unknown')}")

# Test 2: Retrieval (Traced to LangSmith)
print("\n" + "="*60)
print("TEST 2: Retrieval (Traced to LangSmith)")
print("="*60)

from src.flows.node_logic.stage4_retrieval_nodes import retrieve_chunks

retrieval_update = retrieve_chunks(state, rag_engine)
state.update(retrieval_update)  # Merge the update into original state

chunks = state.get("retrieved_chunks", [])
print(f"[OK] Retrieved {len(chunks)} chunks:")
for i, chunk in enumerate(chunks[:3], 1):
    content = chunk.get('content', '')[:80]
    score = chunk.get('similarity', 0.0)
    print(f"   [{i}] Similarity: {score:.3f} - {content}...")

# Test 3: Generation (Traced to LangSmith)
print("\n" + "="*60)
print("TEST 3: Generation (Traced to LangSmith)")
print("="*60)

from src.flows.node_logic.stage5_generation_nodes import generate_draft

generation_update = generate_draft(state, rag_engine)
state.update(generation_update)  # Merge the update into original state
answer = state.get("answer", "")

print(f"[OK] Generated answer ({len(answer)} chars):")
print(f"   {answer[:200]}...")

# Test 4: Full Pipeline
print("\n" + "="*60)
print("TEST 4: Full Conversation Pipeline")
print("="*60)

from src.flows.conversation_flow import run_conversation_flow

fresh_state = {
    "role": "developer",
    "query": "Show me Noah's error handling implementation",
    "session_id": "script-test-full-pipeline",
    "chat_history": []
}

print(f"Running full pipeline for: {fresh_state['query']}")
result = run_conversation_flow(fresh_state, rag_engine, session_id=fresh_state["session_id"])

print(f"\n[OK] Pipeline complete!")
print(f"   Final answer: {result.get('answer', '')[:150]}...")
print(f"   State keys: {len(result.keys())}")

# Summary
print("\n" + "="*60)
print("All Tests Complete!")
print("="*60)
print(f"\nView traces in LangSmith:")
print(f"   https://smith.langchain.com/o/project/noahs-ai-assistant")
print(f"\n   Session IDs to filter:")
print(f"   - script-test-001 (individual nodes)")
print(f"   - script-test-full-pipeline (complete flow)")
print("\n" + "="*60)
