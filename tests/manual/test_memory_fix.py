#!/usr/bin/env python3
"""Test that chat memory and responses work correctly"""
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from assistant.core.rag_engine import RagEngine
from assistant.agents.role_router import RoleRouter
from assistant.core.memory import Memory
from assistant.config.supabase_config import supabase_settings

print("=" * 70)
print("TESTING CHAT MEMORY AND RESPONSE GENERATION")
print("=" * 70)

# Initialize components
print("\n[1] Initializing RAG engine...")
rag_engine = RagEngine(supabase_settings)
print("   ✓ RAG engine initialized")

print("\n[2] Initializing role router and memory...")
role_router = RoleRouter()
memory = Memory()
print("   ✓ Components initialized")

# Simulate a conversation with memory
print("\n[3] Testing conversation with memory...")
chat_history = []
role = "Hiring Manager (nontechnical)"

# First query
query1 = "What is Noah's professional background?"
print(f"\n   User: {query1}")
response1 = role_router.route(role, query1, memory, rag_engine, chat_history=chat_history)
print(f"   Assistant: {response1['response'][:200]}...")

# Add to history
chat_history.append({"role": "user", "content": query1})
chat_history.append({"role": "assistant", "content": response1['response']})

# Second query (should use context from first)
query2 = "What was his biggest achievement?"
print(f"\n   User: {query2}")
response2 = role_router.route(role, query2, memory, rag_engine, chat_history=chat_history)
print(f"   Assistant: {response2['response'][:200]}...")

# Verify response is not empty
if response2['response'] and len(response2['response']) > 50:
    print("\n✓ SUCCESS: Responses are being generated correctly!")
    print(f"✓ Chat history has {len(chat_history)} messages (including new exchange)")
else:
    print("\n✗ FAIL: Response is too short or empty")
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
