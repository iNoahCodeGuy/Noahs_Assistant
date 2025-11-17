"""Quick test to verify vague query expansion doesn't crash."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from assistant.state.conversation_state import ConversationState
from assistant.flows.node_logic.query_classification import classify_query

# Test vague query expansion
def test_vague_queries():
    test_cases = [
        "engineering",
        "skills",
        "hello",
        "Tell me about Noah's experience with RAG"
    ]

    for query in test_cases:
        print(f"\nTesting: '{query}'")
        state = ConversationState(role="Hiring Manager (technical)", query=query)

        try:
            result = classify_query(state)
            print(f"  ✓ Success")
            print(f"    Query type: {result.fetch('query_type', 'unknown')}")
            print(f"    Expanded: {result.fetch('vague_query_expanded', False)}")
            if result.fetch('vague_query_expanded'):
                print(f"    Expanded query: {result.fetch('expanded_query', '')[:100]}...")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("Testing vague query expansion logic...")
    test_vague_queries()
    print("\n✅ All tests completed")
