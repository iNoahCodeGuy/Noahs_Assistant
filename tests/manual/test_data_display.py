"""Test that 'display collected data' query returns the enhanced analytics dashboard."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'assistant'))

from assistant.flows.conversation_flow import run_conversation_flow
from assistant.state.conversation_state import ConversationState
from assistant.core.rag_engine import RagEngine

def test_data_display():
    """Test data display request."""
    rag_engine = RagEngine()

    queries = [
        "display collected data",
        "can you display data analytics?",
        "show me the data collected"
    ]

    for query in queries:
        print(f"\n{'='*80}")
        print(f"TESTING: {query}")
        print('='*80)

        state = ConversationState(
            role="Hiring Manager (technical)",
            query=query
        )

        result = run_conversation_flow(state, rag_engine, session_id="test-123")

        answer_length = len(result.answer)
        has_analytics = "analytics" in result.answer.lower() or "data" in result.answer.lower()
        has_dashboard = "dashboard" in result.answer.lower()
        is_comprehensive = answer_length > 1000  # Should be 11,772 chars

        print(f"\n✅ Query: {query}")
        print(f"📊 Answer length: {answer_length} chars")
        print(f"📈 Contains analytics: {'YES' if has_analytics else 'NO'}")
        print(f"📊 Contains dashboard: {'YES' if has_dashboard else 'NO'}")
        print(f"📏 Comprehensive (>1000 chars): {'YES' if is_comprehensive else 'NO'}")

        print(f"\n🔍 First 500 characters of answer:")
        print("-" * 80)
        print(result.answer[:500])
        print("-" * 80)

        if not is_comprehensive:
            print(f"\n⚠️  WARNING: Answer too short! Expected >11,000 chars, got {answer_length}")
            print(f"\nFull answer:")
            print(result.answer)

if __name__ == "__main__":
    test_data_display()
