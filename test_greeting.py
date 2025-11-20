#!/usr/bin/env python3
"""Quick test for initial greeting - no Docker needed."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assistant.flows.conversation_flow import run_conversation_flow
from assistant.core.rag_engine import RagEngine

def test_initial_greeting():
    """Test that initial greeting shows numbered list correctly."""
    print("\n" + "="*80)
    print("TESTING INITIAL GREETING")
    print("="*80)

    state = {
        "query": "",
        "role": "",
        "session_id": "test-greeting",
        "chat_history": [],
        "session_memory": {}
    }

    print("Input state:")
    print(f"  query: '{state['query']}'")
    print(f"  role: '{state['role']}'")
    print(f"  session_memory: {state['session_memory']}")

    rag_engine = RagEngine()
    result = run_conversation_flow(state, rag_engine, "test-greeting")

    print("\n" + "-"*80)
    print("RESULT:")
    print("-"*80)
    answer = result.get('answer', 'MISSING')
    print(f"Answer length: {len(answer)} chars")
    print(f"Answer preview: {answer[:150]}...")
    print(f"\nFull answer:\n{answer}")
    print("-"*80)

    print("\n" + "-"*80)
    print("STATE FLAGS:")
    print("-"*80)
    print(f"is_greeting: {result.get('is_greeting')}")
    print(f"pipeline_halt: {result.get('pipeline_halt')}")
    print(f"initial_greeting_shown: {result.get('session_memory', {}).get('persona_hints', {}).get('initial_greeting_shown')}")
    print("-"*80)

    print("\n" + "-"*80)
    print("VALIDATION:")
    print("-"*80)

    expected_indicators = [
        "1️⃣ Hiring Manager",
        "2️⃣ Hiring Manager",
        "3️⃣ Software Developer",
        "4️⃣ Just Looking Around",
        "5️⃣ Looking to Confess Crush"
    ]

    has_numbered_list = all(indicator in answer for indicator in expected_indicators)
    has_wrong_message = "Just tell me in your own words" in answer

    if has_numbered_list and not has_wrong_message:
        print("✅ SUCCESS: Initial greeting has numbered list (1️⃣-5️⃣)")
        print("✅ SUCCESS: No 'Just tell me in your own words' message")
        return True
    else:
        print("❌ FAIL: Initial greeting incorrect")
        if not has_numbered_list:
            print(f"   Missing numbered list indicators")
        if has_wrong_message:
            print(f"   Contains unwanted 'Just tell me in your own words' message")
        return False

if __name__ == "__main__":
    success = test_initial_greeting()
    print("\n" + "="*80)
    if success:
        print("✅ TEST PASSED")
    else:
        print("❌ TEST FAILED")
    print("="*80 + "\n")
    sys.exit(0 if success else 1)
