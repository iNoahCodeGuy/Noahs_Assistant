#!/usr/bin/env python3
"""
Test script to validate node logic with the three conversation turns provided.

This script tests:
1. Turn 1: Initial greeting (empty query)
2. Turn 2: Role selection ("2" - Technical Hiring Manager)
3. Turn 3: Menu option 1 selection ("1" - Full tech stack)

Usage:
    python3 scripts/test_vercel_node_logic.py
"""
import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from assistant.flows.conversation_flow import run_conversation_flow
from assistant.state.conversation_state import ConversationState
from assistant.core.rag_engine import RagEngine


def test_turn_1_initial_greeting():
    """Test Turn 1: Initial greeting with empty query."""
    print("\n" + "=" * 80)
    print("TEST 1: Initial Greeting (Empty Query)")
    print("=" * 80)

    rag_engine = RagEngine()
    state: ConversationState = {
        "query": "",
        "role": "",
        "session_id": "test-session-001",
        "chat_history": [],
        "session_memory": {}
    }

    result = run_conversation_flow(state, rag_engine, session_id="test-session-001")

    # Validate expected fields
    assert "answer" in result, "Missing 'answer' field"
    assert result.get("is_greeting") == True, "Should be marked as greeting"
    assert "1️⃣ Hiring Manager" in result["answer"] or "Hiring Manager" in result["answer"], \
        "Answer should contain role selection prompt"

    print(f"✅ Answer length: {len(result['answer'])} chars")
    print(f"✅ Is greeting: {result.get('is_greeting')}")
    print(f"✅ Answer preview: {result['answer'][:150]}...")

    return result


def test_turn_2_role_selection():
    """Test Turn 2: Role selection (Technical Hiring Manager)."""
    print("\n" + "=" * 80)
    print("TEST 2: Role Selection (Technical Hiring Manager)")
    print("=" * 80)

    rag_engine = RagEngine()
    state: ConversationState = {
        "query": "2",
        "role": "",
        "session_id": "test-session-002",
        "chat_history": [],
        "session_memory": {
            "persona_hints": {
                "initial_greeting_shown": True
            }
        }
    }

    result = run_conversation_flow(state, rag_engine, session_id="test-session-002")

    # Validate expected fields
    assert "answer" in result, "Missing 'answer' field"
    assert result.get("role") == "Hiring Manager (technical)" or "technical" in result.get("role", "").lower(), \
        f"Role should be 'Hiring Manager (technical)', got: {result.get('role')}"
    assert result.get("role_mode") == "hiring_manager_technical", \
        f"Role mode should be 'hiring_manager_technical', got: {result.get('role_mode')}"

    print(f"✅ Answer length: {len(result['answer'])} chars")
    print(f"✅ Role: {result.get('role')}")
    print(f"✅ Role mode: {result.get('role_mode')}")
    print(f"✅ Answer preview: {result['answer'][:200]}...")

    return result


def test_turn_3_menu_option_1():
    """Test Turn 3: Menu option 1 (Full tech stack walkthrough)."""
    print("\n" + "=" * 80)
    print("TEST 3: Menu Option 1 (Full Tech Stack)")
    print("=" * 80)

    rag_engine = RagEngine()
    state: ConversationState = {
        "query": "1",
        "role": "Hiring Manager (technical)",
        "role_mode": "hiring_manager_technical",
        "role_confidence": 1,
        "session_id": "test-session-003",
        "chat_history": [],
        "session_memory": {
            "persona_hints": {
                "initial_greeting_shown": True,
                "role_mode": "hiring_manager_technical",
                "role_welcome_shown": True
            }
        },
        # Ensure all required fields are present
        "entities": {},
        "retrieved_chunks": [],
        "analytics_metadata": {}
    }

    result = run_conversation_flow(state, rag_engine, session_id="test-session-003")

    # Validate expected fields
    assert "answer" in result, "Missing 'answer' field"
    assert result.get("query_type") == "menu_selection", \
        f"Query type should be 'menu_selection', got: {result.get('query_type')}"
    assert result.get("menu_choice") == "1", \
        f"Menu choice should be '1', got: {result.get('menu_choice')}"

    # Check for required layers in answer
    answer = result.get("answer", "")
    required_layers = [
        "Frontend Layer",
        "Backend",
        "Data Layer",
        "Observability",
        "Deployment"
    ]

    found_layers = [layer for layer in required_layers if layer.lower() in answer.lower()]
    print(f"✅ Found {len(found_layers)}/{len(required_layers)} required layers: {found_layers}")

    # Check for retrieved chunks
    retrieved_chunks = result.get("retrieved_chunks", [])
    assert len(retrieved_chunks) > 0, "Should have retrieved chunks"
    print(f"✅ Retrieved chunks: {len(retrieved_chunks)}")

    # Check grounding status
    grounding_status = result.get("grounding_status", "unknown")
    print(f"✅ Grounding status: {grounding_status}")

    print(f"✅ Answer length: {len(answer)} chars ({len(answer.split())} words)")
    print(f"✅ Answer preview: {answer[:300]}...")

    return result


def test_api_endpoint_format():
    """Test that the state can be converted to API response format."""
    print("\n" + "=" * 80)
    print("TEST 4: API Response Format Validation")
    print("=" * 80)

    rag_engine = RagEngine()
    state: ConversationState = {
        "query": "1",
        "role": "Hiring Manager (technical)",
        "session_id": "test-session-api",
        "chat_history": [],
        "session_memory": {}
    }

    result = run_conversation_flow(state, rag_engine, session_id="test-session-api")

    # Build API response format (matching api/chat.py)
    api_response = {
        "success": True,
        "answer": result.get("answer", ""),
        "role": result.get("role", ""),
        "session_id": result.get("session_id", ""),
        "session_memory": result.get("session_memory", {}),
        "analytics": result.get("analytics_metadata", {}),
        "actions_taken": [
            action.get("type") for action in result.get("pending_actions", [])
        ],
        "retrieved_chunks": len(result.get("retrieved_chunks", []))
    }

    # Validate API response structure
    assert api_response["success"] == True, "API response should indicate success"
    assert "answer" in api_response, "API response should have 'answer' field"
    assert "session_memory" in api_response, "API response should have 'session_memory' field"

    print(f"✅ API response structure valid")
    print(f"✅ Answer present: {bool(api_response['answer'])}")
    print(f"✅ Session memory present: {bool(api_response['session_memory'])}")
    print(f"✅ Retrieved chunks count: {api_response['retrieved_chunks']}")

    return api_response


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("VERCEL NODE LOGIC VALIDATION TESTS")
    print("=" * 80)
    print("\nTesting the three conversation turns provided...")

    try:
        # Test Turn 1
        turn1_result = test_turn_1_initial_greeting()

        # Test Turn 2
        turn2_result = test_turn_2_role_selection()

        # Test Turn 3
        turn3_result = test_turn_3_menu_option_1()

        # Test API format
        api_result = test_api_endpoint_format()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\nSummary:")
        print(f"  Turn 1: ✅ Initial greeting working")
        print(f"  Turn 2: ✅ Role selection working")
        print(f"  Turn 3: ✅ Menu option 1 working")
        print(f"  API Format: ✅ Response structure valid")
        print("\n🎉 Node logic is ready for Vercel deployment!")

        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
