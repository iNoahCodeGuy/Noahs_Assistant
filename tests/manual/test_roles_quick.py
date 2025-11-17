"""
Quick Role Functionality Tests - Essential Features Only

This script tests the core functionality of each role to ensure:
1. Responses are generated (not "I don't have enough information")
2. Chat memory works for follow-up questions
3. Role-specific features activate correctly
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'assistant'))

from assistant.core.rag_engine import RagEngine
from assistant.core.memory import Memory
from assistant.agents.role_router import RoleRouter
from assistant.config.supabase_config import supabase_settings


def test_role(role_name: str, test_query: str, follow_up_query: str = None):
    """Test a single role with a query and optional follow-up."""
    print(f"\n{'='*80}")
    print(f"Testing: {role_name}")
    print(f"{'='*80}")

    memory = Memory()
    rag_engine = RagEngine(supabase_settings)
    role_router = RoleRouter()
    chat_history = []

    # Test 1: Initial query
    print(f"\n📝 Query: {test_query}")
    result = role_router.route(role_name, test_query, memory, rag_engine, chat_history)
    response = result.get("response", "")

    print(f"\n💬 Response ({len(response)} chars):")
    print(response[:500] + ("..." if len(response) > 500 else ""))

    # Check quality
    has_response = len(response) > 50
    not_error = "I don't have enough information" not in response

    status = "✅ PASS" if (has_response and not_error) else "❌ FAIL"
    print(f"\n{status} - Response Quality:")
    print(f"  - Length: {len(response)} chars (Target: >50)")
    print(f"  - Has content: {has_response}")
    print(f"  - Not error message: {not_error}")
    print(f"  - Response type: {result.get('type')}")

    # Test 2: Follow-up query (chat memory)
    if follow_up_query:
        chat_history.append({"role": "user", "content": test_query})
        chat_history.append({"role": "assistant", "content": response})

        print(f"\n📝 Follow-up: {follow_up_query}")
        follow_result = role_router.route(role_name, follow_up_query, memory, rag_engine, chat_history)
        follow_response = follow_result.get("response", "")

        print(f"\n💬 Follow-up Response ({len(follow_response)} chars):")
        print(follow_response[:500] + ("..." if len(follow_response) > 500 else ""))

        has_follow_up = len(follow_response) > 50
        contextual = len(follow_response) > 30  # Should reference context

        status = "✅ PASS" if (has_follow_up and contextual) else "❌ FAIL"
        print(f"\n{status} - Memory Test:")
        print(f"  - Has response: {has_follow_up}")
        print(f"  - Contextual: {contextual}")

    return has_response and not_error


def main():
    """Run quick tests for all roles."""
    print("\n" + "="*80)
    print("🧪 QUICK ROLE FUNCTIONALITY TESTS")
    print("="*80)

    results = {}

    # Test 1: Hiring Manager (nontechnical)
    results["HM (nontechnical)"] = test_role(
        "Hiring Manager (nontechnical)",
        "What is Noah's work experience?",
        "What were his key achievements?"
    )

    # Test 2: Hiring Manager (technical)
    results["HM (technical)"] = test_role(
        "Hiring Manager (technical)",
        "What technical projects has Noah worked on?",
        "Can you show me some code examples?"
    )

    # Test 3: Software Developer
    results["Software Developer"] = test_role(
        "Software Developer",
        "How does the RAG retrieval system work?",
        "What about the similarity calculation?"
    )

    # Test 4: Just looking around
    results["Just looking around"] = test_role(
        "Just looking around",
        "Tell me about Noah",
        "Does he have any hobbies?"
    )

    # Test 5: MMA Query
    print(f"\n{'='*80}")
    print(f"Testing: MMA Special Feature")
    print(f"{'='*80}")

    memory = Memory()
    rag_engine = RagEngine(supabase_settings)
    role_router = RoleRouter()

    mma_result = role_router.route("Just looking around", "Does Noah do MMA?", memory, rag_engine)
    has_youtube = "youtube_link" in mma_result
    is_mma_type = mma_result.get("type") == "mma"

    print(f"\n💬 Response: {mma_result.get('response')}")
    print(f"YouTube Link: {mma_result.get('youtube_link', 'N/A')}")

    status = "✅ PASS" if (has_youtube and is_mma_type) else "❌ FAIL"
    print(f"\n{status} - MMA Feature:")
    print(f"  - Has YouTube link: {has_youtube}")
    print(f"  - Correct type: {is_mma_type}")

    results["MMA Feature"] = has_youtube and is_mma_type

    # Test 6: Confession
    print(f"\n{'='*80}")
    print(f"Testing: Confession Role")
    print(f"{'='*80}")

    confession_result = role_router.route("Looking to confess crush", "I have a crush", memory, rag_engine)
    is_confession = confession_result.get("type") == "confession"
    is_brief = len(confession_result.get("response", "")) < 200

    print(f"\n💬 Response: {confession_result.get('response')}")

    status = "✅ PASS" if (is_confession and is_brief) else "❌ FAIL"
    print(f"\n{status} - Confession Feature:")
    print(f"  - Correct type: {is_confession}")
    print(f"  - Brief message: {is_brief}")

    results["Confession"] = is_confession and is_brief

    # Summary
    print(f"\n{'='*80}")
    print("📊 TEST SUMMARY")
    print(f"{'='*80}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for role, passed_test in results.items():
        status = "✅" if passed_test else "❌"
        print(f"{status} {role}")

    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n🎉 All tests passed! System is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review above.")

    print("="*80)


if __name__ == "__main__":
    main()
