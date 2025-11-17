#!/usr/bin/env python3
"""Deployment verification script for code display and import explanation features.

Run this before deploying to Vercel to ensure all new features work correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports_kb_exists():
    """Verify imports_kb.csv exists and is readable."""
    print("✓ Checking imports_kb.csv existence...")
    imports_kb_path = Path(__file__).parent / "data" / "imports_kb.csv"

    if not imports_kb_path.exists():
        print(f"❌ FAIL: {imports_kb_path} not found")
        return False

    with open(imports_kb_path, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:  # Header + at least one entry
            print(f"❌ FAIL: imports_kb.csv is empty or malformed")
            return False

    print(f"✅ PASS: Found {len(lines)-1} import entries")
    return True


def test_import_retriever_loads():
    """Verify import_retriever module loads without errors."""
    print("\n✓ Checking import_retriever module...")
    try:
        from assistant.retrieval.import_retriever import (
            get_import_explanation,
            search_import_explanations,
            detect_import_in_query,
            ROLE_TO_TIER
        )
        print("✅ PASS: import_retriever module loads successfully")
        print(f"   - Role mappings: {len(ROLE_TO_TIER)} roles defined")
        return True
    except Exception as e:
        print(f"❌ FAIL: Cannot import import_retriever: {e}")
        return False


def test_content_blocks_functions():
    """Verify new content block functions exist."""
    print("\n✓ Checking content_blocks functions...")
    try:
        from assistant.flows import content_blocks

        required_functions = [
            'format_code_snippet',
            'format_import_explanation',
            'code_display_guardrails'
        ]

        for func_name in required_functions:
            if not hasattr(content_blocks, func_name):
                print(f"❌ FAIL: Missing function {func_name}")
                return False

        print(f"✅ PASS: All {len(required_functions)} content block functions present")
        return True
    except Exception as e:
        print(f"❌ FAIL: Cannot check content_blocks: {e}")
        return False


def test_conversation_nodes_updated():
    """Verify conversation_nodes has new trigger logic."""
    print("\n✓ Checking conversation_nodes updates...")
    try:
        from assistant.flows.conversation_nodes import classify_query, plan_actions
        from assistant.state.conversation_state import ConversationState

        # Test code display trigger
        state = ConversationState(role="Software Developer", query="show me the code")
        result = classify_query(state)

        if not result.fetch("code_display_requested"):
            print("❌ FAIL: Code display trigger not working")
            return False

        # Test import explanation trigger
        state2 = ConversationState(role="Software Developer", query="why use Supabase?")
        result2 = classify_query(state2)

        if not result2.fetch("import_explanation_requested"):
            print("❌ FAIL: Import explanation trigger not working")
            return False

        print("✅ PASS: Conversation node triggers working correctly")
        return True
    except Exception as e:
        print(f"❌ FAIL: Conversation nodes error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_import_retrieval_works():
    """Verify import retrieval actually returns data."""
    print("\n✓ Checking import retrieval...")
    try:
        from assistant.retrieval.import_retriever import (
            get_import_explanation,
            detect_import_in_query
        )

        # Test getting an explanation
        result = get_import_explanation("openai", "Software Developer")

        if not result:
            print("❌ FAIL: No explanation returned for 'openai'")
            return False

        if result["tier"] != "2":
            print(f"❌ FAIL: Wrong tier returned (got {result['tier']}, expected 2)")
            return False

        # Test library detection
        detected = detect_import_in_query("why use supabase for this project?")
        if detected != "supabase":
            print(f"❌ FAIL: Library detection failed (got {detected}, expected supabase)")
            return False

        print("✅ PASS: Import retrieval working correctly")
        print(f"   - Retrieved tier 2 explanation for OpenAI")
        print(f"   - Detected 'supabase' in query")
        return True
    except Exception as e:
        print(f"❌ FAIL: Import retrieval error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_integration():
    """Verify API endpoint would load successfully."""
    print("\n✓ Checking Vercel API integration...")
    try:
        sys.path.insert(0, str(Path(__file__).parent / "api"))

        # Try importing the API handler (without running it)
        from assistant.flows.conversation_flow import run_conversation_flow
        from assistant.core.rag_engine import RagEngine

        print("✅ PASS: Vercel API imports work")
        return True
    except Exception as e:
        print(f"❌ FAIL: API integration error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all deployment verification tests."""
    print("=" * 70)
    print("🚀 VERCEL DEPLOYMENT VERIFICATION")
    print("   Code Display & Import Explanation Features")
    print("=" * 70)

    tests = [
        test_imports_kb_exists,
        test_import_retriever_loads,
        test_content_blocks_functions,
        test_conversation_nodes_updated,
        test_import_retrieval_works,
        test_api_integration,
    ]

    results = []
    for test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"❌ FAIL: Unexpected error in {test_func.__name__}: {e}")
            results.append(False)

    print("\n" + "=" * 70)
    print("📊 VERIFICATION RESULTS")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"\n✅ Passed: {passed}/{total} tests")
    print(f"{'❌' if passed < total else '✅'} Failed: {total - passed}/{total} tests")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - READY FOR DEPLOYMENT!")
        print("\nTo deploy to Vercel:")
        print("  1. git add .")
        print('  2. git commit -m "Add code display and import explanation features"')
        print("  3. git push origin main")
        print("  4. Vercel will auto-deploy from GitHub")
        return 0
    else:
        print("\n⚠️  DEPLOYMENT NOT RECOMMENDED")
        print(f"   {total - passed} test(s) failing - fix issues before deploying")
        return 1


if __name__ == "__main__":
    sys.exit(main())
