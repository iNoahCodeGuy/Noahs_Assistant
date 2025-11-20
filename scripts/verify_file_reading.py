# -*- coding: utf-8 -*-
"""Verification script for file reading functionality.

Tests key components without requiring pytest:
- File path detection
- File reading capability
- Code snippet context expansion
- Documentation chunking

Run: python scripts/verify_file_reading.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from assistant.flows.node_logic.stage2_query_classification import _detect_file_request
from assistant.retrieval.code_index import CodeIndex


def test_file_path_detection():
    """Test file path detection in queries."""
    print("\n" + "="*60)
    print("Testing File Path Detection")
    print("="*60)

    test_cases = [
        ("show me assistant/core/rag_engine.py", "assistant/core/rag_engine.py"),
        ("read assistant/flows/conversation_flow.py", "assistant/flows/conversation_flow.py"),
        ("open the file assistant/state/conversation_state.py", "assistant/state/conversation_state.py"),
        ("how does RAG work?", None),
        ("read the rag engine file", None),
    ]

    all_passed = True
    for query, expected in test_cases:
        result = _detect_file_request(query)
        status = "✅" if result == expected else "❌"
        print(f"{status} Query: '{query}'")
        print(f"   Expected: {expected}, Got: {result}")
        if result != expected:
            all_passed = False

    return all_passed


def test_file_reading():
    """Test file reading capability."""
    print("\n" + "="*60)
    print("Testing File Reading")
    print("="*60)

    code_index = CodeIndex(repo_path=".")

    # Test reading a known file
    test_file = "assistant/retrieval/code_index.py"
    if not Path(test_file).exists():
        print(f"⚠️  Test file not found: {test_file}")
        return False

    result = code_index.read_file_content(test_file)

    if result.get("success"):
        print(f"✅ Successfully read {test_file}")
        print(f"   Lines: {result.get('line_count', 0)}")
        print(f"   Content length: {len(result.get('content', ''))} chars")
        return True
    else:
        print(f"❌ Failed to read {test_file}")
        return False


def test_code_snippet_context():
    """Test code snippet context expansion."""
    print("\n" + "="*60)
    print("Testing Code Snippet Context")
    print("="*60)

    code_index = CodeIndex(repo_path=".")

    test_file = "assistant/retrieval/code_index.py"
    if not Path(test_file).exists():
        print(f"⚠️  Test file not found: {test_file}")
        return False

    # Test with context
    snippet = code_index.get_file_snippet(
        test_file,
        line_start=50,
        line_end=55,
        context_before=5,
        context_after=5
    )

    lines = snippet.split('\n')
    print(f"✅ Retrieved snippet with {len(lines)} lines")
    print(f"   Expected: ~11 lines (5 before + 1 target + 5 after)")

    if len(lines) >= 11:
        print("   ✅ Context expansion working correctly")
        return True
    else:
        print("   ❌ Context expansion may not be working")
        return False


def test_documentation_chunking():
    """Test documentation chunking."""
    print("\n" + "="*60)
    print("Testing Documentation Chunking")
    print("="*60)

    try:
        from scripts.index_documentation import DocumentationIndexer

        indexer = DocumentationIndexer()

        test_content = """# Title

## Section 1
Content for section 1 with some details.

## Section 2
Content for section 2 with more information.

### Subsection 2.1
Nested content here.
"""
        chunks = indexer.chunk_by_sections(test_content, "test.md")

        print(f"✅ Created {len(chunks)} chunks from test content")
        print(f"   Expected: 2-3 chunks")

        if len(chunks) >= 2:
            print("   ✅ Chunking working correctly")
            for i, chunk in enumerate(chunks[:3]):  # Show first 3
                print(f"   Chunk {i+1}: {chunk['section']} ({len(chunk['content'])} chars)")
            return True
        else:
            print("   ❌ Chunking may not be working correctly")
            return False
    except Exception as e:
        print(f"❌ Error testing documentation chunking: {e}")
        return False


def main():
    """Run all verification tests."""
    print("\n" + "="*60)
    print("File Reading Functionality Verification")
    print("="*60)

    results = []

    results.append(("File Path Detection", test_file_path_detection()))
    results.append(("File Reading", test_file_reading()))
    results.append(("Code Snippet Context", test_code_snippet_context()))
    results.append(("Documentation Chunking", test_documentation_chunking()))

    # Summary
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("="*60)
    if all_passed:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
