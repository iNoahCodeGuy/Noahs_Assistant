# -*- coding: utf-8 -*-
"""Automated tests for self-referential query functionality.

Tests that Portfolia can answer questions about herself using codebase
and documentation chunks.

Run: python tests/test_self_referential_queries.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from assistant.core.rag_engine import RagEngine
from assistant.flows.node_logic.stage2_query_classification import classify_intent, _detect_file_request
from assistant.flows.node_logic.stage4_retrieval_nodes import retrieve_chunks
from assistant.state.conversation_state import ConversationState
from assistant.retrieval.pgvector_retriever import PgVectorRetriever


class TestSelfReferentialQueries:
    """Test self-referential query detection and retrieval."""

    def __init__(self):
        self.rag_engine = RagEngine()
        self.passed = 0
        self.failed = 0

    def test(self, name: str, condition: bool, message: str = ""):
        """Run a test and log result."""
        if condition:
            print(f"✅ {name}")
            if message:
                print(f"   {message}")
            self.passed += 1
        else:
            print(f"❌ {name}")
            if message:
                print(f"   {message}")
            self.failed += 1

    def test_self_referential_detection(self):
        """Test that self-referential queries are detected."""
        print("\n" + "=" * 60)
        print("Testing Self-Referential Query Detection")
        print("=" * 60)

        self_referential_queries = [
            "How does your retrieval pipeline work?",
            "How are you built?",
            "What's your architecture?",
            "How do you work?",
            "Show me your code",
        ]

        normal_queries = [
            "How does RAG work?",
            "What is Noah's background?",
            "Tell me about Python",
        ]

        for query in self_referential_queries:
            state: ConversationState = {
                "query": query,
                "role": "Software Developer"
            }
            result = classify_intent(state)
            is_self_ref = result.get("is_self_referential", False)
            self.test(
                f"Detects self-referential: '{query[:40]}...'",
                is_self_ref,
                f"is_self_referential={is_self_ref}"
            )

        for query in normal_queries:
            state: ConversationState = {
                "query": query,
                "role": "Software Developer"
            }
            result = classify_intent(state)
            is_self_ref = result.get("is_self_referential", False)
            self.test(
                f"Does NOT detect as self-referential: '{query[:40]}...'",
                not is_self_ref,
                f"is_self_referential={is_self_ref}"
            )

    def test_retrieval_prioritization(self):
        """Test that self-referential queries prioritize codebase/documentation."""
        print("\n" + "=" * 60)
        print("Testing Retrieval Prioritization")
        print("=" * 60)

        # Test self-referential query
        state: ConversationState = {
            "query": "How does your retrieval pipeline work?",
            "role": "Software Developer",
            "is_self_referential": True
        }

        result = retrieve_chunks(state, self.rag_engine, top_k=4)

        # Check if preferred_source_used is set
        metadata = result.get("analytics_metadata", {})
        preferred_used = metadata.get("preferred_source_used", False)

        self.test(
            "Self-referential query sets preferred_source_used",
            preferred_used,
            f"preferred_source_used={preferred_used}"
        )

        # Check if chunks were retrieved
        chunks = result.get("retrieved_chunks", [])
        has_chunks = len(chunks) > 0

        self.test(
            "Self-referential query retrieves chunks",
            has_chunks,
            f"Retrieved {len(chunks)} chunks"
        )

        # Check if chunks are from codebase/documentation
        if chunks:
            doc_ids = [chunk.get("doc_id") for chunk in chunks if chunk.get("doc_id")]
            has_codebase = "codebase" in doc_ids
            has_documentation = "documentation" in doc_ids

            self.test(
                "Chunks include codebase or documentation",
                has_codebase or has_documentation,
                f"doc_ids found: {set(doc_ids)}"
            )

    def test_codebase_chunks_exist(self):
        """Test that codebase chunks are available for retrieval."""
        print("\n" + "=" * 60)
        print("Testing Codebase Chunks Availability")
        print("=" * 60)

        if not self.rag_engine.pgvector_retriever:
            print("⚠️  pgvector_retriever not available, skipping test")
            return

        # Try to retrieve with doc_id filter
        try:
            chunks = self.rag_engine.pgvector_retriever.retrieve(
                "retrieval pipeline",
                top_k=3,
                doc_id="codebase"
            )

            self.test(
                "Can retrieve codebase chunks",
                len(chunks) > 0,
                f"Retrieved {len(chunks)} codebase chunks"
            )

            if chunks:
                # Verify chunks have correct doc_id
                all_codebase = all(c.get("doc_id") == "codebase" for c in chunks)
                self.test(
                    "All chunks have doc_id='codebase'",
                    all_codebase,
                    f"doc_ids: {[c.get('doc_id') for c in chunks]}"
                )
        except Exception as e:
            self.test(
                "Codebase retrieval works",
                False,
                f"Error: {e}"
            )

    def test_documentation_chunks_exist(self):
        """Test that documentation chunks are available for retrieval."""
        print("\n" + "=" * 60)
        print("Testing Documentation Chunks Availability")
        print("=" * 60)

        if not self.rag_engine.pgvector_retriever:
            print("⚠️  pgvector_retriever not available, skipping test")
            return

        # Try to retrieve with doc_id filter
        try:
            chunks = self.rag_engine.pgvector_retriever.retrieve(
                "architecture",
                top_k=3,
                doc_id="documentation"
            )

            self.test(
                "Can retrieve documentation chunks",
                len(chunks) > 0,
                f"Retrieved {len(chunks)} documentation chunks"
            )

            if chunks:
                # Verify chunks have correct doc_id
                all_docs = all(c.get("doc_id") == "documentation" for c in chunks)
                self.test(
                    "All chunks have doc_id='documentation'",
                    all_docs,
                    f"doc_ids: {[c.get('doc_id') for c in chunks]}"
                )
        except Exception as e:
            self.test(
                "Documentation retrieval works",
                False,
                f"Error: {e}"
            )

    def test_normal_queries_unchanged(self):
        """Test that normal queries still work (regression test)."""
        print("\n" + "=" * 60)
        print("Testing Normal Queries (Regression)")
        print("=" * 60)

        state: ConversationState = {
            "query": "How does RAG work?",
            "role": "Software Developer"
        }

        result = retrieve_chunks(state, self.rag_engine, top_k=4)

        chunks = result.get("retrieved_chunks", [])

        self.test(
            "Normal query retrieves chunks",
            len(chunks) > 0,
            f"Retrieved {len(chunks)} chunks"
        )

        # Normal queries should NOT set preferred_source_used
        metadata = result.get("analytics_metadata", {})
        preferred_used = metadata.get("preferred_source_used", False)

        self.test(
            "Normal query does NOT use preferred sources",
            not preferred_used,
            f"preferred_source_used={preferred_used}"
        )

    def run_all(self):
        """Run all tests."""
        print("\n" + "=" * 60)
        print("Self-Referential Query Functionality Tests")
        print("=" * 60)

        self.test_self_referential_detection()
        self.test_retrieval_prioritization()
        self.test_codebase_chunks_exist()
        self.test_documentation_chunks_exist()
        self.test_normal_queries_unchanged()

        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"Total: {self.passed + self.failed}")
        print("=" * 60)

        return self.failed == 0


if __name__ == "__main__":
    tester = TestSelfReferentialQueries()
    success = tester.run_all()
    sys.exit(0 if success else 1)
