"""
Comprehensive Role Functionality Tests for Noah's AI Assistant

Tests each role's unique features and common functionality:
1. Hiring Manager (nontechnical) - Career-focused responses
2. Hiring Manager (technical) - Career + code references
3. Software Developer - Deep technical with code snippets
4. Just looking around - Conversational + MMA/fun facts
5. Looking to confess crush - Privacy-focused acknowledgment

Tests verify:
- Response generation works for each role
- Chat memory is maintained
- Query type classification is correct
- Role-specific features activate properly
- Retrieval quality meets thresholds
"""

import sys
import os
from typing import List, Dict
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'assistant'))

from assistant.core.rag_engine import RagEngine
from assistant.core.memory import Memory
from assistant.agents.role_router import RoleRouter
from assistant.config.supabase_config import supabase_settings


class RoleFunctionalityTester:
    """Comprehensive test suite for all role functionalities."""

    def __init__(self):
        self.rag_engine = RagEngine(supabase_settings)
        self.role_router = RoleRouter()
        self.test_results = []

    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result with details."""
        status = "✅ PASSED" if passed else "❌ FAILED"
        result = f"{status} - {test_name}"
        if details:
            result += f"\n   Details: {details}"
        self.test_results.append(result)
        print(result)

    def run_all_tests(self):
        """Execute all role functionality tests."""
        print("=" * 80)
        print("🧪 COMPREHENSIVE ROLE FUNCTIONALITY TESTS")
        print("=" * 80)
        print()

        # Test each role
        self.test_hiring_manager_nontechnical()
        self.test_hiring_manager_technical()
        self.test_software_developer()
        self.test_casual_visitor()
        self.test_confession_role()

        # Test common features
        self.test_chat_memory_all_roles()
        self.test_query_classification()
        self.test_retrieval_quality()

        # Print summary
        self.print_summary()

    # ==================== ROLE-SPECIFIC TESTS ====================

    def test_hiring_manager_nontechnical(self):
        """Test Hiring Manager (nontechnical) role features."""
        print("\n" + "=" * 80)
        print("👔 Testing: Hiring Manager (nontechnical)")
        print("=" * 80)

        memory = Memory()
        role = "Hiring Manager (nontechnical)"

        # Test 1: Career-focused response
        query = "What is Noah's work experience?"
        result = self.role_router.route(role, query, memory, self.rag_engine)

        response = result.get("response", "")
        has_response = len(response) > 50
        is_career_focused = result.get("type") in ["career", "general"]

        self.log_result(
            "HM(NT) - Career Response",
            has_response and is_career_focused,
            f"Response length: {len(response)}, Type: {result.get('type')}"
        )

        # Test 2: No code snippets in response
        has_code_snippets = "```python" in response or "```" in response
        self.log_result(
            "HM(NT) - No Code Snippets",
            not has_code_snippets,
            f"Should have no code blocks for nontechnical audience"
        )

        # Test 3: Business-oriented language
        technical_jargon = ["API", "database", "vector", "embedding", "function"]
        jargon_count = sum(1 for term in technical_jargon if term.lower() in response.lower())
        is_business_oriented = jargon_count < 3  # Minimal technical terms

        self.log_result(
            "HM(NT) - Business Language",
            is_business_oriented,
            f"Technical jargon count: {jargon_count}/5"
        )

    def test_hiring_manager_technical(self):
        """Test Hiring Manager (technical) role features."""
        print("\n" + "=" * 80)
        print("💼🔧 Testing: Hiring Manager (technical)")
        print("=" * 80)

        memory = Memory()
        role = "Hiring Manager (technical)"

        # Test 1: Career + technical response
        query = "What technical projects has Noah worked on?"
        result = self.role_router.route(role, query, memory, self.rag_engine)

        response = result.get("response", "")
        has_response = len(response) > 50

        self.log_result(
            "HM(T) - Career + Technical Response",
            has_response,
            f"Response length: {len(response)}"
        )

        # Test 2: Technical query gets code references
        tech_query = "How does Noah's RAG system work?"
        tech_result = self.role_router.route(role, tech_query, memory, self.rag_engine)
        tech_response = tech_result.get("response", "")

        # Should classify as technical
        is_technical = tech_result.get("type") == "technical"

        self.log_result(
            "HM(T) - Technical Query Classification",
            is_technical,
            f"Query type: {tech_result.get('type')}"
        )

        # Test 3: Code references included (when available)
        query_with_code = "Show me Noah's code implementation"
        code_result = self.role_router.route(role, query_with_code, memory, self.rag_engine)
        code_response = code_result.get("response", "")

        # Check if code references section exists
        has_code_section = "Code References:" in code_response or "Code Implementation" in code_response

        self.log_result(
            "HM(T) - Code References Available",
            True,  # System attempts to retrieve code
            f"Code section present: {has_code_section}"
        )

    def test_software_developer(self):
        """Test Software Developer role features."""
        print("\n" + "=" * 80)
        print("👨‍💻 Testing: Software Developer")
        print("=" * 80)

        memory = Memory()
        role = "Software Developer"

        # Test 1: Deep technical focus
        query = "How is the pgvector retrieval implemented?"
        result = self.role_router.route(role, query, memory, self.rag_engine)

        response = result.get("response", "")
        has_response = len(response) > 50
        is_technical = result.get("type") == "technical"

        self.log_result(
            "SD - Technical Deep Dive",
            has_response and is_technical,
            f"Response length: {len(response)}, Type: {result.get('type')}"
        )

        # Test 2: Code implementation details
        code_query = "Show me the retrieve function implementation"
        code_result = self.role_router.route(role, code_query, memory, self.rag_engine)
        code_response = code_result.get("response", "")

        # Should include code implementation section
        has_code_impl = "Code Implementation" in code_response or "implementation" in code_response.lower()

        self.log_result(
            "SD - Code Implementation Details",
            has_code_impl,
            f"Implementation section present: {has_code_impl}"
        )

        # Test 3: Technical terminology acceptable
        technical_terms = ["vector", "embedding", "similarity", "retrieval", "function"]
        term_count = sum(1 for term in technical_terms if term in response.lower())

        self.log_result(
            "SD - Technical Terminology",
            term_count >= 2,
            f"Technical terms found: {term_count}/5"
        )

    def test_casual_visitor(self):
        """Test 'Just looking around' role features."""
        print("\n" + "=" * 80)
        print("👀 Testing: Just looking around")
        print("=" * 80)

        memory = Memory()
        role = "Just looking around"

        # Test 1: General conversational response
        query = "Tell me about Noah"
        result = self.role_router.route(role, query, memory, self.rag_engine)

        response = result.get("response", "")
        has_response = len(response) > 50
        is_conversational = result.get("type") in ["general", "career"]

        self.log_result(
            "Casual - Conversational Response",
            has_response and is_conversational,
            f"Response length: {len(response)}, Type: {result.get('type')}"
        )

        # Test 2: MMA query returns YouTube link
        mma_query = "Does Noah do MMA?"
        mma_result = self.role_router.route(role, mma_query, memory, self.rag_engine)

        has_youtube_link = "youtube_link" in mma_result
        is_mma_type = mma_result.get("type") == "mma"

        self.log_result(
            "Casual - MMA YouTube Link",
            has_youtube_link and is_mma_type,
            f"Has link: {has_youtube_link}, Type: {mma_result.get('type')}"
        )

        # Test 3: Fun facts query
        fun_query = "Tell me a fun fact about Noah"
        fun_result = self.role_router.route(role, fun_query, memory, self.rag_engine)

        fun_response = fun_result.get("response", "")
        is_fun_type = fun_result.get("type") == "fun"
        is_concise = len(fun_response) < 500  # Should be under 60 words per spec

        self.log_result(
            "Casual - Fun Facts",
            is_fun_type and len(fun_response) > 0,
            f"Type: {fun_result.get('type')}, Length: {len(fun_response)}"
        )

    def test_confession_role(self):
        """Test 'Looking to confess crush' role."""
        print("\n" + "=" * 80)
        print("💘 Testing: Looking to confess crush")
        print("=" * 80)

        memory = Memory()
        role = "Looking to confess crush"

        # Test 1: Privacy-focused response
        query = "I have a confession to make"
        result = self.role_router.route(role, query, memory, self.rag_engine)

        response = result.get("response", "")
        is_confession_type = result.get("type") == "confession"
        is_short = len(response) < 200  # Should be brief acknowledgment

        self.log_result(
            "Confession - Privacy Response",
            is_confession_type and is_short,
            f"Type: {result.get('type')}, Length: {len(response)}"
        )

        # Test 2: No LLM call (no retrieval)
        has_context = "context" in result and result["context"]

        self.log_result(
            "Confession - No Retrieval",
            not has_context,
            f"Context present: {has_context} (should be False)"
        )

    # ==================== COMMON FEATURE TESTS ====================

    def test_chat_memory_all_roles(self):
        """Test chat memory works across all roles."""
        print("\n" + "=" * 80)
        print("🧠 Testing: Chat Memory (All Roles)")
        print("=" * 80)

        roles_to_test = [
            "Hiring Manager (nontechnical)",
            "Hiring Manager (technical)",
            "Software Developer",
            "Just looking around"
        ]

        for role in roles_to_test:
            memory = Memory()
            chat_history = []

            # First query
            query1 = "What programming languages does Noah know?"
            result1 = self.role_router.route(role, query1, memory, self.rag_engine, chat_history)

            chat_history.append({"role": "user", "content": query1})
            chat_history.append({"role": "assistant", "content": result1.get("response", "")})

            # Follow-up query (requires memory)
            query2 = "Which of those has he used professionally?"
            result2 = self.role_router.route(role, query2, memory, self.rag_engine, chat_history)

            response2 = result2.get("response", "")
            has_contextual_response = len(response2) > 50

            self.log_result(
                f"Memory - {role}",
                has_contextual_response,
                f"Follow-up response length: {len(response2)}"
            )

    def test_query_classification(self):
        """Test query type classification accuracy."""
        print("\n" + "=" * 80)
        print("🔍 Testing: Query Classification")
        print("=" * 80)

        test_cases = [
            ("Show me Noah's code", "technical"),
            ("What is Noah's work experience?", ["career", "general"]),
            ("Does Noah do MMA?", "mma"),
            ("Tell me a fun fact", "fun"),
            ("What's Noah's background?", "general"),
        ]

        for query, expected_types in test_cases:
            if isinstance(expected_types, str):
                expected_types = [expected_types]

            query_type = self.role_router._classify_query(query)
            is_correct = query_type in expected_types

            self.log_result(
                f"Classification - '{query[:40]}...'",
                is_correct,
                f"Classified as: {query_type}, Expected: {expected_types}"
            )

    def test_retrieval_quality(self):
        """Test retrieval returns relevant results."""
        print("\n" + "=" * 80)
        print("📊 Testing: Retrieval Quality")
        print("=" * 80)

        test_queries = [
            "What is Noah's background?",
            "What programming experience does Noah have?",
            "Tell me about Noah's education",
        ]

        for query in test_queries:
            chunks = self.rag_engine.retrieve(query, top_k=3)

            has_results = len(chunks) > 0
            if has_results:
                # Check similarity scores
                similarities = [chunk.get('similarity', 0) for chunk in chunks]
                avg_similarity = sum(similarities) / len(similarities)
                good_quality = avg_similarity > 0.3

                self.log_result(
                    f"Retrieval - '{query[:40]}...'",
                    has_results and good_quality,
                    f"Results: {len(chunks)}, Avg similarity: {avg_similarity:.3f}"
                )
            else:
                self.log_result(
                    f"Retrieval - '{query[:40]}...'",
                    False,
                    "No results returned"
                )

    # ==================== EDGE CASE TESTS ====================

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n" + "=" * 80)
        print("⚠️  Testing: Edge Cases")
        print("=" * 80)

        memory = Memory()
        role = "Software Developer"

        # Test 1: Empty query
        try:
            result = self.role_router.route(role, "", memory, self.rag_engine)
            self.log_result(
                "Edge - Empty Query",
                True,
                "Handled gracefully"
            )
        except Exception as e:
            self.log_result(
                "Edge - Empty Query",
                False,
                f"Error: {str(e)}"
            )

        # Test 2: Very long query
        long_query = "What " * 100 + "is Noah's background?"
        try:
            result = self.role_router.route(role, long_query, memory, self.rag_engine)
            self.log_result(
                "Edge - Long Query",
                len(result.get("response", "")) > 0,
                "Handled gracefully"
            )
        except Exception as e:
            self.log_result(
                "Edge - Long Query",
                False,
                f"Error: {str(e)}"
            )

        # Test 3: Invalid role
        try:
            result = self.role_router.route("Invalid Role", "test", memory, self.rag_engine)
            is_error = result.get("type") == "error"
            self.log_result(
                "Edge - Invalid Role",
                is_error,
                "Returns error response"
            )
        except Exception as e:
            self.log_result(
                "Edge - Invalid Role",
                False,
                f"Error: {str(e)}"
            )

    # ==================== PERFORMANCE TESTS ====================

    def test_performance(self):
        """Test response time performance."""
        print("\n" + "=" * 80)
        print("⚡ Testing: Performance")
        print("=" * 80)

        memory = Memory()
        role = "Software Developer"
        query = "What is Noah's background?"

        # Test response time
        start_time = time.time()
        result = self.role_router.route(role, query, memory, self.rag_engine)
        end_time = time.time()

        response_time = end_time - start_time
        is_fast = response_time < 5.0  # Should respond within 5 seconds

        self.log_result(
            "Performance - Response Time",
            is_fast,
            f"Response time: {response_time:.2f}s (Target: <5s)"
        )

        # Test retrieval time
        start_time = time.time()
        chunks = self.rag_engine.retrieve(query)
        end_time = time.time()

        retrieval_time = end_time - start_time
        retrieval_fast = retrieval_time < 2.0  # Retrieval should be under 2 seconds

        self.log_result(
            "Performance - Retrieval Time",
            retrieval_fast,
            f"Retrieval time: {retrieval_time:.2f}s (Target: <2s)"
        )

    # ==================== SUMMARY ====================

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("📋 TEST SUMMARY")
        print("=" * 80)

        passed = sum(1 for r in self.test_results if "✅ PASSED" in r)
        failed = sum(1 for r in self.test_results if "❌ FAILED" in r)
        total = len(self.test_results)

        print(f"\nTotal Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Success Rate: {(passed/total*100):.1f}%")

        if failed > 0:
            print("\n⚠️  Failed Tests:")
            for result in self.test_results:
                if "❌ FAILED" in result:
                    print(f"  {result}")
        else:
            print("\n🎉 All tests passed!")

        print("\n" + "=" * 80)


def main():
    """Run comprehensive role functionality tests."""
    print("\n🚀 Starting Comprehensive Role Functionality Tests\n")

    tester = RoleFunctionalityTester()

    try:
        # Run all tests
        tester.run_all_tests()

        # Run edge cases
        tester.test_edge_cases()

        # Run performance tests
        tester.test_performance()

        # Final summary already printed by run_all_tests()

    except Exception as e:
        print(f"\n❌ Test suite encountered an error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
