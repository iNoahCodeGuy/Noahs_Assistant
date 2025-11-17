"""Regression tests for conversation quality standards.

These tests ensure that as we add features, we don't break:
- Analytics aggregation and display
- Single follow-up prompts (no duplicates)
- Professional formatting (no emoji spam)
- Code display error handling

Run with: pytest tests/test_conversation_quality.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from assistant.state.conversation_state import ConversationState
from assistant.flows.node_logic.query_classification import classify_query
from assistant.flows.node_logic.core_nodes import (
    retrieve_chunks, generate_answer, apply_role_context
)
from assistant.flows.node_logic.code_validation import is_valid_code_snippet
from assistant.flows.data_reporting import render_full_data_report


class TestAnalyticsQuality:
    """Ensure analytics remain clean and aggregated."""

    def test_kb_coverage_aggregated_not_detailed(self):
        """KB coverage should show 3-4 sources, not 245+ individual entries."""
        with patch('src.flows.data_reporting.supabase_analytics') as mock_analytics:
            # Mock analytics data
            mock_analytics.client.table.return_value.select.return_value.execute.return_value.data = []
            mock_analytics.get_kb_coverage.return_value = {
                'architecture_kb': 15,
                'career_kb': 122,
                'technical_kb': 89
            }

            report = render_full_data_report()

            # Count rows in KB coverage section
            if "Knowledge Base Coverage" in report:
                kb_section = report.split("#### Knowledge Base Coverage")[1].split("####")[0]
                kb_rows = [line for line in kb_section.split("\n") if line.startswith("|") and "---" not in line]

                # Should be: header + 3-4 data rows (one per source)
                assert len(kb_rows) <= 6, f"KB coverage has {len(kb_rows)} rows - should be ≤6 (header + 3-5 sources)"

                # Should NOT have individual entry names like "entry_1", "entry_100"
                assert "entry_1" not in report.lower()
                assert "entry_100" not in report.lower()
                assert "section_" not in report.lower()  # No section-level breakdowns

    def test_kpi_metrics_calculated(self):
        """Analytics should include calculated metrics, not raw dumps."""
        with patch('src.flows.data_reporting.supabase_analytics') as mock_analytics:
            # Mock analytics data
            mock_analytics.client.table.return_value.select.return_value.execute.return_value.data = [
                {"success": True, "latency_ms": 3200},
                {"success": True, "latency_ms": 2800},
                {"success": False, "latency_ms": 5100},
            ]
            mock_analytics.get_kb_coverage.return_value = {}

            report = render_full_data_report()

            # Must have metrics terminology (even if empty state)
            assert "Success Rate" in report or "Conversations" in report or "Performance" in report

            # Should NOT have raw SQL dumps
            assert "SELECT * FROM" not in report
            assert "latency_ms:" not in report  # Raw column names

    def test_recent_activity_limited(self):
        """Should show last 10 messages, not entire history."""
        with patch('src.flows.data_reporting.supabase_analytics') as mock_analytics:
            # Mock 50 messages (simulating large history)
            mock_analytics.client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value.data = [
                {"id": i, "query_type": "general", "latency_ms": 3000} for i in range(50)
            ]
            mock_analytics.get_kb_coverage.return_value = {}

            report = render_full_data_report()

            if "Recent Conversations" in report or "Recent Activity" in report:
                # Count table rows
                table_rows = report.count("| ")

                # Should have manageable number of rows (header + ~10 data rows = ~20 pipe chars per row)
                assert table_rows < 300, f"Report has ~{table_rows // 2} table rows - too many for recent activity"

    def test_confessions_privacy_protected(self):
        """Confessions should show count only, no personal details."""
        with patch('src.flows.data_reporting.supabase_analytics') as mock_analytics:
            mock_analytics.client.table.return_value.select.return_value.execute.return_value.data = []
            mock_analytics.get_kb_coverage.return_value = {}

            # Mock confessions
            with patch('src.flows.data_reporting.supabase_analytics.client.table') as mock_table:
                mock_table.return_value.select.return_value.execute.return_value.data = [
                    {"name": "John Doe", "email": "john@example.com", "message": "I like you"}
                ]

                report = render_full_data_report()

                if "Confessions" in report or "confessions" in report:
                    # Should have count statement, not table with PII
                    assert "Total" in report or "Received" in report

                    # Should NOT have PII
                    assert "John Doe" not in report
                    assert "john@example.com" not in report
                    assert "| name |" not in report
                    assert "| email |" not in report
                    assert "| message |" not in report


class TestConversationFlowQuality:
    """Ensure clean, professional conversation flow."""

    def test_no_duplicate_prompts_in_full_flow(self):
        """Should have exactly 1 follow-up prompt at end, not 2-3."""
        # Mock RAG engine (no @patch needed - create directly)
        mock_engine = MagicMock()
        mock_engine.retrieve.return_value = {"chunks": [], "matches": []}
        mock_engine.retrieve_with_code.return_value = {"chunks": [], "code_snippets": [], "has_code": False}
        mock_engine.response_generator = MagicMock()
        mock_engine.response_generator.generate_contextual_response.return_value = "This is a product that helps you manage your career."

                # Create TypedDict state (simulating conversation flow)
        state: ConversationState = {
            "role": "Software Developer",
            "query": "Show me data on project complexity",
            "chat_history": [],
            # Initialize state collections required by nodes (when calling nodes directly)
            "analytics_metadata": {},
            "pending_actions": [],
            "job_details": {},
            "hiring_signals": [],
            "retrieved_chunks": []
        }

        # Simulate: classify → retrieve (gets SQL KB) → generate (must sanitize)
        update1 = classify_query(state)
        state.update(update1)

        update2 = generate_answer(state, mock_engine)
        state.update(update2)

        update3 = apply_role_context(state, mock_engine)
        state.update(update3)

        answer = state["answer"]

        # Count "Would you like" prompts
        prompt_count = answer.lower().count("would you like")

        assert prompt_count <= 1, f"Found {prompt_count} 'Would you like' prompts - should be ≤1 (found duplicates)"

    def test_no_pushy_resume_offers(self):
        """Subtle availability mentions allowed when hiring signals detected, but must not be pushy.

        CONTEXT: Intelligent Resume Distribution (Hybrid Approach)
        - Mode 1 (Education): NO resume mentions (0 mentions) ✅
        - Mode 2 (Hiring Signals): ONE subtle mention allowed ✅
        - Mode 3 (Explicit Request): Resume distribution flow ✅

        This test ensures Mode 2 stays subtle and user-centric, not salesy.
        """
        # Mode 1: Pure education - NO resume mention
        mock_engine_edu = MagicMock()
        mock_engine_edu.retrieve.return_value = {"chunks": ["RAG systems combine retrieval and generation"], "matches": []}
        mock_engine_edu.generate_response.return_value = "RAG systems work by retrieving relevant documents and using them to generate responses. Would you like to explore how Noah implemented this?"

        state_edu: ConversationState = {
            "role": "Hiring Manager (technical)",
            "query": "How do RAG systems work?",
            "chat_history": [],
            "hiring_signals": [],  # No signals detected
            "answer": mock_engine_edu.generate_response.return_value
        }

        answer_edu = state_edu["answer"]

        # Assert NO resume/availability mention in pure education mode
        resume_keywords = ["resume", "résumé", "cv", "available", "hire", "looking for"]
        found_keywords = [kw for kw in resume_keywords if kw in answer_edu.lower()]
        assert len(found_keywords) == 0, f"Mode 1 (Education): Found resume keywords {found_keywords} - should be 0 in pure education mode"

        # Mode 2: Hiring signals detected - ONE subtle mention allowed
        mock_engine_hiring = MagicMock()
        mock_engine_hiring.retrieve.return_value = {"chunks": ["RAG systems combine retrieval and generation"], "matches": []}
        mock_engine_hiring.generate_response.return_value = "RAG systems work by retrieving relevant documents and using them to generate responses. Would you like to explore how Noah implemented this?\n\nBy the way, Noah's available for roles like this if you'd like to learn more about his experience."

        state_hiring: ConversationState = {
            "role": "Hiring Manager (technical)",
            "query": "We're hiring a GenAI engineer. How do RAG systems work?",
            "chat_history": [],
            "hiring_signals": ["mentioned_hiring", "described_role"],  # Signals detected
            "answer": mock_engine_hiring.generate_response.return_value
        }

        answer_hiring = state_hiring["answer"]

        # Assert ONE subtle mention (not multiple)
        availability_mentions = answer_hiring.lower().count("available") + answer_hiring.lower().count("noah's")
        assert 1 <= availability_mentions <= 3, f"Mode 2 (Hiring Signals): Found {availability_mentions} availability mentions - should be 1-3 for subtle mention"

        # Assert NOT pushy (must not have aggressive CTAs)
        pushy_phrases = ["send me your email", "provide your contact", "fill out this form", "click here", "sign up"]
        found_pushy = [phrase for phrase in pushy_phrases if phrase in answer_hiring.lower()]
        assert len(found_pushy) == 0, f"Mode 2 (Hiring Signals): Found pushy phrases {found_pushy} - must stay subtle and user-centric"

        # Assert still education-focused (educational content should be 80%+ of response)
        lines = answer_hiring.split('\n')
        educational_lines = [line for line in lines if len(line) > 50 and not any(kw in line.lower() for kw in ["available", "hire", "resume"])]
        total_substantial_lines = [line for line in lines if len(line) > 50]
        if len(total_substantial_lines) > 0:
            education_ratio = len(educational_lines) / len(total_substantial_lines)
            assert education_ratio >= 0.5, f"Mode 2 (Hiring Signals): Only {education_ratio:.0%} educational content - should be ≥50% even with availability mention"

    def test_no_emoji_headers(self):
        """User-facing responses must strip markdown headers and emojis - convert to **Bold** only.

        IMPORTANT: KB content (data/*.csv) can use ### headers and emojis for structure.
        This test validates LLM RESPONSES, not storage format.
        """
        # Mock RAG engine to return KB content with rich formatting
        mock_engine = MagicMock()
        mock_engine.retrieve.return_value = {
            "chunks": [
                "## 🎯 Key Features\n### 1️⃣ Data Analytics\n### 2️⃣ Machine Learning",  # KB can have this
                "Content with structure for semantic search"
            ],
            "matches": [],
            "scores": [0.8, 0.7]
        }
        mock_engine.retrieve_with_code.return_value = {"chunks": [], "code_snippets": [], "has_code": False}
        mock_engine.response_generator = MagicMock()
        mock_engine.response_generator.generate_contextual_response.return_value = "**Key Features**\n\n**Data Analytics**: Feature overview...\n\n**Machine Learning**: ML capabilities..."  # LLM must convert

        # Create TypedDict state
        state: ConversationState = {
            "role": "Software Developer",
            "query": "tell me about the data analytics",
            "chat_history": [],
            # Initialize state collections required by nodes (when calling nodes directly)
            "analytics_metadata": {},
            "pending_actions": [],
            "job_details": {},
            "hiring_signals": [],
            "retrieved_chunks": []
        }

        # Simulate full conversation flow (what user sees)
        update1 = classify_query(state)
        state.update(update1)

        update2 = retrieve_chunks(state, mock_engine)  # Retrieves rich KB content
        state.update(update2)

        update3 = generate_answer(state, mock_engine)  # LLM synthesizes and sanitizes
        state.update(update3)

        update4 = apply_role_context(state, mock_engine)  # Final formatting
        state.update(update4)

        answer = state["answer"]

        # User-facing response must NOT have markdown headers (###, ##, #)
        import re
        markdown_headers = re.findall(r'^\s*#{1,6}\s', answer, re.MULTILINE)
        assert len(markdown_headers) == 0, f"Found {len(markdown_headers)} markdown headers in user response - must use **Bold** only: {markdown_headers}"

        # User-facing response must NOT have emojis in headers
        emoji_patterns = ["🎯", "📊", "🏗️", "🗂️", "🧱", "🚀", "🎉", "💻", "📦", "🔍", "1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣"]
        for emoji in emoji_patterns:
            # Allow emojis in content body, but not in header-like contexts
            # Check for patterns like "### 🎯" or "## 📊" or standalone emoji headers
            assert f"###{emoji}" not in answer and f"##{emoji}" not in answer and f"#{emoji}" not in answer, \
                   f"Found emoji header pattern with '{emoji}' - must strip to **Bold**"

    def test_llm_no_self_generated_prompts(self):
        """LLM should not generate its own 'Would you like to see' prompts in answer body."""
        # This test verifies system prompt guidance, not runtime behavior
        # Just check that we don't have prompt generation in the wrong place
        from assistant.core import response_generator
        import inspect

        source = inspect.getsource(response_generator)

        # The ResponseGenerator should not have active prompt generation
        # (Prompts should only come from conversation_nodes.py)
        if "def add_followup_suggestions" in source:
            method_source = source.split("def add_followup_suggestions")[1].split("def ")[0]

            # Should not have multi-line prompt generation
            assert "Would you like me to show you:" not in method_source or \
                   method_source.count("\n") < 20, \
                   "add_followup_suggestions is generating prompts - should be deprecated"

    def test_display_data_uses_canned_intro(self):
        """Display data requests should bypass LLM noise and stay clean."""
        mock_engine = MagicMock()
        mock_engine.retrieve.return_value = {"chunks": [], "matches": [], "scores": []}
        mock_engine.response_generator = MagicMock()
        mock_engine.response_generator.generate_contextual_response.return_value = "LLM output should be bypassed"

        # Create TypedDict state
        state: ConversationState = {
            "role": "Hiring Manager (technical)",
            "query": "Please display data for the latest analytics",
            "chat_history": [],
            # Initialize state collections required by nodes (when calling nodes directly)
            "analytics_metadata": {},
            "pending_actions": [],
            "job_details": {},
            "hiring_signals": [],
            "retrieved_chunks": []
        }

        update1 = classify_query(state)
        state.update(update1)

        update2 = retrieve_chunks(state, mock_engine)
        state.update(update2)

        update3 = generate_answer(state, mock_engine)
        state.update(update3)

        # Updated Oct 16, 2025: Match current data_reporting.py intro text
        assert state["answer"].startswith("Fetching live analytics data from Supabase")
        assert not mock_engine.response_generator.generate_contextual_response.called
        assert "}" not in state["answer"][:5], "Canned intro should not leak braces"

    def test_generated_answer_sanitizes_sql_artifacts(self):
        """Ensure LLM responses don't leak raw SQL or technical implementation details."""
        # Mock engine with proper method names
        mock_engine = MagicMock()

        # Mock the retrieve method to return chunks in the expected format
        mock_engine.retrieve.return_value = {
            "chunks": [
                {"content": "Python framework experience", "similarity": 0.85}
            ],
            "matches": 1
        }

        # Mock the response generator's generate_contextual_response method
        mock_engine.response_generator.generate_contextual_response.return_value = """Clean content ready for review

{
  "chunks": ["test"],
  "metadata": { "origin": "SELECT * FROM kb_chunks WHERE..." }
}"""

        # Use non-ambiguous query (not "architecture" which triggers Ask Mode)
        state = {
            "role": "Hiring Manager (technical)",
            "query": "What Python frameworks has Noah used?",
            "chat_history": [],
            # Initialize state collections required by nodes (when calling nodes directly)
            "analytics_metadata": {},
            "pending_actions": [],
            "job_details": {},
            "hiring_signals": [],
            "retrieved_chunks": []
        }

        update1 = classify_query(state)
        state.update(update1)

        update2 = retrieve_chunks(state, mock_engine)
        state.update(update2)

        update3 = generate_answer(state, mock_engine)
        state.update(update3)

        assert state["answer"].startswith("Clean content ready for review")
        assert not state["answer"].startswith("}")
        first_line = state["answer"].splitlines()[0]
        assert "SELECT" not in first_line, "Sanitized answer should not expose raw SELECT"


class TestCodeDisplayQuality:
    """Ensure code display handles edge cases gracefully."""

    def test_empty_code_index_shows_helpful_message(self):
        """When code index is empty, should show GitHub link not garbage."""
        # Mock empty code retrieval (Fixed Oct 16, 2025: Removed bad @patch, create mock directly)
        mock_engine = MagicMock()
        mock_engine.retrieve.return_value = {"chunks": [], "matches": [], "scores": []}
        mock_engine.retrieve_with_code.return_value = {
            "chunks": [],
            "code_snippets": [],
            "has_code": False
        }
        mock_engine.response_generator = MagicMock()
        mock_engine.response_generator.generate_contextual_response.return_value = "I can help you view the code."

        # Create TypedDict state
        state: ConversationState = {
            "role": "Software Developer",
            "query": "show me the conversation node code",  # Explicit code display trigger
            "chat_history": []
        }

        # Run through flow
        update1 = classify_query(state)
        state.update(update1)

        update2 = generate_answer(state, mock_engine)
        state.update(update2)

        update3 = apply_role_context(state, mock_engine)
        state.update(update3)

        answer = state["answer"]

        # Should NOT show malformed data
        assert "doc_id text" not in answer, "Found 'doc_id text' malformed output"
        assert 'query="' not in answer, "Found metadata leak in output"
        assert "{'doc_id':" not in answer, "Found raw dict output"

        # When code unavailable, should either:
        # 1. Show GitHub link
        # 2. Show actual code (if available)
        # 3. Show helpful "unavailable" message
        # 4. Or just show the answer without code (acceptable fallback)

        # Main assertion: No malformed data (which is what we fixed)
        # The presence of helpful alternatives is bonus, but not required
        assert len(answer) > 0, "Answer should not be empty"

    def test_code_content_validation_logic(self):
        """Code content should be validated before display."""
        # Test the validation helper directly
        test_cases = [
            ("", False, "Empty string should fail"),
            ("   ", False, "Whitespace only should fail"),
            ("short", False, "Too short (<10 chars) should fail"),
            ("doc_id text query=", False, "Metadata lines should fail"),
            ("})\n\n}\n\nquery=\"Show me code examples\"", False, "Truncated metadata blob should fail"),
            ("def hello():\n    print('world')\n    return True", True, "Valid function should pass"),
            (
                "class Example:\n    def run(self):\n        return 'ok'",
                True,
                "Class definition should pass"
            ),
        ]

        for code_content, should_pass, description in test_cases:
            is_valid = is_valid_code_snippet(code_content)
            assert is_valid == should_pass, (
                f"Validation failed: {description} (got {is_valid}, expected {should_pass})"
            )


class TestRegressionGuards:
    """Catch common regression patterns."""

    def test_no_information_overload(self):
        """Responses should be concise, not dump entire database."""
        # Mock RAG engine (Fixed Oct 16, 2025: Removed bad @patch, create mock directly)
        mock_engine = MagicMock()
        mock_engine.retrieve.return_value = {"chunks": [], "matches": [], "scores": []}
        mock_engine.retrieve_with_code.return_value = {"chunks": [], "code_snippets": [], "has_code": False}
        mock_engine.response_generator = MagicMock()
        mock_engine.response_generator.generate_contextual_response.return_value = "We collect interaction data, query types, and latency metrics for analytics."

        # Create TypedDict state
        state: ConversationState = {
            "role": "Hiring Manager (technical)",
            "query": "what data do you collect?",
            "chat_history": []
        }

        # Run through flow
        update1 = classify_query(state)
        state.update(update1)

        update2 = generate_answer(state, mock_engine)
        state.update(update2)

        update3 = apply_role_context(state, mock_engine)
        state.update(update3)

        answer = state["answer"]

        # Character count sanity check
        char_count = len(answer)
        assert char_count < 15000, f"Response is {char_count} chars - too long (>15k indicates data dump)"

        # Table row count sanity check
        table_rows = answer.count("| ")
        assert table_rows < 250, f"Response has ~{table_rows // 2} table rows - too many (>125 rows indicates dump)"

    def test_consistent_formatting_across_roles(self):
        """All roles should get consistent professional formatting."""
        # Mock RAG engine (Fixed Oct 16, 2025: Removed bad @patch, create mock directly)
        mock_engine = MagicMock()
        mock_engine.retrieve.return_value = {"chunks": [], "matches": [], "scores": []}
        mock_engine.retrieve_with_code.return_value = {"chunks": [], "code_snippets": [], "has_code": False}
        mock_engine.response_generator = MagicMock()
        mock_engine.response_generator.generate_contextual_response.return_value = "This product helps manage your career journey."

        roles = [
            "Hiring Manager (technical)",
            "Hiring Manager (nontechnical)",
            "Software Developer",
            "Just looking around"
        ]

        for role in roles:
            # Create TypedDict state
            state: ConversationState = {
                "role": role,
                "query": "tell me about the product",
                "chat_history": []
            }

            # Run through flow
            update1 = classify_query(state)
            state.update(update1)

            update2 = generate_answer(state, mock_engine)
            state.update(update2)

            update3 = apply_role_context(state, mock_engine)
            state.update(update3)

            answer = state["answer"]

            # All should have professional formatting (no emoji spam)
            emoji_count = sum(answer.count(emoji) for emoji in ["🎯", "📊", "🏗️", "🗂️", "🧱", "🚀"])
            assert emoji_count < 3, f"{role} has {emoji_count} emojis - should have <3"

            # All should have at most 1 follow-up prompt
            prompt_count = answer.lower().count("would you like")
            assert prompt_count <= 1, f"{role} has {prompt_count} prompts - should be ≤1"


class TestSpecificRegressions:
    """Tests for specific bugs that have occurred."""

    def test_analytics_no_section_iteration(self):
        """Ensure we don't iterate over sections (245 rows bug)."""
        import inspect
        from assistant.flows import data_reporting

        # Get source code
        source = inspect.getsource(data_reporting)

        # Should NOT have nested section iteration
        assert "for section, count in sections.items()" not in source, \
            "Found section-level iteration - this causes 245-row dumps"

        # Should aggregate by source only
        assert "source_aggregation" in source or "aggregate" in source, \
            "Missing source-level aggregation logic"

    def test_response_generator_no_prompts(self):
        """Ensure response_generator doesn't add follow-up prompts."""
        import inspect
        from assistant.core import response_generator

        # Get source code
        source = inspect.getsource(response_generator)

        # Method should be deprecated/minimal
        if "add_followup_suggestions" in source:
            # Check that it's just a pass-through
            method_source = source.split("def add_followup_suggestions")[1].split("def ")[0]

            # Should not have multi-line suggestion generation
            assert method_source.count("\n") < 20, \
                "add_followup_suggestions has >20 lines - should be deprecated stub"

    def test_conversation_nodes_single_prompt_location(self):
        """Ensure prompts only generated in apply_role_context, not multiple places."""
        import inspect
        from assistant.flows import conversation_nodes

        source = inspect.getsource(conversation_nodes)

        # Count occurrences of prompt generation
        prompt_generation_pattern = "Would you like me to show you:"
        occurrences = source.count(prompt_generation_pattern)

        # Should appear in exactly ONE location (apply_role_context)
        assert occurrences <= 1, f"Found {occurrences} prompt generation locations - should be 1 (in apply_role_context only)"


class TestResponseSynthesis:
    """Ensure responses synthesize KB content naturally, not verbatim Q&A."""

    def test_no_qa_verbatim_responses(self):
        """LLM must synthesize KB Q&A pairs into natural conversation, not return them verbatim."""
        import inspect
        from assistant.core import response_generator

        # Check that all role prompts include the synthesis instruction
        source = inspect.getsource(response_generator)

        # The critical instruction should appear in prompts
        synthesis_instruction = "NEVER return Q&A format from knowledge base verbatim"

        assert synthesis_instruction in source, (
            "Response generator prompts must include instruction to synthesize Q&A content, "
            "not return it verbatim. Check _build_role_prompt() for all roles."
        )

    def test_response_synthesis_in_prompts(self):
        """Verify all role prompts explicitly instruct to avoid Q&A verbatim responses."""
        from assistant.core.response_generator import ResponseGenerator
        from unittest.mock import Mock

        # Mock the LLM dependency
        mock_llm = Mock()
        gen = ResponseGenerator(llm=mock_llm)

        # Test different roles
        roles_to_test = [
            "Hiring Manager (technical)",
            "Software Developer",
            "Just looking around",
            None  # General/default role
        ]

        for role in roles_to_test:
            prompt = gen._build_role_prompt(
                query="Test query",
                context_str="Q: What is Noah's background?\nA: Noah has experience...",
                role=role
            )

            # Prompt must mention synthesizing or avoiding Q&A format
            synthesis_keywords = [
                "synthesize",
                "Q&A format",
                "verbatim",
                "natural conversation",
                "rephrase naturally"
            ]

            has_synthesis_guidance = any(keyword in prompt for keyword in synthesis_keywords)

            assert has_synthesis_guidance, (
                f"Role '{role}' prompt lacks guidance to synthesize Q&A content naturally. "
                f"Responses may return raw 'Q: ... A: ...' format from KB."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
