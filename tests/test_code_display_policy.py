"""Test code display and import explanation functionality.

This test suite validates:
1. Code display triggers detect appropriate queries (classify_query flags)
2. Import explanation triggers detect stack justification questions
3. Action planning queues the right enrichment actions (plan_actions)
4. Import explanation retrieval returns the correct tier
5. Code/import formatting helpers produce correct markdown

Current architecture notes:
- ConversationState is a TypedDict (plain dict). Nodes read and write state
  keys directly; classify_query and plan_actions merge their updates into the
  state dict and return it.
- plan_actions queues {"type": "include_code_reference"} for code display and
  {"type": "explain_imports"} for import/stack questions (the old
  "display_code_snippet" action type no longer exists).
"""

import pytest
from assistant.state.conversation_state import ConversationState
from assistant.flows.conversation_nodes import classify_query, plan_actions
from assistant.retrieval.import_retriever import (
    get_import_explanation,
    search_import_explanations,
    detect_import_in_query,
    ROLE_TO_TIER
)
from assistant.flows import content_blocks


def _make_state(role: str, query: str) -> ConversationState:
    """Build a minimal conversation state dict for node-level tests."""
    return {"role": role, "query": query, "chat_history": []}


class TestCodeDisplayTriggers:
    """Test that code display requests are detected correctly."""

    def test_explicit_show_code_request(self):
        """Explicit 'show code' should trigger code display."""
        state = _make_state("Software Developer", "show me the code for retrieval")
        result = classify_query(state)
        assert result.get("code_display_requested") is True
        assert result.get("query_type") == "technical"

    def test_show_implementation_request(self):
        """'show implementation' should trigger code display."""
        state = _make_state("Hiring Manager (technical)", "show implementation of the API")
        result = classify_query(state)
        assert result.get("code_display_requested") is True

    def test_how_do_you_implementation_request(self):
        """'how do you' questions should trigger code display."""
        state = _make_state("Software Developer", "how do you call the Supabase API?")
        result = classify_query(state)
        assert result.get("code_display_requested") is True

    def test_no_trigger_for_general_question(self):
        """General questions should not trigger code display."""
        state = _make_state("Software Developer", "what is Noah's experience?")
        result = classify_query(state)
        assert result.get("code_display_requested") is not True

    def test_code_display_action_planned_for_technical_role(self):
        """Code reference action should be added for technical roles when triggered."""
        state = _make_state("Software Developer", "show me the retrieval code")
        state = classify_query(state)
        state = plan_actions(state)

        action_types = {action["type"] for action in state["pending_actions"]}
        assert "include_code_reference" in action_types


class TestImportExplanationTriggers:
    """Test that import explanation requests are detected correctly."""

    def test_why_use_import_trigger(self):
        """'why use X' should trigger import explanation."""
        state = _make_state("Software Developer", "why use Supabase?")
        result = classify_query(state)
        assert result.get("import_explanation_requested") is True

    def test_explain_imports_trigger(self):
        """'explain imports' should trigger import explanation."""
        state = _make_state("Hiring Manager (technical)", "explain your imports")
        result = classify_query(state)
        assert result.get("import_explanation_requested") is True

    def test_trade_off_question_trigger(self):
        """Trade-off questions should trigger import explanation."""
        state = _make_state("Software Developer", "what are the trade-offs of using pgvector?")
        result = classify_query(state)
        assert result.get("import_explanation_requested") is True

    def test_import_explanation_action_planned(self):
        """Import explanation action should be added when triggered (developer flow)."""
        state = _make_state("Software Developer", "why did you choose OpenAI?")
        state = classify_query(state)
        state = plan_actions(state)

        action_types = {action["type"] for action in state["pending_actions"]}
        assert "explain_imports" in action_types


class TestImportRetrieval:
    """Test import explanation retrieval functions."""

    def test_get_import_explanation_tier1(self):
        """Should retrieve tier 1 explanation for technical hiring manager."""
        result = get_import_explanation("openai", "Hiring Manager (technical)")
        assert result is not None
        assert result["tier"] == "1"
        assert result["import"] == "openai"
        assert "audience" in result

    def test_get_import_explanation_tier2(self):
        """Should retrieve tier 2 explanation for software developer."""
        result = get_import_explanation("supabase", "Software Developer")
        assert result is not None
        assert result["tier"] == "2"
        assert result["import"] == "supabase"

    def test_detect_import_in_query(self):
        """Should detect import names in queries."""
        assert detect_import_in_query("why use supabase?") == "supabase"
        assert detect_import_in_query("explain openai integration") == "openai"
        assert detect_import_in_query("what about pgvector?") == "pgvector"
        assert detect_import_in_query("tell me about Noah") is None

    def test_search_import_explanations(self):
        """Should find relevant imports based on query keywords."""
        results = search_import_explanations(
            "why use a vector database?",
            role="Software Developer",
            top_k=3
        )
        assert len(results) > 0
        # Should find pgvector or supabase (both relate to vector/database)
        import_names = [r["import"] for r in results]
        assert any(name in ["pgvector", "supabase"] for name in import_names)

    def test_role_to_tier_mapping(self):
        """Should map roles to correct tiers."""
        assert ROLE_TO_TIER["Software Developer"] == "2"
        assert ROLE_TO_TIER["Hiring Manager (technical)"] == "1"
        assert ROLE_TO_TIER["Just looking around"] == "1"


class TestCodeFormatting:
    """Test code snippet formatting functions."""

    def test_format_code_snippet_basic(self):
        """Should format code with file path and prompt."""
        result = content_blocks.format_code_snippet(
            code="def hello():\n    return 'world'",
            file_path="src/example.py",
            language="python"
        )
        assert "src/example.py" in result
        assert "def hello()" in result
        assert "Would you like to see" in result

    def test_format_code_snippet_with_description(self):
        """Should include description when provided."""
        result = content_blocks.format_code_snippet(
            code="print('test')",
            file_path="src/test.py",
            description="This prints a test message"
        )
        assert "This prints a test message" in result

    def test_format_code_snippet_includes_branch(self):
        """Should include git branch in metadata."""
        result = content_blocks.format_code_snippet(
            code="pass",
            file_path="src/file.py",
            branch="feature/new"
        )
        assert "feature/new" in result

    def test_code_display_guardrails(self):
        """Should return guardrails message."""
        result = content_blocks.code_display_guardrails()
        assert "sensitive values" in result.lower() or "api keys" in result.lower()
        assert "10-40 lines" in result


class TestImportFormatting:
    """Test import explanation formatting functions."""

    def test_format_import_explanation_tier1(self):
        """Should format tier 1 explanation without enterprise details."""
        result = content_blocks.format_import_explanation(
            import_name="openai",
            tier="1",
            explanation="OpenAI provides GPT-4 API for LLM capabilities."
        )
        assert "OPENAI" in result
        assert "OpenAI provides GPT-4" in result

    def test_format_import_explanation_tier2_with_concerns(self):
        """Should format tier 2 with enterprise concerns."""
        result = content_blocks.format_import_explanation(
            import_name="supabase",
            tier="2",
            explanation="Supabase combines database and auth.",
            enterprise_concern="Limited high-availability options."
        )
        assert "SUPABASE" in result
        assert "Enterprise Concerns" in result
        assert "Limited high-availability" in result

    def test_format_import_explanation_tier3_full(self):
        """Should format tier 3 with all enterprise context."""
        result = content_blocks.format_import_explanation(
            import_name="pgvector",
            tier="3",
            explanation="pgvector provides vector search in Postgres.",
            enterprise_concern="Slower than dedicated vector DBs at scale.",
            enterprise_alternative="Pinecone or Weaviate for production.",
            when_to_switch="When managing >1M vectors."
        )
        assert "PGVECTOR" in result
        assert "Enterprise Concerns" in result
        assert "Enterprise Alternative" in result
        assert "When to Switch" in result
        assert "Pinecone" in result


class TestEndToEndFlow:
    """Test complete code display and import explanation flow."""

    def test_developer_asks_how_do_you(self):
        """Software developer asking 'how do you' should get code + import actions."""
        state = _make_state("Software Developer", "how do you retrieve from pgvector?")

        # Classify should detect code display and import explanation
        state = classify_query(state)
        assert state.get("code_display_requested") is True
        assert state.get("import_explanation_requested") is True

        # Plan should add both actions
        state = plan_actions(state)
        action_types = {action["type"] for action in state["pending_actions"]}
        assert "include_code_reference" in action_types
        assert "explain_imports" in action_types

    def test_hiring_manager_asks_why_supabase(self):
        """Technical hiring manager should get tier 1 explanation."""
        state = _make_state(
            "Hiring Manager (technical)",
            "why did you use Supabase instead of separate services?"
        )

        state = classify_query(state)
        assert state.get("import_explanation_requested") is True

        # Should get tier 1 explanation
        explanation = get_import_explanation("supabase", state["role"])
        assert explanation["tier"] == "1"

    def test_casual_user_no_code_display(self):
        """Casual users shouldn't trigger code display even with technical words."""
        state = _make_state("Just looking around", "how does this work?")

        state = classify_query(state)
        state = plan_actions(state)

        action_types = {action["type"] for action in state["pending_actions"]}
        # Should not include code display for casual role
        assert "include_code_reference" not in action_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
