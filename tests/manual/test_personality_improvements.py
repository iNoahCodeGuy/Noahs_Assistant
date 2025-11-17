"""Quick validation that Phase 1 personality improvements work correctly.

Tests:
1. Warmth amplifiers are present in role prompts
2. Performance metrics helper functions correctly
3. Metrics injection works as expected
4. No regressions in quality standards
"""

from assistant.flows.node_logic.performance_metrics import PerformanceMetrics
from assistant.core.response_generator import ResponseGenerator
from assistant.core.langchain_compat import ChatOpenAI
from assistant.config.supabase_config import supabase_settings

def test_warmth_in_prompts():
    """Verify warmth amplifiers were added to all role prompts."""
    # Initialize with minimal LLM (not used for prompt generation)
    llm = ChatOpenAI(
        model=supabase_settings.openai_model,
        temperature=0.7,
        api_key=supabase_settings.openai_api_key
    )
    generator = ResponseGenerator(llm=llm)

    # Technical HM prompt should have warmth
    tech_hm_prompt = generator._build_role_prompt("test query", "context", role="Hiring Manager (technical)")
    assert "I'm excited to help" in tech_hm_prompt, "Missing warmth in Technical HM prompt"
    assert "CONVERSATIONAL WARMTH GUIDELINES" in tech_hm_prompt, "Missing warmth guidelines"
    assert "colleague who built something cool" in tech_hm_prompt, "Missing personality descriptor"

    # Developer prompt should have warmth
    dev_prompt = generator._build_role_prompt("test query", "context", role="Software Developer")
    assert "I'm excited to walk you through" in dev_prompt, "Missing warmth in Developer prompt"
    assert "pair programming with a colleague" in dev_prompt, "Missing dev personality"

    # Casual prompt should have warmth
    casual_prompt = generator._build_role_prompt("test query", "context", role="Just looking around")
    assert "genuinely excited to explain" in casual_prompt, "Missing warmth in Casual prompt"
    assert "friendly and inviting" in casual_prompt, "Missing casual warmth guidelines"

    print("✅ All role prompts have warmth amplifiers")

def test_performance_metrics():
    """Verify performance metrics helper returns expected formats."""

    # RAG metrics
    rag = PerformanceMetrics.get_rag_metrics()
    assert "~1.2s average" in rag, "Missing latency"
    assert "$0.0003/query" in rag, "Missing cost per query"
    assert "100k queries/day" in rag, "Missing scale"

    # Embedding metrics
    embed = PerformanceMetrics.get_embedding_metrics()
    assert "150ms/query" in embed, "Missing embedding latency"
    assert "text-embedding-3-small" in embed, "Missing model name"

    # Storage metrics
    storage = PerformanceMetrics.get_storage_metrics()
    assert "245 career highlights" in storage, "Missing data size"
    assert "pgvector" in storage, "Missing tech stack"

    # Cost breakdown
    cost = PerformanceMetrics.get_cost_breakdown()
    assert "$40/month" in cost or "$45/month" in cost, "Missing total cost"

    # Scale metrics
    scale = PerformanceMetrics.get_scale_metrics()
    assert "At 1k users" in scale, "Missing scale projection"

    print("✅ All performance metrics return expected formats")

def test_metrics_injection():
    """Verify metrics inject correctly into responses."""

    # Test injection before "Would you like..."
    response1 = "I use RAG to retrieve information. Would you like to learn more?"
    injected1 = PerformanceMetrics.inject_into_response(response1, "rag")
    assert "**Performance**" in injected1, "Metrics not injected"
    assert injected1.index("**Performance**") < injected1.index("Would you like"), \
        "Metrics not before follow-up"

    # Test injection before "Want to see..."
    response2 = "Here's how it works. Want to see the code?"
    injected2 = PerformanceMetrics.inject_into_response(response2, "embedding")
    assert "**Embedding Speed**" in injected2, "Embedding metrics not injected"

    # Test injection at end if no follow-up
    response3 = "This is how RAG works."
    injected3 = PerformanceMetrics.inject_into_response(response3, "storage")
    assert "**Storage**" in injected3, "Storage metrics not injected"

    # Test no duplicate injection
    response4 = "I use RAG. **Performance**: ~1.2s average. Would you like more?"
    injected4 = PerformanceMetrics.inject_into_response(response4, "rag")
    assert response4 == injected4, "Duplicate metrics injected"

    print("✅ Metrics injection works correctly")

def test_quality_standards_maintained():
    """Verify warmth changes don't break existing quality rules."""
    llm = ChatOpenAI(
        model=supabase_settings.openai_model,
        temperature=0.7,
        api_key=supabase_settings.openai_api_key
    )
    generator = ResponseGenerator(llm=llm)

    # Check prompts still have critical rules
    tech_hm_prompt = generator._build_role_prompt("test query", "context", role="Hiring Manager (technical)")
    assert "CRITICAL RULES:" in tech_hm_prompt, "Missing critical rules section"
    assert "THIRD PERSON about Noah" in tech_hm_prompt, "Missing first/third person rule"
    assert "Strip markdown headers" in tech_hm_prompt, "Missing strip headers rule"

    print("✅ Quality standards maintained in prompts")

if __name__ == "__main__":
    print("\n🧪 Testing Phase 1 Personality Improvements\n")

    try:
        test_warmth_in_prompts()
        test_performance_metrics()
        test_metrics_injection()
        test_quality_standards_maintained()

        print("\n✅ All personality improvement tests passed!")
        print("\nNext steps:")
        print("1. Run full test suite: pytest tests/ -v")
        print("2. Commit changes: git add src/core/response_generator.py src/flows/performance_metrics.py")
        print("3. Test with real queries in Streamlit UI")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1)
