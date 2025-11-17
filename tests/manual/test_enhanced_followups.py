"""
Quick validation test for enhanced follow-up system.
Tests that all roles get follow-up suggestions after responses.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'assistant'))

from assistant.core.response_generator import ResponseGenerator
from assistant.core.rag_factory import RagEngineFactory
from assistant.config.supabase_config import supabase_settings

def test_followups_for_all_roles():
    """Test that all roles receive follow-up suggestions."""

    # Create response generator
    factory = RagEngineFactory(supabase_settings)
    llm, degraded = factory.create_llm()
    generator = ResponseGenerator(llm=llm, qa_chain=None, degraded_mode=degraded)

    # Test roles
    roles = [
        "Software Developer",
        "Hiring Manager (technical)",
        "Hiring Manager (nontechnical)",
        "Just looking around",
        "Looking to confess crush"
    ]

    # Test query
    query = "How does this AI assistant work?"
    base_response = "Noah built this using LangGraph, RAG, and pgvector for semantic search."

    print("=" * 80)
    print("FOLLOW-UP SUGGESTIONS TEST")
    print("=" * 80)

    for role in roles:
        print(f"\n{'='*80}")
        print(f"ROLE: {role}")
        print(f"{'='*80}")

        # Generate response with follow-up
        enhanced_response = generator._add_technical_followup(base_response, query, role)

        # Check if follow-up was added
        has_followup = len(enhanced_response) > len(base_response)
        followup_marker = "**" in enhanced_response and ("What would you like" in enhanced_response or "Would you like" in enhanced_response or "Want to explore" in enhanced_response or "Explore more" in enhanced_response)

        print(f"\nBase response length: {len(base_response)} chars")
        print(f"Enhanced response length: {len(enhanced_response)} chars")
        print(f"Follow-up added: {'✅ YES' if has_followup else '❌ NO'}")
        print(f"Follow-up format valid: {'✅ YES' if followup_marker else '❌ NO'}")

        print(f"\nFull enhanced response:")
        print("-" * 80)
        print(enhanced_response)
        print("-" * 80)

    print("\n" + "="*80)
    print("TEST: Enterprise Adaptation Context")
    print("="*80)

    # Test enterprise query
    enterprise_query = "How would this work for a large enterprise company?"
    enterprise_response = "Noah would scale this by adding Redis caching, managed vector DB, and SSO."

    for role in ["Hiring Manager (nontechnical)", "Software Developer"]:
        print(f"\nRole: {role}")
        enhanced = generator._add_technical_followup(enterprise_response, enterprise_query, role)
        has_enterprise = "enterprise" in enhanced.lower() or "10,000+" in enhanced or "stack" in enhanced.lower()
        print(f"Enterprise suggestions: {'✅ INCLUDED' if has_enterprise else '❌ MISSING'}")
        print(f"Follow-up:\n{enhanced[len(enterprise_response):]}")

if __name__ == "__main__":
    test_followups_for_all_roles()
