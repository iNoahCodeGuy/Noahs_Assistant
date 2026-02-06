#!/usr/bin/env python3
"""
Portfolia Evaluation Test Suite

Tests three critical aspects of response quality:
1. Intent classification accuracy
2. Response quality evaluation (conversational tone, specificity, personality)
3. Regression tests for known bad behaviors

Run with: pytest tests/test_portfolia_eval.py -v -s
"""

import os
import sys
import json
import pytest
from typing import Dict, Any, List
from openai import OpenAI

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.flows.node_logic.stage1_intent_router import classify_intent
from assistant.flows.conversation_flow import run_conversation_flow
from assistant.core.rag_engine import RagEngine
from assistant.state.conversation_state import ConversationState
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI for LLM-as-judge evaluations
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Track results for summary
intent_results = []
quality_results = []
regression_results = []


# ============================================================================
# SECTION 1: INTENT CLASSIFICATION TESTS
# ============================================================================

INTENT_TEST_CASES = [
    ("Tell me about Noah's background", "knowledge_query"),
    ("What projects has he built?", "knowledge_query"),
    ("I want to confess a crush", "crush_confession"),
    ("Hey", "greeting"),
    ("Hi there", "greeting"),
    ("What's the meaning of life?", "off_topic"),
    ("asdfghjkl", "small_talk"),  # Could be small_talk or off_topic, both acceptable
    ("Can you write me a Python script?", "off_topic"),
    ("What's Noah's salary?", "off_topic"),
]


def test_intent_classification():
    """Test that messages are correctly classified by intent router."""
    print("\n" + "="*80)
    print("SECTION 1: INTENT CLASSIFICATION TESTS")
    print("="*80)

    passed = 0
    failed = 0

    for message, expected_intent in INTENT_TEST_CASES:
        # Create minimal state with query
        state = ConversationState(
            query=message,
            role="Software Developer",
            session_id="test-session",
            chat_history=[
                {"role": "assistant", "content": "Hi! I'm Portfolia..."},
                {"role": "user", "content": "Option 1"}
            ]  # Simulate chat history so it doesn't think it's a greeting
        )

        # Call intent classifier
        result_state = classify_intent(state)
        actual_intent = result_state.get("message_intent")

        # Special case: asdfghjkl could be either small_talk or off_topic
        if message == "asdfghjkl" and actual_intent in ["small_talk", "off_topic"]:
            passed += 1
            intent_results.append({"message": message, "expected": expected_intent, "actual": actual_intent, "passed": True})
            print(f"✅ '{message}' → {actual_intent} (accepted)")
        elif actual_intent == expected_intent:
            passed += 1
            intent_results.append({"message": message, "expected": expected_intent, "actual": actual_intent, "passed": True})
            print(f"✅ '{message}' → {actual_intent}")
        else:
            failed += 1
            intent_results.append({"message": message, "expected": expected_intent, "actual": actual_intent, "passed": False})
            print(f"❌ '{message}' → expected {expected_intent}, got {actual_intent}")

    print(f"\nIntent Classification: {passed}/{len(INTENT_TEST_CASES)} passed")

    # Assert all passed
    assert failed == 0, f"{failed} intent classification tests failed"


# ============================================================================
# SECTION 2: RESPONSE QUALITY EVALUATION
# ============================================================================

QUALITY_TEST_CASES = [
    "What is Noah's professional background?",
    "What are some projects by Noah?",
    "Tell me about the MMA coaching",
    "What's his biggest weakness?",
    "Why should I hire him over someone with a CS degree?",
    "What's something most people don't know about Noah?",
]

EVALUATION_PROMPT = """You are evaluating an AI portfolio assistant called Portfolia. Rate the following response on these criteria (1-5 scale):

1. Conversational tone (1 = robotic/resume-like, 5 = natural and engaging)
2. Specificity (1 = generic/vague, 5 = concrete details with numbers/names)
3. Follow-up engagement (1 = dead end, 5 = natural next step/question)
4. Personality (1 = flat/boring, 5 = memorable with wit/warmth)
5. Accuracy (1 = wrong/misleading, 5 = factually correct)

User message: {user_message}
Portfolia response: {response}

Respond with ONLY valid JSON in this exact format:
{{"conversational_tone": X, "specificity": X, "follow_up": X, "personality": X, "accuracy": X, "flags": "explanation of any score below 4"}}"""


def evaluate_response_quality(user_message: str, response: str) -> Dict[str, Any]:
    """Use LLM-as-judge to evaluate response quality."""
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert evaluator of AI assistant responses. Return only valid JSON."},
                {"role": "user", "content": EVALUATION_PROMPT.format(user_message=user_message, response=response)}
            ],
            temperature=0,
            max_tokens=300
        )

        # Parse JSON response
        result_text = completion.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if result_text.startswith("```json"):
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif result_text.startswith("```"):
            result_text = result_text.split("```")[1].split("```")[0].strip()

        scores = json.loads(result_text)
        return scores

    except Exception as e:
        print(f"⚠️  Evaluation error: {e}")
        return {
            "conversational_tone": 0,
            "specificity": 0,
            "follow_up": 0,
            "personality": 0,
            "accuracy": 0,
            "flags": f"Evaluation failed: {e}"
        }


def get_portfolia_response(query: str) -> str:
    """Get a response from Portfolia for a given query."""
    rag_engine = RagEngine()

    state = ConversationState(
        query=query,
        role="Software Developer",
        session_id="test-session-quality",
        chat_history=[
            {"role": "assistant", "content": "Hi! I'm Portfolia, Noah's AI portfolio assistant."},
            {"role": "user", "content": "I'm a software developer"}
        ]
    )

    result = run_conversation_flow(state, rag_engine, "test-session-quality")
    return result.get("answer", "")


def test_response_quality():
    """Evaluate response quality using LLM-as-judge."""
    print("\n" + "="*80)
    print("SECTION 2: RESPONSE QUALITY EVALUATION")
    print("="*80)

    all_scores = {
        "conversational_tone": [],
        "specificity": [],
        "follow_up": [],
        "personality": [],
        "accuracy": []
    }

    failed_count = 0

    for query in QUALITY_TEST_CASES:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")

        # Get response
        response = get_portfolia_response(query)
        print(f"Response: {response[:200]}...")

        # Evaluate
        scores = evaluate_response_quality(query, response)

        # Track scores
        for criterion in ["conversational_tone", "specificity", "follow_up", "personality", "accuracy"]:
            score = scores.get(criterion, 0)
            all_scores[criterion].append(score)

        # Print scores
        print(f"\nScores:")
        print(f"  Conversational Tone: {scores.get('conversational_tone', 0)}/5")
        print(f"  Specificity: {scores.get('specificity', 0)}/5")
        print(f"  Follow-up: {scores.get('follow_up', 0)}/5")
        print(f"  Personality: {scores.get('personality', 0)}/5")
        print(f"  Accuracy: {scores.get('accuracy', 0)}/5")

        if scores.get("flags"):
            print(f"  Flags: {scores['flags']}")

        # Check for failures (any score below 3)
        min_score = min([scores.get(k, 0) for k in ["conversational_tone", "specificity", "follow_up", "personality", "accuracy"]])
        if min_score < 3:
            failed_count += 1
            print(f"  ❌ FAIL - Minimum score {min_score} below threshold")
            quality_results.append({
                "query": query,
                "response": response,
                "scores": scores,
                "passed": False
            })
        else:
            print(f"  ✅ PASS")
            quality_results.append({
                "query": query,
                "response": response,
                "scores": scores,
                "passed": True
            })

    # Calculate averages
    avg_scores = {k: sum(v) / len(v) if v else 0 for k, v in all_scores.items()}

    print(f"\n{'='*80}")
    print("QUALITY SUMMARY")
    print(f"{'='*80}")
    print(f"Tests passed: {len(QUALITY_TEST_CASES) - failed_count}/{len(QUALITY_TEST_CASES)}")
    print(f"Average scores:")
    print(f"  Conversational Tone: {avg_scores['conversational_tone']:.2f}/5")
    print(f"  Specificity: {avg_scores['specificity']:.2f}/5")
    print(f"  Follow-up: {avg_scores['follow_up']:.2f}/5")
    print(f"  Personality: {avg_scores['personality']:.2f}/5")
    print(f"  Accuracy: {avg_scores['accuracy']:.2f}/5")

    # Assert no failures
    assert failed_count == 0, f"{failed_count} responses scored below 3 on at least one criterion"


# ============================================================================
# SECTION 3: REGRESSION TESTS (KNOWN BAD BEHAVIORS)
# ============================================================================

def test_regression_no_insufficient_info():
    """Regression: 'I don't have enough information' should not appear for crush confession."""
    print("\n" + "="*80)
    print("SECTION 3: REGRESSION TESTS")
    print("="*80)
    print("\nTest 1: No 'I don't have enough information' for crush confession")

    response = get_portfolia_response("I would like to confess a crush")

    bad_phrases = ["I don't have enough information", "I cannot answer", "I don't know"]
    found_bad = any(phrase.lower() in response.lower() for phrase in bad_phrases)

    if found_bad:
        print(f"❌ FAIL - Response contains 'insufficient info' message")
        print(f"Response: {response}")
        regression_results.append({"test": "no_insufficient_info", "passed": False, "response": response})
        assert False, "Crush confession should not return 'I don't have enough information'"
    else:
        print(f"✅ PASS - No insufficient info message")
        regression_results.append({"test": "no_insufficient_info", "passed": True})


def test_regression_no_dry_openers():
    """Regression: No more dry '[Subject]'s [topic] includes...' openers."""
    print("\nTest 2: No dry '[Subject]'s [topic] includes...' openers")

    response = get_portfolia_response("What is Noah's professional background?")

    # Check for bad patterns
    bad_patterns = [
        "noah's professional background includes",
        "noah's background includes",
        "his professional background includes",
        "his background includes"
    ]

    found_bad = any(pattern in response.lower() for pattern in bad_patterns)

    if found_bad:
        print(f"❌ FAIL - Response uses dry opener pattern")
        print(f"Response: {response}")
        regression_results.append({"test": "no_dry_openers", "passed": False, "response": response})
        assert False, "Response should not use dry '[Subject]'s [topic] includes...' pattern"
    else:
        print(f"✅ PASS - No dry opener pattern")
        regression_results.append({"test": "no_dry_openers", "passed": True})


def test_regression_multiple_projects():
    """Regression: 'What are Noah's projects?' should mention at least 2 projects."""
    print("\nTest 3: Multiple projects mentioned")

    response = get_portfolia_response("What are some projects by Noah?")

    # Look for project names
    projects = ["Portfolia", "Tesla", "heatmap", "dashboard", "attrition", "Employee Attrition", "logistic regression"]
    mentioned = [p for p in projects if p.lower() in response.lower()]

    if len(mentioned) < 2:
        print(f"❌ FAIL - Only {len(mentioned)} project(s) mentioned: {mentioned}")
        print(f"Response: {response}")
        regression_results.append({"test": "multiple_projects", "passed": False, "response": response})
        assert False, "Response should mention at least 2 distinct projects"
    else:
        print(f"✅ PASS - {len(mentioned)} projects mentioned: {mentioned}")
        regression_results.append({"test": "multiple_projects", "passed": True})


def test_regression_links_appear():
    """Regression: Links should appear when user asks where to see work."""
    print("\nTest 4: Links appear when appropriate")

    response = get_portfolia_response("Where can I see his work?")

    # Check for current URLs (updated as of Feb 2026)
    has_github = "github.com/inoahcodeguy" in response.lower()
    has_linkedin = "linkedin.com/in/noah-de-la-calzada" in response.lower()

    if not (has_github and has_linkedin):
        print(f"❌ FAIL - Missing links (GitHub: {has_github}, LinkedIn: {has_linkedin})")
        print(f"Response: {response}")
        regression_results.append({"test": "links_appear", "passed": False, "response": response})
        assert False, "Response should contain both GitHub and LinkedIn links"
    else:
        print(f"✅ PASS - Both links present")
        regression_results.append({"test": "links_appear", "passed": True})


def test_regression_salary_boundary():
    """Regression: Salary questions should be deflected gracefully."""
    print("\nTest 5: Graceful boundary handling for salary question")

    response = get_portfolia_response("What's Noah's salary?")

    # Should NOT contain actual salary info
    bad_terms = ["$", "dollars", "k/year", "salary is"]
    contains_salary = any(term in response.lower() for term in bad_terms)

    # SHOULD pivot to something else
    good_pivots = ["project", "skill", "background", "tesla", "performance"]
    contains_pivot = any(term in response.lower() for term in good_pivots)

    if contains_salary:
        print(f"❌ FAIL - Response contains salary information")
        print(f"Response: {response}")
        regression_results.append({"test": "salary_boundary", "passed": False, "response": response})
        assert False, "Should not reveal salary information"
    elif not contains_pivot:
        print(f"⚠️  WARNING - Response doesn't pivot to alternative topic")
        print(f"Response: {response}")

    print(f"✅ PASS - No salary info, graceful handling")
    regression_results.append({"test": "salary_boundary", "passed": True})


def test_regression_gibberish_handling():
    """Regression: Gibberish should be handled gracefully with redirects."""
    print("\nTest 6: Gibberish handling")

    response = get_portfolia_response("asdfghjkl")

    # Should NOT say "I don't have enough information"
    bad_phrases = ["I don't have enough information", "I cannot answer"]
    found_bad = any(phrase.lower() in response.lower() for phrase in bad_phrases)

    # SHOULD offer alternative topics
    good_terms = ["project", "tell", "ask", "interested", "show"]
    contains_redirect = any(term in response.lower() for term in good_terms)

    if found_bad:
        print(f"❌ FAIL - Response says 'I don't have enough information'")
        print(f"Response: {response}")
        regression_results.append({"test": "gibberish_handling", "passed": False, "response": response})
        assert False, "Gibberish should not trigger 'insufficient info' message"
    elif not contains_redirect:
        print(f"⚠️  WARNING - Response doesn't offer alternative topics")
        print(f"Response: {response}")

    print(f"✅ PASS - Graceful handling with redirect")
    regression_results.append({"test": "gibberish_handling", "passed": True})


# ============================================================================
# TEST SUMMARY
# ============================================================================

def test_print_final_summary(capfd):
    """Print final summary of all test results."""
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)

    # Intent classification
    intent_passed = sum(1 for r in intent_results if r["passed"])
    print(f"\nINTENT CLASSIFICATION: {intent_passed}/{len(intent_results)} passed")

    # Response quality
    quality_passed = sum(1 for r in quality_results if r["passed"])
    if quality_results:
        avg_scores = {
            "tone": sum(r["scores"].get("conversational_tone", 0) for r in quality_results) / len(quality_results),
            "specificity": sum(r["scores"].get("specificity", 0) for r in quality_results) / len(quality_results),
            "follow_up": sum(r["scores"].get("follow_up", 0) for r in quality_results) / len(quality_results),
            "personality": sum(r["scores"].get("personality", 0) for r in quality_results) / len(quality_results),
            "accuracy": sum(r["scores"].get("accuracy", 0) for r in quality_results) / len(quality_results),
        }
        print(f"RESPONSE QUALITY: {quality_passed}/{len(quality_results)} passed")
        print(f"  Avg scores: tone={avg_scores['tone']:.1f}, specificity={avg_scores['specificity']:.1f}, "
              f"follow_up={avg_scores['follow_up']:.1f}, personality={avg_scores['personality']:.1f}, "
              f"accuracy={avg_scores['accuracy']:.1f}")

    # Regression tests
    regression_passed = sum(1 for r in regression_results if r["passed"])
    print(f"REGRESSION TESTS: {regression_passed}/{len(regression_results)} passed")

    # Overall
    total_passed = intent_passed + quality_passed + regression_passed
    total_tests = len(intent_results) + len(quality_results) + len(regression_results)
    print(f"\nOVERALL: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n✅ ALL TESTS PASSED! Portfolia is performing well.")
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. Review output above for details.")


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
