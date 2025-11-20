#!/usr/bin/env python3
"""
Test script to validate the deployed Vercel endpoints.

This script tests:
1. Health check endpoint
2. Chat endpoint with the three conversation turns

Usage:
    python3 scripts/test_vercel_deployment.py [base_url]

Example:
    python3 scripts/test_vercel_deployment.py https://your-app.vercel.app
"""
import sys
import json
import requests
from typing import Optional

def test_health_check(base_url: str) -> bool:
    """Test the health check endpoint."""
    print("\n" + "=" * 80)
    print("TEST: Health Check Endpoint")
    print("=" * 80)

    try:
        response = requests.get(f"{base_url}/api/health", timeout=10)
        response.raise_for_status()
        data = response.json()

        assert data.get("status") == "healthy", f"Expected 'healthy', got {data.get('status')}"
        print(f"✅ Health check passed: {data.get('message')}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_turn_1_initial_greeting(base_url: str) -> bool:
    """Test Turn 1: Initial greeting."""
    print("\n" + "=" * 80)
    print("TEST: Turn 1 - Initial Greeting")
    print("=" * 80)

    try:
        payload = {
            "query": "",
            "role": "",
            "session_id": "vercel-test-001",
            "chat_history": [],
            "session_memory": {}
        }

        response = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        assert data.get("success") == True, "Request should succeed"
        assert "answer" in data, "Response should have 'answer' field"
        assert "1️⃣" in data["answer"] or "Hiring Manager" in data["answer"], \
            "Answer should contain role selection prompt"

        print(f"✅ Turn 1 passed")
        print(f"   Answer length: {len(data.get('answer', ''))} chars")
        print(f"   Answer preview: {data.get('answer', '')[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Turn 1 failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text[:200]}")
        return False


def test_turn_2_role_selection(base_url: str) -> bool:
    """Test Turn 2: Role selection."""
    print("\n" + "=" * 80)
    print("TEST: Turn 2 - Role Selection (Technical Hiring Manager)")
    print("=" * 80)

    try:
        payload = {
            "query": "2",
            "role": "",
            "session_id": "vercel-test-002",
            "chat_history": [],
            "session_memory": {
                "persona_hints": {
                    "initial_greeting_shown": True
                }
            }
        }

        response = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        assert data.get("success") == True, "Request should succeed"
        assert "answer" in data, "Response should have 'answer' field"
        role = data.get("role", "")
        assert "technical" in role.lower() or "hiring" in role.lower(), \
            f"Role should be technical hiring manager, got: {role}"

        print(f"✅ Turn 2 passed")
        print(f"   Role: {role}")
        print(f"   Answer length: {len(data.get('answer', ''))} chars")
        return True
    except Exception as e:
        print(f"❌ Turn 2 failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text[:200]}")
        return False


def test_turn_3_menu_option_1(base_url: str) -> bool:
    """Test Turn 3: Menu option 1."""
    print("\n" + "=" * 80)
    print("TEST: Turn 3 - Menu Option 1 (Full Tech Stack)")
    print("=" * 80)

    try:
        payload = {
            "query": "1",
            "role": "Hiring Manager (technical)",
            "session_id": "vercel-test-003",
            "chat_history": [],
            "session_memory": {
                "persona_hints": {
                    "initial_greeting_shown": True,
                    "role_mode": "hiring_manager_technical",
                    "role_welcome_shown": True
                }
            }
        }

        response = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60  # Longer timeout for LLM generation
        )
        response.raise_for_status()
        data = response.json()

        assert data.get("success") == True, "Request should succeed"
        assert "answer" in data, "Response should have 'answer' field"
        answer = data.get("answer", "")

        # Check for required layers
        required_layers = ["Frontend", "Backend", "Data", "Observability", "Deployment"]
        found_layers = [layer for layer in required_layers if layer in answer]

        assert len(found_layers) >= 3, f"Should find at least 3 layers, found: {found_layers}"
        assert data.get("retrieved_chunks", 0) > 0, "Should have retrieved chunks"

        print(f"✅ Turn 3 passed")
        print(f"   Found {len(found_layers)}/{len(required_layers)} required layers")
        print(f"   Retrieved chunks: {data.get('retrieved_chunks', 0)}")
        print(f"   Answer length: {len(answer)} chars ({len(answer.split())} words)")
        return True
    except Exception as e:
        print(f"❌ Turn 3 failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text[:200]}")
        return False


def main():
    """Run all deployment tests."""
    base_url = sys.argv[1] if len(sys.argv) > 1 else "https://your-app.vercel.app"

    if base_url == "https://your-app.vercel.app":
        print("⚠️  Using default URL. Provide your Vercel URL as argument:")
        print("   python3 scripts/test_vercel_deployment.py https://your-app.vercel.app")
        print("")

    print("=" * 80)
    print("VERCEL DEPLOYMENT VALIDATION TESTS")
    print("=" * 80)
    print(f"Base URL: {base_url}")

    results = []

    # Test health check
    results.append(("Health Check", test_health_check(base_url)))

    # Test Turn 1
    results.append(("Turn 1 - Initial Greeting", test_turn_1_initial_greeting(base_url)))

    # Test Turn 2
    results.append(("Turn 2 - Role Selection", test_turn_2_role_selection(base_url)))

    # Test Turn 3
    results.append(("Turn 3 - Menu Option 1", test_turn_3_menu_option_1(base_url)))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")

    print("")
    if passed == total:
        print(f"🎉 All {total} tests passed! Deployment is working correctly.")
        return 0
    else:
        print(f"⚠️  {passed}/{total} tests passed. Review failures above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
