"""
Test script for API endpoints - run locally before deploying.
"""
import requests
import json
from typing import Dict, Any

# Local Vercel dev server
BASE_URL = "http://localhost:3000/api"


def test_chat_endpoint():
    """Test /api/chat endpoint."""
    print("\n" + "="*80)
    print("Testing /api/chat endpoint")
    print("="*80)
    
    payload = {
        "query": "What is your Python experience?",
        "role": "Hiring Manager (technical)",
        "session_id": "test-123",
        "chat_history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help you today?"}
        ],
        "user_email": "test@example.com",
        "user_name": "Test User"
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()["success"] == True
    print("✅ Chat endpoint test passed!")


def test_email_endpoint():
    """Test /api/email endpoint."""
    print("\n" + "="*80)
    print("Testing /api/email endpoint")
    print("="*80)
    
    payload = {
        "type": "resume",
        "to_email": "recruiter@example.com",
        "to_name": "Test Recruiter",
        "message": "Please find my resume attached"
    }
    
    response = requests.post(f"{BASE_URL}/email", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()["success"] == True
    print("✅ Email endpoint test passed!")


def test_feedback_endpoint():
    """Test /api/feedback endpoint."""
    print("\n" + "="*80)
    print("Testing /api/feedback endpoint")
    print("="*80)
    
    payload = {
        "message_id": "msg_test_123",
        "rating": 5,
        "comment": "Excellent AI assistant!",
        "contact_requested": True,
        "user_email": "feedback@example.com",
        "user_name": "Happy User",
        "user_phone": "+15551234567"
    }
    
    response = requests.post(f"{BASE_URL}/feedback", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()["success"] == True
    print("✅ Feedback endpoint test passed!")


def test_confess_endpoint():
    """Test /api/confess endpoint."""
    print("\n" + "="*80)
    print("Testing /api/confess endpoint")
    print("="*80)
    
    # Test anonymous confession
    payload = {
        "message": "I think you're an amazing developer and wanted to let you know!",
        "is_anonymous": True
    }
    
    response = requests.post(f"{BASE_URL}/confess", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()["success"] == True
    print("✅ Confess endpoint (anonymous) test passed!")
    
    # Test named confession
    payload = {
        "message": "Your AI portfolio is inspiring! Would love to connect.",
        "is_anonymous": False,
        "name": "Sarah Johnson",
        "email": "sarah@example.com",
        "phone": "+15559876543"
    }
    
    response = requests.post(f"{BASE_URL}/confess", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()["success"] == True
    print("✅ Confess endpoint (named) test passed!")


def test_error_handling():
    """Test error handling."""
    print("\n" + "="*80)
    print("Testing error handling")
    print("="*80)
    
    # Missing required field
    payload = {"role": "Software Developer"}  # Missing 'query'
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 400
    assert response.json()["success"] == False
    print("✅ Error handling test passed!")


if __name__ == "__main__":
    print("\n🧪 API Endpoint Tests")
    print("Make sure Vercel dev server is running: vercel dev\n")
    
    try:
        test_chat_endpoint()
        test_email_endpoint()
        test_feedback_endpoint()
        test_confess_endpoint()
        test_error_handling()
        
        print("\n" + "="*80)
        print("🎉 All API tests passed!")
        print("="*80 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to local server.")
        print("Make sure Vercel dev server is running: vercel dev\n")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
