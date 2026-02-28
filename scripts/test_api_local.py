"""
Test API endpoints locally without Node.js/Vercel CLI.
This script directly tests the core handler logic.
"""
import sys
import os
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_chat_endpoint():
    """Test /api/chat endpoint."""
    print("\n" + "="*80)
    print("Testing /api/chat endpoint")
    print("="*80)
    
    from api.chat import handler
    
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
    
    request = create_mock_request("chat", payload)
    
    try:
        # Call the handler's do_POST method
        handler_instance = handler(request, ("localhost", 3000), None)
        handler_instance.do_POST()
        
        print(f"Status: {request.response_status}")
        if request.response_body:
            response = json.loads(request.response_body.decode('utf-8'))
            print(f"Response: {json.dumps(response, indent=2)}")
            
            if response.get("success"):
                print("✅ Chat endpoint test passed!")
                return True
            else:
                print(f"❌ Chat endpoint returned error: {response.get('error')}")
                return False
        else:
            print("❌ No response body received")
            return False
            
    except Exception as e:
        print(f"❌ Chat endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_email_endpoint():
    """Test /api/email endpoint."""
    print("\n" + "="*80)
    print("Testing /api/email endpoint")
    print("="*80)
    
    from api.email import handler
    
    payload = {
        "type": "resume",
        "to_email": "recruiter@example.com",
        "to_name": "Test Recruiter",
        "message": "Please find my resume attached"
    }
    
    request = create_mock_request("email", payload)
    
    try:
        handler_instance = handler(request, ("localhost", 3000), None)
        handler_instance.do_POST()
        
        print(f"Status: {request.response_status}")
        if request.response_body:
            response = json.loads(request.response_body.decode('utf-8'))
            print(f"Response: {json.dumps(response, indent=2)}")
            
            if response.get("success"):
                print("✅ Email endpoint test passed!")
                return True
            else:
                print(f"❌ Email endpoint returned error: {response.get('error')}")
                return False
        else:
            print("❌ No response body received")
            return False
            
    except Exception as e:
        print(f"❌ Email endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feedback_endpoint():
    """Test /api/feedback endpoint."""
    print("\n" + "="*80)
    print("Testing /api/feedback endpoint")
    print("="*80)
    
    from api.feedback import handler
    
    payload = {
        "message_id": "msg_test_123",
        "rating": 5,
        "comment": "Excellent AI assistant!",
        "contact_requested": True,
        "user_email": "feedback@example.com",
        "user_name": "Happy User",
        "user_phone": "+15551234567"
    }
    
    request = create_mock_request("feedback", payload)
    
    try:
        handler_instance = handler(request, ("localhost", 3000), None)
        handler_instance.do_POST()
        
        print(f"Status: {request.response_status}")
        if request.response_body:
            response = json.loads(request.response_body.decode('utf-8'))
            print(f"Response: {json.dumps(response, indent=2)}")
            
            if response.get("success"):
                print("✅ Feedback endpoint test passed!")
                return True
            else:
                print(f"❌ Feedback endpoint returned error: {response.get('error')}")
                return False
        else:
            print("❌ No response body received")
            return False
            
    except Exception as e:
        print(f"❌ Feedback endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_confess_endpoint():
    """Test /api/confess endpoint."""
    print("\n" + "="*80)
    print("Testing /api/confess endpoint (anonymous)")
    print("="*80)
    
    from api.confess import handler
    
    # Test anonymous confession
    payload = {
        "message": "I think you're an amazing developer and wanted to let you know!",
        "is_anonymous": True
    }
    
    request = create_mock_request("confess", payload)
    
    try:
        handler_instance = handler(request, ("localhost", 3000), None)
        handler_instance.do_POST()
        
        print(f"Status: {request.response_status}")
        if request.response_body:
            response = json.loads(request.response_body.decode('utf-8'))
            print(f"Response: {json.dumps(response, indent=2)}")
            
            if response.get("success"):
                print("✅ Confess endpoint (anonymous) test passed!")
                
                # Test named confession
                print("\n" + "-"*80)
                print("Testing /api/confess endpoint (named)")
                print("-"*80)
                
                payload2 = {
                    "message": "Your AI portfolio is inspiring! Would love to connect.",
                    "is_anonymous": False,
                    "name": "Sarah Johnson",
                    "email": "sarah@example.com",
                    "phone": "+15559876543"
                }
                
                request2 = create_mock_request("confess", payload2)
                handler_instance2 = handler(request2, ("localhost", 3000), None)
                handler_instance2.do_POST()
                
                print(f"Status: {request2.response_status}")
                if request2.response_body:
                    response2 = json.loads(request2.response_body.decode('utf-8'))
                    print(f"Response: {json.dumps(response2, indent=2)}")
                    
                    if response2.get("success"):
                        print("✅ Confess endpoint (named) test passed!")
                        return True
                    else:
                        print(f"❌ Named confession returned error: {response2.get('error')}")
                        return False
                
                return True
            else:
                print(f"❌ Confess endpoint returned error: {response.get('error')}")
                return False
        else:
            print("❌ No response body received")
            return False
            
    except Exception as e:
        print(f"❌ Confess endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling."""
    print("\n" + "="*80)
    print("Testing error handling (missing required field)")
    print("="*80)
    
    from api.chat import handler
    
    # Missing required field 'query'
    payload = {"role": "Software Developer"}
    
    request = create_mock_request("chat", payload)
    
    try:
        handler_instance = handler(request, ("localhost", 3000), None)
        handler_instance.do_POST()
        
        print(f"Status: {request.response_status}")
        if request.response_body:
            response = json.loads(request.response_body.decode('utf-8'))
            print(f"Response: {json.dumps(response, indent=2)}")
            
            if request.response_status == 400 and not response.get("success"):
                print("✅ Error handling test passed!")
                return True
            else:
                print(f"❌ Expected 400 error but got {request.response_status}")
                return False
        else:
            print("❌ No response body received")
            return False
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🧪 Local API Endpoint Tests (No Node.js required)")
    print("="*80)
    print("Testing Python handler logic directly...")
    print("="*80)
    
    results = []
    
    # Run all tests
    results.append(("Chat", test_chat_endpoint()))
    results.append(("Email", test_email_endpoint()))
    results.append(("Feedback", test_feedback_endpoint()))
    results.append(("Confess", test_confess_endpoint()))
    results.append(("Error Handling", test_error_handling()))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        print("="*80 + "\n")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Check output above for details.")
        print("="*80 + "\n")
        sys.exit(1)
