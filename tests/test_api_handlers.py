# -*- coding: utf-8 -*-
"""
Unit tests for API handlers - tests logic without running server.
"""
import sys
import os
from io import BytesIO
from unittest.mock import Mock, patch, MagicMock

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.chat import handler as ChatHandler
from api.email import handler as EmailHandler
from api.feedback import handler as FeedbackHandler
from api.confess import handler as ConfessHandler


def create_mock_request(data):
    """Create mock HTTP request."""
    request = Mock()
    request.headers = {'Content-Length': str(len(str(data)))}
    request.rfile = BytesIO(str(data).encode('utf-8'))
    request.wfile = BytesIO()
    return request


def test_chat_handler():
    """Test chat API handler."""
    print("\n" + "="*80)
    print("Testing Chat Handler")
    print("="*80)
    
    # Mock dependencies
    with patch('api.chat.RagEngine') as MockRagEngine, \
         patch('api.chat.run_conversation_flow') as mock_flow:
        
        # Setup mocks
        mock_state = Mock()
        mock_state.answer = "I have 5+ years of Python experience..."
        mock_state.role = "Hiring Manager (technical)"
        mock_state.session_id = "test-123"
        mock_state.analytics = {"latency_ms": 1200}
        mock_state.pending_actions = [{"type": "offer_resume_prompt"}]
        mock_state.retrieved_chunks = ["chunk1", "chunk2", "chunk3"]
        
        mock_flow.return_value = mock_state
        
        # Create handler
        handler = ChatHandler()
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        handler.wfile = BytesIO()
        
        # Create request data
        import json
        request_data = {
            "query": "What is your Python experience?",
            "role": "Hiring Manager (technical)",
            "session_id": "test-123"
        }
        
        handler.headers = {'Content-Length': str(len(json.dumps(request_data)))}
        handler.rfile = BytesIO(json.dumps(request_data).encode('utf-8'))
        
        # Execute
        handler.do_POST()
        
        # Verify
        assert mock_flow.called, "Flow should be called"
        print("✅ Chat handler test passed!")


def test_email_handler():
    """Test email API handler."""
    print("\n" + "="*80)
    print("Testing Email Handler")
    print("="*80)
    
    with patch('api.email.get_resend_service') as mock_resend, \
         patch('api.email.get_storage_service') as mock_storage, \
         patch('api.email.get_twilio_service') as mock_twilio:
        
        # Setup mocks
        mock_resend.return_value.send_resume_email.return_value = {"status": "sent"}
        mock_storage.return_value.get_signed_url.return_value = "https://example.com/resume.pdf"
        mock_twilio.return_value.send_contact_alert.return_value = {"status": "sent"}
        
        # Create handler
        handler = EmailHandler()
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        handler.wfile = BytesIO()
        
        # Create request
        import json
        request_data = {
            "type": "resume",
            "to_email": "test@example.com",
            "to_name": "Test User"
        }
        
        handler.headers = {'Content-Length': str(len(json.dumps(request_data)))}
        handler.rfile = BytesIO(json.dumps(request_data).encode('utf-8'))
        
        # Execute
        handler.do_POST()
        
        # Verify
        assert mock_resend.called, "Resend service should be called"
        assert mock_storage.called, "Storage service should be called"
        print("✅ Email handler test passed!")


def test_feedback_handler():
    """Test feedback API handler."""
    print("\n" + "="*80)
    print("Testing Feedback Handler")
    print("="*80)
    
    with patch('api.feedback.supabase_analytics') as mock_analytics, \
         patch('api.feedback.get_twilio_service') as mock_twilio:
        
        # Setup mocks
        mock_analytics.log_feedback.return_value = "fb_123"
        mock_twilio.return_value.send_contact_alert.return_value = {"status": "sent"}
        
        # Create handler
        handler = FeedbackHandler()
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        handler.wfile = BytesIO()
        
        # Create request
        import json
        request_data = {
            "message_id": "msg_123",
            "rating": 5,
            "comment": "Great assistant!",
            "contact_requested": True,
            "user_email": "user@example.com",
            "user_name": "Happy User"
        }
        
        handler.headers = {'Content-Length': str(len(json.dumps(request_data)))}
        handler.rfile = BytesIO(json.dumps(request_data).encode('utf-8'))
        
        # Execute
        handler.do_POST()
        
        # Verify
        assert mock_analytics.log_feedback.called, "Feedback should be logged"
        assert mock_twilio.called, "SMS should be sent for contact request"
        print("✅ Feedback handler test passed!")


def test_confess_handler():
    """Test confession API handler."""
    print("\n" + "="*80)
    print("Testing Confession Handler")
    print("="*80)
    
    with patch('api.confess.get_supabase_client') as mock_supabase, \
         patch('api.confess.get_twilio_service') as mock_twilio:
        
        # Setup mocks
        mock_table = Mock()
        mock_table.insert.return_value.execute.return_value.data = [{"id": "conf_123"}]
        mock_supabase.return_value.table.return_value = mock_table
        mock_twilio.return_value.send_contact_alert.return_value = {"status": "sent"}
        
        # Create handler
        handler = ConfessHandler()
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        handler.wfile = BytesIO()
        
        # Create request
        import json
        request_data = {
            "message": "I think you're amazing and wanted to tell you!",
            "is_anonymous": False,
            "name": "Sarah",
            "email": "sarah@example.com"
        }
        
        handler.headers = {'Content-Length': str(len(json.dumps(request_data)))}
        handler.rfile = BytesIO(json.dumps(request_data).encode('utf-8'))
        
        # Execute
        handler.do_POST()
        
        # Verify
        assert mock_supabase.called, "Confession should be stored"
        assert mock_twilio.called, "SMS should be sent"
        print("✅ Confession handler test passed!")


if __name__ == "__main__":
    print("\n🧪 API Handler Unit Tests")
    print("="*80)
    
    try:
        test_chat_handler()
        test_email_handler()
        test_feedback_handler()
        test_confess_handler()
        
        print("\n" + "="*80)
        print("🎉 All API handler tests passed!")
        print("="*80 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
