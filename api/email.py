"""
Vercel serverless function for email API endpoint.
Sends resume or LinkedIn link via Resend service.
"""
from http.server import BaseHTTPRequestHandler
import json
import sys
import os
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from assistant.services.resend_service import get_resend_service
from assistant.services.storage_service import get_storage_service
from assistant.services.twilio_service import get_twilio_service


class handler(BaseHTTPRequestHandler):
    """Vercel serverless function handler for /api/email endpoint."""

    def do_POST(self):
        """Handle POST request to send email."""
        try:
            # Parse request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)

            # Validate required fields
            required_fields = ['type', 'to_email', 'to_name']
            for field in required_fields:
                if field not in data:
                    self._send_error(400, f"Missing required field: {field}")
                    return

            email_type = data.get('type')  # 'resume' or 'linkedin'
            to_email = data.get('to_email')
            to_name = data.get('to_name')
            message = data.get('message', '')

            # Validate email type
            if email_type not in ['resume', 'linkedin']:
                self._send_error(400, "Invalid email type. Must be 'resume' or 'linkedin'")
                return

            # Get services
            resend_service = get_resend_service()
            if not resend_service or not resend_service.enabled:
                self._send_error(503, "Email service unavailable")
                return

            if email_type == 'resume':
                # Get signed URL for resume
                try:
                    storage_service = get_storage_service()
                    resume_url = storage_service.get_signed_url('resumes/noah_resume.pdf')
                except Exception as e:
                    logger.error(f"Failed to get resume URL: {e}")
                    self._send_error(500, "Failed to retrieve resume")
                    return

                # Send resume email
                result = resend_service.send_resume_email(
                    to_email=to_email,
                    to_name=to_name,
                    resume_url=resume_url,
                    message=message
                )

                # Send SMS notification to Noah (best effort)
                try:
                    twilio_service = get_twilio_service()
                    if twilio_service and twilio_service.enabled:
                        twilio_service.send_contact_alert(
                            from_name=to_name,
                            from_email=to_email,
                            message_preview=f"Resume sent to {to_email}"
                        )
                except Exception as e:
                    logger.error(f"Failed to send SMS notification: {e}")
                    # Don't fail the request if SMS fails

                response = {
                    'success': True,
                    'type': 'resume',
                    'status': result.get('status', 'sent'),
                    'resume_url': resume_url
                }

            elif email_type == 'linkedin':
                # Send LinkedIn link (simpler email)
                linkedin_url = os.getenv('LINKEDIN_URL', 'https://linkedin.com/in/noahdelacalzada')

                result = resend_service.send_contact_notification(
                    from_name='Portfolia - Noah AI Assistant',
                    from_email='assistant@noahdelacalzada.com',
                    message=f"LinkedIn Profile: {linkedin_url}\n\nSent to: {to_name} ({to_email})",
                    user_role='system',
                    phone=None
                )

                response = {
                    'success': True,
                    'type': 'linkedin',
                    'status': result.get('status', 'sent'),
                    'linkedin_url': linkedin_url
                }

            # Send success response
            self._send_json(200, response)

        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON in request body")
        except Exception as e:
            self._send_error(500, f"Internal server error: {str(e)}")

    def do_OPTIONS(self):
        """Handle CORS preflight request."""
        self._send_cors_headers()
        self.send_response(200)
        self.end_headers()

    def _send_json(self, status_code: int, data: Dict[str, Any]):
        """Send JSON response with CORS headers."""
        self.send_response(status_code)
        self._send_cors_headers()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def _send_error(self, status_code: int, message: str):
        """Send error response."""
        self._send_json(status_code, {
            'success': False,
            'error': message
        })

    def _send_cors_headers(self):
        """Add CORS headers for cross-origin requests."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
