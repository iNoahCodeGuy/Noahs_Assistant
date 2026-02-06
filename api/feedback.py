"""
Vercel serverless function for feedback API endpoint.
Logs user ratings, comments, and contact requests to Supabase.
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

from assistant.analytics.supabase_analytics import supabase_analytics
from assistant.services.twilio_service import get_twilio_service


class handler(BaseHTTPRequestHandler):
    """Vercel serverless function handler for /api/feedback endpoint."""

    def do_POST(self):
        """Handle POST request to submit feedback."""
        try:
            # Parse request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)

            # Extract feedback data
            message_id = data.get('message_id')
            rating = data.get('rating')  # 1-5 stars
            comment = data.get('comment', '')
            contact_requested = data.get('contact_requested', False)
            user_email = data.get('user_email')
            user_name = data.get('user_name')
            user_phone = data.get('user_phone')

            # Validate rating if provided
            if rating is not None:
                try:
                    rating = int(rating)
                    if rating < 1 or rating > 5:
                        self._send_error(400, "Rating must be between 1 and 5")
                        return
                except (ValueError, TypeError):
                    self._send_error(400, "Invalid rating value")
                    return

            # Log feedback to Supabase
            feedback_id = supabase_analytics.log_feedback(
                message_id=message_id,
                rating=rating or 0,  # Default to 0 if no rating provided
                comment=comment,
                contact_requested=contact_requested,
                user_email=user_email,
                user_name=user_name,
                user_phone=user_phone
            )

            # Send SMS notification if contact requested
            if contact_requested:
                try:
                    twilio_service = get_twilio_service()
                    if twilio_service and twilio_service.enabled:
                        twilio_service.send_contact_alert(
                            from_name=user_name or 'Anonymous',
                            from_email=user_email or 'no-email',
                            message_preview=f"Contact request: {comment[:100] if comment else 'No message'}",
                            is_urgent=True
                        )
                    else:
                        logger.warning("Twilio service not enabled, skipping SMS notification")
                except Exception as e:
                    logger.error(f"Failed to send SMS notification: {e}")
                    # Don't fail the request if SMS fails

            # Build response
            response = {
                'success': True,
                'feedback_id': feedback_id,
                'message': 'Feedback received. Thank you!' if not contact_requested else 'Feedback received. Noah will reach out soon!'
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
