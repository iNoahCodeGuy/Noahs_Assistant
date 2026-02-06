"""
Vercel serverless function for confession API endpoint.
Handles anonymous or named confession submissions with SMS alerts.
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

from assistant.analytics.supabase_analytics import get_supabase_client
from assistant.services.twilio_service import get_twilio_service


class handler(BaseHTTPRequestHandler):
    """Vercel serverless function handler for /api/confess endpoint."""

    def do_POST(self):
        """Handle POST request to submit confession."""
        try:
            # Parse request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)

            # Validate required fields
            if 'message' not in data:
                self._send_error(400, "Missing required field: message")
                return

            # Extract confession data
            message = data.get('message')
            is_anonymous = data.get('is_anonymous', True)
            name = data.get('name') if not is_anonymous else None
            email = data.get('email') if not is_anonymous else None
            phone = data.get('phone') if not is_anonymous else None

            # Validate message length
            if len(message.strip()) < 10:
                self._send_error(400, "Message must be at least 10 characters")
                return

            if len(message) > 1000:
                self._send_error(400, "Message must be less than 1000 characters")
                return

            # Store confession in Supabase
            supabase = get_supabase_client()
            confession_data = {
                'message': message.strip(),
                'is_anonymous': is_anonymous,
                'name': name,
                'email': email,
                'phone': phone
            }

            result = supabase.table('confessions').insert(confession_data).execute()
            confession_id = result.data[0]['id'] if result.data else None

            # Send SMS notification to Noah (best effort)
            try:
                twilio_service = get_twilio_service()
                if twilio_service and twilio_service.enabled:
                    if is_anonymous:
                        sms_preview = f"💌 Anonymous confession: {message[:100]}..."
                    else:
                        sms_preview = f"💌 Confession from {name or 'someone'}: {message[:80]}..."

                    twilio_service.send_contact_alert(
                        from_name=name or 'Anonymous Admirer',
                        from_email=email or 'anonymous@confession.com',
                        message_preview=sms_preview,
                        is_urgent=False
                    )
                else:
                    logger.warning("Twilio service not enabled, skipping SMS notification")
            except Exception as e:
                logger.error(f"Failed to send SMS notification: {e}")
                # Don't fail the request if SMS fails

            # Build response
            response = {
                'success': True,
                'confession_id': confession_id,
                'message': 'Your confession has been delivered to Noah! 💌' if is_anonymous else f'Thank you {name}! Noah will reach out soon. 💌'
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
