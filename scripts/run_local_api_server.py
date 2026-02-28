"""
Local test server for API endpoints - simulates Vercel functions locally.
Run this instead of 'vercel dev' if you don't have Vercel CLI installed.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.chat import handler as ChatHandler
from api.email import handler as EmailHandler
from api.feedback import handler as FeedbackHandler
from api.confess import handler as ConfessHandler


class LocalAPIServer(BaseHTTPRequestHandler):
    """Local server that routes to API handlers."""
    
    def do_POST(self):
        """Route POST requests to appropriate handler."""
        if self.path == '/api/chat':
            ChatHandler.do_POST(self)
        elif self.path == '/api/email':
            EmailHandler.do_POST(self)
        elif self.path == '/api/feedback':
            FeedbackHandler.do_POST(self)
        elif self.path == '/api/confess':
            ConfessHandler.do_POST(self)
        else:
            self._send_error(404, f"Endpoint not found: {self.path}")
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        if self.path.startswith('/api/'):
            self._send_cors_headers()
            self.send_response(200)
            self.end_headers()
        else:
            self._send_error(404, f"Endpoint not found: {self.path}")
    
    def _send_error(self, status_code: int, message: str):
        """Send error response."""
        self._send_cors_headers()
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({
            'success': False,
            'error': message
        }).encode('utf-8'))
    
    def _send_cors_headers(self):
        """Add CORS headers."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')


def run_server(port=3000):
    """Start local API server."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, LocalAPIServer)
    
    print(f"\n🚀 Local API server running on http://localhost:{port}")
    print(f"\nEndpoints available:")
    print(f"  POST http://localhost:{port}/api/chat")
    print(f"  POST http://localhost:{port}/api/email")
    print(f"  POST http://localhost:{port}/api/feedback")
    print(f"  POST http://localhost:{port}/api/confess")
    print(f"\nPress Ctrl+C to stop\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
        httpd.server_close()


if __name__ == '__main__':
    run_server()
