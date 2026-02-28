"""
Vercel Serverless Function - Health Check
"""
from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            "status": "healthy",
            "message": "Noah's AI Assistant API is running",
            "version": "1.0.0",
            "endpoints": {
                "/api/health": "This endpoint",
                "/api/chat": "Chat endpoint (coming soon)"
            }
        }
        
        self.wfile.write(json.dumps(response).encode())
        return
