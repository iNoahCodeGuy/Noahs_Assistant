"""
Vercel serverless function for live analytics display.
GET /api/analytics - Returns live data from Supabase with PII redaction.

Security: Uses service role key (server-side only), rate limited 6 req/min.
"""
from http.server import BaseHTTPRequestHandler
import json
import sys
import os
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from assistant.config.supabase_config import get_supabase_client

# Rate limiting (simple in-memory for demo; use Redis in production)
_rate_limit_store: Dict[str, List[float]] = {}
RATE_LIMIT_REQUESTS = 6
RATE_LIMIT_WINDOW = 60  # seconds


def redact_pii(text: Optional[str]) -> str:
    """Redact emails and phone numbers from text.

    Args:
        text: String that may contain PII

    Returns:
        Text with emails and phones replaced with [redacted]
    """
    if not text:
        return "—"

    # Redact emails
    email_pattern = r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b'
    text = re.sub(email_pattern, '[redacted]', text, flags=re.IGNORECASE)

    # Redact phone numbers
    phone_pattern = r'\+?\d[\d\s().-]{7,}\d'
    text = re.sub(phone_pattern, '[redacted]', text)

    return text


def check_rate_limit(ip: str) -> bool:
    """Check if IP has exceeded rate limit.

    Args:
        ip: Client IP address

    Returns:
        True if within limit, False if exceeded
    """
    now = datetime.now().timestamp()

    # Clean old entries
    if ip in _rate_limit_store:
        _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if now - t < RATE_LIMIT_WINDOW]
    else:
        _rate_limit_store[ip] = []

    # Check limit
    if len(_rate_limit_store[ip]) >= RATE_LIMIT_REQUESTS:
        return False

    # Add current request
    _rate_limit_store[ip].append(now)
    return True


def fetch_table_data(client: Any, table_name: str, columns: List[str], limit: int = 50, timeout: int = 2500) -> Dict[str, Any]:
    """Fetch data from a Supabase table with timeout and error handling.

    Args:
        client: Supabase client
        table_name: Name of the table
        columns: List of column names to select
        limit: Maximum number of rows to return
        timeout: Timeout in milliseconds

    Returns:
        Dict with 'data' list and optional 'error' string
    """
    try:
        selection = ",".join(columns)
        result = client.table(table_name).select(selection).order(
            "created_at" if "created_at" in columns else "id", desc=True
        ).limit(limit).execute()

        return {"data": result.data or []}
    except Exception as e:
        logger.error(f"Error fetching {table_name}: {e}")
        return {"data": [], "error": str(e)}


def fetch_inventory(client: Any) -> Dict[str, int]:
    """Fetch row counts for all tables.

    Args:
        client: Supabase client

    Returns:
        Dict with table names as keys and counts as values
    """
    inventory = {}
    tables = ["messages", "retrieval_logs", "feedback", "confessions", "kb_chunks", "sms_logs"]

    for table in tables:
        try:
            result = client.table(table).select("*", count="exact", head=True).execute()
            inventory[table] = result.count or 0
        except Exception as e:
            logger.warning(f"Could not count {table}: {e}")
            inventory[table] = 0

    return inventory


def fetch_kb_coverage(client: Any) -> Optional[List[Dict[str, Any]]]:
    """Fetch KB coverage summary via RPC.

    Args:
        client: Supabase client

    Returns:
        List of {source, count} dicts or None if RPC doesn't exist
    """
    try:
        result = client.rpc("kb_coverage_summary").execute()
        return result.data or []
    except Exception as e:
        logger.warning(f"KB coverage RPC not available: {e}")
        return None


class handler(BaseHTTPRequestHandler):
    """Vercel serverless function handler for /api/analytics endpoint."""

    def do_GET(self):
        """Handle GET request for analytics data."""
        try:
            # Rate limiting
            client_ip = self.headers.get('X-Forwarded-For', '').split(',')[0].strip() or \
                        self.headers.get('X-Real-IP', '') or \
                        self.client_address[0]

            if not check_rate_limit(client_ip):
                self._send_json(429, {
                    "error": "Rate limit exceeded. Maximum 6 requests per minute."
                })
                return

            # Initialize Supabase client (server-side with service key)
            try:
                client = get_supabase_client()
            except Exception as e:
                logger.error(f"Supabase initialization failed: {e}")
                self._send_json(500, {
                    "error": "Analytics temporarily unavailable; would you like a cached summary instead?"
                })
                return

            # Fetch inventory
            inventory = fetch_inventory(client)

            # Fetch dataset details
            messages_data = fetch_table_data(
                client, "messages",
                ["id", "role_mode", "user_query", "latency_ms", "token_count", "created_at", "success"]
            )

            retrieval_logs_data = fetch_table_data(
                client, "retrieval_logs",
                ["message_id", "chunk_id", "similarity_score", "grounded", "created_at"]
            )

            feedback_data = fetch_table_data(
                client, "feedback",
                ["message_id", "rating", "comment", "contact_requested", "created_at"]
            )

            # Redact PII in feedback comments
            if feedback_data.get("data"):
                for row in feedback_data["data"]:
                    if "comment" in row:
                        row["comment"] = redact_pii(row["comment"])

            confessions_data = fetch_table_data(
                client, "confessions",
                ["id", "is_anonymous", "created_at"],
                limit=5  # Only last 5 for privacy
            )

            kb_chunks_data = fetch_table_data(
                client, "kb_chunks",
                ["id", "section", "created_at"],
                limit=20  # Sample only
            )

            kb_coverage = fetch_kb_coverage(client)

            # Build response
            response = {
                "inventory": inventory,
                "messages": messages_data,
                "retrieval_logs": retrieval_logs_data,
                "feedback": feedback_data,
                "confessions": confessions_data,
                "kb_chunks": kb_chunks_data,
                "kb_coverage": kb_coverage,
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }

            # Log analytics view
            try:
                client.table("tool_invocations").insert({
                    "tool": "analytics_view",
                    "args_hash": "",
                    "duration_ms": 0,  # Will be calculated client-side
                    "status": "success"
                }).execute()
            except Exception as e:
                logger.warning(f"Could not log analytics view: {e}")

            self._send_json(200, response)

        except Exception as e:
            logger.error(f"Error processing analytics request: {str(e)}")
            self._send_json(500, {
                "error": f"Internal server error: {str(e)}"
            })

    def do_OPTIONS(self):
        """Handle CORS preflight request."""
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def _send_json(self, status_code: int, data: Dict[str, Any]):
        """Send JSON response with CORS headers."""
        self.send_response(status_code)
        self._send_cors_headers()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def _send_cors_headers(self):
        """Add CORS headers for cross-origin requests."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
