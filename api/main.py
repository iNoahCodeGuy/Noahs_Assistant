"""FastAPI chat endpoint for Portfolia AI Assistant.

Run locally with:
    uvicorn api.main:app --reload --port 8000
"""

import logging
import os
import sys
import time
import traceback
import uuid
from collections import defaultdict, deque
from copy import deepcopy
from typing import Deque, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add project root to path so assistant package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from assistant.core.rag_engine import RagEngine
from assistant.flows.conversation_flow import run_conversation_flow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Error monitoring (Sentry) ---
# Active only when SENTRY_DSN is set (Railway env). Local dev, CI, and the
# hermetic test suite run without it — and without the import.
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    import sentry_sdk

    sentry_sdk.init(
        dsn=SENTRY_DSN,
        environment=os.getenv("SENTRY_ENVIRONMENT", "production"),
        # Errors only. LangSmith owns request tracing; Sentry transactions
        # on top of it would be a second, overlapping trace store.
        traces_sample_rate=0.0,
    )


def _report_exception(exc: Exception) -> None:
    """Forward to Sentry when configured; silent no-op otherwise."""
    if SENTRY_DSN:
        import sentry_sdk

        sentry_sdk.capture_exception(exc)


app = FastAPI(title="Portfolia API")

# --- CORS ---
# Strict allow-list. The production domains are public constants, so they
# are defaults here — FRONTEND_URL adds extra origins (normalized: CORS
# origin matching is exact, so a trailing slash in the env var would
# silently block the site). No credentials cross-origin (API is cookie-free).
from assistant.config.supabase_config import get_supabase_client, supabase_settings

origins = {
    "http://localhost:3000",
    "https://noahdelacalzada.com",
    "https://www.noahdelacalzada.com",
    "https://portfoliafrontend.vercel.app",
}
if supabase_settings.frontend_url:
    origins.add(supabase_settings.frontend_url.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(origins),
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Anything that escapes a route: report, log, return clean JSON
    (Starlette's default is a plain-text 500 and no alerting)."""
    logger.error(f"Unhandled error on {request.url.path}: {exc}")
    logger.error(traceback.format_exc())
    _report_exception(exc)
    return JSONResponse(
        status_code=500,
        content={"success": False, "detail": "Internal server error."},
    )


@app.get("/health")
def health():
    """Liveness plus a cheap DB probe.

    The keep-alive workflow (.github/workflows/keepalive.yml) hits this on a
    schedule: the query keeps the Supabase free tier from auto-pausing (the
    2026-07-05 outage), and a failed probe turns a silent retrieval
    degradation into a red workflow run instead of fallback answers.
    """
    try:
        result = (
            get_supabase_client()
            .table("kb_chunks")
            .select("id", count="exact")
            .limit(1)
            .execute()
        )
        kb_chunks = result.count
    except Exception as e:
        logger.error(f"/health probe failed: {e}")
        _report_exception(e)
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "kb_chunks": None},
        )
    return {"status": "ok", "kb_chunks": kb_chunks}


# --- In-memory session store ---
sessions: dict[str, dict] = {}

# --- Shared RAG engine (initialized once at startup) ---
rag_engine = RagEngine()


# --- Abuse guards ---
# Every /chat call triggers an embedding + two LLM calls, so unbounded
# anonymous traffic is unbounded spend. Sliding-window limiter per client
# IP; in-memory is correct for the single-instance Railway deployment
# (upgrade path: Upstash/Redis if this ever scales horizontally).
MAX_MESSAGE_CHARS = 4000
RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60

_request_log: Dict[str, Deque[float]] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limited(ip: str) -> bool:
    """Record this request; True if the client exceeded the window limit."""
    now = time.time()
    log = _request_log[ip]
    while log and now - log[0] > RATE_LIMIT_WINDOW_SECONDS:
        log.popleft()
    if len(log) >= RATE_LIMIT_REQUESTS:
        return True
    log.append(now)
    return False


# --- Request / Response models ---
class ChatRequest(BaseModel):
    message: Optional[str] = None
    query: Optional[str] = None  # Frontend may send "query" instead of "message"
    session_id: Optional[str] = None
    role: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    response: str = ""
    answer: str = ""  # Mirror of response — frontend reads this field
    session_id: str = ""
    success: bool = True
    # Structured form signal from the pipeline's state machines. The
    # frontend renders forms from this instead of pattern-matching the
    # answer text: "crush" | "contact" | None.
    form: Optional[str] = None


# --- Menu mappings ---
MENU_MAP = {
    "Learn more about Noah": "1",
    "See what Noah has built": "2",
    "How I relate to Enterprise AI": "3",
    "Just looking around": "Just looking around",  # free-text, pass through
    "Confess a crush": "4",
}

ROLE_MAP = {
    "Learn more about Noah": "Learn more about Noah",
    "See what Noah has built": "See what Noah has built",
    "How I relate to Enterprise AI": "How I relate to Enterprise AI",
    "Just looking around": "Just looking around",
    "Confess a crush": "Looking to confess crush",
}


# Use sync def so FastAPI runs it in a threadpool (run_conversation_flow is blocking I/O)
@app.post("/chat")
def chat(req: ChatRequest, request: Request):
    # Abuse guards run before any embedding/LLM call is possible
    if _rate_limited(_client_ip(request)):
        raise HTTPException(
            status_code=429,
            detail="Too many requests — give it a minute and try again.",
        )

    user_message = req.message or req.query or ""
    if len(user_message) > MAX_MESSAGE_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Message too long (max {MAX_MESSAGE_CHARS} characters).",
        )

    try:
        session_id = req.session_id or str(uuid.uuid4())

        # Retrieve or create session state
        if session_id not in sessions:
            sessions[session_id] = {
                "chat_history": [],
                "session_memory": {},
                "role": "Just looking around",
            }
        session = sessions[session_id]

        # If frontend sent chat_history, use it (stateless mode)
        # Otherwise fall back to server-side session store
        if req.chat_history is not None:
            session["chat_history"] = req.chat_history

        # Set role from button text or frontend-provided role
        original_message = user_message
        role = ROLE_MAP.get(original_message)
        if not role and req.role:
            role = req.role
        if not role:
            role = session.get("role", "Just looking around")
        session["role"] = role

        # Convert menu button text to pipeline format
        message = MENU_MAP.get(original_message, original_message)

        # Build pipeline state
        state = {
            "role": role,
            "query": message,
            "chat_history": deepcopy(session["chat_history"]),
            "session_id": session_id,
            "session_memory": deepcopy(session["session_memory"]),
        }

        logger.info(f"API: role={state['role']}, query={state['query'][:80]}, session_id={session_id}")
        result = run_conversation_flow(state, rag_engine, session_id=session_id)

        # Persist updated session state
        session["chat_history"] = result.get("chat_history", [])
        session["session_memory"] = result.get("session_memory", {})
        session["role"] = result.get("role", role)

        answer = result.get("answer", "")

        # Structured form signal from the pipeline's own state machines —
        # the frontend renders forms from this instead of sniffing the
        # answer text for marker phrases.
        form = None
        if result.get("awaiting_crush_choice"):
            form = "crush"
        elif result.get("hm_capture_step") == "awaiting_hm_details":
            form = "contact"

        return {
            "success": True,
            "response": answer,
            "answer": answer,
            "session_id": session_id,
            "form": form,
        }

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        logger.error(traceback.format_exc())
        # The user gets a graceful fallback, but the error must still page
        # someone — a masked 200 with no alerting is how outages go unnoticed.
        _report_exception(e)
        return {
            "success": False,
            "response": "Something went wrong. Try again in a moment.",
            "answer": "Something went wrong. Try again in a moment.",
            "session_id": req.session_id or "",
            "form": None,
        }
