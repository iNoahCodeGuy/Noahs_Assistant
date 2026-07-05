"""FastAPI chat endpoint for Portfolia AI Assistant.

Run locally with:
    uvicorn api.main:app --reload --port 8000
"""

import logging
import os
import sys
import traceback
import uuid
from copy import deepcopy
from typing import Optional, List, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path so assistant package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from assistant.core.rag_engine import RagEngine
from assistant.flows.conversation_flow import run_conversation_flow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Portfolia API")

# --- CORS ---
# Strict allow-list when FRONTEND_URL is configured; permissive fallback
# otherwise so a missing env var degrades to "open" rather than breaking
# the live site. No credentials cross-origin (the API is cookie-free).
from assistant.config.supabase_config import supabase_settings

origins = ["http://localhost:3000"]
if supabase_settings.frontend_url:
    origins.append(supabase_settings.frontend_url)
else:
    logger_cors = logging.getLogger(__name__)
    logger_cors.warning("FRONTEND_URL not set — CORS falls back to allow all origins")
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# --- In-memory session store ---
sessions: dict[str, dict] = {}

# --- Shared RAG engine (initialized once at startup) ---
rag_engine = RagEngine()


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
def chat(req: ChatRequest):
    try:
        # Accept both "message" and "query" field names
        user_message = req.message or req.query or ""
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
        return {
            "success": False,
            "response": "Something went wrong. Try again in a moment.",
            "answer": "Something went wrong. Try again in a moment.",
            "session_id": req.session_id or "",
            "form": None,
        }
