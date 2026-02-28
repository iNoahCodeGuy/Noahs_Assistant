"""FastAPI chat endpoint for Portfolia AI Assistant.

Run locally with:
    uvicorn api.main:app --reload --port 8000
"""

import logging
import os
import sys
import uuid
from copy import deepcopy

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path so assistant package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from assistant.core.rag_engine import RagEngine
from assistant.flows.conversation_flow import run_conversation_flow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Portfolia API")

# --- CORS ---
origins = [
    "http://localhost:3000",
]
vercel_url = os.getenv("FRONTEND_URL")
if vercel_url:
    origins.append(vercel_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# --- In-memory session store ---
sessions: dict[str, dict] = {}

# --- Shared RAG engine (initialized once at startup) ---
rag_engine = RagEngine()


# --- Request / Response models ---
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


# Use sync def so FastAPI runs it in a threadpool (run_conversation_flow is blocking I/O)
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    # Retrieve or create session state
    session = sessions.get(session_id, {
        "chat_history": [],
        "session_memory": {},
        "role": "Just looking around",
    })

    # Build pipeline state — deepcopy to prevent the pipeline from mutating stored state
    state = {
        "role": session["role"],
        "query": req.message,
        "chat_history": deepcopy(session["chat_history"]),
        "session_id": session_id,
        "session_memory": deepcopy(session["session_memory"]),
    }

    result = run_conversation_flow(state, rag_engine, session_id=session_id)

    # Persist updated session state
    sessions[session_id] = {
        "chat_history": result.get("chat_history", []),
        "session_memory": result.get("session_memory", {}),
        "role": result.get("role", session["role"]),
    }

    return ChatResponse(
        response=result.get("answer", ""),
        session_id=session_id,
    )
