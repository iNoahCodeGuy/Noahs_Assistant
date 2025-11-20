#!/usr/bin/env python3
"""
Local LangGraph server using FastAPI - NO DOCKER NEEDED!

This runs the LangGraph graph directly on your local machine, so code changes
are picked up immediately without rebuilding Docker containers.

Usage:
    python3 scripts/run_langgraph_local_fastapi.py

Then connect LangGraph Studio to: http://127.0.0.1:2024
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    from assistant.flows.conversation_flow import graph
    from assistant.state.conversation_state import ConversationState
    from assistant.core.rag_engine import RagEngine

    if graph is None:
        print("❌ Error: Graph is None. Check conversation_flow.py")
        sys.exit(1)

    app = FastAPI(title="LangGraph Local Server", version="1.0.0")

    # Enable CORS for LangGraph Studio
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize RAG engine once
    rag_engine = RagEngine()

    @app.get("/")
    def root():
        return {
            "status": "ok",
            "message": "LangGraph Local Server (No Docker)",
            "version": "1.0.0"
        }

    @app.get("/info")
    def info():
        return {
            "status": "ok",
            "version": "1.0.0",
            "graphs": ["conversation_flow"],
            "server": "local-fastapi"
        }

    @app.post("/threads/{thread_id}/runs")
    async def create_run(thread_id: str, request: Dict[str, Any]):
        """Create a new run (LangGraph Studio endpoint)"""
        try:
            # Extract input from request
            input_data = request.get("input", {})

            # Create initial state matching ConversationState
            state: ConversationState = {
                "query": input_data.get("query", ""),
                "role": input_data.get("role", ""),
                "session_id": thread_id or input_data.get("session_id", "local-test"),
                "chat_history": input_data.get("chat_history", []),
                "session_memory": input_data.get("session_memory", {}),
                "answer": "",
                "is_greeting": False,
                "pipeline_halt": False,
            }

            # Invoke graph
            result = graph.invoke(state)

            return {
                "run_id": f"run-{thread_id}",
                "status": "success",
                "output": result
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/threads/{thread_id}/runs/{run_id}/stream")
    async def stream_run(thread_id: str, run_id: str):
        """Stream run updates (simplified for local dev)"""
        return {"status": "completed"}

    @app.get("/graphs")
    def list_graphs():
        """List available graphs"""
        return {
            "graphs": [
                {
                    "name": "conversation_flow",
                    "description": "Main conversation flow with 18-node pipeline"
                }
            ]
        }

    print("\n" + "=" * 60)
    print("🚀 Starting LangGraph Local Server (NO DOCKER)")
    print("=" * 60)
    print("📊 Graph loaded successfully")
    print("🌐 Server will be available at: http://127.0.0.1:2024")
    print("🔗 Connect LangGraph Studio to: http://127.0.0.1:2024")
    print("=" * 60)
    print("💡 Code changes will be picked up immediately!")
    print("   (Just restart this script to reload)")
    print("=" * 60 + "\n")

    # Run with auto-reload for development
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=2024,
        reload=True,  # Auto-reload on code changes
        reload_dirs=[str(project_root / "assistant")],  # Watch assistant/ directory
        log_level="info"
    )

except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\n📦 Install with:")
    print("   pip install fastapi uvicorn[standard]")
    if "dotenv" in str(e):
        print("   pip install python-dotenv")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting server: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
