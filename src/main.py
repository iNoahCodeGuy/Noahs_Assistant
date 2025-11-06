"""Main entry point for Portfolia (Noah's AI Assistant) Streamlit application.

This file orchestrates the complete user interaction flow:

1. **Role Selection** (one-time, persisted in session)
   - Users must select their role before chatting
   - Role determines retrieval strategy and response style
   - Stored in Streamlit session state (survives page reruns)

2. **Multi-Turn Chat Interface**
   - Built with Streamlit's chat components
   - Maintains conversation history in session
   - Each message triggers RAG pipeline

3. **Request Processing**
   - Route to appropriate agent based on role
   - Retrieve relevant knowledge (pgvector or FAISS)
   - Generate contextually-aware response
   - Format for target audience (technical vs business)

4. **Analytics Logging**
   - Every interaction logged to Supabase
   - Tracks: query, response, latency, tokens, role
   - Enables evaluation metrics and debugging

Streamlit Session State Variables:
    role (str): User's selected role (None until first selection)
        Options: "Hiring Manager (technical)", "Software Developer", etc.

    chat_history (List[Dict]): Conversation messages for display
        Format: [{"role": "user"|"assistant", "content": str}, ...]

    session_id (str): UUID for tracking conversation in analytics
        Generated once per session, persists across reruns

Why Role Selection First:
    Different roles need different knowledge sources and response styles:
    - Software Developer: Prioritize code index, technical depth
    - Hiring Manager (technical): Career KB + code snippets, dual-audience format
    - Hiring Manager (nontechnical): Career KB only, business-focused
    - Casual Visitor: Lightweight retrieval, conversational tone

    Without knowing the role, we can't optimize retrieval or formatting.

Usage:
    streamlit run src/main.py

    Then:
    1. Select your role from dropdown
    2. Click "Confirm Role"
    3. Start chatting!

Environment Variables Required:
    OPENAI_API_KEY: For embeddings and LLM generation
    SUPABASE_URL: Your Supabase project URL
    SUPABASE_SERVICE_KEY: Service role key (bypasses RLS)
"""
import os
import sys
from pathlib import Path

# Add parent directory to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import time
import uuid
from datetime import datetime
from src.core.rag_engine import RagEngine
from src.core.memory import Memory
from src.agents.role_router import RoleRouter
from src.agents.response_formatter import ResponseFormatter
from src.analytics.supabase_analytics import supabase_analytics, UserInteractionData
from src.config.supabase_config import supabase_settings
from src.state.conversation_state import ConversationState
from src.flows.conversation_flow import run_conversation_flow
from src.flows.node_logic.greetings import get_role_greeting

ROLE_OPTIONS = [
    "Hiring Manager (nontechnical)",  # Business-focused, career KB only
    "Hiring Manager (technical)",     # Dual-audience: code + plain English
    "Software Developer",              # Deep technical, code index priority
    "Just looking around",             # Casual visitor, lightweight retrieval
    "Looking to confess crush"         # Fun mode, guarded PII handling
]

USE_LANGGRAPH_FLOW = os.getenv("LANGGRAPH_FLOW_ENABLED", "true").lower() == "true"

def init_state():
    """Initialize Streamlit session state variables.

    Session state persists across Streamlit reruns (when user interacts).
    This ensures role selection and chat history survive page updates.

    Why check 'not in': Streamlit reruns entire script on every interaction.
    Without this guard, we'd reset state to None on every rerun.
    """
    if "role" not in st.session_state:
        st.session_state.role = None  # Will be set once user selects role
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list[dict(role="user"/"assistant", content=str)]
    if "session_id" not in st.session_state:
        # Generate unique ID for this conversation (for analytics tracking)
        st.session_state.session_id = str(uuid.uuid4())

def main():
    """Main application flow: validate config → chat loop (no role selection UI)."""
    init_state()

    # Validate Supabase configuration
    supabase_settings.validate_configuration()

    memory = Memory()
    rag_engine = RagEngine(supabase_settings)
    role_router = RoleRouter()
    response_formatter = ResponseFormatter()

    st.title("Portfolia - Noah's AI Assistant")

    # Show current role in sidebar if inferred
    if st.session_state.role:
        st.sidebar.markdown(f"**Detected Role:** {st.session_state.role}")
        if st.sidebar.button("Reset Conversation"):
            st.session_state.role = None
            st.session_state.chat_history = []
            st.rerun()

    # Display prior messages
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Chat input (multi-turn)
    user_input = st.chat_input("Ask a question...")
    if user_input:
        # Start timing for analytics
        start_time = time.time()

        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            if USE_LANGGRAPH_FLOW:
                state = ConversationState(
                    role=st.session_state.role,
                    query=user_input,
                    chat_history=st.session_state.chat_history.copy(),
                )
                state = run_conversation_flow(
                    state,
                    rag_engine,
                    session_id=st.session_state.session_id,
                )
                raw_response = {
                    "response": state.answer or "I need a moment to find that info.",
                    "type": state.fetch("query_type", "general"),
                    "context": state.retrieved_chunks,
                }
                latency_ms = int((time.time() - start_time) * 1000)
            else:
                raw_response = role_router.route(
                    st.session_state.role,
                    user_input,
                    memory,
                    rag_engine,
                    chat_history=st.session_state.chat_history
                )
                formatted_latency = time.time() - start_time
                latency_ms = int(formatted_latency * 1000)

                query_type = "general"
                lowered = user_input.lower()
                if any(keyword in lowered for keyword in ["code", "implementation", "architecture"]):
                    query_type = "technical"
                elif any(keyword in lowered for keyword in ["career", "experience", "background"]):
                    query_type = "career"
                elif any(keyword in lowered for keyword in ["mma", "fight", "fighting"]):
                    query_type = "mma"

                interaction_data = UserInteractionData(
                    session_id=st.session_state.session_id,
                    role_mode=st.session_state.role,
                    query=user_input,
                    answer=response_formatter.format(raw_response),
                    query_type=query_type,
                    latency_ms=latency_ms,
                    tokens_prompt=None,
                    tokens_completion=None,
                    success=True
                )
                supabase_analytics.log_interaction(interaction_data)

            formatted = response_formatter.format(raw_response)

            st.session_state.chat_history.append({"role": "assistant", "content": formatted})

            with st.chat_message("assistant"):
                st.markdown(formatted)

        except Exception as e:
            # Log failed interaction
            response_time = time.time() - start_time
            latency_ms = int(response_time * 1000)

            interaction_data = UserInteractionData(
                session_id=st.session_state.session_id,
                role_mode=st.session_state.role,
                query=user_input,
                answer=f"Error: {str(e)}",
                query_type="error",
                latency_ms=latency_ms,
                success=False
            )

            supabase_analytics.log_interaction(interaction_data)

            st.error(f"Sorry, I encountered an error: {str(e)}")

    # Supabase Analytics panel
    with st.expander("System Health", expanded=False):
        health_status = supabase_analytics.health_check()
        if health_status["status"] == "healthy":
            st.success("✅ Analytics system healthy")
            st.metric("Total Messages", health_status["total_messages"])
            st.metric("Recent (24h)", health_status["recent_messages_24h"])
        else:
            st.error("❌ Analytics system unhealthy")
            st.error(health_status.get("error", "Unknown error"))

    # Simple clear chat
    if st.sidebar.button("Clear Chat"):
        st.session_state.chat_history = []

    # Special UI for Confession role
    if st.session_state.role == "Looking to confess crush":
        st.subheader("Anonymous / Named Confession")
        st.markdown("Provide a message. Name is optional. We store only what you submit. No hidden PII capture.")
        with st.form("confession_form"):
            name = st.text_input("Name (optional)")
            message = st.text_area("Your message", max_chars=500, help="Max 500 characters")
            consent = st.checkbox("I consent to storing this submitted content.", value=False)
            submitted = st.form_submit_button("Submit Confession")
        if submitted:
            if not consent:
                st.warning("Consent required to store the message.")
            elif not message.strip():
                st.warning("Message cannot be empty.")
            else:
                import csv, os, datetime
                from config.supabase_config import supabase_settings

                path = supabase_settings.confessions_path
                os.makedirs(os.path.dirname(path), exist_ok=True)
                write_header = not os.path.exists(path)
                with open(path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(["timestamp", "name", "message", "consent"])
                    w.writerow([datetime.datetime.now(datetime.timezone.utc).isoformat(), name.strip(), message.strip(), "yes"])
                st.success("Confession stored. 💌")
        st.stop()

if __name__ == "__main__":
    main()
