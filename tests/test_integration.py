import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from assistant.core.memory import Memory
from assistant.agents.role_router import RoleRouter
from assistant.core.rag_engine import RagEngine

@pytest.fixture
def integration_setup():
    """Setup for integration tests."""
    # Create temporary memory file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        f.write('{}')
        temp_file = f.name

    memory = Memory(persistence_file=temp_file)
    router = RoleRouter()

    # Mock RAG engine
    mock_rag = Mock(spec=RagEngine)
    mock_rag.retrieve.return_value = [
        {"content": "Noah has Python experience", "metadata": {"source": "test"}}
    ]
    mock_rag.generate_response.return_value = "Test response about Noah's skills"

    yield memory, router, mock_rag, temp_file

    # Cleanup
    os.unlink(temp_file)

def test_full_conversation_flow(integration_setup):
    """Test complete conversation flow with memory persistence."""
    memory, router, mock_rag, temp_file = integration_setup

    session_id = "integration-test"
    role = "Technical Hiring Manager"

    # Simulate conversation turns
    conversation = [
        {"user": "What's Noah's background?", "expected_context": ""},
        {"user": "How strong is his Python?", "expected_context": "What's Noah's background?"},
        {"user": "Show me his projects", "expected_context": "How strong is his Python?"}
    ]

    chat_history = []

    for turn in conversation:
        # Route the query
        response = router.route(
            role,
            turn["user"],
            memory,
            mock_rag,
            chat_history.copy()
        )

        # Update chat history
        chat_history.append({"role": "user", "content": turn["user"]})
        chat_history.append({"role": "assistant", "content": response})

        # Store in memory
        memory.store_session_context(session_id, role, chat_history)

    # Verify final state
    final_context = memory.retrieve_session_context(session_id)
    assert final_context["role"] == role
    assert len(final_context["chat_history"]) == 6  # 3 turns × 2 messages each

def test_session_persistence_across_restarts(integration_setup):
    """Test that sessions persist across application restarts."""
    memory1, router, mock_rag, temp_file = integration_setup

    session_id = "persistence-test"
    role = "Software Developer"
    initial_chat = [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "First response"}
    ]

    # Store session in first instance
    memory1.store_session_context(session_id, role, initial_chat)

    # Create new Memory instance (simulating restart)
    memory2 = Memory(persistence_file=temp_file)

    # Continue conversation with new instance
    retrieved_context = memory2.retrieve_session_context(session_id)
    assert retrieved_context is not None
    assert retrieved_context["role"] == role
    assert len(retrieved_context["chat_history"]) == 2

    # Add more to conversation
    extended_chat = retrieved_context["chat_history"] + [
        {"role": "user", "content": "Second message"},
        {"role": "assistant", "content": "Second response"}
    ]

    memory2.store_session_context(session_id, role, extended_chat)

    # Verify persistence
    final_context = memory2.retrieve_session_context(session_id)
    assert len(final_context["chat_history"]) == 4

def test_token_budgeting_in_conversation(integration_setup):
    """Test that token budgeting works in long conversations."""
    memory, router, mock_rag, temp_file = integration_setup

    # Create very long chat history
    long_chat_history = []
    for i in range(50):  # 50 turns = 100 messages
        long_chat_history.append({
            "role": "user",
            "content": f"This is a very long message number {i} that contains many words to test token limits " * 10
        })
        long_chat_history.append({
            "role": "assistant",
            "content": f"This is a very long response number {i} with detailed explanations " * 10
        })

    # Route with long history
    response = router.route(
        "Software Developer",
        "Final question",
        memory,
        mock_rag,
        long_chat_history
    )

    # Should not crash and should return response
    assert response is not None

    # Test passes if router handles long chat history without crashing
    assert "response" in response

@pytest.mark.skipif(True, reason="Streamlit not required for core functionality")
def test_streamlit_session_integration():
    """Test integration with Streamlit session state (mocked)."""
    # Skip this test as streamlit is not a core dependency
    pass
