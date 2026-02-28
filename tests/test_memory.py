import pytest
import json
import tempfile
import os
from assistant.core.memory import Memory

@pytest.fixture
def temp_memory_file():
    """Create temporary memory file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        f.write('{}')
        temp_file = f.name
    yield temp_file
    os.unlink(temp_file)

@pytest.fixture
def memory_instance(temp_memory_file):
    """Create Memory instance with temporary file."""
    return Memory(persistence_file=temp_memory_file)

def test_memory_initialization(memory_instance):
    """Test Memory class initializes correctly."""
    assert memory_instance.session_data == {}
    assert os.path.exists(memory_instance.persistence_file)

def test_store_and_retrieve_session_context(memory_instance):
    """Test storing and retrieving session context."""
    session_id = "test-session-123"
    role = "Software Developer"
    chat_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What's Noah's Python skill level?"}
    ]

    # Store session context
    memory_instance.store_session_context(session_id, role, chat_history)

    # Retrieve session context
    retrieved = memory_instance.retrieve_session_context(session_id)

    assert retrieved is not None
    assert retrieved["role"] == role
    assert len(retrieved["chat_history"]) == 3
    assert retrieved["chat_history"][0]["content"] == "Hello"
    assert "timestamp" in retrieved

def test_chat_history_truncation(memory_instance):
    """Test that chat history is truncated to last 10 messages."""
    session_id = "test-truncation"
    role = "Technical Manager"

    # Create 15 messages
    chat_history = []
    for i in range(15):
        chat_history.append({"role": "user", "content": f"Message {i}"})
        chat_history.append({"role": "assistant", "content": f"Response {i}"})

    memory_instance.store_session_context(session_id, role, chat_history)
    retrieved = memory_instance.retrieve_session_context(session_id)

    # Should only keep last 10 messages
    assert len(retrieved["chat_history"]) == 10
    assert retrieved["chat_history"][-1]["content"] == "Response 14"

def test_working_memory(memory_instance):
    """Test working memory functionality."""
    # Add to working memory
    memory_instance.add_to_working_memory("user_preferences", {"theme": "dark"})
    memory_instance.add_to_working_memory("last_query", "Python skills")

    # Retrieve from working memory
    preferences = memory_instance.get_from_working_memory("user_preferences")
    last_query = memory_instance.get_from_working_memory("last_query")
    missing = memory_instance.get_from_working_memory("nonexistent", "default")

    assert preferences == {"theme": "dark"}
    assert last_query == "Python skills"
    assert missing == "default"

def test_clear_session(memory_instance):
    """Test clearing session data."""
    session_id = "test-clear"
    role = "Hiring Manager"
    chat_history = [{"role": "user", "content": "Test message"}]

    # Store then clear
    memory_instance.store_session_context(session_id, role, chat_history)
    assert memory_instance.retrieve_session_context(session_id) is not None

    memory_instance.clear_session(session_id)
    assert memory_instance.retrieve_session_context(session_id) is None

def test_persistence_across_instances(temp_memory_file):
    """Test that data persists across Memory instances."""
    session_id = "persistent-test"
    role = "Software Developer"
    chat_history = [{"role": "user", "content": "Persistent message"}]

    # Create first instance and store data
    memory1 = Memory(persistence_file=temp_memory_file)
    memory1.store_session_context(session_id, role, chat_history)

    # Create second instance and retrieve data
    memory2 = Memory(persistence_file=temp_memory_file)
    retrieved = memory2.retrieve_session_context(session_id)

    assert retrieved is not None
    assert retrieved["role"] == role
    assert retrieved["chat_history"][0]["content"] == "Persistent message"
