import pytest
from unittest.mock import Mock, MagicMock
from assistant.agents.role_router import RoleRouter
from assistant.core.memory import Memory
from assistant.core.rag_engine import RagEngine

@pytest.fixture
def mock_rag_engine():
    """Mock RAG engine for testing."""
    mock_engine = Mock(spec=RagEngine)
    mock_engine.retrieve_career_info.return_value = [
        {"content": "Noah has intermediate Python skills", "metadata": {"source": "career_kb.csv"}}
    ]
    mock_engine.retrieve_code_info.return_value = [
        {"content": "def test_function():", "metadata": {"file": "test.py"}}
    ]
    mock_engine.retrieve_with_code.return_value = {
        "code_snippets": [{
            "content": "def example():",
            "file": "example.py",
            "name": "example_function",
            "citation": "example.py:1",
            "github_url": "https://github.com/user/repo/blob/main/example.py",
            "type": "function"
        }],
        "matches": ["Career context"]
    }
    mock_engine.generate_response_with_context.return_value = "Generated response with context"
    mock_engine.generate_technical_response.return_value = "Technical response"
    # Use only methods that actually exist in RagEngine
    mock_engine.generate_response.return_value = "General response"
    return mock_engine

@pytest.fixture
def mock_memory():
    """Mock Memory for testing."""
    return Mock(spec=Memory)

@pytest.fixture
def role_router():
    """Create RoleRouter instance."""
    return RoleRouter()

def test_role_router_initialization(role_router):
    """Test RoleRouter initializes correctly."""
    assert role_router is not None
    assert hasattr(role_router, 'settings')

def test_nontechnical_manager_routing(role_router, mock_memory, mock_rag_engine):
    """Test routing for nontechnical hiring managers."""
    user_input = "What's Noah's experience?"

    response = role_router.route(
        "Hiring Manager (nontechnical)",
        user_input,
        mock_memory,
        mock_rag_engine
    )

    assert response is not None
    assert "response" in response
    assert response["type"] == "career"
    mock_rag_engine.retrieve_career_info.assert_called_once()

def test_technical_manager_routing(role_router, mock_memory, mock_rag_engine):
    """Test routing for technical hiring managers."""
    user_input = "Show me Noah's technical stack and code implementation"

    response = role_router.route(
        "Hiring Manager (technical)",
        user_input,
        mock_memory,
        mock_rag_engine
    )

    assert response is not None
    assert "response" in response
    # Technical query should trigger technical handling
    assert response["type"] == "technical"

def test_developer_routing(role_router, mock_memory, mock_rag_engine):
    """Test routing for software developers."""
    user_input = "Explain Noah's code architecture"

    response = role_router.route(
        "Software Developer",
        user_input,
        mock_memory,
        mock_rag_engine
    )

    assert response is not None
    assert "response" in response
    mock_rag_engine.retrieve_code_info.assert_called_once()

def test_casual_visitor_routing(role_router, mock_memory, mock_rag_engine):
    """Test routing for casual visitors."""
    user_input = "Tell me something interesting about Noah"

    response = role_router.route(
        "Just looking around",
        user_input,
        mock_memory,
        mock_rag_engine
    )

    assert response is not None
    assert "response" in response

def test_confession_routing(role_router, mock_memory, mock_rag_engine):
    """Test routing for confession role."""
    user_input = "I have something to tell Noah"

    response = role_router.route(
        "Looking to confess crush",
        user_input,
        mock_memory,
        mock_rag_engine
    )

    assert response is not None
    assert "response" in response
    assert response["type"] == "confession"
    assert "💌" in response["response"]

def test_context_preservation_across_turns(role_router, mock_memory, mock_rag_engine):
    """Test that the router handles optional chat_history parameter."""
    user_input = "How advanced is his Python?"
    chat_history = [
        {"role": "user", "content": "What languages does Noah use?"},
        {"role": "assistant", "content": "Noah uses Python primarily."}
    ]

    response = role_router.route(
        "Software Developer",
        user_input,
        mock_memory,
        mock_rag_engine,
        chat_history
    )

    assert response is not None
    assert "response" in response
