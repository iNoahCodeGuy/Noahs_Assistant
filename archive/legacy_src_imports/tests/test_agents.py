import pytest
from unittest.mock import Mock
from src.agents.role_router import RoleRouter
from src.agents.response_formatter import ResponseFormatter
from src.core.memory import Memory
from src.core.rag_engine import RagEngine

@pytest.fixture
def role_router():
    return RoleRouter()

@pytest.fixture
def response_formatter():
    return ResponseFormatter()

@pytest.fixture
def mock_memory():
    return Mock(spec=Memory)

@pytest.fixture
def mock_rag_engine():
    mock = Mock(spec=RagEngine)
    mock.retrieve_code_info.return_value = []
    mock.retrieve_career_info.return_value = []
    mock.retrieve_with_code.return_value = {"code_snippets": [], "matches": []}
    mock.generate_response_with_context.return_value = "Generated response"
    mock.generate_technical_response.return_value = "Technical response"
    return mock

def test_role_router_initialization(role_router):
    assert role_router is not None

def test_route_nontechnical_hiring_manager(role_router, mock_memory, mock_rag_engine):
    response = role_router.route("Hiring Manager (nontechnical)", "test query", mock_memory, mock_rag_engine)
    assert "response" in response

def test_route_technical_hiring_manager(role_router, mock_memory, mock_rag_engine):
    response = role_router.route("Hiring Manager (technical)", "test query", mock_memory, mock_rag_engine)
    assert "response" in response

def test_route_software_developer(role_router, mock_memory, mock_rag_engine):
    response = role_router.route("Software Developer", "test query", mock_memory, mock_rag_engine)
    assert "response" in response

def test_route_casual_visitor(role_router, mock_memory, mock_rag_engine):
    response = role_router.route("Just looking around", "test query", mock_memory, mock_rag_engine)
    assert "response" in response

def test_route_confess_crush(role_router, mock_memory, mock_rag_engine):
    response = role_router.route("Looking to confess crush", "test query", mock_memory, mock_rag_engine)
    assert "response" in response

def test_response_formatter_initialization(response_formatter):
    assert response_formatter is not None

def test_format_technical_response(response_formatter):
    test_data = {"response": "Technical details here.", "type": "technical", "context": []}
    formatted_response = response_formatter.format(test_data)
    assert "Technical details here." in formatted_response

def test_format_plain_english_response(response_formatter):
    test_data = {"response": "Plain English summary here.", "type": "general", "context": []}
    formatted_response = response_formatter.format(test_data)
    assert "Plain English summary here." in formatted_response

def test_format_mixed_response(response_formatter):
    test_data = {"response": "Technical details with summary", "type": "technical", "context": []}
    formatted_response = response_formatter.format(test_data)
    assert "Technical details" in formatted_response
