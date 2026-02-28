"""
Pytest configuration and shared fixtures for Noah's AI Assistant tests.

This file contains common test fixtures, utilities, and configuration
used across the test suite.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Add src to Python path for imports
src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    mock = Mock()
    mock.openai_api_key = "test-key"
    mock.disable_auto_rebuild = True
    mock.log_level = "INFO"
    mock.max_tokens = 1000
    mock.temperature = 0.7
    return mock


@pytest.fixture
def mock_rag_engine():
    """Mock RAG engine for testing."""
    mock = Mock()
    mock.query.return_value = {
        'answer': 'Test response',
        'sources': ['test_source.py:1-10'],
        'code_snippets': []
    }
    mock.retrieve_with_code.return_value = {
        'response': 'Test response with code',
        'code_snippets': [
            {
                'file_path': 'test_file.py',
                'content': 'def test_function():\n    pass',
                'citation': 'test_file.py:1-2',
                'line_start': 1,
                'line_end': 2
            }
        ],
        'has_code': True
    }
    mock.code_index_version.return_value = "test_version_123"
    # Mock pgvector retriever instead of vector_store
    mock.pgvector_retriever = Mock()
    mock.pgvector_retriever.retrieve.return_value = [
        {'content': 'Test career chunk', 'similarity': 0.85}
    ]
    mock.pgvector_retriever.health_check.return_value = {'status': 'healthy'}
    return mock


@pytest.fixture
def mock_role_router():
    """Mock role router for testing."""
    mock = Mock()
    mock.route.return_value = {
        'response': 'Test routed response',
        'role_specific_data': {}
    }
    mock.get_role_context.return_value = {
        'technical_depth': 'high',
        'include_code': True
    }
    return mock


@pytest.fixture
def mock_memory():
    """Mock memory system for testing."""
    mock = Mock()
    mock.get_context.return_value = "Previous conversation context"
    mock.add_message.return_value = None
    mock.get_conversation_history.return_value = [
        {'role': 'user', 'content': 'Previous question'},
        {'role': 'assistant', 'content': 'Previous answer'}
    ]
    return mock


@pytest.fixture
def mock_analytics():
    """Mock analytics system for testing."""
    mock = Mock()
    mock.get_most_common_questions.return_value = [
        {
            'question': 'What is Noah\'s background?',
            'frequency': 15,
            'success_rate': 0.85,
            'roles': ['Hiring Manager']
        },
        {
            'question': 'Show me the code architecture',
            'frequency': 12,
            'success_rate': 0.92,
            'roles': ['Software Developer']
        }
    ]
    mock.get_user_behavior_insights.return_value = {
        'total_interactions': 100,
        'role_distribution': {
            'Software Developer': 45,
            'Hiring Manager (technical)': 30,
            'Hiring Manager (nontechnical)': 15,
            'Just looking around': 10
        },
        'performance_by_role': {
            'Software Developer': {
                'success_rate': 0.92,
                'avg_response_time': 2.1
            }
        },
        'query_patterns_by_role': {
            'Software Developer': {
                'technical': 35,
                'career': 10
            }
        }
    }
    mock.get_content_effectiveness_report.return_value = {
        'top_content': [
            {'content_id': 'career_1', 'accesses': 25, 'relevance': 0.9}
        ],
        'performance_by_type': {
            'career': {
                'unique_items': 15,
                'total_accesses': 100,
                'avg_relevance': 0.85
            }
        }
    }
    return mock


@pytest.fixture
def sample_code_snippets():
    """Sample code snippets for testing."""
    return [
        {
            'file_path': 'src/core/rag_engine.py',
            'content': '''def retrieve_with_code(self, query: str, role: str = None):
    """Retrieve documents and code snippets based on query."""
    results = self.pgvector_retriever.retrieve(query)
    return self._format_results(results)''',
            'citation': 'src/core/rag_engine.py:25-28',
            'line_start': 25,
            'line_end': 28,
            'relevance_score': 0.95
        },
        {
            'file_path': 'src/agents/role_router.py',
            'content': '''def route(self, role: str, query: str, context: str):
    """Route query based on user role."""
    if role == "Software Developer":
        return self._technical_response(query, context)
    return self._general_response(query, context)''',
            'citation': 'src/agents/role_router.py:15-19',
            'line_start': 15,
            'line_end': 19,
            'relevance_score': 0.88
        }
    ]


@pytest.fixture
def test_environment():
    """Set up test environment variables."""
    test_env = {
        'OPENAI_API_KEY': 'test-key-12345',
        'TESTING': 'true',
        'LOG_LEVEL': 'DEBUG'
    }
    
    with patch.dict(os.environ, test_env):
        yield test_env


@pytest.fixture
def temp_test_files(tmp_path):
    """Create temporary test files."""
    # Create test source files
    test_src = tmp_path / "test_src"
    test_src.mkdir()
    
    # Create a test Python file
    test_file = test_src / "test_module.py"
    test_file.write_text('''"""Test module for testing."""

def example_function():
    """Example function for testing code retrieval."""
    return "Hello, World!"

class ExampleClass:
    """Example class for testing."""
    
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
''')
    
    return {
        'src_dir': test_src,
        'test_file': test_file
    }


@pytest.fixture(scope="session")
def performance_baseline():
    """Performance baseline metrics for testing."""
    return {
        'max_query_time': 10.0,  # seconds
        'max_init_time': 30.0,   # seconds
        'min_success_rate': 0.8,  # 80%
        'max_memory_usage': 500   # MB
    }


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid or "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.nodeid or "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark unit tests (default for everything else)
        elif not any(marker.name in ['integration', 'performance'] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# Utility functions for tests
def assert_valid_response(response: Dict[str, Any]):
    """Assert that a response has the expected structure."""
    assert isinstance(response, dict)
    assert 'answer' in response or 'response' in response
    
    if 'code_snippets' in response:
        assert isinstance(response['code_snippets'], list)
        for snippet in response['code_snippets']:
            assert 'file_path' in snippet
            assert 'content' in snippet
            assert 'citation' in snippet


def assert_valid_code_snippet(snippet: Dict[str, Any]):
    """Assert that a code snippet has the expected structure."""
    required_fields = ['file_path', 'content', 'citation']
    for field in required_fields:
        assert field in snippet, f"Code snippet missing required field: {field}"
    
    # Validate citation format (file:line or file:start-end)
    citation = snippet['citation']
    assert ':' in citation, "Citation should contain ':' separator"
    
    file_part, line_part = citation.split(':', 1)
    assert file_part.endswith('.py'), "Citation should reference a Python file"


def create_mock_interaction(role: str = "Software Developer", 
                          query: str = "Test query",
                          success: bool = True) -> Dict[str, Any]:
    """Create a mock user interaction for testing."""
    return {
        'timestamp': '2024-01-01T12:00:00',
        'user_role': role,
        'query': query,
        'response_time': 2.5,
        'success': success,
        'code_snippets_count': 3 if role == "Software Developer" else 0,
        'citation_accuracy': success
    }