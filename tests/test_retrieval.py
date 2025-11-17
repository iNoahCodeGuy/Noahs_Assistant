import pytest
from assistant.retrieval.career_kb import CareerKnowledgeBase
from assistant.retrieval.code_index import CodeIndex

@pytest.fixture
def setup_career_kb():
    return CareerKnowledgeBase('data/career_kb.csv')

@pytest.fixture
def setup_code_index():
    return CodeIndex('vector_stores/code_index')

def test_career_kb_loading(setup_career_kb):
    assert setup_career_kb.data is not None
    assert len(setup_career_kb.data) > 0

def test_career_kb_query(setup_career_kb):
    result = setup_career_kb.query('Software Engineer')
    assert result is not None
    # The query should return a result that contains "software engineer" in the content
    # Since it finds "Where does Noah see himself in 3 years?" which mentions software engineer in the answer
    assert 'title' in result
    assert result['title'] is not None
    # Check that the result actually contains information about software engineering
    full_text = str(result.get('Answer', '')) + str(result.get('title', ''))
    assert 'software engineer' in full_text.lower()

def test_code_index_loading(setup_code_index):
    assert setup_code_index.index is not None

def test_code_index_query(setup_code_index):
    result = setup_code_index.query('def example_function')
    assert result is not None
    assert 'example_function' in result['code']
