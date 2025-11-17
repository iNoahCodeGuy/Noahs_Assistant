import pytest
from src.core.rag_engine import RagEngine
from src.retrieval.career_kb import CareerKB
from src.retrieval.code_index import CodeIndex

@pytest.fixture
def rag_engine():
    career_kb = CareerKB('data/career_kb.csv')
    code_index = CodeIndex('vector_stores/code_index')
    return RagEngine(career_kb, code_index)

def test_rag_engine_initialization(rag_engine):
    assert rag_engine is not None
    assert isinstance(rag_engine.career_kb, CareerKB)
    assert isinstance(rag_engine.code_index, CodeIndex)

def test_rag_engine_retrieval(rag_engine):
    query = "What are the key skills for a software developer?"
    response = rag_engine.retrieve(query)
    assert response is not None
    assert isinstance(response, dict)
    assert "skills" in response

def test_rag_engine_embedding(rag_engine):
    text = "Example text for embedding."
    embedding = rag_engine.embed(text)
    assert embedding is not None
    assert len(embedding) > 0

def test_rag_engine_integration(rag_engine):
    query = "Explain the tech stack used in the project."
    response = rag_engine.generate_response(query)
    assert response is not None
    assert "tech stack" in response.lower()
