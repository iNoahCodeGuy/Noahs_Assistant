"""End-to-end tests for the 22-node functional pipeline.

All tests are hermetic: the Anthropic client is faked at the stage1 boundary,
retrieval/generation run through a dummy engine, and analytics logging is
stubbed. Nodes return partial dicts that run_conversation_flow merges via
state.update(result) — these tests exercise that contract through the real
pipeline loop.
"""

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

import assistant.flows.node_logic.stage1_intent_router as stage1
from assistant.flows.conversation_flow import run_conversation_flow
from assistant.flows.node_logic.stage0_session_management import (
    initialize_conversation_state,
)


class DummyResponseGenerator:
    """Matches the interface generate_draft/format_answer actually use."""

    def __init__(self, response: str):
        self._response = response
        self.calls: List[Dict[str, Any]] = []

    def generate_contextual_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        role: str = None,
        chat_history: List[Dict[str, str]] = None,
        extra_instructions: str = None,
        model_name: str = None,
    ) -> str:
        self.calls.append({"query": query, "n_chunks": len(context or [])})
        return f"{self._response} (answering: {query[:60]})"

    def _enforce_first_person(self, text: str) -> str:
        return text


class DummyRagEngine:
    """Minimal engine satisfying every rag_engine.* call in the pipeline."""

    def __init__(self, chunks: List[Dict[str, Any]], response_text: str):
        self._chunks = chunks
        self.response_generator = DummyResponseGenerator(response_text)
        self.pgvector_retriever = None  # forces the plain retrieve() path
        self.retrieve_calls: List[str] = []

    def retrieve(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        self.retrieve_calls.append(query)
        return {
            "chunks": self._chunks,
            "matches": [c["content"] for c in self._chunks],
            "scores": [c.get("similarity", 0.0) for c in self._chunks],
            "skills": [],
            "raw": [],
        }

    def retrieve_with_code(self, query: str, role: str = None) -> List[Dict[str, Any]]:
        return []


class FakeAnthropic:
    """Stands in for anthropic.Anthropic inside stage1_intent_router."""

    canned_response = "knowledge_query|neutral"

    def __init__(self, api_key: str = None):
        self.messages = self

    def create(self, **kwargs):
        return SimpleNamespace(
            content=[SimpleNamespace(text=self.canned_response)]
        )


@pytest.fixture
def dummy_engine() -> DummyRagEngine:
    return DummyRagEngine(
        chunks=[
            {
                "content": "Noah works at Tesla and builds ML projects.",
                "section": "career_overview",
                "doc_id": "career",
                "similarity": 0.82,
            },
            {
                "content": "He coaches BJJ at Xtreme Couture since 2021.",
                "section": "career_coaching",
                "doc_id": "career",
                "similarity": 0.71,
            },
        ],
        response_text="Noah has a strong track record.",
    )


@pytest.fixture
def hermetic_pipeline(monkeypatch: pytest.MonkeyPatch):
    """Fake the LLM classifier and silence analytics side effects."""
    monkeypatch.setattr(stage1, "Anthropic", FakeAnthropic)
    monkeypatch.setattr(
        "assistant.flows.conversation_flow.log_and_notify",
        lambda state, **kwargs: state,
    )
    FakeAnthropic.canned_response = "knowledge_query|neutral"
    return monkeypatch


def _base_state(query: str) -> Dict[str, Any]:
    # role_welcome_shown models a session past the scripted welcome turn —
    # without it, classify_role_mode answers every query with the welcome
    # template and halts before retrieval (by design).
    return {
        "role": "Learn more about Noah",
        "query": query,
        "chat_history": [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "Welcome — what brings you here?"},
        ],
        "session_memory": {
            "persona_hints": {"role_welcome_shown": True, "role_mode": "explorer"}
        },
    }


def test_knowledge_query_end_to_end(hermetic_pipeline, dummy_engine):
    state = _base_state("What is Noah's professional background?")
    result = run_conversation_flow(state, dummy_engine, session_id="test-e2e")

    assert result["answer"], "pipeline must produce an answer"
    assert "Noah has a strong track record." in result["answer"]
    assert result["retrieved_chunks"], "knowledge queries must hit retrieval"
    assert dummy_engine.retrieve_calls, "engine.retrieve should have been called"


def test_pipeline_appends_turn_to_chat_history(hermetic_pipeline, dummy_engine):
    state = _base_state("What is Noah's professional background?")
    history_before = len(state["chat_history"])
    result = run_conversation_flow(state, dummy_engine, session_id="test-history")

    history = result["chat_history"]
    assert len(history) == history_before + 2, "query + answer should be appended"
    assert history[-2]["content"] == "What is Noah's professional background?"
    assert history[-1]["content"] == result["answer"]


def test_greeting_short_circuits_retrieval(hermetic_pipeline, dummy_engine):
    state = _base_state("hi")
    result = run_conversation_flow(state, dummy_engine, session_id="test-greeting")

    assert result["answer"], "greeting must still produce an answer"
    assert not dummy_engine.retrieve_calls, "greetings must never hit retrieval"


def test_crush_intent_skips_rag(hermetic_pipeline, dummy_engine):
    FakeAnthropic.canned_response = "crush_confession|crush"
    state = _base_state("I would like to confess a crush")
    result = run_conversation_flow(state, dummy_engine, session_id="test-crush")

    assert result["answer"], "crush flow must produce an answer"
    assert not dummy_engine.retrieve_calls, "crush flow must never hit retrieval"
    assert "Message for Noah:" in result["answer"], (
        "crush flow should present the confession form (state-machine marker)"
    )


def test_initialize_clears_volatile_fields():
    """Regression: per-turn fields must not leak across turns (serverless reuse)."""
    stale = {
        "role": "Learn more about Noah",
        "query": "next question",
        "chat_history": [],
        "session_memory": {},
        # leftovers from a previous turn:
        "answer": "old answer",
        "pipeline_halt": True,
        "skip_rag": True,
        "message_intent": "greeting",
        "is_greeting": True,
        "clarification_needed": True,
    }
    result = initialize_conversation_state(stale)
    merged = {**stale, **(result or {})}

    assert merged["answer"] == ""
    assert merged["pipeline_halt"] is False
    assert merged["skip_rag"] is False
    assert merged["is_greeting"] is False
    assert merged["clarification_needed"] is False
