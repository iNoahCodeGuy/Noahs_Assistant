"""Tests around role-specific LangGraph enrichments for technical audiences."""

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from src.state.conversation_state import ConversationState
from src.flows import conversation_nodes as nodes


class DummyResponseGenerator:
    def __init__(self, text: str):
        self._text = text

    def generate_basic_response(self, query: str, fallback_docs: List[str], chat_history: List[Dict[str, str]] | None = None) -> str:
        return self._text


@dataclass
class DummyRagEngine:
    code_snippets: List[Dict[str, Any]]
    response_text: str = "Here is the latest information."

    def retrieve(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        return {"matches": [], "scores": [], "chunks": []}

    def retrieve_with_code(self, query: str, role: str | None = None) -> Dict[str, Any]:
        return {
            "code_snippets": self.code_snippets,
            "has_code": bool(self.code_snippets),
        }

    @property
    def response_generator(self) -> DummyResponseGenerator:
        return DummyResponseGenerator(self.response_text)


@pytest.fixture
def developer_engine() -> DummyRagEngine:
    snippet = {
        "content": "def run_conversation_flow(state, rag_engine):\n    return state",
        "citation": "src/flows/conversation_flow.py:10-15",
        "github_url": "https://github.com/noahcal/noahs-ai-assistant/blob/main/src/flows/conversation_flow.py#L10-L15",
    }
    return DummyRagEngine(code_snippets=[snippet])


def _build_chat_history(turns: int) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    for idx in range(turns):
        history.append({"role": "user", "content": f"Question {idx}"})
        history.append({"role": "assistant", "content": f"Answer {idx}"})
    return history


def test_technical_hiring_manager_enrichments(developer_engine: DummyRagEngine) -> None:
    state = ConversationState(
        role="Hiring Manager (technical)",
        query="Explain the architecture and enterprise fit",
        chat_history=_build_chat_history(2),
    )

    nodes.classify_query(state)
    nodes.plan_actions(state)
    state.set_answer("Base technical answer.")

    nodes.apply_role_context(state, developer_engine)

    enriched = state.answer or ""
    assert "Architecture Snapshot" in enriched
    assert "Enterprise Fit" in enriched
    assert "Data Collection Overview" in enriched


def test_software_developer_receives_code_and_updates(developer_engine: DummyRagEngine) -> None:
    state = ConversationState(
        role="Software Developer",
        query="Show me the latest implementation details",
        chat_history=[],
    )

    nodes.classify_query(state)
    nodes.plan_actions(state)
    state.set_answer("Developer focused answer.")

    nodes.apply_role_context(state, developer_engine)

    output = state.answer or ""
    assert "```python" in output
    assert "Source:" in output
    assert "Staying Current" in output
    assert "Data Collection Overview" in output


def test_non_technical_manager_offered_resume_prompt(developer_engine: DummyRagEngine) -> None:
    state = ConversationState(
        role="Hiring Manager (nontechnical)",
        query="Tell me more about Noah's experience",
        chat_history=_build_chat_history(2),
    )

    nodes.classify_query(state)
    nodes.plan_actions(state)

    assert any(action["type"] == "offer_resume_prompt" for action in state.pending_actions)

    state.set_answer("Career overview.")
    nodes.apply_role_context(state, developer_engine)

    assert "Would you like me to email you my resume" in (state.answer or "")
