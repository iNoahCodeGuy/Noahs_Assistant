"""Code display enrichment: technical code requests get real code blocks appended.

Historical note: this file used to test role-specific answer enrichments
("Architecture Snapshot", "Enterprise Fit", "Data Collection Overview",
"Staying Current", and unconditional resume-offer prompts). Those blocks were
removed along with the role-based pipeline, so those tests were deleted.

What survives is the code-display path, which still runs in the universal
pipeline:
    classify_query sets code_display_requested
    → plan_actions queues {"type": "include_code_reference"}
    → format_answer calls rag_engine.retrieve_with_code() and appends a
      formatted markdown code block (with file citation) to the answer.

Nodes return partial update dicts (LangGraph-style); callers merge them via
state.update(result).
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from assistant.state.conversation_state import ConversationState
from assistant.flows import conversation_nodes as nodes


@dataclass
class DummyRagEngine:
    """Stub engine exposing only what format_answer's code path uses."""

    code_snippets: List[Dict[str, Any]]

    def retrieve_with_code(self, query: str, role: str | None = None) -> Dict[str, Any]:
        return {
            "code_snippets": self.code_snippets,
            "has_code": bool(self.code_snippets),
        }


@pytest.fixture
def developer_engine() -> DummyRagEngine:
    snippet = {
        "content": "def run_conversation_flow(state, rag_engine):\n    return state",
        "citation": "src/flows/conversation_flow.py:10-15",
        "github_url": "https://github.com/noahcal/noahs-ai-assistant/blob/main/src/flows/conversation_flow.py#L10-L15",
    }
    return DummyRagEngine(code_snippets=[snippet])


def test_software_developer_code_request_appends_code_block(developer_engine: DummyRagEngine) -> None:
    """A developer asking for implementation details gets a cited code block."""
    state: ConversationState = {
        "role": "Software Developer",
        "query": "Show me the latest implementation details",
        "chat_history": [],
    }

    state = nodes.classify_query(state)
    state = nodes.plan_actions(state)

    assert state.get("code_display_requested") is True
    action_types = {action["type"] for action in state["pending_actions"]}
    assert "include_code_reference" in action_types

    # format_answer builds the final answer from draft_answer and returns a
    # partial update dict that the pipeline merges into state.
    state["draft_answer"] = "Developer focused answer."
    state.update(nodes.format_answer(state, developer_engine))

    output = state.get("answer") or ""
    assert "Developer focused answer." in output
    assert "```python" in output
    assert "src/flows/conversation_flow.py" in output
    assert "def run_conversation_flow" in output
