import os
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from assistant.state.conversation_state import ConversationState
from assistant.flows import conversation_nodes as nodes
from assistant.flows.conversation_flow import run_conversation_flow


class DummyResponseGenerator:
    def __init__(self, response: str):
        self._response = response

    def generate_basic_response(self, query: str, fallback_docs: List[str], chat_history: List[Dict[str, str]] | None = None) -> str:
        history_note = f" ({len(chat_history)} messages)" if chat_history else ""
        return f"{self._response} | {query}{history_note}"


@dataclass
class DummyRagEngine:
    retrieve_result: Dict[str, Any]
    response_text: str

    def retrieve(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        return self.retrieve_result

    @property
    def response_generator(self) -> DummyResponseGenerator:
        return DummyResponseGenerator(self.response_text)


@pytest.fixture
def base_state() -> ConversationState:
    return ConversationState(
        role="Hiring Manager (nontechnical)",
        query="Tell me about Noah's career",
        chat_history=
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Can you share more?"},
        ],
    )


@pytest.fixture
def dummy_engine() -> DummyRagEngine:
    return DummyRagEngine(
        retrieve_result={
            "matches": ["Match A", "Match B"],
            "scores": [0.95, 0.74],
            "chunks": [
                {"content": "Match A", "doc_id": "career", "similarity": 0.95},
                {"content": "Match B", "doc_id": "career", "similarity": 0.74},
            ],
        },
        response_text="Noah has a strong track record.",
    )


def test_classify_query_sets_type(base_state: ConversationState) -> None:
    nodes.classify_query(base_state)
    assert base_state.fetch("query_type") == "career"


def test_retrieve_chunks_stores_context(base_state: ConversationState, dummy_engine: DummyRagEngine) -> None:
    nodes.retrieve_chunks(base_state, dummy_engine)
    assert len(base_state.retrieved_chunks) == 2
    assert base_state.fetch("retrieval_matches") == ["Match A", "Match B"]
    assert base_state.fetch("retrieval_scores") == [0.95, 0.74]


def test_generate_answer_uses_response_generator(base_state: ConversationState, dummy_engine: DummyRagEngine) -> None:
    nodes.retrieve_chunks(base_state, dummy_engine)
    nodes.generate_answer(base_state, dummy_engine)
    assert base_state.answer
    assert "Tell me about Noah's career" in base_state.answer


@pytest.mark.parametrize(
    "role,query,chat_history,expected",
    [
        (
            "Hiring Manager (nontechnical)",
            "Tell me about Noah",
            [
                {"role": "user", "content": "Intro"},
                {"role": "assistant", "content": "Reply"},
                {"role": "user", "content": "Follow up"},
            ],
            "offer_resume_prompt",
        ),
        (
            "Hiring Manager (technical)",
            "Explain the architecture",
            [],
            "provide_data_tables",
        ),
        (
            "Software Developer",
            "Show me the code",
            [],
            "include_code_snippets",
        ),
        (
            "Just looking around",
            "Give me fun facts",
            [],
            "share_fun_facts",
        ),
        (
            "Looking to confess crush",
            "I have a secret",
            [],
            "collect_confession",
        ),
    ],
)
def test_plan_actions_appends_expected_action(role: str, query: str, chat_history: List[Dict[str, str]], expected: str, dummy_engine: DummyRagEngine) -> None:
    state = ConversationState(role=role, query=query, chat_history=chat_history)
    nodes.classify_query(state)
    nodes.plan_actions(state)
    action_types = [action["type"] for action in state.pending_actions]
    assert expected in action_types


def test_log_and_notify_records_metadata(monkeypatch: pytest.MonkeyPatch, base_state: ConversationState) -> None:
    logged_payloads: List[Dict[str, Any]] = []

    class DummyAnalytics:
        @staticmethod
        def log_interaction(data):
            logged_payloads.append({
                "role_mode": data.role_mode,
                "query": data.query,
                "latency_ms": data.latency_ms,
            })
            return 99

    monkeypatch.setattr(nodes, "supabase_analytics", DummyAnalytics)
    base_state.answer = "Career summary"
    nodes.classify_query(base_state)
    result_state = nodes.log_and_notify(base_state, session_id="test-session", latency_ms=123)
    assert result_state.analytics_metadata["message_id"] == 99
    assert logged_payloads[0]["role_mode"] == "Hiring Manager (nontechnical)"
    assert logged_payloads[0]["latency_ms"] == 123


def test_run_conversation_flow_happy_path(base_state: ConversationState, dummy_engine: DummyRagEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(os.environ, "LANGGRAPH_FLOW_ENABLED", "true")

    logged = {}

    class DummyAnalytics:
        @staticmethod
        def log_interaction(data):
            logged["query_type"] = data.query_type
            logged["latency_ms"] = data.latency_ms
            return 101

    monkeypatch.setattr(nodes, "supabase_analytics", DummyAnalytics)

    state = run_conversation_flow(
        state=base_state,
        rag_engine=dummy_engine,
        session_id="abc123",
    )

    assert state.answer.startswith("Noah has a strong track record.")
    assert "Would you like me to email you my resume" in state.answer
    assert state.retrieved_chunks
    assert state.pending_actions[0]["type"] == "offer_resume_prompt"
    assert state.analytics_metadata["message_id"] == 101
    assert logged["query_type"] == "career"
    assert isinstance(logged["latency_ms"], int)


def test_execute_actions_send_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    state = ConversationState(
        role="Hiring Manager (technical)",
        query="Please email your resume",
        chat_history=[],
    )
    state.pending_actions = [
        {"type": "send_resume", "email": "hiring@company.com", "name": "Hiring Team"},
        {"type": "notify_resume_sent"},
    ]
    state.stash("user_email", "hiring@company.com")

    class DummyStorage:
        def __init__(self):
            self.calls = []

        def get_signed_url(self, file_path: str, expires_in: int = 86400) -> str:
            self.calls.append((file_path, expires_in))
            return "https://signed.example.com/resume.pdf"

    class DummyResend:
        def __init__(self):
            self.sent = []

        def send_resume_email(self, to_email: str, to_name: str, resume_url: str, message=None) -> Dict[str, Any]:
            self.sent.append((to_email, to_name, resume_url, message))
            return {"status": "sent"}

    class DummyTwilio:
        def __init__(self):
            self.alerts = []

        def send_contact_alert(self, **payload) -> Dict[str, Any]:
            self.alerts.append(payload)
            return {"status": "sent"}

    dummy_storage = DummyStorage()
    dummy_resend = DummyResend()
    dummy_twilio = DummyTwilio()

    monkeypatch.setattr(nodes, "get_storage_service", lambda: dummy_storage)
    monkeypatch.setattr(nodes, "get_resend_service", lambda: dummy_resend)
    monkeypatch.setattr(nodes, "get_twilio_service", lambda: dummy_twilio)

    nodes.execute_actions(state)

    assert dummy_storage.calls == [("resumes/noah_resume.pdf", 86400)]
    assert dummy_resend.sent == [
        ("hiring@company.com", "Hiring Team", "https://signed.example.com/resume.pdf", None)
    ]
    assert state.analytics_metadata["resume_email_status"] == "sent"
    assert dummy_twilio.alerts[0]["message_preview"].startswith("Resume dispatched")


def test_execute_actions_contact_notifications(monkeypatch: pytest.MonkeyPatch) -> None:
    state = ConversationState(
        role="Hiring Manager (nontechnical)",
        query="Please reach out to me about the role",
        chat_history=[],
    )
    state.pending_actions = [
        {"type": "notify_contact_request", "urgent": True},
    ]
    state.stash("user_name", "Alex")
    state.stash("user_email", "alex@example.com")
    state.stash("user_phone", "+15551234")

    class DummyResend:
        def __init__(self):
            self.notifications = []

        def send_contact_notification(self, **payload) -> Dict[str, Any]:
            self.notifications.append(payload)
            return {"status": "sent"}

    class DummyTwilio:
        def __init__(self):
            self.alerts = []

        def send_contact_alert(self, **payload) -> Dict[str, Any]:
            self.alerts.append(payload)
            return {"status": "sent"}

    dummy_resend = DummyResend()
    dummy_twilio = DummyTwilio()

    monkeypatch.setattr(nodes, "get_resend_service", lambda: dummy_resend)
    monkeypatch.setattr(nodes, "get_twilio_service", lambda: dummy_twilio)

    nodes.execute_actions(state)

    assert dummy_resend.notifications[0]["from_name"] == "Alex"
    assert dummy_resend.notifications[0]["user_role"] == "Hiring Manager (nontechnical)"
    assert dummy_twilio.alerts[0]["is_urgent"] is True
