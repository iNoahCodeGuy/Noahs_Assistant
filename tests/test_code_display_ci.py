"""CI-style tests focused on action execution for role-based flows."""

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from assistant.state.conversation_state import ConversationState
from assistant.flows import conversation_nodes as nodes


@dataclass
class DummyRagEngine:
    def retrieve(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        return {"matches": [], "scores": [], "chunks": []}

    def retrieve_with_code(self, query: str, role: str | None = None) -> Dict[str, Any]:
        return {"code_snippets": [], "has_code": False}

    @property
    def response_generator(self):
        class _Generator:
            def generate_basic_response(self, query: str, fallback_docs: List[str], chat_history: List[Dict[str, str]] | None = None) -> str:
                return "Happy to help with that."

        return _Generator()


class DummyResend:
    def __init__(self):
        self.sent: List[Dict[str, Any]] = []

    def send_resume_email(self, to_email: str, to_name: str, resume_url: str, message: str | None = None) -> Dict[str, Any]:
        payload = {
            "to_email": to_email,
            "to_name": to_name,
            "resume_url": resume_url,
            "message": message,
        }
        self.sent.append(payload)
        return {"status": "sent"}

    def send_contact_notification(self, **payload: Any) -> Dict[str, Any]:
        self.sent.append(payload)
        return {"status": "sent"}


class DummyTwilio:
    def __init__(self):
        self.alerts: List[Dict[str, Any]] = []

    def send_contact_alert(self, **payload: Any) -> Dict[str, Any]:
        self.alerts.append(payload)
        return {"status": "sent"}


class DummyStorage:
    def __init__(self):
        self.requests: List[Dict[str, Any]] = []

    def get_signed_url(self, file_path: str, expires_in: int = 86400) -> str:
        self.requests.append({"file_path": file_path, "expires_in": expires_in})
        return "https://signed.example.com/resume.pdf"


@pytest.fixture
def dummy_engine() -> DummyRagEngine:
    return DummyRagEngine()


def test_resume_request_triggers_email_sms_and_prompt(monkeypatch: pytest.MonkeyPatch, dummy_engine: DummyRagEngine) -> None:
    state = ConversationState(
        role="Hiring Manager (nontechnical)",
        query="Could you email me your resume?",
        chat_history=[{"role": "user", "content": "Hello"}],
    )

    nodes.classify_query(state)
    nodes.plan_actions(state)
    state.stash("user_email", "hiring@example.com")
    state.stash("user_name", "Alex Recruiter")
    state.set_answer("Career overview here.")

    resend = DummyResend()
    twilio = DummyTwilio()
    storage = DummyStorage()

    monkeypatch.setattr(nodes, "get_resend_service", lambda: resend)
    monkeypatch.setattr(nodes, "get_twilio_service", lambda: twilio)
    monkeypatch.setattr(nodes, "get_storage_service", lambda: storage)

    nodes.apply_role_context(state, dummy_engine)
    nodes.execute_actions(state)

    assert any(action["type"] == "send_resume" for action in state.pending_actions)
    assert resend.sent[0]["to_email"] == "hiring@example.com"
    assert storage.requests[0]["file_path"] == "resumes/noah_resume.pdf"
    assert twilio.alerts[0]["message_preview"].startswith("Resume dispatched")
    assert "Would you like Noah to reach out?" in (state.answer or "")


def test_linkedin_request_prompts_follow_up(dummy_engine: DummyRagEngine) -> None:
    state = ConversationState(
        role="Hiring Manager (nontechnical)",
        query="Can you share your LinkedIn profile?",
        chat_history=[{"role": "user", "content": "Hi"}],
    )

    nodes.classify_query(state)
    nodes.plan_actions(state)
    state.set_answer("Absolutely, here are the details.")

    nodes.apply_role_context(state, dummy_engine)

    assert any(action["type"] == "send_linkedin" for action in state.pending_actions)
    assert "LinkedIn profile" in (state.answer or "")
    assert "Would you like Noah to reach out?" in (state.answer or "")


def test_contact_request_sends_notifications(monkeypatch: pytest.MonkeyPatch, dummy_engine: DummyRagEngine) -> None:
    state = ConversationState(
        role="Hiring Manager (technical)",
        query="Please reach out to me about this opportunity",
        chat_history=[],
    )

    nodes.classify_query(state)
    nodes.plan_actions(state)
    state.stash("user_name", "Jordan Hiring")
    state.stash("user_email", "jordan@example.com")
    state.stash("user_phone", "+15551234567")
    state.set_answer("Technical overview ready.")

    resend = DummyResend()
    twilio = DummyTwilio()

    monkeypatch.setattr(nodes, "get_resend_service", lambda: resend)
    monkeypatch.setattr(nodes, "get_twilio_service", lambda: twilio)
    monkeypatch.setattr(nodes, "get_storage_service", lambda: DummyStorage())

    nodes.apply_role_context(state, dummy_engine)
    nodes.execute_actions(state)

    assert any(action["type"] == "notify_contact_request" for action in state.pending_actions)
    assert resend.sent[0]["from_name"] == "Jordan Hiring"
    assert twilio.alerts[0]["from_email"] == "jordan@example.com"
