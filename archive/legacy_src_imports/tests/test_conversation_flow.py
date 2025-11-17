import os
from typing import Any, Dict, List

import pytest

from src.state.conversation_state import ConversationState
from src.flows import conversation_nodes as nodes
from src.flows.node_logic import action_execution
from src.flows import content_blocks
from src.flows.conversation_flow import run_conversation_flow


class DummyResponseGenerator:
    def __init__(self, response: str):
        self._response = response

    def generate_basic_response(self, query: str, fallback_docs: List[str], chat_history: List[Dict[str, str]] | None = None) -> str:
        history_note = f" ({len(chat_history)} messages)" if chat_history else ""
        return f"{self._response} | {query}{history_note}"

    def generate_contextual_response(self, query: str, context: List[Any], role: str, chat_history: List[Dict[str, str]] | None = None, extra_instructions: str | None = None) -> str:
        history_note = f" ({len(chat_history)} messages)" if chat_history else ""
        return f"{self._response} | {query}{history_note}"


class DummyRagEngine:
    def __init__(self, retrieve_result: Dict[str, Any], response_text: str):
        self.retrieve_result = retrieve_result
        self.response_text = response_text

    def retrieve(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        return self.retrieve_result

    @property
    def response_generator(self) -> DummyResponseGenerator:
        return DummyResponseGenerator(self.response_text)

    def retrieve_with_code(self, query: str, role: str) -> Dict[str, Any]:
        return {
            "code_snippets": [
                {
                    "content": "def demo():\n    return 'demo'",
                    "citation": "src/demo.py",
                }
            ]
        }


@pytest.fixture
def base_state() -> ConversationState:
    state: ConversationState = {
        "role": "Hiring Manager (nontechnical)",
        "query": "Tell me about Noah's career",
        "chat_history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Can you share more?"},
        ],
        "answer": "",
        "retrieved_chunks": [],
        "pending_actions": [],
        "analytics_metadata": {},
        "hiring_signals": [],
        "resume_sent": False,
        "resume_explicitly_requested": False,
        "job_details": {},
    }
    nodes.initialize_conversation_state(state)
    nodes.classify_role_mode(state)
    return state


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
    assert base_state.get("query_type") == "career"


def test_retrieve_chunks_stores_context(base_state: ConversationState, dummy_engine: DummyRagEngine) -> None:
    nodes.initialize_conversation_state(base_state)
    nodes.retrieve_chunks(base_state, dummy_engine)
    assert len(base_state["retrieved_chunks"]) == 2
    assert base_state.get("retrieval_scores") == [0.95, 0.74]
    assert base_state["analytics_metadata"]["retrieval_count"] == 2


def test_generate_answer_uses_response_generator(base_state: ConversationState, dummy_engine: DummyRagEngine) -> None:
    nodes.initialize_conversation_state(base_state)
    nodes.retrieve_chunks(base_state, dummy_engine)
    nodes.generate_answer(base_state, dummy_engine)
    assert base_state["answer"]
    assert "Tell me about Noah's career" in base_state["answer"]


def test_greeting_short_circuit(dummy_engine: DummyRagEngine) -> None:
    state: ConversationState = {
        "role": "Software Developer",
        "query": "hi",
        "chat_history": [],
        "answer": "",
        "retrieved_chunks": [],
        "pending_actions": [],
        "analytics_metadata": {},
        "hiring_signals": [],
        "resume_sent": False,
        "resume_explicitly_requested": False,
        "job_details": {},
    }

    result = run_conversation_flow(state, dummy_engine, session_id="greet-1")
    assert result.get("is_greeting") is True
    assert "depth_level" not in result


def test_depth_controller_sets_engineering_layout(base_state: ConversationState) -> None:
    base_state["role"] = "Software Developer"
    base_state["query"] = "How does the LangGraph orchestration work?"
    nodes.classify_intent(base_state)
    nodes.depth_controller(base_state)
    nodes.display_controller(base_state)
    assert base_state["layout_variant"] == "engineering"
    assert base_state["display_toggles"]["code"] is True


def test_format_answer_variants(dummy_engine: DummyRagEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        content_blocks,
        "cost_latency_grounded_block",
        lambda: (["Average latency ~1.2 s", "Grounded responses ≈94%"], "Test metrics"),
    )
    state: ConversationState = {
        "role": "Software Developer",
        "query": "How does the RAG flow stay grounded?",
        "chat_history": [],
        "draft_answer": "Retrieval keeps things grounded. Sources: 1. kb",
        "pending_actions": [
            {"type": "include_sequence_diagram"},
            {"type": "include_code_reference"},
            {"type": "include_metrics_block"},
        ],
        "analytics_metadata": {},
        "hiring_signals": [],
        "depth_level": 3,
        "layout_variant": "engineering",
        "followup_variant": "engineering",
    }
    nodes.format_answer(state, dummy_engine)
    assert "Engineering Sequence" in state["answer"]
    assert "Where next?" in state["answer"]
    assert any("LangGraph" in line for line in state["followup_prompts"])

    business_state: ConversationState = {
        "role": "Hiring Manager (nontechnical)",
        "query": "What are the latency and cost numbers?",
        "chat_history": [],
        "draft_answer": "Here's the snapshot from analytics. Sources: 1. kb",
        "pending_actions": [
            {"type": "include_metrics_block"},
            {"type": "include_adaptation_diagram"},
        ],
        "analytics_metadata": {},
        "hiring_signals": [],
        "depth_level": 2,
        "layout_variant": "business",
        "followup_variant": "business",
    }
    nodes.format_answer(business_state, dummy_engine)
    assert "Cost · Latency · Grounding" in business_state["answer"]
    assert any("cost savings" in item for item in business_state["followup_prompts"])


def test_display_controller_heuristics() -> None:
    state: ConversationState = {
        "role": "Software Developer",
        "query": "How does LangGraph coordinate the nodes?",
        "chat_history": [],
        "pending_actions": [],
        "analytics_metadata": {},
        "hiring_signals": [],
    }
    nodes.initialize_conversation_state(state)
    nodes.classify_role_mode(state)
    nodes.classify_intent(state)
    nodes.depth_controller(state)
    nodes.display_controller(state)
    assert state["display_toggles"]["code"] is True

    business_state: ConversationState = {
        "role": "Hiring Manager (nontechnical)",
        "query": "What's the reliability and latency picture?",
        "chat_history": [],
        "pending_actions": [],
        "analytics_metadata": {},
        "hiring_signals": [],
    }
    nodes.initialize_conversation_state(business_state)
    nodes.classify_role_mode(business_state)
    nodes.classify_intent(business_state)
    nodes.depth_controller(business_state)
    nodes.display_controller(business_state)
    toggles = business_state["display_toggles"]
    assert toggles["data"] is True
    assert toggles["diagram"] is True


def test_resume_prompt_requires_signal_or_depth() -> None:
    state: ConversationState = {
        "role": "Hiring Manager (technical)",
        "query": "Tell me about the platform",
        "chat_history": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Can you explain the stack?"},
        ],
        "pending_actions": [],
        "analytics_metadata": {},
        "hiring_signals": [],
        "depth_level": 2,
        "layout_variant": "engineering",
    }
    nodes.initialize_conversation_state(state)
    nodes.classify_role_mode(state)
    nodes.classify_intent(state)
    nodes.plan_actions(state)
    assert "offer_resume_prompt" not in {a["type"] for a in state["pending_actions"]}

    state["hiring_signals_strong"] = True
    nodes.plan_actions(state)
    assert "offer_resume_prompt" in {a["type"] for a in state["pending_actions"]}


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

        @staticmethod
        def log_retrieval(data):
            logged_payloads.append({"retrieval_logged": True, "grounded": data.grounded})

    # Import and monkeypatch in logging_nodes where supabase_analytics is actually used
    from src.flows.node_logic import logging_nodes
    monkeypatch.setattr(logging_nodes, "supabase_analytics", DummyAnalytics)

    base_state["answer"] = "Career summary"
    base_state["grounding_status"] = "ok"
    nodes.classify_query(base_state)
    result = nodes.log_and_notify(base_state, session_id="test-session", latency_ms=123)

    # log_and_notify returns state with analytics metadata stored in analytics_metadata dict
    assert result["analytics_metadata"]["message_id"] == 99
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

        @staticmethod
        def log_retrieval(data):
            logged["retrieval"] = {
                "grounded": data.grounded,
                "scores": data.scores,
            }

    from src.flows.node_logic import logging_nodes
    monkeypatch.setattr(logging_nodes, "supabase_analytics", DummyAnalytics)

    state = run_conversation_flow(
        state=base_state,
        rag_engine=dummy_engine,
        session_id="abc123",
    )

    assert "Noah has a strong track record." in state["answer"]
    assert "Where next?" in state["answer"]
    assert state.get("followup_prompts")

    assert state["retrieved_chunks"]
    assert {a["type"] for a in state.get("pending_actions", [])} == {"include_adaptation_diagram"}
    assert state["analytics_metadata"]["message_id"] == 101
    assert logged["query_type"] == "career"
    assert isinstance(logged["latency_ms"], int)


def test_execute_actions_send_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    state: ConversationState = {
        "role": "Hiring Manager (technical)",
        "query": "Please email your resume",
        "chat_history": [],
        "answer": "",
        "retrieved_chunks": [],
        "pending_actions": [
            {"type": "send_resume", "email": "hiring@company.com", "name": "Hiring Team"},
            {"type": "notify_resume_sent"},
        ],
        "analytics_metadata": {},
        "hiring_signals": [],
        "resume_sent": False,
        "resume_explicitly_requested": False,
        "job_details": {},
        "user_email": "hiring@company.com",
    }

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

        def send_contact_notification(self, **payload) -> Dict[str, Any]:
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

    # Reset cached services before monkeypatching (important for global executor)
    action_execution._action_executor.reset_services()

    monkeypatch.setattr(action_execution, "get_storage_service", lambda: dummy_storage)
    monkeypatch.setattr(action_execution, "get_resend_service", lambda: dummy_resend)
    monkeypatch.setattr(action_execution, "get_twilio_service", lambda: dummy_twilio)

    nodes.execute_actions(state)

    assert dummy_storage.calls == [("resumes/noah_resume.pdf", 86400)]
    assert dummy_resend.sent == [
        ("hiring@company.com", "Hiring Team", "https://signed.example.com/resume.pdf", None)
    ]
    assert state["analytics_metadata"]["resume_email_status"] == "sent"
    assert dummy_twilio.alerts[0]["message_preview"].startswith("Resume dispatched")


def test_execute_actions_contact_notifications(monkeypatch: pytest.MonkeyPatch) -> None:
    state: ConversationState = {
        "role": "Hiring Manager (nontechnical)",
        "query": "Please reach out to me about the role",
        "chat_history": [],
        "answer": "",
        "retrieved_chunks": [],
        "pending_actions": [
            {"type": "notify_contact_request", "urgent": True},
        ],
        "analytics_metadata": {},
        "hiring_signals": [],
        "resume_sent": False,
        "resume_explicitly_requested": False,
        "job_details": {},
        "user_name": "Alex",
        "user_email": "alex@example.com",
        "user_phone": "+15551234",
    }

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

    # Reset cached services before monkeypatching (important for global executor)
    action_execution._action_executor.reset_services()

    monkeypatch.setattr(action_execution, "get_resend_service", lambda: dummy_resend)
    monkeypatch.setattr(action_execution, "get_twilio_service", lambda: dummy_twilio)

    nodes.execute_actions(state)

    assert dummy_resend.notifications[0]["from_name"] == "Alex"
    assert dummy_resend.notifications[0]["user_role"] == "Hiring Manager (nontechnical)"
    assert dummy_twilio.alerts[0]["is_urgent"] is True
