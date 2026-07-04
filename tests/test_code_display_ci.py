"""CI-style test for action planning + execution on explicit resource requests.

Historical note: this file used to test the role-based resume distribution flow
(resume email via Resend, "Resume dispatched" SMS via Twilio, contact-request
notifications built from contact info stashed in state). Those behaviors were
removed with the universal pipeline: nothing populates user_email/user_name in
state anymore, and direct contact requests are intercepted by the stage-1
intent router (contact form + pipeline halt) before plan_actions ever runs.
Those tests were deleted.

What survives is the LinkedIn resource request, which still runs end-to-end
without external services:
    classify_query marks the query as an action_request
    → plan_actions queues send_linkedin + ask_reach_out
    → format_answer renders the LinkedIn URL and reach-out prompt
    → execute_actions records the linkedin_offer analytics flag.
"""

from typing import Any, Dict

from assistant.state.conversation_state import ConversationState
from assistant.flows import conversation_nodes as nodes
from assistant.flows.node_logic.stage6_formatting_nodes import LINKEDIN_URL


class DummyRagEngine:
    """Minimal stub; format_answer's action_request path never touches it."""

    def retrieve_with_code(self, query: str, role: str | None = None) -> Dict[str, Any]:
        return {"code_snippets": [], "has_code": False}


def test_linkedin_request_prompts_follow_up() -> None:
    state: ConversationState = {
        "role": "Hiring Manager (nontechnical)",
        "query": "Can you share your LinkedIn profile?",
        "chat_history": [{"role": "user", "content": "Hi"}],
    }

    state = nodes.classify_query(state)
    assert state.get("query_type") == "action_request"

    state = nodes.plan_actions(state)
    action_types = {action["type"] for action in state["pending_actions"]}
    assert "send_linkedin" in action_types
    assert "ask_reach_out" in action_types

    # format_answer's action_request path renders the resource links directly
    state["draft_answer"] = "Absolutely, here are the details."
    state.update(nodes.format_answer(state, DummyRagEngine()))

    answer = state.get("answer") or ""
    assert LINKEDIN_URL in answer
    assert "Would you like Noah to reach out directly?" in answer

    # execute_actions handles send_linkedin without any external service:
    # it only records the offer in analytics metadata.
    state = nodes.execute_actions(state)
    assert state["analytics_metadata"]["linkedin_offer"] is True
