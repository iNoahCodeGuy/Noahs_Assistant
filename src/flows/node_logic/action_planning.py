"""Action planning logic.

This module decides what follow-up actions to take based on:
- The user's role (hiring manager, developer, etc.)
- What they asked about (technical, career, data, etc.)
- How many turns into the conversation we are

Actions planned here get executed later by action_execution.py.

Junior dev note: This is like a "shopping list" builder. We figure out
what we need to do (send resume, show code, offer LinkedIn) and add it
to a list. The actual work happens in a later step.
"""

from typing import Any
from src.state.conversation_state import ConversationState
from src.flows.node_logic.query_classification import _is_data_display_request


def plan_actions(state: ConversationState) -> ConversationState:
    """Decide what follow-up actions to take for this query.

    This looks at:
    - state.role: Who is the user? (Hiring Manager, Developer, etc.)
    - query_type: What did they ask about? (technical, career, data, etc.)
    - user_turns: How many times have they asked questions?
    - Special flags: Did they explicitly ask for code/resume/analytics?

    Then it builds a list of actions to execute, like:
    - "send_resume": Email Noah's resume
    - "render_live_analytics": Show data tables
    - "include_code_reference": Retrieve and display code snippet
    - "include_metrics_block": Show cost/latency/grounding snapshot

    Args:
        state: Current conversation state

    Returns:
        Updated state with pending_actions list populated
    """
    # Clear any old actions from previous turns
    state["pending_actions"] = []

    # Get context about the query
    query_type = state.get("query_type", "general")
    lowered = state["query"].lower()
    user_turns = sum(1 for message in state["chat_history"] if message.get("role") == "user")

    # Presentation metadata from depth/display controllers
    toggles = state.get("display_toggles", {})
    layout_variant = state.get("layout_variant", "mixed")

    # Check for special request flags
    code_display_requested = state.get("code_display_requested", False)
    import_explanation_requested = state.get("import_explanation_requested", False)

    # Helper to add actions to the list
    def add_action(action_type: str, **extras: Any) -> None:
        state["pending_actions"].append({"type": action_type, **extras})

    # Detect specific user requests
    resume_requested = any(key in lowered for key in ["send resume", "email resume", "resume", "cv"])
    linkedin_requested = any(key in lowered for key in ["linkedin", "link me", "profile"])
    contact_requested = any(key in lowered for key in ["reach out", "contact me", "call me", "follow up"])

    # Handle data display requests (show analytics/metrics)
    if _is_data_display_request(lowered):
        add_action("render_live_analytics")
        state["data_display_requested"] = True

    # Handle direct requests for resources
    if resume_requested:
        add_action("send_resume")
        add_action("ask_reach_out")
        add_action("notify_resume_sent")
        state["offer_sent"] = True

    if linkedin_requested:
        add_action("send_linkedin")
        if not state.get("offer_sent"):
            add_action("ask_reach_out")
            state["offer_sent"] = True

    if contact_requested:
        add_action("notify_contact_request")
        state["contact_requested"] = True

    # Detect product/how-it-works questions
    product_question = any(term in lowered for term in [
        "how does this work", "how does it work", "how does", "how is this",
        "what is this", "what does this", "explain this",
        "how is this built", "tell me about this", "what's this"
    ]) or ("product" in lowered and any(word in lowered for word in ["how", "what", "explain", "work"]))

    # Handle code and import explanation requests
    if code_display_requested and state["role"] in ["Hiring Manager (technical)", "Software Developer"]:
        add_action("include_code_reference")

    if import_explanation_requested:
        add_action("explain_imports")

    # Use display toggles to drive supporting artifacts
    if toggles.get("code"):
        add_action("include_code_reference")

    if toggles.get("data"):
        add_action("include_metrics_block")

    if toggles.get("diagram"):
        if layout_variant == "engineering":
            add_action("include_sequence_diagram")
        else:
            add_action("include_adaptation_diagram")

    # Add QA strategy for product questions (all technical roles)
    if product_question and state["role"] in ["Hiring Manager (technical)", "Software Developer"]:
        add_action("include_qa_strategy")

    # Role-specific action planning
    if state["role"] == "Hiring Manager (nontechnical)":
        if query_type == "technical" or code_display_requested or import_explanation_requested or product_question:
            add_action("suggest_technical_role_switch")

    elif state["role"] == "Just looking around":
        if query_type == "mma":
            add_action("share_mma_link")
        else:
            add_action("share_fun_facts")

    elif state["role"] == "Looking to confess crush":
        add_action("collect_confession")

    # Resume gating (teach first, sell later)
    resume_gate_open = state.get("hiring_signals_strong", False) or state.get("depth_level", 1) >= 3
    if (
        not resume_requested
        and not linkedin_requested
        and resume_gate_open
        and state["role"] in ["Hiring Manager (technical)", "Hiring Manager (nontechnical)"]
    ):
        add_action("offer_resume_prompt")

    return state
