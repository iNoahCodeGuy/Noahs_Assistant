"""Role-specific conversation nodes.

This module collects onboarding and follow-up helpers that are exclusive to
certain personas. The first implementation focuses on technical hiring
managers, giving them a guided menu that emphasises enterprise relevance.

Functions intentionally return the shared ConversationState so they can slot
into the LangGraph pipeline without additional wrappers.
"""

from __future__ import annotations

from typing import Dict

from assistant.state.conversation_state import ConversationState


_HM_TECH_MENU = (
    "What would you like first?\n"
    "1️⃣ Technical backend walkthrough\n"
    "2️⃣ Knowledge-base design\n"
    "3️⃣ Enterprise adaptation & scale\n"
    "4️⃣ Relevant certifications Noah has earned\n"
    "5️⃣ Real-world agentic system examples"
)

_SELECTION_MAP = {
    "1": "1",
    "1️⃣": "1",
    "2": "2",
    "2️⃣": "2",
    "3": "3",
    "3️⃣": "3",
    "4": "4",
    "4️⃣": "4",
    "5": "5",
    "5️⃣": "5",
}


def onboard_hiring_manager_technical(state: ConversationState) -> ConversationState:
    """Deliver the technical hiring manager onboarding prompt.

    The onboarding message frames Portfolia's technical scope and invites the
    user to select where they want to dive in first. It also flips enterprise
    toggles so downstream answers highlight scale, governance, and business
    value.
    """
    session_memory = state.setdefault("session_memory", {})
    persona_hints: Dict[str, str] = session_memory.setdefault("persona_hints", {})
    persona_hints["role_mode"] = "hiring_manager_technical"
    persona_hints["hm_technical_onboarded"] = True

    state["role"] = "Hiring Manager (technical)"
    state["role_mode"] = "hiring_manager_technical"
    state["role_confidence"] = max(state.get("role_confidence", 0.0), 1.0)
    state["is_greeting"] = False
    state["relate_to_enterprise"] = True
    state["awaiting_hm_tech_menu"] = True

    welcome = (
        "Welcome! Since you’re a technical hiring manager, I can show you what’s in my knowledge base—\n"
        "• Noah’s architecture and LangGraph orchestration\n"
        "• The AI pipeline (retrievers, vector store, evaluation)\n"
        "• How agentic assistants like me translate into enterprise value\n\n"
        f"{_HM_TECH_MENU}"
    )

    state["answer"] = welcome
    state["pipeline_halt"] = True
    return state


def explain_enterprise_adaptation(state: ConversationState) -> ConversationState:
    """Explain how Portfolia's architecture maps to enterprise rollouts."""
    message = (
        "Here’s how Portfolia scales inside an enterprise stack:\n\n"
        "• **Multi-tenant isolation:** Supabase schemas + scoped API keys keep customer data sealed.\n"
        "• **Governance hooks:** Every retrieval/generation call is logged through LangSmith + Supabase analytics.\n"
        "• **Operational guardrails:** Feature flags toggle advanced behaviours (code display, action execution) per tenant.\n"
        "• **Rollout strategy:** Blue/green Streamlit or Vercel deployments with LangSmith shadow evaluation before full release.\n\n"
    )

    if state.get("relate_to_enterprise"):
        message += (
            "Enterprise tie-in: This pattern lets hiring leaders see reliability data along with candidate summaries, so the assistant stays audit-ready while accelerating evaluation cycles.\n\n"
        )

    message += _HM_TECH_MENU

    state["answer"] = message
    state["pipeline_halt"] = True
    state["awaiting_hm_tech_menu"] = True
    state["is_greeting"] = False
    return state


def show_certifications(state: ConversationState) -> ConversationState:
    """List Noah's technical certificates and link them back to enterprise readiness."""
    message = (
        "Noah keeps sharpening the fundamentals and applied skills through focused credentials:\n\n"
        "• DeepLearning.AI — *LangChain for LLM Application Development*\n"
        "• DeepLearning.AI — *Building Systems with the ChatGPT API*\n"
        "• AWS Machine Learning Foundations (Udacity x AWS)\n"
        "• Google Cloud Skills Boost — *Generative AI for Developers*\n\n"
    )

    if state.get("relate_to_enterprise"):
        message += (
            "Enterprise tie-in: These programs emphasise production-grade practices (observability, evaluation, cost controls), so the same checkpoints show up in my LangGraph flow when we discuss candidate fit.\n\n"
        )

    message += _HM_TECH_MENU

    state["answer"] = message
    state["pipeline_halt"] = True
    state["awaiting_hm_tech_menu"] = True
    state["is_greeting"] = False
    return state


def show_enterprise_pattern_example(state: ConversationState) -> ConversationState:
    """Describe a support-copilot style deployment that mirrors Portfolia."""
    message = (
        "Picture a Support Copilot built on the same stack:\n\n"
        "1. **LangGraph orchestration** routes each ticket through retrieval, troubleshooting flows, and escalation guards.\n"
        "2. **Supabase pgvector** houses product knowledge, policy notes, and past resolutions for grounding.\n"
        "3. **CRM hooks (Salesforce or HubSpot)** log the assistant’s draft response and capture customer sentiment.\n"
        "4. **Action bus** triggers Slack or PagerDuty alerts when human approval is required.\n\n"
    )

    if state.get("relate_to_enterprise"):
        message += (
            "Enterprise tie-in: Teams see faster handle times and consistent compliance, while leadership gets LangSmith dashboards for deflection rate, CSAT, and content drift.\n\n"
        )

    message += _HM_TECH_MENU

    state["answer"] = message
    state["pipeline_halt"] = True
    state["awaiting_hm_tech_menu"] = True
    state["is_greeting"] = False
    return state


def handle_hm_technical_menu_selection(state: ConversationState) -> ConversationState:
    """Process numeric selections after the technical HM onboarding menu."""
    selection_raw = state.get("query", "").strip().lower()
    selection = _SELECTION_MAP.get(selection_raw) or _SELECTION_MAP.get(selection_raw.replace(" ", ""))

    if not selection:
        if selection_raw:
            # Treat any free-form follow-up as a normal query and exit the menu loop
            state.pop("pipeline_halt", None)
            state["awaiting_hm_tech_menu"] = False
            return state

        state["answer"] = (
            "I didn’t catch that selection. Please choose 1–5 so I can surface the right view.\n\n"
            f"{_HM_TECH_MENU}"
        )
        state["pipeline_halt"] = True
        state["awaiting_hm_tech_menu"] = True
        return state

    if selection in {"1", "2"}:
        state.pop("pipeline_halt", None)
        state["awaiting_hm_tech_menu"] = False
        if selection == "1":
            state["query"] = "Please walk me through Portfolia's technical backend architecture."
        else:
            state["query"] = "Show me how Noah structured the knowledge base for Portfolia."
        return state

    if selection == "3":
        return explain_enterprise_adaptation(state)
    if selection == "4":
        return show_certifications(state)
    if selection == "5":
        return show_enterprise_pattern_example(state)

    # Fallback (should not hit due to mapping)
    return state


def route_hiring_manager_technical(state: ConversationState) -> ConversationState:
    """Entry point for technical hiring manager onboarding and menu handling."""
    persona_hints = state.setdefault("session_memory", {}).setdefault("persona_hints", {})

    if state.get("awaiting_hm_tech_menu"):
        return handle_hm_technical_menu_selection(state)

    if state.get("role_mode") == "hiring_manager_technical" and not persona_hints.get("hm_technical_onboarded"):
        return onboard_hiring_manager_technical(state)

    return state
