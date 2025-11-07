"""Action planning logic.

This module decides what follow-up actions to take based on:
- The user's role (hiring manager, developer, etc.)
- What they asked about (technical, career, data, etc.)
- How many turns into the conversation we are
- Detected hiring signals (merged from resume_distribution.py)

Actions planned here get executed later by action_execution.py.

Junior dev note: This is like a "shopping list" builder. We figure out
what we need to do (send resume, show code, offer LinkedIn) and add it
to a list. The actual work happens in a later step.

Merged logic:
- detect_hiring_signals: Passively scans queries for hiring intent
- handle_resume_request: Detects explicit resume requests (Mode 3)
"""

import re
from typing import Any
from src.state.conversation_state import ConversationState
from src.flows.node_logic.query_classification import _is_data_display_request


def plan_actions(state: ConversationState) -> ConversationState:
    """Decide what follow-up actions to take for this query.

    This looks at:
    - state.role: Who is the user? (Hiring Manager, Developer, etc.)
    - query_type: What did they asked about? (technical, career, data, etc.)
    - user_turns: How many times have they asked questions?
    - Special flags: Did they explicitly ask for code/resume/analytics?
    - Hiring signals: Passively detected hiring intent (merged from detect_hiring_signals)

    Then it builds a list of actions to execute, like:
    - "send_resume": Email Noah's resume
    - "render_live_analytics": Show data tables
    - "include_code_reference": Retrieve and display code snippet
    - "include_metrics_block": Show cost/latency/grounding snapshot

    Args:
        state: Current conversation state

    Returns:
        Updated state with pending_actions list populated, hiring_signals tracked

    Merged functionality:
        - Hiring signal detection (Mode 2 enabler)
        - Explicit resume request detection (Mode 3 trigger)
    """
    # MERGED: Detect hiring signals first (from detect_hiring_signals node)
    _detect_hiring_signals(state)
    
    # MERGED: Check for explicit resume requests (from handle_resume_request node)
    _check_explicit_resume_request(state)
    
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


# ============================================================================
# MERGED HIRING DETECTION LOGIC (from resume_distribution.py)
# ============================================================================


def _detect_hiring_signals(state: ConversationState) -> None:
    """Passively detect hiring signals in user query (internal helper, merged from detect_hiring_signals node).
    
    Scans for indicators that the user is actively hiring:
    - mentioned_hiring: "we're hiring", "looking for", "need someone"
    - described_role: "GenAI engineer", "ML specialist", specific title
    - team_context: "our team", "my team", organizational mention
    - asked_timeline: "when available", "start date", urgency mention
    - budget_mentioned: "salary range", "compensation", financial discussion
    
    Updates state with hiring_signals list and strength metadata (≥2 signals → hiring_signals_strong=True).
    """
    query_lower = state["query"].lower()
    hiring_signals = state.get("hiring_signals", [])

    # Pattern 1: Mentioned hiring explicitly
    hiring_patterns = [
        r'\b(hiring|looking for|need someone|recruiting|seeking)\b',
        r'\b(open position|job opening|role available)\b',
        r'\b(candidates|applicants)\b'
    ]
    if any(re.search(pattern, query_lower) for pattern in hiring_patterns):
        if "mentioned_hiring" not in hiring_signals:
            hiring_signals.append("mentioned_hiring")

    # Pattern 2: Described specific role
    role_patterns = [
        r'\b(engineer|developer|architect|specialist|lead)\b',
        r'\b(genai|gen ai|generative ai|ml|machine learning|ai)\b.*\b(engineer|developer|role)\b',
        r'\b(full.?stack|backend|frontend|data|software)\b.*\b(engineer|developer)\b'
    ]
    if any(re.search(pattern, query_lower) for pattern in role_patterns):
        if "described_role" not in hiring_signals:
            hiring_signals.append("described_role")

    # Pattern 3: Team context mentioned
    team_patterns = [
        r'\b(our team|my team|the team)\b',
        r'\b(organization|company|startup|enterprise)\b',
        r'\b(we are|we\'re)\b.*\b(building|creating|developing)\b'
    ]
    if any(re.search(pattern, query_lower) for pattern in team_patterns):
        if "team_context" not in hiring_signals:
            hiring_signals.append("team_context")

    # Pattern 4: Timeline/urgency mentioned
    timeline_patterns = [
        r'\b(when available|start date|immediately|asap)\b',
        r'\b(timeline|schedule|availability|available)\b',
        r'\b(notice period|can start|when.*start)\b'
    ]
    if any(re.search(pattern, query_lower) for pattern in timeline_patterns):
        if "asked_timeline" not in hiring_signals:
            hiring_signals.append("asked_timeline")

    # Pattern 5: Budget/compensation mentioned
    budget_patterns = [
        r'\b(salary|compensation|budget|rate)\b',
        r'\b(pay|payment|\$\d+k?|k salary)\b',
        r'\b(benefits|equity|stock)\b'
    ]
    if any(re.search(pattern, query_lower) for pattern in budget_patterns):
        if "budget_mentioned" not in hiring_signals:
            hiring_signals.append("budget_mentioned")

    # Update state with signals and strength metadata
    state["hiring_signals"] = hiring_signals
    strength = len(hiring_signals)
    state["hiring_signals_strength"] = strength
    state["hiring_signals_strong"] = strength >= 2


def _check_explicit_resume_request(state: ConversationState) -> None:
    """Detect explicit resume requests and trigger Mode 3 (internal helper, merged from handle_resume_request node).
    
    Scans for explicit requests like:
    - "Can I get your resume?"
    - "Send me your CV"
    - "Is Noah available?"
    - "Share Noah's resume"
    
    Sets state.resume_explicitly_requested = True, which triggers:
    1. Email collection flow (no qualification needed)
    2. Sends resume immediately after email provided
    3. Bypasses subtle mention logic (user asked directly)
    """
    query_lower = state["query"].lower()

    # Pattern 1: Direct resume request
    resume_patterns = [
        r'\b(can i get|send me|share|forward|email me)\b.*\b(resume|cv|curriculum vitae)\b',
        r'\b(resume|cv)\b.*\b(available|access|view|see)\b',
        r'\byour resume\b',
        r'\bnoah\'s resume\b'
    ]
    if any(re.search(pattern, query_lower) for pattern in resume_patterns):
        state["resume_explicitly_requested"] = True
        return

    # Pattern 2: Availability inquiry
    availability_patterns = [
        r'\bis noah available\b',
        r'\bcan noah\b.*\b(interview|meet|talk|discuss)\b',
        r'\bavailable for\b.*\b(hire|hiring|role|position|work)\b'
    ]
    if any(re.search(pattern, query_lower) for pattern in availability_patterns):
        state["resume_explicitly_requested"] = True
        return

    # Pattern 3: Contact request
    contact_patterns = [
        r'\bcontact noah\b',
        r'\bconnect with noah\b',
        r'\btalk to noah\b.*\b(about|regarding)\b.*\b(role|position|opportunity)\b'
    ]
    if any(re.search(pattern, query_lower) for pattern in contact_patterns):
        state["resume_explicitly_requested"] = True
        return
