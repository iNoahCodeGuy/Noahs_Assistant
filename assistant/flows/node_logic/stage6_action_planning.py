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

import logging
import re
from typing import Any
from assistant.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)
from assistant.flows.node_logic.stage2_query_classification import _is_data_display_request


def plan_actions(state: ConversationState) -> ConversationState:
    """Plan follow-up actions based on user role and query context.

    🎯 PURPOSE: Decide what extra stuff to add to the response (code snippets,
    resume offers, analytics, etc.) based on who's asking and what they want.

    📋 ACTION MATRIX (what you get for asking as each role):
    ┌─────────────────────────────┬──────────────┬─────────────────────────────┐
    │ Role                        │ Query Type   │ Actions Added               │
    ├─────────────────────────────┼──────────────┼─────────────────────────────┤
    │ Hiring Manager (technical)  │ technical    │ • code snippets             │
    │                             │              │ • metrics (cost/latency)    │
    │                             │              │ • architecture diagrams     │
    │                             │              │ • QA strategy (if asked)    │
    │                             │ career       │ • resume offer (after 2+    │
    │                             │              │   turns OR hiring signals)  │
    ├─────────────────────────────┼──────────────┼─────────────────────────────┤
    │ Hiring Manager (nontechnical)│ technical   │ • suggest role switch       │
    │                             │ career       │ • resume offer (same rules) │
    ├─────────────────────────────┼──────────────┼─────────────────────────────┤
    │ Software Developer          │ technical    │ • code snippets             │
    │                             │              │ • architecture diagrams     │
    │                             │              │ • import explanations       │
    │                             │              │ • QA strategy (if asked)    │
    ├─────────────────────────────┼──────────────┼─────────────────────────────┤
    │ Just looking around         │ mma          │ • MMA fight video link      │
    │                             │ any other    │ • fun facts about Noah      │
    ├─────────────────────────────┼──────────────┼─────────────────────────────┤
    │ Looking to confess crush    │ any          │ • confession collection     │
    └─────────────────────────────┴──────────────┴─────────────────────────────┘

    🔄 EXECUTION FLOW (3 phases):
    1. ALWAYS RUN: Cross-role concerns (hiring signals, explicit requests)
    2. ROLE-SPECIFIC: Main action planning (different for each role)
    3. ALWAYS RUN: Resume gating (teach first, sell later)

    💡 JUNIOR DEV TIP: Think of this as building a shopping list. We decide
    what to include, then action_execution.py actually does the work.

    Args:
        state: Current conversation state with role, query, history

    Returns:
        Updated state with pending_actions list populated
    """
    state["pending_actions"] = []

    # ============================================================================
    # PHASE 1: ALWAYS RUN - Cross-role concerns
    # ============================================================================
    # These happen for ALL roles, regardless of who's asking

    _detect_hiring_signals(state)  # Scan for "we're hiring", "looking for engineers"
    _check_explicit_resume_request(state)  # Detect "send me your resume"
    _handle_direct_requests(state)  # Handle resume/LinkedIn/contact requests
    _handle_edge_case_meta_teaching(state)  # Offer meta-teaching for edge cases

    # ============================================================================
    # PHASE 2: ROLE-SPECIFIC - Main action planning
    # ============================================================================
    # Different logic for each role type

    if state["role"] == "Hiring Manager (technical)":
        _plan_hm_technical_actions(state)
    elif state["role"] == "Hiring Manager (nontechnical)":
        _plan_hm_nontechnical_actions(state)
    elif state["role"] == "Software Developer":
        _plan_developer_actions(state)
    elif state["role"] == "Just looking around":
        _plan_explorer_actions(state)
    elif state["role"] == "Looking to confess crush":
        _plan_confession_actions(state)

    # ============================================================================
    # PHASE 3: ALWAYS RUN - Resume gating
    # ============================================================================
    # Offer resume only after we've demonstrated value (teach first, sell later)

    _maybe_offer_resume(state)

    return state


# ==============================================================================
# PHASE 1 HELPERS: Cross-role concerns (run for everyone)
# ==============================================================================


def _handle_direct_requests(state: ConversationState) -> None:
    """Handle explicit user requests that work for all roles.

    Junior dev: These are when users directly ask for something specific:
    - "Can I get your resume?"
    - "Show me your LinkedIn"
    - "Can you contact me about this role?"

    We detect these keywords and add the appropriate actions.
    """
    query = state.get("query", "")
    if not query:
        return

    lowered = query.lower()

    # Detect what they're asking for
    resume_requested = any(key in lowered for key in ["send resume", "email resume", "resume", "cv"])
    linkedin_requested = any(key in lowered for key in ["linkedin", "linked in", "link me", "profile"])
    github_requested = any(key in lowered for key in ["github", "git hub", "repository", "repo", "code repository"])
    contact_requested = any(key in lowered for key in ["reach out", "contact me", "call me", "follow up"])
    data_requested = _is_data_display_request(lowered)

    # Add actions based on what they asked for
    if resume_requested:
        state["pending_actions"].append({"type": "send_resume"})
        state["pending_actions"].append({"type": "ask_reach_out"})
        state["pending_actions"].append({"type": "notify_resume_sent"})
        state["offer_sent"] = True

    if linkedin_requested:
        state["pending_actions"].append({"type": "send_linkedin"})
        if not state.get("offer_sent"):
            state["pending_actions"].append({"type": "ask_reach_out"})
            state["offer_sent"] = True

    if github_requested:
        state["pending_actions"].append({"type": "send_github"})
        logger.info("GitHub link requested - added send_github action")

    if contact_requested:
        state["pending_actions"].append({"type": "notify_contact_request"})
        state["contact_requested"] = True

    # Check for analytics query type OR data display keywords
    analytics_query = state.get("query_type") == "analytics"
    if data_requested or analytics_query:
        state["pending_actions"].append({"type": "render_live_analytics"})
        state["data_display_requested"] = True
        logger.info("Analytics/data display requested - added render_live_analytics action")


def _handle_edge_case_meta_teaching(state: ConversationState) -> None:
    """Offer meta-teaching explanation for edge cases (technical users only).

    When an edge case is detected, offer to explain how the detection works
    to technical users. This turns edge cases into teaching moments.

    Args:
        state: Conversation state with edge_case_detected flag
    """
    if not state.get("edge_case_detected"):
        return

    # Only offer to technical users
    role_mode = state.get("role_mode", "")
    is_technical = role_mode in ["software_developer", "hiring_manager_technical"]

    if is_technical:
        edge_case_type = state.get("edge_case_type", "edge_case")
        state["pending_actions"].append({
            "type": "offer_edge_case_meta_teaching",
            "edge_case_type": edge_case_type
        })
        logger.info(f"Meta-teaching offer added for edge case: {edge_case_type}")


# ==============================================================================
# PHASE 2 HELPERS: Role-specific action planning
# ==============================================================================


def _plan_hm_technical_actions(state: ConversationState) -> None:
    """Plan actions for technical hiring managers.

    These folks want to see BOTH business value AND technical depth:
    - Code snippets (to verify technical chops)
    - Architecture diagrams (to understand system design)
    - Metrics (to see cost/performance awareness)
    - QA strategy (if they ask "how does this work?")

    Junior dev: Think of this as "impress the technical interviewer" mode.
    """
    query_type = state.get("query_type", "general")
    query = state.get("query", "")
    if not query:
        return

    lowered = query.lower()
    toggles = state.get("display_toggles", {})
    layout_variant = state.get("layout_variant", "mixed")
    menu_choice = state.get("menu_choice")

    # DEBUG: Log all variables for action planning
    logger.info(f"🎯 HM Technical Actions: query_type={query_type}, menu_choice={menu_choice}, role={state.get('role_mode')}")

    # Detect if they're asking "how does this work?"
    product_question = any(term in lowered for term in [
        "how does this work", "how does it work", "how is this built",
        "tell me about this", "explain this", "what's this"
    ]) or ("product" in lowered and any(word in lowered for word in ["how", "what", "explain"]))

    # Special handling for menu option 1 (full tech stack) - include architecture code
    if query_type == "menu_selection" and menu_choice == "1":
        logger.info(f"✅ Adding architecture code reference for menu option 1")
        state["pending_actions"].append({
            "type": "include_code_reference",
            "context": "architecture"
        })
    else:
        logger.debug(f"Architecture code NOT added: query_type={query_type}, menu_choice={menu_choice}")

    # Always add technical artifacts for technical queries
    if query_type == "technical" or toggles.get("code"):
        state["pending_actions"].append({"type": "include_code_reference"})

    if toggles.get("data"):
        state["pending_actions"].append({"type": "include_metrics_block"})

    if toggles.get("diagram"):
        diagram_type = "include_sequence_diagram" if layout_variant == "engineering" else "include_adaptation_diagram"
        state["pending_actions"].append({"type": diagram_type})

    # Add QA strategy for "how does this work" questions
    if product_question:
        state["pending_actions"].append({"type": "include_qa_strategy"})


def _plan_hm_nontechnical_actions(state: ConversationState) -> None:
    """Plan actions for nontechnical hiring managers.

    These folks are more business-focused. If they ask technical questions,
    we gently suggest switching to the technical HM role for better answers.

    Junior dev: This is like "wrong department, let me transfer you" logic.
    """
    query_type = state.get("query_type", "general")
    query = state.get("query", "")
    if not query:
        return

    lowered = query.lower()

    # Check if they're asking technical stuff
    code_display_requested = state.get("code_display_requested", False)
    import_explanation_requested = state.get("import_explanation_requested", False)
    product_question = any(term in lowered for term in [
        "how does this work", "how is this built", "architecture"
    ])

    # Suggest role switch if they're asking technical questions
    if query_type == "technical" or code_display_requested or import_explanation_requested or product_question:
        state["pending_actions"].append({"type": "suggest_technical_role_switch"})


def _plan_developer_actions(state: ConversationState) -> None:
    """Plan actions for software developers.

    These folks want ALL the technical details:
    - Code snippets (they want to see implementation)
    - Import explanations (understand dependencies)
    - Architecture diagrams (see system design)
    - QA strategy (understand testing approach)

    Junior dev: This is "show me everything" mode - developers want depth.
    """
    query_type = state.get("query_type", "general")
    query = state.get("query", "")
    if not query:
        return

    lowered = query.lower()
    toggles = state.get("display_toggles", {})
    layout_variant = state.get("layout_variant", "mixed")

    code_display_requested = state.get("code_display_requested", False)
    import_explanation_requested = state.get("import_explanation_requested", False)

    # Detect product questions
    product_question = any(term in lowered for term in [
        "how does this work", "how is this built", "architecture"
    ])

    # Add code and technical artifacts
    if code_display_requested or toggles.get("code"):
        state["pending_actions"].append({"type": "include_code_reference"})

    if import_explanation_requested:
        state["pending_actions"].append({"type": "explain_imports"})

    if toggles.get("data"):
        state["pending_actions"].append({"type": "include_metrics_block"})

    if toggles.get("diagram"):
        state["pending_actions"].append({"type": "include_sequence_diagram"})

    if product_question:
        state["pending_actions"].append({"type": "include_qa_strategy"})


def _plan_explorer_actions(state: ConversationState) -> None:
    """Plan actions for casual visitors ("Just looking around").

    These folks are here to browse, not hire. Keep it light and fun:
    - If they mention MMA/fighting → share Noah's MMA fight video
    - Otherwise → share fun facts about Noah

    Junior dev: This is "casual conversation" mode - no hard sell.
    """
    query_type = state.get("query_type", "general")
    chat_history = state.get("chat_history", [])

    if query_type == "mma":
        state["pending_actions"].append({"type": "share_mma_link"})
    elif len(chat_history) <= 2:
        # Only share fun facts on the first interaction, not every response
        state["pending_actions"].append({"type": "share_fun_facts"})


def _plan_confession_actions(state: ConversationState) -> None:
    """Plan actions for confession mode.

    This is a fun Easter egg mode - user wants to leave an anonymous message.
    We collect it and store it safely.

    Junior dev: This is the "fun mode" - just collect the message.
    """
    state["pending_actions"].append({"type": "collect_confession"})


# ==============================================================================
# PHASE 3 HELPER: Resume gating
# ==============================================================================


def _calculate_engagement_score(state: ConversationState) -> int:
    """Calculate engagement score based on user's exploration depth.

    This score measures how engaged the user is with Portfolia's content,
    which helps determine when to naturally offer resume/LinkedIn.

    Scoring:
    - topics_explored * 2: More topics = more engaged
    - depth_level * 3: Deeper questions = more serious interest
    - questions_asked * 1: More turns = sustained engagement
    - code_viewed * 5: Code interest = strong technical engagement
    - diagram_viewed * 3: Architecture interest = serious evaluation

    Args:
        state: Conversation state with session_memory

    Returns:
        Engagement score (higher = more engaged)
    """
    session_memory = state.get("session_memory", {})

    # Count topics explored
    topics = session_memory.get("topics", [])
    topics_explored = len(topics)

    # Get depth level
    depth_level = state.get("depth_level", 1)

    # Count turns from chat history
    chat_history = state.get("chat_history", [])
    questions_asked = len([m for m in chat_history
                          if (isinstance(m, dict) and m.get("role") == "user") or
                             (hasattr(m, "type") and m.type == "human")])

    # Check if user viewed code (from display_toggles or answered "show code")
    display_toggles = state.get("display_toggles", {})
    code_viewed = 1 if display_toggles.get("code") else 0

    # Check if user viewed diagrams
    diagram_viewed = 1 if display_toggles.get("diagram") else 0

    # Check if they asked about specific technical areas (high intent)
    query_lower = state.get("query", "").lower()
    technical_interest_keywords = ["implementation", "code", "how does", "show me", "walk through"]
    technical_bonus = 3 if any(kw in query_lower for kw in technical_interest_keywords) else 0

    # Calculate total score
    engagement_score = (
        topics_explored * 2 +
        depth_level * 3 +
        questions_asked * 1 +
        code_viewed * 5 +
        diagram_viewed * 3 +
        technical_bonus
    )

    # Store in state for analytics
    state["engagement_score"] = engagement_score

    return engagement_score


def _has_seen_value_demonstration(state: ConversationState) -> bool:
    """Check if user has seen enough value to warrant resume offer.

    Before offering resume, ensure user has seen at least:
    - One code snippet (demonstrates real engineering)
    - One real metric (demonstrates production quality)
    - One enterprise pattern (demonstrates business value)

    This prevents premature resume offers.

    Args:
        state: Conversation state

    Returns:
        True if user has seen sufficient value demonstration
    """
    session_memory = state.get("session_memory", {})

    # Check discussed topics for value indicators
    topics = session_memory.get("topics", [])
    discussed_files = session_memory.get("discussed_files", [])

    # Value checklist
    seen_code = len(discussed_files) > 0 or "implementation" in topics or "code" in topics
    seen_metrics = "cost" in topics or "observability" in topics or state.get("depth_level", 1) >= 2
    seen_enterprise = "enterprise" in topics or any("adapt" in t or "customer" in t for t in topics)

    # At minimum, need 2 of 3 value demonstrations
    value_count = sum([seen_code, seen_metrics, seen_enterprise])

    return value_count >= 2


def _maybe_offer_resume(state: ConversationState) -> None:
    """Offer resume only after demonstrating value (teach first, sell later).

    Resume gate opens when ANY of these conditions are met:
    1. Strong hiring signals detected (≥2 signals: "we're hiring", "need engineer", etc.)
    2. Deep engagement (depth_level ≥3, meaning they've asked follow-up questions)
    3. High engagement score (≥15, meaning sustained interaction across multiple topics)

    Additional requirement: Must have demonstrated value first.

    Junior dev: This is like "let them try before you sell" - we don't push
    the resume immediately, we wait until they're engaged.

    Only applies to hiring managers (technical and nontechnical).
    """
    # Skip if resume/linkedin/github was already explicitly requested this turn
    # (handled in _handle_direct_requests - don't offer what we're already providing)
    action_types = {a.get("type") for a in state.get("pending_actions", [])}
    if "send_resume" in action_types or "send_linkedin" in action_types or "send_github" in action_types:
        logger.debug("Resume/LinkedIn/GitHub already requested - skipping offer_resume_prompt")
        return

    # Get query defensively
    query = state.get("query", "")
    if not query:
        return

    lowered = query.lower()
    resume_requested = any(key in lowered for key in ["send resume", "email resume"])
    linkedin_requested = any(key in lowered for key in ["linkedin", "linked in", "link me"])

    # Don't offer if they already asked directly (handled in _handle_direct_requests)
    if resume_requested or linkedin_requested:
        return

    # Only offer to hiring managers
    if state["role"] not in ["Hiring Manager (technical)", "Hiring Manager (nontechnical)"]:
        return

    # Calculate engagement score
    engagement_score = _calculate_engagement_score(state)

    # Check if gate is open (strong signals OR deep engagement OR high engagement score)
    hiring_signals_strong = state.get("hiring_signals_strong", False)
    depth_level = state.get("depth_level", 1)

    # Gate conditions
    has_hiring_signals = hiring_signals_strong
    has_deep_engagement = depth_level >= 3
    has_high_engagement_score = engagement_score >= 15

    resume_gate_open = has_hiring_signals or has_deep_engagement or has_high_engagement_score

    # Additional check: must have demonstrated value
    has_value = _has_seen_value_demonstration(state)

    if resume_gate_open and has_value:
        state["pending_actions"].append({"type": "offer_resume_prompt"})
        logger.info(f"📧 Resume gate opened: signals={has_hiring_signals}, depth={depth_level}, engagement={engagement_score}")
    elif resume_gate_open and not has_value:
        logger.debug(f"Resume gate conditions met but value not yet demonstrated: engagement={engagement_score}")


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
    query = state.get("query", "")
    if not query:
        return

    query_lower = query.lower()
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
    query = state.get("query", "")
    if not query:
        return

    query_lower = query.lower()

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
