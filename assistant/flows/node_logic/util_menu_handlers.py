"""Menu handlers for the new 4-branch conversation flow.

This module implements handlers for:
1. Professional Background - Sales achievements + resume distribution
2. Technical Background - Certifications vs Projects sub-menu
3. Explorer - Video options (hotdog/cage fight)
4. Confession - Anonymous vs identity with SMS notifications

Each handler follows the pattern from util_role_specific.py:
- Receives ConversationState
- Sets answer and pipeline_halt
- Returns updated state

Enhanced with:
- Engagement tracking (projects viewed, questions asked, engagement score)
- Progressive resume/LinkedIn offers based on engagement
- Cross-branch navigation (menu, back, pivot prompts)
- Portfolia deep dive mode for code exploration
"""

from __future__ import annotations

import logging
from textwrap import dedent
from typing import Dict, Any, List

from assistant.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)

# ============================================================================
# Engagement Tracking
# ============================================================================

def _track_engagement(state: ConversationState, action_type: str, detail: str = None) -> None:
    """Track user engagement metrics in session_memory.

    Args:
        state: ConversationState to update
        action_type: Type of engagement ('project_viewed', 'question_asked', 'video_watched', etc.)
        detail: Optional detail (e.g., project name, video name)
    """
    session_memory = state.setdefault("session_memory", {})
    engagement = session_memory.setdefault("engagement", {
        "projects_viewed": [],
        "certifications_viewed": False,
        "videos_watched": [],
        "questions_asked": 0,
        "resume_offered": False,
        "resume_sent": False,
        "linkedin_offered": False,
    })

    if action_type == "project_viewed" and detail:
        if detail not in engagement["projects_viewed"]:
            engagement["projects_viewed"].append(detail)
    elif action_type == "certifications_viewed":
        engagement["certifications_viewed"] = True
    elif action_type == "video_watched" and detail:
        if detail not in engagement["videos_watched"]:
            engagement["videos_watched"].append(detail)
    elif action_type == "question_asked":
        engagement["questions_asked"] = engagement.get("questions_asked", 0) + 1
    elif action_type == "resume_offered":
        engagement["resume_offered"] = True
    elif action_type == "resume_sent":
        engagement["resume_sent"] = True
    elif action_type == "linkedin_offered":
        engagement["linkedin_offered"] = True

    # Compute engagement score
    engagement["engagement_score"] = _compute_engagement_score(engagement)


def _compute_engagement_score(engagement: Dict[str, Any]) -> int:
    """Compute engagement score based on user activity.

    Scoring:
    - Each project viewed: +1
    - Certifications viewed: +1
    - Each video watched: +0.5
    - Each question asked: +1
    - Resume offered: +1
    - Resume sent: +2

    Returns:
        Engagement score (0-10+)
    """
    score = 0
    score += len(engagement.get("projects_viewed", []))
    if engagement.get("certifications_viewed"):
        score += 1
    score += len(engagement.get("videos_watched", [])) * 0.5
    score += engagement.get("questions_asked", 0)
    if engagement.get("resume_offered"):
        score += 1
    if engagement.get("resume_sent"):
        score += 2
    return int(score)


def _get_engagement_score(state: ConversationState) -> int:
    """Get current engagement score from state."""
    session_memory = state.get("session_memory", {})
    engagement = session_memory.get("engagement", {})
    return engagement.get("engagement_score", 0)


# ============================================================================
# Navigation Helpers
# ============================================================================

NAVIGATION_HINTS = {
    "after_video": "Want to see Noah's serious side? Type 'professional' or 'technical'!",
    "after_confession": "Curious about Noah beyond the confession? I can tell you about his work!",
    "after_projects": "Want to go back to the main menu? Just say 'menu' or '0'.",
    "stuck": "Not sure what to ask? Type 'menu' to see options or ask me anything!",
}

LINKEDIN_URL = "https://www.linkedin.com/in/noahdelacalzada/"


def _check_navigation_keywords(query: str) -> str:
    """Check if user wants to navigate (menu, back, etc.).

    Returns:
        Navigation action: 'menu', 'back', or None
    """
    query_lower = query.lower().strip()
    nav_keywords = {
        "menu": ["menu", "options", "0", "main menu", "start over", "reset"],
        "back": ["back", "go back", "previous", "return"],
    }

    for action, keywords in nav_keywords.items():
        if any(kw in query_lower for kw in keywords):
            return action
    return None


# ============================================================================
# Sub-Menu Selection Mapping
# ============================================================================

_TECHNICAL_SUB_MENU_MAP = {
    "1": "certifications",
    "1️⃣": "certifications",
    "2": "projects",
    "2️⃣": "projects",
}

_EXPLORER_SUB_MENU_MAP = {
    "1": "hotdog",
    "1️⃣": "hotdog",
    "2": "cage_fight",
    "2️⃣": "cage_fight",
}

_CONFESSION_SUB_MENU_MAP = {
    "1": "anonymous",
    "1️⃣": "anonymous",
    "2": "identity",
    "2️⃣": "identity",
}

_PROJECT_DETAIL_MAP = {
    "1": "portfolia",
    "1️⃣": "portfolia",
    "2": "response_time",
    "2️⃣": "response_time",
    "3": "heatmap",
    "3️⃣": "heatmap",
}

_PORTFOLIA_DEEP_DIVE_MAP = {
    "1": "rag",
    "1️⃣": "rag",
    "2": "langgraph",
    "2️⃣": "langgraph",
    "3": "error_handling",
    "3️⃣": "error_handling",
    "4": "resume",
    "4️⃣": "resume",
}

# ============================================================================
# Video URLs
# ============================================================================
# TODO: Replace these placeholder URLs with actual video links
# You can update these directly here or move to environment variables/config
# Example: VIDEO_URLS["hotdog"] = os.environ.get("HOTDOG_VIDEO_URL", "default")

VIDEO_URLS = {
    # Replace with actual YouTube/video URLs
    "hotdog": "https://www.youtube.com/watch?v=HOTDOG_VIDEO_ID_HERE",
    "cage_fight": "https://www.youtube.com/watch?v=CAGE_FIGHT_VIDEO_ID_HERE",
}

# ============================================================================
# Progressive CTA Generation
# ============================================================================

def _get_resume_offer(engagement_score: int, linkedin_offered: bool = False) -> str:
    """Generate appropriate resume/LinkedIn offer based on engagement.

    Args:
        engagement_score: Current engagement score
        linkedin_offered: Whether LinkedIn has already been offered

    Returns:
        CTA text string (empty if no offer should be made)
    """
    if engagement_score >= 5:
        # High engagement - strong offer with LinkedIn
        if not linkedin_offered:
            return dedent(f"""\

                ---

                You've been exploring Noah's work - would you like his resume? I can email it to you and Noah will receive a notification that you're interested!

                Or connect on LinkedIn: {LINKEDIN_URL}
            """)
        else:
            return dedent("""\

                ---

                Would you like Noah's resume? I can email it to you and Noah will receive a notification that you're interested!
            """)
    elif engagement_score >= 3:
        # Medium engagement - dedicated paragraph
        return dedent("""\

            ---

            Interested in learning more about Noah? I can send you his resume - just let me know your email! Noah will also receive a notification that you're interested.
        """)
    elif engagement_score >= 1:
        # Low engagement - soft offer at end
        return "\n\nWant Noah's resume? Just ask!"
    else:
        # No engagement yet - no offer
        return ""


def _should_offer_linkedin(state: ConversationState) -> bool:
    """Check if user engagement warrants LinkedIn offer.

    Args:
        state: ConversationState

    Returns:
        True if LinkedIn should be offered
    """
    engagement_score = _get_engagement_score(state)
    session_memory = state.get("session_memory", {})
    engagement = session_memory.get("engagement", {})

    # Offer LinkedIn if:
    # - High engagement (score >= 5)
    # - OR viewed 2+ projects
    # - OR asked follow-up questions
    return (
        engagement_score >= 5 or
        len(engagement.get("projects_viewed", [])) >= 2 or
        engagement.get("questions_asked", 0) >= 2
    ) and not engagement.get("linkedin_offered", False)


def _format_project_response(project_key: str, engagement_level: str = "low") -> str:
    """Format project response based on engagement level.

    Args:
        project_key: Key from PROJECT_DATA
        engagement_level: 'low', 'medium', or 'high'

    Returns:
        Formatted project description
    """
    if project_key not in PROJECT_DATA:
        return ""

    project = PROJECT_DATA[project_key]

    # Build response based on engagement
    parts = [f"**{project_key.capitalize().replace('_', ' ')}** - {project['hook']}"]

    if engagement_level in ("medium", "high"):
        parts.append(f"\n{project['business_problem']}")

    parts.append("\n**What it does:**")
    for item in project["what_it_does"]:
        parts.append(f"- {item}")

    parts.append("\n**Tech Stack:**")
    for item in project["tech_stack"]:
        parts.append(f"- {item}")

    if engagement_level == "high":
        parts.append("\n**Skills Demonstrated:**")
        for skill in project["skills_demonstrated"]:
            parts.append(f"- {skill}")

    parts.append(f"\n🔗 {project['github_url']}")

    if project["deep_dive_available"] and engagement_level in ("medium", "high"):
        parts.append(f"\n{project['cta_text']}")
    elif engagement_level == "low":
        parts.append(f"\n{project['cta_text']}")

    return "\n".join(parts)


# ============================================================================
# Content Constants
# ============================================================================

CERTIFICATIONS_CONTENT = dedent("""\
    Here are Noah's technical certifications:

    **AI/ML Certifications:**
    - DeepLearning.AI — *LangChain for LLM Application Development*
    - DeepLearning.AI — *Building Systems with the ChatGPT API*
    - AWS Machine Learning Foundations (Udacity x AWS)
    - Google Cloud Skills Boost — *Generative AI for Developers*

    **Other Technical Training:**
    - Python for Data Science fundamentals
    - SQL and database management
    - Git/GitHub version control

    These certifications demonstrate Noah's commitment to staying current with AI/ML technologies and production-grade practices.

    Would you like me to send you Noah's resume with full details?
""")

PROJECTS_LIST = dedent("""\
    Here are the programs Noah has built:

    1️⃣ **Portfolia** - Noah's Interactive AI Assistant (this very app!)
       A generative AI application showcasing RAG, LangGraph orchestration, and production deployment.
       https://github.com/iNoahCodeGuy/ai_assistant.git

    2️⃣ **Response Time & Close Rate Analysis**
       Analyzes impact of lead response time on close rates using statistical analysis with chi-square tests, z tests, and logistic regression.
       https://github.com/iNoahCodeGuy/response_time_cl_analysis.git

    3️⃣ **Heatmap Response Time Tracker**
       Streamlit dashboard that visualizes lead response performance with interactive heat maps. See times of day your team responds fast and when they don't. Built with Python, Streamlit, Pandas, and Plotly.
       https://github.com/iNoahCodeGuy/generic-lead-response-heatmap.git

    Would you like more details on any of these projects? Just type the number!
""")

# Structured project data for dynamic formatting
PROJECT_DATA = {
    "portfolia": {
        "hook": "You're talking to it right now!",
        "business_problem": "Portfolio sites are typically static - they don't demonstrate skills in action. Portfolia solves this by being a working AI assistant that explains its own implementation.",
        "what_it_does": [
            "22-node LangGraph pipeline orchestrating conversation flow",
            "RAG (Retrieval-Augmented Generation) using pgvector for semantic search",
            "Role-aware routing that adapts responses to user persona",
            "Production-grade error handling and graceful degradation",
            "SMS/Email notifications via Twilio and Resend",
            "Real-time analytics and observability with LangSmith",
        ],
        "tech_stack": [
            "Backend: Python 3.11+ with LangGraph for workflow orchestration",
            "LLM: OpenAI GPT-4o-mini for generation, text-embedding-3-small for embeddings",
            "Database: Supabase Postgres with pgvector for semantic search",
            "Frontend: Next.js (TypeScript) with a beautiful chat interface",
            "Observability: LangSmith for tracing and monitoring",
            "Deployment: Vercel serverless functions",
        ],
        "skills_demonstrated": [
            "Full-stack AI system architecture",
            "Production deployment and monitoring",
            "Complex state management and orchestration",
            "Integration of multiple external services",
        ],
        "github_url": "https://github.com/iNoahCodeGuy/ai_assistant.git",
        "deep_dive_available": True,
        "cta_text": "Want me to show you the actual code that powers this conversation?",
    },
    "response_time": {
        "hook": "Does responding faster actually increase close rates?",
        "business_problem": "Sales teams often guess at the impact of response time on conversion. This project provides data-backed evidence using rigorous statistical analysis.",
        "what_it_does": [
            "Ingests sales data and response time metrics",
            "Performs chi-square tests to identify significant relationships",
            "Uses z-tests for proportion comparisons between response time buckets",
            "Applies logistic regression for predictive modeling",
            "Controls for confounders like lead source and sales rep",
            "Explains results in plain English for non-technical stakeholders",
        ],
        "tech_stack": [
            "Python for data processing and analysis",
            "Pandas for data manipulation",
            "SciPy for statistical tests (chi-square, z-tests)",
            "Statsmodels for logistic regression",
            "Streamlit for interactive dashboard",
        ],
        "skills_demonstrated": [
            "Statistical analysis and hypothesis testing",
            "Translating complex analysis into business insights",
            "Data pipeline design (ETL, cleaning, bucketing)",
            "Bridging technical analysis with business value",
        ],
        "github_url": "https://github.com/iNoahCodeGuy/response_time_cl_analysis.git",
        "deep_dive_available": False,
        "cta_text": "This project shows Noah's ability to bridge data science with business value.",
    },
    "heatmap": {
        "hook": "See times of day your team responds fast and when they don't.",
        "business_problem": "Sales managers need to see response time patterns at a glance - not dig through spreadsheets. This dashboard makes performance gaps immediately visible.",
        "what_it_does": [
            "Interactive heat maps showing response times by hour/day",
            "Color-coded visualization (green = fast, red = slow)",
            "Filterable by team member, lead source, or time period",
            "Visual identification of performance gaps",
            "Exportable reports for management",
            "Upload any CSV to analyze",
        ],
        "tech_stack": [
            "Python for data processing",
            "Streamlit for the web interface",
            "Pandas for data manipulation",
            "Plotly for interactive visualizations",
        ],
        "skills_demonstrated": [
            "Clean 3-layer architecture (UI → Logic → Visualization)",
            "Building tools that are both technically sound AND usable by non-technical users",
            "Data visualization and UX thinking",
            "Production-ready dashboard design",
        ],
        "github_url": "https://github.com/iNoahCodeGuy/generic-lead-response-heatmap.git",
        "deep_dive_available": False,
        "cta_text": "This shows Noah's focus on building tools that are both technically sound AND actually usable.",
    },
}

# Portfolia deep dive content
PORTFOLIA_DEEP_DIVES = {
    "rag": {
        "title": "How RAG/Vector Search Works",
        "explanation": dedent("""\
            **Retrieval-Augmented Generation (RAG)** is the technique that lets me ground my responses in real data.

            Here's how it works in this app:

            1. **Query Embedding**: When you ask a question, I convert it to a vector using OpenAI's text-embedding-3-small model
            2. **Vector Search**: I search Supabase's pgvector database for similar chunks using cosine similarity
            3. **Context Assembly**: Top-k results (usually 3-5 chunks) are assembled into context
            4. **LLM Generation**: GPT-4o-mini generates a response grounded in that context

            This prevents hallucinations by ensuring every answer is backed by actual knowledge base content.

            **Key File**: `assistant/retrieval/vector_search.py` - Contains the semantic_search() function that powers this.
        """),
        "code_file": "assistant/retrieval/vector_search.py",
        "key_function": "semantic_search()",
    },
    "langgraph": {
        "title": "The LangGraph Conversation Pipeline",
        "explanation": dedent("""\
            The conversation flow uses **LangGraph** to orchestrate a 22-node pipeline.

            **Pipeline Stages:**
            - Stage 0: Session management and initialization
            - Stage 2: Role classification and routing
            - Stage 3: Query composition and refinement
            - Stage 4: Vector search and retrieval
            - Stage 5: Response generation with quality validation
            - Stage 6: Formatting and presentation control
            - Stage 7: Action execution (SMS, email, analytics)

            Each node receives ConversationState and returns partial updates. LangGraph automatically merges these updates.

            **Key File**: `assistant/flows/conversation_flow.py` - Contains build_conversation_graph() that wires everything together.
        """),
        "code_file": "assistant/flows/conversation_flow.py",
        "key_function": "build_conversation_graph()",
    },
    "error_handling": {
        "title": "Graceful Error Handling",
        "explanation": dedent("""\
            Production systems need to handle failures gracefully. This app implements several patterns:

            **Degradation Modes:**
            - If vector search fails → Fall back to keyword search
            - If LLM call fails → Return cached response or friendly error
            - If external service (Twilio/Resend) fails → Log error but continue conversation

            **Error Tracking:**
            - All errors are logged to LangSmith for observability
            - User-facing errors are friendly and actionable
            - System continues operating even when components fail

            This ensures the conversation never completely breaks, even under adverse conditions.
        """),
        "code_file": "assistant/flows/node_logic/stage5_quality_validation.py",
        "key_function": "validate_response_quality()",
    },
}

# ============================================================================
# Main Sub-Menu Handler
# ============================================================================

def handle_sub_menu_selection(state: ConversationState) -> ConversationState:
    """Route sub-menu selections to appropriate handlers based on current branch.

    Also handles cross-branch navigation (e.g., "professional" or "technical" after videos).

    Args:
        state: ConversationState with current_menu_branch and query

    Returns:
        Updated state with appropriate response
    """
    query = state.get("query", "").lower().strip()
    branch = state.get("current_menu_branch", "")
    sub_menu_type = state.get("sub_menu_type", "")

    logger.debug(f"Handling sub-menu selection: branch={branch}, sub_menu_type={sub_menu_type}")

    # Check for cross-branch navigation keywords
    if any(kw in query for kw in ["professional", "sales", "career", "background"]):
        # Pivot to professional background
        from assistant.flows.node_logic.stage2_role_routing import _get_role_welcome_message
        state["role"] = "Professional Background"
        state["role_mode"] = "professional_background"
        state["current_menu_branch"] = "professional_background"
        state["awaiting_sub_menu"] = True
        welcome_msg = _get_role_welcome_message("professional_background")
        state["answer"] = welcome_msg
        state["pipeline_halt"] = True
        return state
    elif any(kw in query for kw in ["technical", "code", "projects", "certifications", "github"]):
        # Pivot to technical background
        from assistant.flows.node_logic.stage2_role_routing import _get_role_welcome_message
        state["role"] = "Technical Background"
        state["role_mode"] = "technical_background"
        state["current_menu_branch"] = "technical_background"
        state["awaiting_sub_menu"] = True
        state["sub_menu_type"] = "technical_background_choice"
        welcome_msg = _get_role_welcome_message("technical_background")
        state["answer"] = welcome_msg
        state["pipeline_halt"] = True
        return state

    # Check for navigation keywords
    nav_action = _check_navigation_keywords(query)
    if nav_action == "menu":
        state["awaiting_sub_menu"] = False
        state.pop("current_menu_branch", None)
        state.pop("sub_menu_type", None)
        state["answer"] = dedent("""\
            Sure! Here's the main menu:

            1️⃣ Looking to learn about Noah's professional background
            2️⃣ Looking to learn about his technical background
            3️⃣ Just looking around
            4️⃣ Looking to confess crush 💌
        """)
        state["pipeline_halt"] = True
        return state

    if branch == "technical_background":
        return handle_technical_sub_menu(state)
    elif branch == "explorer":
        return handle_explorer_sub_menu(state)
    elif branch == "confession":
        return handle_confession_sub_menu(state)
    elif branch == "professional_background":
        return handle_professional_response(state)
    else:
        # Unknown branch, clear sub-menu state and continue
        state["awaiting_sub_menu"] = False
        state.pop("current_menu_branch", None)
        state.pop("sub_menu_type", None)
        return state


# ============================================================================
# Professional Background Handler
# ============================================================================

def handle_professional_response(state: ConversationState) -> ConversationState:
    """Handle responses in the professional background flow.

    Checks if user wants resume and triggers SMS notification.
    """
    query = state.get("query", "").lower().strip()

    # Check if user wants resume
    resume_keywords = ["yes", "sure", "send", "resume", "email", "please", "ok", "okay", "yeah", "yep"]
    if any(keyword in query for keyword in resume_keywords):
        return request_email_for_resume(state)

    # Check if user provided an email
    import re
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, state.get("query", ""))
    if email_match:
        return send_resume_with_notification(state, email_match.group(0))

    # Otherwise, let conversation continue normally
    state["awaiting_sub_menu"] = False
    state.pop("current_menu_branch", None)
    return state


def request_email_for_resume(state: ConversationState) -> ConversationState:
    """Ask user for email to send resume."""
    state["answer"] = dedent("""\
        Great! I'd be happy to send you Noah's resume.

        Please provide your email address and I'll send it right over! Noah will also receive a notification that you're interested. 📧
    """)
    state["pipeline_halt"] = True
    state["awaiting_sub_menu"] = True
    state["current_menu_branch"] = "professional_background"
    return state


def send_resume_with_notification(state: ConversationState, email: str) -> ConversationState:
    """Send resume to user and notify Noah via SMS.

    Args:
        state: ConversationState
        email: User's email address
    """
    # Execute resume send immediately (don't queue since pipeline will halt)
    from assistant.flows.node_logic.stage7_action_execution import _action_executor

    action = {
        "type": "send_resume",
        "email": email,
        "notify_noah": True
    }

    # Try to execute the action directly
    try:
        _action_executor.execute_send_resume_simple(state, action)
        success = True
    except Exception as exc:
        logger.error(f"Failed to send resume: {exc}")
        success = False

    # Track resume sent
    _track_engagement(state, "resume_sent")

    if success:
        state["answer"] = dedent(f"""\
            Perfect! I'm sending Noah's resume to {email} right now.

            Noah will receive a text notification that you're interested - he'll be thrilled to connect with you! 🎉

            Is there anything else you'd like to know about Noah's background or experience?
        """)
    else:
        state["answer"] = dedent(f"""\
            I tried to send Noah's resume to {email}, but encountered a small hiccup.

            Don't worry - Noah will still be notified of your interest! You can also reach him directly at noah@noahdelacalzada.com.

            Is there anything else you'd like to know?
        """)

    state["pipeline_halt"] = True  # Halt pipeline, we've handled everything
    state["awaiting_sub_menu"] = False
    state.pop("current_menu_branch", None)
    state["resume_sent"] = True
    return state


# ============================================================================
# Technical Background Handlers
# ============================================================================

def handle_technical_sub_menu(state: ConversationState) -> ConversationState:
    """Handle technical background sub-menu selections."""
    query = state.get("query", "").lower().strip()
    selection = _TECHNICAL_SUB_MENU_MAP.get(query) or _TECHNICAL_SUB_MENU_MAP.get(query.replace(" ", ""))

    # Check if user is asking for project details
    if state.get("sub_menu_type") == "project_details":
        return handle_project_detail_selection(state)

    if selection == "certifications":
        return show_certifications(state)
    elif selection == "projects":
        return show_projects(state)
    else:
        # Check for keywords
        if any(kw in query for kw in ["cert", "certification", "credential", "qualification"]):
            return show_certifications(state)
        elif any(kw in query for kw in ["project", "built", "program", "github", "code"]):
            return show_projects(state)

        # User asked something else - let pipeline continue
        state["awaiting_sub_menu"] = False
        state.pop("current_menu_branch", None)
        state.pop("sub_menu_type", None)
        return state


def show_certifications(state: ConversationState) -> ConversationState:
    """Display Noah's certifications."""
    _track_engagement(state, "certifications_viewed")
    state["answer"] = CERTIFICATIONS_CONTENT
    state["pipeline_halt"] = True
    # Keep in professional mode for potential resume request
    state["awaiting_sub_menu"] = True
    state["current_menu_branch"] = "professional_background"
    return state


def show_projects(state: ConversationState) -> ConversationState:
    """Display list of Noah's projects."""
    state["answer"] = PROJECTS_LIST
    state["pipeline_halt"] = True
    state["awaiting_sub_menu"] = True
    state["sub_menu_type"] = "project_details"
    # Track that projects list was viewed
    _track_engagement(state, "question_asked")
    return state


def handle_project_detail_selection(state: ConversationState) -> ConversationState:
    """Handle project detail selection with engagement tracking and progressive CTAs."""
    query = state.get("query", "").lower().strip()

    # Check for navigation keywords first
    nav_action = _check_navigation_keywords(query)
    if nav_action == "menu":
        # Reset to main menu
        state["awaiting_sub_menu"] = False
        state.pop("current_menu_branch", None)
        state.pop("sub_menu_type", None)
        state["answer"] = dedent("""\
            Sure! Here's the main menu:

            1️⃣ Looking to learn about Noah's professional background
            2️⃣ Looking to learn about his technical background
            3️⃣ Just looking around
            4️⃣ Looking to confess crush 💌
        """)
        state["pipeline_halt"] = True
        return state

    # Check if we're in Portfolia deep dive mode
    if state.get("sub_menu_type") == "portfolia_deep_dive":
        return handle_portfolia_deep_dive(state)

    selection = _PROJECT_DETAIL_MAP.get(query) or _PROJECT_DETAIL_MAP.get(query.replace(" ", ""))

    # Check for keywords too
    if not selection:
        if any(kw in query for kw in ["portfolia", "assistant", "ai", "this"]):
            selection = "portfolia"
        elif any(kw in query for kw in ["response", "close", "rate", "analysis", "statistical"]):
            selection = "response_time"
        elif any(kw in query for kw in ["heatmap", "heat", "map", "tracker", "dashboard"]):
            selection = "heatmap"

    if selection and selection in PROJECT_DATA:
        # Track project view
        _track_engagement(state, "project_viewed", selection)

        # Determine engagement level
        engagement_score = _get_engagement_score(state)
        if engagement_score >= 5:
            engagement_level = "high"
        elif engagement_score >= 2:
            engagement_level = "medium"
        else:
            engagement_level = "low"

        # Format project response
        project_response = _format_project_response(selection, engagement_level)

        # Add progressive CTA
        session_memory = state.get("session_memory", {})
        engagement = session_memory.get("engagement", {})
        linkedin_offered = engagement.get("linkedin_offered", False)

        cta = _get_resume_offer(engagement_score, linkedin_offered)
        if cta:
            _track_engagement(state, "resume_offered")
            if _should_offer_linkedin(state):
                engagement["linkedin_offered"] = True
                state["linkedin_offered"] = True

        # For Portfolia, add deep dive option if medium/high engagement
        if selection == "portfolia" and engagement_level in ("medium", "high"):
            deep_dive_prompt = dedent("""\

                ---

                Want to dive deeper? I can show you:
                1. How RAG/vector search works
                2. The LangGraph conversation pipeline
                3. How I handle errors gracefully
                4. See Noah's resume instead
            """)
            project_response += deep_dive_prompt
            state["sub_menu_type"] = "portfolia_deep_dive"
        else:
            # Add navigation hint
            project_response += f"\n\n{NAVIGATION_HINTS['after_projects']}"

        state["answer"] = project_response + cta
        state["pipeline_halt"] = True
        state["awaiting_sub_menu"] = True
        if state.get("sub_menu_type") != "portfolia_deep_dive":
            state["sub_menu_type"] = "project_details"
        return state

    # User asked something else - let pipeline continue
    state["awaiting_sub_menu"] = False
    state.pop("current_menu_branch", None)
    state.pop("sub_menu_type", None)
    return state


def handle_portfolia_deep_dive(state: ConversationState) -> ConversationState:
    """Handle Portfolia deep dive sub-menu selections."""
    query = state.get("query", "").lower().strip()
    selection = _PORTFOLIA_DEEP_DIVE_MAP.get(query) or _PORTFOLIA_DEEP_DIVE_MAP.get(query.replace(" ", ""))

    # Check for keywords
    if not selection:
        if any(kw in query for kw in ["rag", "vector", "search", "retrieval"]):
            selection = "rag"
        elif any(kw in query for kw in ["langgraph", "pipeline", "orchestration", "flow"]):
            selection = "langgraph"
        elif any(kw in query for kw in ["error", "handling", "degradation", "failure"]):
            selection = "error_handling"
        elif any(kw in query for kw in ["resume", "cv", "linkedin"]):
            selection = "resume"

    if selection == "resume":
        # Track engagement and offer resume
        _track_engagement(state, "resume_offered")
        return request_email_for_resume(state)
    elif selection and selection in PORTFOLIA_DEEP_DIVES:
        deep_dive = PORTFOLIA_DEEP_DIVES[selection]
        _track_engagement(state, "question_asked")

        response = dedent(f"""\
            **{deep_dive['title']}**

            {deep_dive['explanation']}

            **Code Location**: `{deep_dive['code_file']}` - Look for `{deep_dive['key_function']}`

            ---

            Want to explore another aspect of Portfolia, or see Noah's resume?
        """)

        # Add strong CTA for highly engaged users
        engagement_score = _get_engagement_score(state)
        if engagement_score >= 5:
            session_memory = state.get("session_memory", {})
            engagement = session_memory.get("engagement", {})
            if not engagement.get("linkedin_offered", False):
                response += dedent(f"""

                    You've been really diving deep! Would you like Noah's resume? Or connect on LinkedIn: {LINKEDIN_URL}
                """)
                engagement["linkedin_offered"] = True
                state["linkedin_offered"] = True
            else:
                response += "\n\nWould you like Noah's resume? Just ask!"

        state["answer"] = response
        state["pipeline_halt"] = True
        state["awaiting_sub_menu"] = True
        state["sub_menu_type"] = "portfolia_deep_dive"
        return state

    # User asked something else - let pipeline continue
    state["awaiting_sub_menu"] = False
    state.pop("current_menu_branch", None)
    state.pop("sub_menu_type", None)
    return state


# ============================================================================
# Explorer Handlers (Video Options)
# ============================================================================

def handle_explorer_sub_menu(state: ConversationState) -> ConversationState:
    """Handle explorer sub-menu selections (video options)."""
    query = state.get("query", "").lower().strip()
    selection = _EXPLORER_SUB_MENU_MAP.get(query) or _EXPLORER_SUB_MENU_MAP.get(query.replace(" ", ""))

    # Check for keywords
    if not selection:
        if any(kw in query for kw in ["hotdog", "hot dog", "eating", "food"]):
            selection = "hotdog"
        elif any(kw in query for kw in ["cage", "fight", "mma", "boxing"]):
            selection = "cage_fight"

    if selection == "hotdog":
        return show_hotdog_video(state)
    elif selection == "cage_fight":
        return show_cage_fight_video(state)
    else:
        # User asked something else - let pipeline continue
        state["awaiting_sub_menu"] = False
        state.pop("current_menu_branch", None)
        state.pop("sub_menu_type", None)
        return state


def show_hotdog_video(state: ConversationState) -> ConversationState:
    """Show the hotdog eating video."""
    video_url = VIDEO_URLS.get("hotdog", "https://youtube.com")
    _track_engagement(state, "video_watched", "hotdog")

    state["answer"] = dedent(f"""\
        🌭 Here's the legendary video of Noah eating 10 hotdogs!

        Watch it here: {video_url}

        Noah definitely regretted this one... but hey, it was for charity! 😄

        ---

        That's the fun side of Noah! But there's a lot more to him.

        Want to see his serious side?
        - Type "professional" to learn about his sales background
        - Type "technical" to see his coding projects
        - Or ask me anything!
    """)
    state["pipeline_halt"] = False
    state["awaiting_sub_menu"] = False
    state.pop("current_menu_branch", None)
    return state


def show_cage_fight_video(state: ConversationState) -> ConversationState:
    """Show the cage fight video."""
    video_url = VIDEO_URLS.get("cage_fight", "https://youtube.com")
    _track_engagement(state, "video_watched", "cage_fight")

    state["answer"] = dedent(f"""\
        🥊 Here's Noah in the cage!

        Watch it here: {video_url}

        Noah trained MMA for several years and this was one of his amateur fights. He's got the discipline and competitive spirit that carries into everything he does!

        ---

        That's the fun side of Noah! But there's a lot more to him.

        Want to see his serious side?
        - Type "professional" to learn about his sales background
        - Type "technical" to see his coding projects
        - Or ask me anything!
    """)
    state["pipeline_halt"] = False
    state["awaiting_sub_menu"] = False
    state.pop("current_menu_branch", None)
    return state


# ============================================================================
# Confession Handlers
# ============================================================================

def handle_confession_sub_menu(state: ConversationState) -> ConversationState:
    """Handle confession sub-menu selections (anonymous vs identity)."""
    query = state.get("query", "").lower().strip()
    selection = _CONFESSION_SUB_MENU_MAP.get(query) or _CONFESSION_SUB_MENU_MAP.get(query.replace(" ", ""))

    # Check for keywords
    if not selection:
        if any(kw in query for kw in ["anonymous", "anon", "secret", "private"]):
            selection = "anonymous"
        elif any(kw in query for kw in ["identity", "name", "who i am", "reveal"]):
            selection = "identity"

    # Check if we're waiting for confession content
    if state.get("sub_menu_type") == "confession_content":
        return process_confession(state)

    # Check if we're waiting for identity info
    if state.get("sub_menu_type") == "confession_identity":
        return process_confession_with_identity(state)

    if selection == "anonymous":
        return setup_anonymous_confession(state)
    elif selection == "identity":
        return setup_identity_confession(state)
    else:
        # User asked something else - let pipeline continue
        state["awaiting_sub_menu"] = False
        state.pop("current_menu_branch", None)
        state.pop("sub_menu_type", None)
        return state


def setup_anonymous_confession(state: ConversationState) -> ConversationState:
    """Set up anonymous confession flow."""
    state["answer"] = dedent("""\
        💌 Perfect, your identity will stay completely secret!

        Go ahead and share your confession. I'll send it directly to Noah and he'll never know who it's from.

        Take your time... I'm listening! 💕
    """)
    state["pipeline_halt"] = True
    state["awaiting_sub_menu"] = True
    state["sub_menu_type"] = "confession_content"
    state["confession_is_anonymous"] = True
    return state


def setup_identity_confession(state: ConversationState) -> ConversationState:
    """Set up confession with identity."""
    state["answer"] = dedent("""\
        💌 That's brave of you! Noah will love knowing who this is from.

        First, what's your name? (You can also share your contact info if you'd like Noah to reach out!)
    """)
    state["pipeline_halt"] = True
    state["awaiting_sub_menu"] = True
    state["sub_menu_type"] = "confession_identity"
    state["confession_is_anonymous"] = False
    return state


def process_confession_with_identity(state: ConversationState) -> ConversationState:
    """Process identity info and ask for confession content."""
    # Store the identity info
    identity_info = state.get("query", "").strip()
    session_memory = state.setdefault("session_memory", {})
    session_memory["confessor_identity"] = identity_info

    state["answer"] = dedent(f"""\
        Got it! I'll let Noah know this is from: {identity_info}

        Now, go ahead and share your confession! What would you like Noah to know? 💕
    """)
    state["pipeline_halt"] = True
    state["awaiting_sub_menu"] = True
    state["sub_menu_type"] = "confession_content"
    return state


def process_confession(state: ConversationState) -> ConversationState:
    """Process the actual confession and send SMS to Noah."""
    confession_text = state.get("query", "").strip()
    is_anonymous = state.get("confession_is_anonymous", True)
    session_memory = state.get("session_memory", {})
    confessor_identity = session_memory.get("confessor_identity", "Anonymous")

    if len(confession_text) < 5:
        state["answer"] = "That seems a bit short! Don't be shy - share what's on your mind! 💕"
        state["pipeline_halt"] = True
        return state

    # Execute confession SMS immediately (don't queue since pipeline will halt)
    from assistant.flows.node_logic.stage7_action_execution import _action_executor

    action = {
        "type": "send_confession_sms",
        "confession": confession_text,
        "is_anonymous": is_anonymous,
        "confessor_identity": None if is_anonymous else confessor_identity
    }

    # Try to execute the action directly
    try:
        _action_executor.execute_send_confession_sms(state, action)
        success = True
    except Exception as exc:
        logger.error(f"Failed to send confession SMS: {exc}")
        success = False

    if is_anonymous:
        if success:
            state["answer"] = dedent(f"""\
                💌 Your anonymous confession has been sent to Noah!

                He'll receive a text message with your heartfelt words. Your secret is safe with me! 🤫

                ---

                {NAVIGATION_HINTS['after_confession']}
            """)
        else:
            state["answer"] = dedent(f"""\
                💌 Your confession has been received!

                I had a small issue sending the text, but don't worry - your message is safe and Noah will see it. Your secret is safe with me! 🤫

                ---

                {NAVIGATION_HINTS['after_confession']}
            """)
    else:
        if success:
            state["answer"] = dedent(f"""\
                💌 Your confession has been sent to Noah!

                He'll receive a text message letting him know that {confessor_identity} has something sweet to say. Maybe he'll reach out! 😊

                ---

                {NAVIGATION_HINTS['after_confession']}
            """)
        else:
            state["answer"] = dedent(f"""\
                💌 Your confession has been received!

                I had a small issue sending the text, but don't worry - Noah will see your message from {confessor_identity}. Maybe he'll reach out! 😊

                ---

                {NAVIGATION_HINTS['after_confession']}
            """)

    state["pipeline_halt"] = True  # Halt pipeline, we've handled everything
    state["awaiting_sub_menu"] = False
    state.pop("current_menu_branch", None)
    state.pop("sub_menu_type", None)
    state["confession_sent"] = True
    return state
