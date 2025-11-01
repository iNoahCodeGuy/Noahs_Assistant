"""Conversation nodes orchestrator - imports modular functions.

This module acts as the central import point for all conversation pipeline nodes.
The actual implementations live in focused modules:

- query_classification.py: classify_query() - intent detection
- core_nodes.py: retrieve_chunks, generate_answer, apply_role_context, log_and_notify
- action_planning.py: plan_actions() - build action shopping list
- action_execution.py: execute_actions() - perform side effects

For junior developers: This file is now just a convenience wrapper that re-exports
functions from specialized modules. Each module has detailed docstrings explaining
what that part of the pipeline does.

See docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md for the full conversation flow diagram.
"""

from __future__ import annotations

# Import all conversation nodes from their focused modules
from src.flows.query_classification import classify_query
from src.flows.core_nodes import (
    retrieve_chunks,
    generate_answer,
    apply_role_context,
    log_and_notify
)
from src.flows.action_planning import plan_actions
from src.flows.action_execution import execute_actions

# Re-export everything so existing code doesn't break
__all__ = [
    "classify_query",
    "retrieve_chunks",
    "generate_answer",
    "plan_actions",
    "apply_role_context",
    "execute_actions",
    "log_and_notify",
]


# Legacy stubs for compatibility (deprecated)
def search_import_explanations(*args, **kwargs):
        return []

    def detect_import_in_query(*args, **kwargs):
        return None

    def get_import_explanation(*args, **kwargs):
        return None


CODE_VALIDATION_KEYWORDS = [
    "def ", "class ", "import ", "from ", "return ", "async ", "await ", "try:", "except ", "lambda "
]


def _is_valid_code_snippet(code: str) -> bool:
    """Heuristic validation to ensure retrieved code looks like Python source."""
    if not code:
        return False

    stripped = code.strip()

    if len(stripped) < 20:
        return False

    if "doc_id" in stripped or "query=" in stripped or stripped.startswith("{'"):
        return False

    if stripped[0] in {"}", ")"} and stripped.count("\n") < 3:
        return False

    if "\n" not in stripped:
        return False

    if not any(keyword in stripped for keyword in CODE_VALIDATION_KEYWORDS):
        return False

    return True


SANITIZE_PREFIX_PATTERNS = [
    re.compile(r"^[\}\)\]\{\(\[]+$"),
    re.compile(r"^SELECT\.?$", re.IGNORECASE),
    re.compile(r"^FROM\.?$", re.IGNORECASE),
]


def _sanitize_generated_answer(answer: str) -> str:
    """Strip leading SQL/artifact noise that can leak from retrieval context."""
    if not answer:
        return answer

    lines = answer.splitlines()
    sanitized_lines = []
    dropping_prefix = True

    for line in lines:
        stripped = line.strip()

        if dropping_prefix and (not stripped):
            continue

        if dropping_prefix and any(pattern.match(stripped) for pattern in SANITIZE_PREFIX_PATTERNS):
            continue

        dropping_prefix = False
        sanitized_lines.append(line)

    if not sanitized_lines:
        return ""

    return "\n".join(sanitized_lines).lstrip()


RESUME_DOWNLOAD_URL = os.getenv("RESUME_DOWNLOAD_URL", "https://example.com/noah-resume.pdf")
LINKEDIN_URL = os.getenv("LINKEDIN_URL", "https://linkedin.com/in/noahdelacalzada")

DATA_DISPLAY_KEYWORDS = [
    "display data",
    "show data",
    "show me the data",
    "display analytics",
    "data collected",
    "collected data",
    "display collected data",
    "share the data",
    "show analytics",
    "can you display",
]


def _is_data_display_request(lowered_query: str) -> bool:
    """Check if query requests data/analytics display."""
    return any(keyword in lowered_query for keyword in DATA_DISPLAY_KEYWORDS)


def classify_query(state: ConversationState) -> ConversationState:
    """Classify the incoming query and stash the result on state.

    Detects:
    - Code display requests (show/display code, how do you, implementation details)
    - Import/stack questions (why use X, what imports, explain dependencies)
    - Technical queries (architecture, retrieval, pipeline)
    - Career queries (resume, experience, achievements)
    - MMA queries (fight references)
    - Data display requests (show analytics, display data)
    - Fun queries (fun facts, hobbies)
    """
    lowered = state.query.lower()

    # Code display triggers (explicit requests)
    code_display_keywords = [
        "show code", "display code", "show me code", "show the code",
        "show implementation", "display implementation",
        "how do you", "how does it", "how is it",
        "show me the", "show retrieval", "show api",
        "code snippet", "code example", "source code"
    ]
    if any(keyword in lowered for keyword in code_display_keywords):
        state.stash("code_display_requested", True)
        state.stash("query_type", "technical")

    # Import/stack explanation triggers
    import_keywords = [
        "why use", "why choose", "why did you use", "why did you choose",
        "what imports", "explain imports", "your imports", "dependencies",
        "why supabase", "why openai", "why langchain", "why vercel",
        "why pgvector", "why twilio", "why resend",
        "justify", "trade-off", "alternative", "vs", "instead of",
        "enterprise", "production", "scale", "library", "libraries"
    ]
    # Also check for specific library mentions in query
    library_names = ["supabase", "openai", "pgvector", "langchain", "langgraph",
                     "vercel", "resend", "twilio", "langsmith", "streamlit"]

    if any(keyword in lowered for keyword in import_keywords):
        state.stash("import_explanation_requested", True)
        state.stash("query_type", "technical")
    # If query mentions a library name directly, likely asking about it
    elif any(lib in lowered for lib in library_names) and any(word in lowered for word in ["why", "what", "how", "explain"]):
        state.stash("import_explanation_requested", True)
        state.stash("query_type", "technical")

    # Query type classification
    if any(re.search(r"\\b" + k + r"\\b", lowered) for k in ["mma", "fight", "ufc", "bout", "cage"]):
        state.stash("query_type", "mma")
    elif any(term in lowered for term in ["fun fact", "hobby", "interesting fact", "hot dog"]):
        state.stash("query_type", "fun")
    elif _is_data_display_request(lowered):
        state.stash("data_display_requested", True)
        state.stash("query_type", "data")
    # Detect "how does [product/system/chatbot] work" queries as technical
    elif any(term in lowered for term in ["code", "technical", "stack", "architecture", "implementation", "retrieval"]) \
         or (("how does" in lowered or "how did" in lowered or "explain how" in lowered)
             and any(word in lowered for word in ["product", "system", "chatbot", "assistant", "rag", "pipeline", "work", "built"])):
        state.stash("query_type", "technical")
    elif any(term in lowered for term in ["career", "resume", "cv", "experience", "achievement", "work"]):
        state.stash("query_type", "career")
    else:
        state.stash("query_type", "general")
    return state


def retrieve_chunks(state: ConversationState, rag_engine: RagEngine, top_k: int = 4) -> ConversationState:
    """Retrieve knowledge base chunks and store them on state."""
    results = rag_engine.retrieve(state.query, top_k=top_k)
    state.add_retrieved_chunks(results.get("chunks", []))
    state.stash("retrieval_matches", results.get("matches", []))
    state.stash("retrieval_scores", results.get("scores", []))
    return state


def generate_answer(state: ConversationState, rag_engine: RagEngine) -> ConversationState:
    """Generate an assistant response using retrieved context.

    Uses contextual response generation which includes:
    - Role-aware prompting
    - Third-person language enforcement
    - Technical follow-up questions (for technical roles)

    For data display requests, fetches LIVE analytics from /api/analytics endpoint.
    """
    # Get retrieved chunks for context
    retrieved_chunks = state.retrieved_chunks or []

    # For data display requests, use LIVE analytics from API
    if state.fetch("data_display_requested", False):
        # Mark that we need to fetch live analytics
        # This will be handled in apply_role_context
        state.set_answer("Fetching live analytics data from Supabase...")
        return state

    # Use contextual response generator (includes follow-ups)
    answer = rag_engine.response_generator.generate_contextual_response(
        query=state.query,
        context=retrieved_chunks,
        role=state.role
    )

    state.set_answer(_sanitize_generated_answer(answer))
    return state


def plan_actions(state: ConversationState) -> ConversationState:
    """Derive follow-up actions (resume offers, tables, notifications, code display)."""
    state.pending_actions.clear()
    query_type = state.fetch("query_type", "general")
    lowered = state.query.lower()
    user_turns = sum(1 for message in state.chat_history if message.get("role") == "user")

    # Check for code/import request flags
    code_display_requested = state.fetch("code_display_requested", False)
    import_explanation_requested = state.fetch("import_explanation_requested", False)

    # Detect product/how-it-works questions
    product_question = any(term in lowered for term in [
        "how does this work", "how does it work", "how does", "how is this",
        "what is this", "what does this", "explain this",
        "how is this built", "tell me about this", "what's this"
    ]) or ("product" in lowered and any(word in lowered for word in ["how", "what", "explain", "work"]))

    def add_action(action_type: str, **extras: Any) -> None:
        state.append_pending_action({"type": action_type, **extras})

    resume_requested = any(key in lowered for key in ["send resume", "email resume", "resume", "cv"])
    linkedin_requested = any(key in lowered for key in ["linkedin", "link me", "profile"])
    contact_requested = any(key in lowered for key in ["reach out", "contact me", "call me", "follow up"])

    if _is_data_display_request(lowered):
        add_action("render_live_analytics")  # Changed from render_data_report
        state.stash("data_display_requested", True)

    if resume_requested:
        add_action("send_resume")
        add_action("ask_reach_out")
        add_action("notify_resume_sent")
        state.stash("offer_sent", True)
    if linkedin_requested:
        add_action("send_linkedin")
        if not state.fetch("offer_sent"):
            add_action("ask_reach_out")
            state.stash("offer_sent", True)
    if contact_requested:
        add_action("notify_contact_request")
        state.stash("contact_requested", True)

    # Code display and import explanations for technical roles
    if code_display_requested and state.role in ["Hiring Manager (technical)", "Software Developer"]:
        add_action("display_code_snippet")

    if import_explanation_requested:
        add_action("explain_imports")

    # Add QA strategy for product questions (all technical roles)
    if product_question and state.role in ["Hiring Manager (technical)", "Software Developer"]:
        add_action("include_qa_strategy")

    if state.role == "Hiring Manager (nontechnical)":
        # Suggest role switch for technical questions
        if query_type == "technical" or code_display_requested or import_explanation_requested or product_question:
            add_action("suggest_technical_role_switch")
        if not resume_requested and not linkedin_requested and user_turns >= 2:
            add_action("offer_resume_prompt")
        if not resume_requested and not linkedin_requested and user_turns >= 2:
            add_action("offer_resume_prompt")
    elif state.role == "Hiring Manager (technical)":
        if query_type in {"technical", "data"}:
            add_action("include_purpose_overview")
            add_action("include_architecture_overview")
            add_action("summarize_data_strategy")
            add_action("provide_data_tables")
            add_action("explain_enterprise_usage")
            add_action("explain_stack_currency")
            add_action("highlight_enterprise_adaptability")
            # Auto-display code for technical queries if not already requested
            if not code_display_requested:
                add_action("include_code_snippets")
        if not resume_requested and not linkedin_requested and user_turns >= 2:
            add_action("offer_resume_prompt")
    elif state.role == "Software Developer":
        if query_type in {"technical", "data"}:
            add_action("include_purpose_overview")
            add_action("include_code_snippets")
            add_action("summarize_data_strategy")
            add_action("provide_data_tables")
            add_action("explain_stack_currency")
            add_action("highlight_enterprise_adaptability")
    elif state.role == "Just looking around":
        if query_type == "mma":
            add_action("share_mma_link")
        else:
            add_action("share_fun_facts")
    elif state.role == "Looking to confess crush":
        add_action("collect_confession")

    return state


def apply_role_context(state: ConversationState, rag_engine: RagEngine) -> ConversationState:
    """Tailor the generated answer with role-specific enrichments.

    This node appends structured content blocks based on planned actions:
    - Product purpose and enterprise signal
    - Full data analytics report
    - Architecture and stack explanations
    - Fun facts and MMA links
    - Resume and LinkedIn offers
    - Code snippets for developers

    Args:
        state: Current conversation state
        rag_engine: RAG engine for code retrieval

    Returns:
        Updated state with enriched answer
    """
    if not state.answer:
        return state

    components = [state.answer]
    actions = {action["type"] for action in state.pending_actions}
    query_type = state.fetch("query_type", "general")

    # Enterprise-focused content blocks
    if "include_purpose_overview" in actions:
        components.append(
            "\n\n" + content_blocks.format_section("Product Purpose", content_blocks.purpose_block())
        )

    if "include_qa_strategy" in actions:
        components.append(
            "\n\n" + content_blocks.format_section("Quality Assurance", content_blocks.qa_strategy_block())
        )

    # NEW: Live analytics rendering from /api/analytics endpoint
    if "render_live_analytics" in actions:
        try:
            import requests
            from src.flows.analytics_renderer import render_live_analytics
            from src.config.supabase_config import supabase_settings

            # Determine the analytics API URL
            if supabase_settings.is_vercel:
                # Production: use the deployed URL
                analytics_url = "https://noahsaiassistant.vercel.app/api/analytics"
            else:
                # Local: use localhost
                analytics_url = "http://localhost:3000/api/analytics"

            # Fetch live analytics
            response = requests.get(analytics_url, timeout=3)
            response.raise_for_status()
            analytics_data = response.json()

            # Render with role-appropriate detail level
            analytics_report = render_live_analytics(
                analytics_data,
                state.role,
                focus=None  # Could detect focus from query
            )

            # Replace placeholder with actual analytics
            components = [analytics_report]

        except Exception as e:
            logger.error(f"Failed to fetch live analytics: {e}")
            # Fallback to cached/KB version
            components.append(
                "\n\n⚠️ Live analytics temporarily unavailable. Would you like to see a cached summary?"
            )

    # Legacy: Old data report (keep for compatibility)
    if "render_data_report" in actions and "render_live_analytics" not in actions:
        report = state.fetch("data_report")
        if not report:
            report = render_full_data_report()
            state.stash("data_report", report)
        components.append(
            "\n\n" + content_blocks.format_section("Data Insights & Full Dataset", report)
        )

    if "provide_data_tables" in actions:
        components.append(
            "\n\n" + content_blocks.format_section("Data Collection Overview", content_blocks.data_collection_table())
        )

    if "include_architecture_overview" in actions:
        components.append(
            "\n\n" + content_blocks.format_section("Architecture Snapshot", content_blocks.architecture_snapshot())
        )

    if "summarize_data_strategy" in actions:
        components.append(
            "\n\n" + content_blocks.format_section("Data Management Strategy", content_blocks.data_strategy_block())
        )

    if "explain_enterprise_usage" in actions:
        components.append(
            "\n\n" + content_blocks.format_section("Enterprise Fit", content_blocks.enterprise_fit_explanation())
        )

    if "explain_stack_currency" in actions:
        components.append(
            "\n\n" + content_blocks.format_section("Stack Importance", content_blocks.stack_importance_explanation())
        )

    if "suggest_technical_role_switch" in actions:
        components.append(content_blocks.role_switch_suggestion("Hiring Manager (technical)"))

    if "suggest_developer_role_switch" in actions:
        components.append(content_blocks.role_switch_suggestion("Software Developer"))

    if "highlight_enterprise_adaptability" in actions:
        components.append(
            "\n\n" + content_blocks.format_section("Enterprise Adaptability", content_blocks.enterprise_adaptability_block())
        )

    # Casual content blocks
    if "share_fun_facts" in actions:
        components.append(
            "\n\n" + content_blocks.format_section("Fun Facts About Noah", content_blocks.fun_facts_block())
        )

    if "share_mma_link" in actions or query_type == "mma":
        components.append("\n\n" + content_blocks.mma_fight_link())

    # Resource offers
    if "send_linkedin" in actions:
        components.append(f"\n\nHere is Noah's LinkedIn profile: {LINKEDIN_URL}")
        state.stash("offer_sent", True)

    if "send_resume" in actions:
        resume_link = state.fetch("resume_signed_url", RESUME_DOWNLOAD_URL)
        components.append(f"\n\nDownload Noah's resume: {resume_link}")
        state.stash("offer_sent", True)

    # Code snippets for developers
    if "include_code_snippets" in actions or "display_code_snippet" in actions:
        try:
            results = rag_engine.retrieve_with_code(state.query, role=state.role)
            snippets = results.get("code_snippets", []) if results else []
        except Exception as e:  # pragma: no cover - defensive guard
            logger.warning(f"Code retrieval failed: {e}")
            snippets = []

        if snippets:
            snippet = snippets[0]
            code_content = snippet.get("content", "")
            citation = snippet.get("citation", "codebase")

            if _is_valid_code_snippet(code_content):
                # Use formatted code display with enterprise prompt
                formatted_code = content_blocks.format_code_snippet(
                    code=code_content,
                    file_path=citation,
                    language="python",
                    description="Implementation showing the core logic referenced in your question"
                )
                components.append(f"\n\n**Code Implementation**\n{formatted_code}")
                components.append(content_blocks.code_display_guardrails())
            else:
                # Code index is empty or malformed - provide helpful message
                components.append(
                    "\n\n"
                    + content_blocks.format_section(
                        "Code Display Unavailable",
                        "The code index is being refreshed right now. In the meantime you can:\n"
                        "- Browse the codebase on GitHub: https://github.com/iNoahCodeGuy/ai_assistant\n"
                        "- Ask for architecture explanations or design walkthroughs\n"
                        "- Request high-level summaries of how components interact"
                    )
                )
        else:
            # No code found for query
            if "display_code_snippet" in actions:
                # User explicitly asked to see code
                components.append(
                    "\n\n"
                    + content_blocks.format_section(
                        "No Matching Code",
                        "I couldn't find code matching that request. You can:\n"
                        "- Browse the full codebase: https://github.com/iNoahCodeGuy/ai_assistant\n"
                        "- Ask for an architectural overview or diagram\n"
                        "- Request insights into specific features or services"
                    )
                )

    # Import explanations for stack questions
    if "explain_imports" in actions:
        # Detect which import they're asking about
        import_name = detect_import_in_query(state.query)

        if import_name:
            # Get tier-appropriate explanation
            explanation_data = get_import_explanation(import_name, state.role)
            if explanation_data:
                formatted_explanation = content_blocks.format_import_explanation(
                    import_name=explanation_data["import"],
                    tier=explanation_data["tier"],
                    explanation=explanation_data["explanation"],
                    enterprise_concern=explanation_data.get("enterprise_concern", ""),
                    enterprise_alternative=explanation_data.get("enterprise_alternative", ""),
                    when_to_switch=explanation_data.get("when_to_switch", "")
                )
                components.append(f"\n\n{formatted_explanation}")
        else:
            # Search for relevant imports based on query
            relevant_imports = search_import_explanations(state.query, state.role, top_k=3)
            if relevant_imports:
                components.append("\n\n**Stack Justifications**\n")
                for imp_data in relevant_imports:
                    formatted = content_blocks.format_import_explanation(
                        import_name=imp_data["import"],
                        tier=imp_data["tier"],
                        explanation=imp_data["explanation"],
                        enterprise_concern=imp_data.get("enterprise_concern", ""),
                        enterprise_alternative=imp_data.get("enterprise_alternative", ""),
                        when_to_switch=imp_data.get("when_to_switch", "")
                    )
                    components.append(f"\n{formatted}\n")


    # Interactive prompts
    if "offer_resume_prompt" in actions and not state.fetch("offer_sent"):
        components.append("\n\nWould you like me to email you my resume or share my LinkedIn profile?")

    if "ask_reach_out" in actions:
        components.append("\n\nWould you like Noah to reach out?")

    if "collect_confession" in actions:
        components.append(
            "\n\n💌 Your message is safe. You can submit anonymously or include your name and contact info, and I'll pass it along with a private SMS to Noah."
        )

    # Technical follow-up prompt
    if (
        query_type in {"technical", "data"}
        and state.role in {"Hiring Manager (technical)", "Software Developer"}
    ):
        components.append(
            "\n\nWould you like me to go into further detail about the logic behind the architecture, display data collected, or go deeper on how a project like this could be adapted into enterprise use?"
        )

    state.set_answer("".join(components))
    return state


# execute_actions is imported from action_execution module
# (see imports at top of file)


def log_and_notify(
    state: ConversationState,
    session_id: str,
    latency_ms: int,
    success: bool = True
) -> ConversationState:
    """Persist analytics and trigger downstream notifications."""
    try:
        interaction = UserInteractionData(
            session_id=session_id,
            role_mode=state.role,
            query=state.query,
            answer=state.answer or "",
            query_type=state.fetch("query_type", "general"),
            latency_ms=latency_ms,
            success=success
        )
        message_id = supabase_analytics.log_interaction(interaction)
        state.update_analytics("message_id", message_id)
    except Exception as exc:
        logger.error("Failed logging analytics: %s", exc)

    # Additional notification hooks will be wired as nodes mature.
    return state
