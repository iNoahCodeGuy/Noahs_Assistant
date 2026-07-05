"""Followup and hook generation helpers (extracted from stage6_formatting_nodes).

Owns the conversation-arc and pillar data (CONVERSATION_ARCS, MAIN_PILLARS) plus
the followup builders: conversation flow analysis, pillar exploration tracking,
topic extraction, next-topic prediction, and synthesis planning.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

from assistant.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)


def _build_subcategory_followups(active_subcats: List[str]) -> List[str]:
    """Generate followup questions based on active technical subcategories.

    Merged from logging_nodes.suggest_followups for inline followup generation.
    Maps subcategory focus to natural next-step questions.

    Args:
        active_subcats: List of active subcategory names

    Returns:
        List of contextual followup questions
    """
    suggestions = []

    if "stack_depth" in active_subcats:
        suggestions.append("Want to compare LangChain vs LlamaIndex trade-offs?")
        suggestions.append("Should I break down the requirements.txt dependencies?")

    if "architecture_depth" in active_subcats:
        suggestions.append("Want the LangGraph node flow diagram?")
        suggestions.append("Should I map this architecture to your team's stack?")

    if "data_pipeline_depth" in active_subcats:
        suggestions.append("Curious about the pgvector RPC implementation?")
        suggestions.append("Want to see the embedding generation flow?")

    if "state_management_depth" in active_subcats:
        suggestions.append("Should I show the ConversationState transitions?")
        suggestions.append("Want to trace a query through the pipeline?")

    # Default fallback if no matches
    if not suggestions:
        suggestions = [
            "Want me to walk through the LangGraph node transitions in detail?",
            "Curious how the Supabase pgvector query works under load?",
            "Should we map this architecture to your internal stack?",
        ]

    return suggestions[:3]  # Return at most 3


def _analyze_conversation_flow(chat_history: List[Dict], session_memory: Dict) -> Dict:
    """Analyze conversation pattern to infer progression.

    Detects conversation patterns like:
    - orchestration → enterprise (Turn 3: orchestration, Turn 4: enterprise)
    - architecture → implementation (architectural discussion → code details)
    - general → specific (broad question → specific follow-up)

    Args:
        chat_history: List of message dicts with 'role' and 'content' keys
        session_memory: Dict with topics, entities, persona_hints, etc.

    Returns:
        Dict with:
        - pattern: Detected pattern name (or None)
        - topics: List of accumulated topics
        - has_progression: Boolean indicating if progression detected

    Example:
        >>> chat_history = [
        ...     {"role": "user", "content": "explain orchestration"},
        ...     {"role": "assistant", "content": "..."},
        ...     {"role": "user", "content": "how is this relevant to enterprise?"}
        ... ]
        >>> session_memory = {"topics": ["orchestration", "technical"]}
        >>> _analyze_conversation_flow(chat_history, session_memory)
        {"pattern": "orchestration_to_enterprise", "topics": [...], "has_progression": True}
    """
    patterns = []
    topics = session_memory.get("topics", []) if session_memory else []
    topics_text = " ".join(topics).lower() if topics else ""

    # Need at least 2 messages to detect patterns
    if not chat_history or len(chat_history) < 2:
        return {
            "pattern": None,
            "topics": topics,
            "has_progression": False
        }

    # Get last few messages for pattern analysis
    recent_messages = chat_history[-4:] if len(chat_history) > 4 else chat_history
    # Extract content from messages (support both dict format and LangGraph message objects)
    recent_contents = []
    for msg in recent_messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
        elif hasattr(msg, "content"):
            content = msg.content if hasattr(msg, "content") else ""
        else:
            content = ""
        recent_contents.append(content.lower())
    recent_content = " ".join(recent_contents)

    # Detect pattern: orchestration → enterprise
    # User asked about orchestration (Turn 3), now asking about enterprise/adaptation (Turn 4)
    if "orchestration" in topics_text or "orchestration" in recent_content:
        # Check recent messages for enterprise/adaptation mentions (support both formats)
        # Expanded keywords to catch "customer support", "adapt", etc.
        enterprise_keywords = ["enterprise", "customer support", "adapt", "use case", "production", "scales"]
        enterprise_mentioned = False
        for msg in recent_messages[-2:]:
            if isinstance(msg, dict):
                content = msg.get("content", "").lower()
            elif hasattr(msg, "content"):
                content = msg.content.lower() if hasattr(msg, "content") else ""
            else:
                content = ""
            if any(kw in content for kw in enterprise_keywords):
                enterprise_mentioned = True
                break
        if enterprise_mentioned:
            patterns.append("orchestration_to_enterprise")

    # Detect pattern: architecture → implementation
    # User asked about architecture, now asking about implementation/code
    if "architecture" in recent_content or "architecture" in topics_text:
        if any(kw in recent_content for kw in ["code", "implementation", "how is this built", "show me"]):
            patterns.append("architecture_to_implementation")

    # Detect pattern: general → specific
    # User started with general question, now asking specific follow-up
    if len(chat_history) >= 4:
        # Compare first query vs recent queries (support both dict format and LangGraph message objects)
        first_msg = chat_history[0]
        recent_msg = chat_history[-1]

        # Extract content and role from first message
        if isinstance(first_msg, dict):
            first_content = first_msg.get("content", "")
            first_role = first_msg.get("role", "") or first_msg.get("type", "")
        else:
            first_content = getattr(first_msg, "content", "") if hasattr(first_msg, "content") else ""
            first_role = getattr(first_msg, "role", "") or getattr(first_msg, "type", "") if hasattr(first_msg, "role") or hasattr(first_msg, "type") else ""
        first_query = first_content.lower() if (first_role == "user" or first_role == "human") else ""

        # Extract content and role from recent message
        if isinstance(recent_msg, dict):
            recent_content = recent_msg.get("content", "")
            recent_role = recent_msg.get("role", "") or recent_msg.get("type", "")
        else:
            recent_content = getattr(recent_msg, "content", "") if hasattr(recent_msg, "content") else ""
            recent_role = getattr(recent_msg, "role", "") or getattr(recent_msg, "type", "") if hasattr(recent_msg, "role") or hasattr(recent_msg, "type") else ""
        recent_query = recent_content.lower() if (recent_role == "user" or recent_role == "human") else ""

        # First query is general (short, simple), recent query is specific (longer, more keywords)
        if first_query and recent_query:
            first_word_count = len(first_query.split())
            recent_word_count = len(recent_query.split())
            if first_word_count <= 5 and recent_word_count > 5:
                patterns.append("general_to_specific")

    return {
        "pattern": patterns[0] if patterns else None,
        "topics": topics,
        "has_progression": len(patterns) > 0
    }


def _should_offer_synthesis(session_memory: Dict, conversation_turn: int, conversation_phase: str) -> bool:
    """Check if we should offer to synthesize across topics.

    Synthesis is offered:
    - In synthesis phase (8+ turns, 4+ topics)
    - OR at 5+ turns with 3+ topics (if not in synthesis phase yet)
    - But not too frequently (wait 3 turns between offers)

    Args:
        session_memory: Session memory with topics
        conversation_turn: Current conversation turn
        conversation_phase: Current conversation phase

    Returns:
        True if synthesis should be offered, False otherwise
    """
    topics = session_memory.get("topics", []) if session_memory else []

    # Always offer in synthesis phase
    if conversation_phase == "synthesis":
        last_synthesis = session_memory.get("last_synthesis_turn", 0) if session_memory else 0
        if conversation_turn - last_synthesis >= 3:
            return True

    # Offer at 5+ turns with 3+ topics
    if conversation_turn >= 5 and len(topics) >= 3:
        last_synthesis = session_memory.get("last_synthesis_turn", 0) if session_memory else 0
        if conversation_turn - last_synthesis >= 3:
            return True

    return False


# Extended conversation arcs for indefinite interactions (100+ turns)
# Each tuple: (current_topic, [next_topic_options])
# Arcs create guided journeys through the knowledge base
CONVERSATION_ARCS = {
    "hiring_manager_technical": [
        # Phase 1: Architecture Foundation (turns 1-10)
        ("tech_stack", ["orchestration", "rag_pipeline", "data_layer"]),
        ("architecture", ["orchestration", "rag_pipeline", "observability"]),
        ("orchestration", ["langgraph_nodes", "state_management", "error_handling"]),
        ("rag_pipeline", ["retrieval", "generation", "grounding"]),
        ("data_layer", ["pgvector", "embeddings", "supabase"]),

        # Phase 2: Deep Dives (turns 10-25)
        ("langgraph_nodes", ["enterprise_adaptation", "cost_analysis", "testing"]),
        ("state_management", ["memory_accumulation", "session_persistence", "scaling"]),
        ("retrieval", ["similarity_search", "query_composition", "reranking"]),
        ("generation", ["prompt_engineering", "model_selection", "streaming"]),
        ("observability", ["langsmith", "analytics", "cost_tracking"]),

        # Phase 3: Enterprise Application (turns 25-50)
        ("enterprise_adaptation", ["customer_support", "internal_docs", "sales_enablement"]),
        ("customer_support", ["role_adaptation", "action_handlers", "metrics"]),
        ("internal_docs", ["knowledge_ingestion", "access_control", "search_quality"]),
        ("sales_enablement", ["crm_integration", "lead_scoring", "personalization"]),
        ("cost_analysis", ["token_optimization", "caching", "batch_processing"]),

        # Phase 4: Production Concerns (turns 50-75)
        ("deployment", ["vercel_serverless", "edge_functions", "cold_starts"]),
        ("scaling", ["horizontal_scaling", "rate_limiting", "load_balancing"]),
        ("testing", ["qa_strategy", "golden_datasets", "regression_testing"]),
        ("monitoring", ["alerting", "dashboards", "incident_response"]),

        # Phase 5: Noah's Background (turns 75+)
        ("noahs_background", ["projects", "philosophy", "tesla_experience"]),
        ("projects", ["portfolia_deep_dive", "other_projects", "github"]),
        ("philosophy", ["ai_ethics", "learning_approach", "career_goals"]),

        # Circular paths for indefinite conversation
        ("synthesis", ["new_territory", "deeper_dive", "noah_connection"]),
        ("new_territory", ["tech_stack", "enterprise_adaptation", "noahs_background"]),
    ],
    "software_developer": [
        # Phase 1: Code Architecture
        ("architecture", ["code_structure", "module_design", "dependencies"]),
        ("code_structure", ["flows", "nodes", "state_management"]),
        ("module_design", ["rag_engine", "response_generator", "retrievers"]),

        # Phase 2: Implementation Details
        ("implementation", ["langgraph_flow", "pgvector_queries", "prompt_templates"]),
        ("langgraph_flow", ["node_logic", "edge_conditions", "state_transitions"]),
        ("pgvector_queries", ["similarity_search", "indexing", "optimization"]),

        # Phase 3: Quality & Testing
        ("debugging", ["tracing", "logging", "error_handling"]),
        ("testing", ["pytest", "mocking", "integration_tests"]),
        ("performance", ["latency", "caching", "async_patterns"]),

        # Phase 4: DevOps
        ("deployment", ["vercel", "environment_config", "secrets_management"]),
        ("monitoring", ["langsmith", "supabase_analytics", "alerts"]),

        # Circular
        ("synthesis", ["architecture", "implementation", "noahs_code"]),
    ],
    "hiring_manager_nontechnical": [
        # Phase 1: Career Overview
        ("career", ["achievements", "growth_trajectory", "unique_value"]),
        ("achievements", ["tesla_impact", "project_outcomes", "metrics"]),

        # Phase 2: Business Value
        ("business_impact", ["roi_examples", "efficiency_gains", "team_value"]),
        ("team_fit", ["collaboration", "communication", "leadership"]),

        # Phase 3: Conversion
        ("resume", ["skills_summary", "experience_highlights", "contact"]),
        ("contact", ["availability", "next_steps", "references"]),

        # Circular
        ("synthesis", ["career", "business_impact", "unique_value"]),
    ],
    "explorer": [
        # Phase 1: Casual Exploration
        ("casual", ["what_is_this", "how_it_works", "who_built_it"]),
        ("what_is_this", ["architecture_overview", "cool_features", "demo"]),

        # Phase 2: Deeper Interest
        ("architecture", ["rag_basics", "ai_explained", "under_the_hood"]),
        ("behind_scenes", ["how_i_think", "my_memory", "my_limitations"]),

        # Phase 3: Fun Stuff
        ("fun_facts", ["mma_background", "chess_ai_story", "hidden_features"]),
        ("mma_background", ["fight_record", "training", "discipline"]),

        # Circular
        ("synthesis", ["casual", "architecture", "fun_facts"]),
    ]
}


# ============================================================================
# MAIN KNOWLEDGE PILLARS - Core topics to guide users through
# ============================================================================

MAIN_PILLARS = {
    "hiring_manager_technical": {
        "pillars": [
            # 5 pillars for technical hiring managers
            ("orchestration", "Explore the orchestration layer: how nodes, states, and safeguards work together"),
            ("tech_stack", "See my full tech stack: frontend, backend, observability"),
            ("enterprise", "See how this architecture adapts for enterprise use cases"),
            ("data_pipeline", "Learn about data pipeline management: embeddings, vector storage, analytics"),
            ("noahs_background", "Learn about Noah's technical background and projects"),
        ],
        "topic_to_pillar": {
            # Orchestration pillar (the "brain")
            "orchestration": "orchestration", "langgraph": "orchestration", "nodes": "orchestration",
            "state_management": "orchestration", "memory": "orchestration", "pipeline": "orchestration",
            "flow": "orchestration", "langgraph_nodes": "orchestration", "state": "orchestration",
            "safeguards": "orchestration", "conversation_flow": "orchestration",

            # Tech stack pillar (architecture, frontend, backend, observability)
            "architecture": "tech_stack", "tech_stack": "tech_stack", "frontend": "tech_stack",
            "backend": "tech_stack", "observability": "tech_stack", "langsmith": "tech_stack",
            "rag_pipeline": "tech_stack", "rag": "tech_stack", "supabase": "tech_stack",

            # Data pipeline pillar (separate from tech_stack)
            "data_pipeline": "data_pipeline", "data_flow": "data_pipeline", "data flow": "data_pipeline",
            "embedding": "data_pipeline", "embeddings": "data_pipeline", "vector": "data_pipeline",
            "pgvector": "data_pipeline", "data_layer": "data_pipeline", "data": "data_pipeline",
            "knowledge_base": "data_pipeline", "knowledge base": "data_pipeline", "migration": "data_pipeline",
            "ingest": "data_pipeline", "ingestion": "data_pipeline", "chunking": "data_pipeline",
            "analytics": "data_pipeline", "logging": "data_pipeline", "tracking": "data_pipeline",
            "metrics": "data_pipeline", "dashboard": "data_pipeline",

            # Enterprise pillar
            "enterprise": "enterprise", "enterprise_adaptation": "enterprise", "customer_support": "enterprise",
            "internal_docs": "enterprise", "sales_enablement": "enterprise", "deployment": "enterprise",
            "cost_analysis": "enterprise", "scaling": "enterprise", "cost": "enterprise",
            "production": "enterprise", "adapt": "enterprise",

            # Noah's background pillar
            "noahs_background": "noahs_background", "career": "noahs_background", "noah": "noahs_background",
            "resume": "noahs_background", "projects": "noahs_background", "tesla": "noahs_background",
            "background": "noahs_background", "skills": "noahs_background", "experience": "noahs_background",
        }
    },
    "software_developer": {
        "pillars": [
            # Lead with implementation (code-focused users want to see code first)
            ("implementation", "Explore the LangGraph implementation: nodes, state, pgvector queries"),
            ("architecture", "See the code architecture: modules, flows, design patterns"),
            ("debugging", "See the debugging and testing strategies"),
            ("noahs_background", "Learn about Noah's technical background"),
        ],
        "topic_to_pillar": {
            # Implementation pillar (first - show the code)
            "implementation": "implementation", "langgraph_flow": "implementation", "pgvector_queries": "implementation",
            "code": "implementation", "langgraph": "implementation", "stategraph": "implementation",
            # Data pipeline keywords for developers
            "data_pipeline": "implementation", "embedding": "implementation", "retrieval": "implementation",
            "vector_search": "implementation", "similarity": "implementation",

            # Architecture pillar
            "architecture": "architecture", "code_structure": "architecture", "module_design": "architecture",
            "design_patterns": "architecture", "modules": "architecture", "flows": "architecture",

            # Debugging pillar
            "debugging": "debugging", "testing": "debugging", "performance": "debugging",
            "pytest": "debugging", "error_handling": "debugging", "tracing": "debugging",

            # Noah's background
            "noahs_background": "noahs_background", "career": "noahs_background", "noah": "noahs_background",
        }
    },
    "hiring_manager_nontechnical": {
        "pillars": [
            ("career", "See Noah's career journey and achievements"),
            ("business_impact", "Understand the business value Noah delivers"),
            ("team_fit", "See how Noah works with teams"),
            ("resume", "Get Noah's resume and contact info"),
        ],
        "topic_to_pillar": {
            "career": "career", "achievements": "career", "growth_trajectory": "career",
            "business_impact": "business_impact", "roi_examples": "business_impact", "efficiency_gains": "business_impact",
            "team_fit": "team_fit", "collaboration": "team_fit", "communication": "team_fit",
            "resume": "resume", "contact": "resume", "availability": "resume",
        }
    },
    "explorer": {
        "pillars": [
            ("what_is_this", "Discover what I am and how I work"),
            ("architecture", "See the technology behind me"),
            ("fun_facts", "Learn some fun facts about Noah"),
            ("noahs_background", "Learn about Noah's background"),
        ],
        "topic_to_pillar": {
            "casual": "what_is_this", "what_is_this": "what_is_this", "how_it_works": "what_is_this",
            "architecture": "architecture", "behind_scenes": "architecture", "rag_basics": "architecture",
            "fun_facts": "fun_facts", "mma_background": "fun_facts", "hidden_features": "fun_facts",
            "noahs_background": "noahs_background", "who_built_it": "noahs_background",
        }
    },
}


def _get_explored_pillars(session_memory: Dict, role_mode: str) -> Set[str]:
    """Get which main pillars have been explored based on topics discussed.

    Args:
        session_memory: Session memory dict containing topics list
        role_mode: User's role mode for pillar mapping

    Returns:
        Set of explored pillar keys
    """
    topics = session_memory.get("topics", [])
    pillar_config = MAIN_PILLARS.get(role_mode, {})
    topic_to_pillar = pillar_config.get("topic_to_pillar", {})

    explored = set()
    for topic in topics:
        topic_lower = topic.lower()
        if topic_lower in topic_to_pillar:
            explored.add(topic_to_pillar[topic_lower])

    return explored


def _get_unexplored_pillars(session_memory: Dict, role_mode: str) -> List[Tuple[str, str]]:
    """Get unexplored pillars with user-friendly prompts.

    Args:
        session_memory: Session memory dict containing topics list
        role_mode: User's role mode for pillar mapping

    Returns:
        List of (pillar_key, user_friendly_prompt) tuples for unexplored pillars
    """
    pillar_config = MAIN_PILLARS.get(role_mode, {})
    all_pillars = pillar_config.get("pillars", [])
    explored = _get_explored_pillars(session_memory, role_mode)

    return [(key, prompt) for key, prompt in all_pillars if key not in explored]


def _get_depth_options_for_pillar(pillar: str, role_mode: str = "hiring_manager_technical") -> List[str]:
    """Get specific depth options for each pillar.

    Used when user is exploring a pillar to offer meaningful next steps
    that go deeper into that specific area.

    Args:
        pillar: The pillar key (orchestration, tech_stack, enterprise, data_pipeline, noahs_background)
        role_mode: User's role mode for context

    Returns:
        List of specific depth prompts for the pillar
    """
    depth_map = {
        "orchestration": [
            "Dive into specific nodes like retrieve_chunks or generate_draft",
            "See how state flows through the pipeline"
        ],
        "tech_stack": [
            "Explore the Supabase setup in detail",
            "See the LangSmith observability implementation"
        ],
        "enterprise": [
            "See customer support adaptation patterns",
            "Explore production safeguards"
        ],
        "data_pipeline": [
            "See how embeddings are generated and stored",
            "Explore the analytics logging system"
        ],
        "noahs_background": [
            "See Noah's GitHub projects",
            "Learn about his certifications and training path"
        ],
    }
    return depth_map.get(pillar, [f"Go deeper into {pillar}"])


def _extract_current_topic_from_query(query: str, role_mode: str = "hiring_manager_technical") -> Optional[str]:
    """Extract the pillar being discussed in current query.

    This is used to:
    1. Filter the current topic OUT of follow-up suggestions (don't suggest what user just asked)
    2. Mark the current pillar as explored BEFORE generating follow-ups

    Args:
        query: User's current query
        role_mode: User's role mode for context

    Returns:
        Pillar key if detected (orchestration, tech_stack, enterprise, data_pipeline, noahs_background), None otherwise
    """
    query_lower = query.lower()

    # Noah's background pillar keywords
    if any(kw in query_lower for kw in [
        "noah", "background", "career", "resume", "tesla", "experience",
        "skills", "certifications", "projects", "github", "linkedin",
        "who built", "about noah", "noah's"
    ]):
        return "noahs_background"

    # Orchestration pillar keywords
    if any(kw in query_lower for kw in [
        "orchestration", "nodes", "state management", "memory", "langgraph",
        "pipeline", "flow", "safeguards", "conversation flow", "state graph",
        "how do you think", "how does the pipeline"
    ]):
        return "orchestration"

    # Data pipeline pillar keywords (separate from tech_stack)
    if any(kw in query_lower for kw in [
        "data pipeline", "embedding", "embeddings", "vector", "pgvector",
        "data flow", "chunking", "analytics", "ingestion", "ingest",
        "knowledge base", "migration", "metrics", "logging", "tracking"
    ]):
        return "data_pipeline"

    # Tech stack pillar keywords (architecture, frontend, backend, observability)
    if any(kw in query_lower for kw in [
        "tech stack", "architecture", "frontend", "backend", "rag",
        "supabase", "observability", "langsmith", "full stack", "retrieval"
    ]):
        return "tech_stack"

    # Enterprise pillar keywords
    if any(kw in query_lower for kw in [
        "enterprise", "adapt", "customer support", "scaling", "production",
        "deployment", "cost", "internal docs", "sales enablement",
        "how would this scale", "enterprise use"
    ]):
        return "enterprise"

    # For software_developer role, check implementation-specific keywords
    if role_mode == "software_developer":
        if any(kw in query_lower for kw in ["code", "implementation", "show me the code"]):
            return "implementation"
        if any(kw in query_lower for kw in ["debug", "test", "error"]):
            return "debugging"

    return None


def _predict_next_topics(role_mode: str, current_topics: List[str], conversation_phase: str) -> List[str]:
    """Predict likely next topics based on conversation arc and phase.

    Args:
        role_mode: User's role mode
        current_topics: List of current topics (use last 3-5)
        conversation_phase: Current conversation phase

    Returns:
        List of predicted next topics
    """
    arcs = CONVERSATION_ARCS.get(role_mode, [])

    # Use most recent topic to predict next
    if current_topics:
        recent_topic = current_topics[-1].lower()
        for topic, next_topics in arcs:
            if topic in recent_topic:
                # Filter based on phase
                if conversation_phase == "synthesis":
                    # In synthesis phase, suggest connecting topics
                    return ["synthesis", "big_picture", "connect_dots"]
                return next_topics

    return []


def _generate_synthesis_response(state: ConversationState) -> str:
    """Generate synthesis response connecting multiple turns.

    Uses conversational style (not teaching) to connect dots.

    Args:
        state: Conversation state with chat_history, session_memory

    Returns:
        Synthesis response string
    """
    session_memory = state.get("session_memory", {})
    topics = session_memory.get("topics", [])
    chat_history = state.get("chat_history", [])

    # Extract turn-by-turn summary
    turn_summaries = []
    turn_num = 1
    for i in range(0, len(chat_history), 2):
        if i < len(chat_history):
            user_msg = chat_history[i]
            if isinstance(user_msg, dict):
                query = user_msg.get("content", "")[:100]
            else:
                query = getattr(user_msg, "content", "")[:100] if hasattr(user_msg, "content") else ""

            if query:
                turn_summaries.append(f"Turn {turn_num}: {query}")
                turn_num += 1

    # Synthesize topics
    topics_text = ", ".join(topics[-5:]) if topics else "various topics"

    response = f"""We have covered a lot of ground. Here is how it connects.

{chr(10).join(turn_summaries[-5:]) if turn_summaries else "We have explored several topics."}

The through-line across {topics_text} is [synthesis explanation]. The most interesting thread to pull on next is [suggest one natural next step based on conversation arc]."""

    return response


def _build_followups(variant: str, intent: str = "general", active_subcats: List[str] = None, role_mode: str = "explorer", chat_history: List[Dict] = None, session_memory: Dict = None, quality_warnings: str = None, guidance_flags: List[str] = None, conversation_phase: str = None, query: str = None) -> List[str]:
    """Generate role-specific followup prompt suggestions with conversation guidance and phase awareness.

    Enhanced with conversation guidance for quality issues, natural progression, and phase-aware suggestions.
    Uses sliding window analysis (last 6 messages) for scalability with 100+ turns.

    CRITICAL: Filters out the current topic from suggestions - we don't suggest what user just asked about!

    Args:
        variant: "engineering" | "business" | "mixed" (layout variant)
        intent: Query intent/type (technical, data, career, etc.)
        active_subcats: Active technical subcategories for precision targeting
        role_mode: User's role (explorer, software_developer, etc.)
        chat_history: Conversation history (only last 6 messages analyzed)
        session_memory: Session memory with topics (only last 10 topics used)
        quality_warnings: Quality warning string from validate_answer_quality
        guidance_flags: List of guidance flags from validate_conversation_guidance
        conversation_phase: Current phase: discovery/exploration/synthesis/extended
        query: Current user query (used to filter current topic from suggestions)

    Returns:
        List of 3 followup prompt strings

    Example:
        >>> _build_followups("engineering", "technical", guidance_flags=["stuck_need_redirection"])
        ["Want to explore a different aspect?", ...]
    """
    active_subcats = active_subcats or []
    chat_history = chat_history or []
    session_memory = session_memory or {}
    guidance_flags = guidance_flags or []

    # Confession mode gets no followups (privacy)
    if role_mode == "confession":
        return []

    # =========================================================================
    # PRIORITY 0: PILLAR-AWARE GUIDANCE (guide users through 4 main pillars)
    # =========================================================================

    # CRITICAL: Get current topic from query to EXCLUDE from suggestions
    # We should never suggest what the user just asked about!
    current_topic_pillar = _extract_current_topic_from_query(query or "", role_mode) if query else None

    unexplored_pillars = _get_unexplored_pillars(session_memory, role_mode)
    explored_pillars = _get_explored_pillars(session_memory, role_mode)
    topics = session_memory.get("topics", []) if session_memory else []

    # Filter OUT the current topic from unexplored pillars
    # Don't suggest "Learn about Noah's background" when user just asked about Noah!
    if current_topic_pillar:
        unexplored_pillars = [(key, prompt) for key, prompt in unexplored_pillars if key != current_topic_pillar]
        logger.debug(f"Filtered current topic '{current_topic_pillar}' from follow-up suggestions")

    # If current topic WAS the only unexplored pillar, user is about to complete all 5
    # Offer depth on current topic + synthesis instead of falling through to generic
    if not unexplored_pillars and current_topic_pillar:
        logger.info(f"User exploring last pillar '{current_topic_pillar}' - offer depth + synthesis")
        depth_options = _get_depth_options_for_pillar(current_topic_pillar, role_mode)
        return [
            depth_options[0] if depth_options else f"Go deeper into {current_topic_pillar}",
            "See how all these pieces connect end-to-end",
            "Ready to see Noah's resume and LinkedIn?"
        ]

    # Check for pillar depth warning (user has been in same pillar for 3+ turns)
    pillar_depth = session_memory.get("pillar_depth", {}) if session_memory else {}
    deep_pillar = None
    for pillar, depth in pillar_depth.items():
        if depth >= 3 and pillar != current_topic_pillar:
            deep_pillar = pillar
            break

    # If user has gone deep (3+ turns) on one pillar, actively guide to others
    if deep_pillar and unexplored_pillars:
        logger.info(f"User deep in '{deep_pillar}' pillar - actively guiding to other pillars")
        pillar_prompts = []
        pillar_prompts.append(f"You've explored {deep_pillar} in depth! {unexplored_pillars[0][1]}")
        if len(unexplored_pillars) > 1:
            pillar_prompts.append(unexplored_pillars[1][1])
        pillar_prompts.append("See how all the pieces connect end-to-end")
        return pillar_prompts[:3]

    # =========================================================================
    # FIX 6: SMART FOLLOWUPS - Always: 1 depth option + 1-2 unexplored pillars
    # Per spec: "would you like me to list all my nodes and states or would you
    # like to see how enterprises use similar programs for customer support?"
    # =========================================================================
    if current_topic_pillar or topics:
        smart_followups = []

        # 1. First option: Depth into current topic
        if current_topic_pillar:
            depth_options = _get_depth_options_for_pillar(current_topic_pillar, role_mode)
            if depth_options:
                smart_followups.append(depth_options[0])
            else:
                smart_followups.append(f"Go deeper into {current_topic_pillar}")
        elif topics:
            # Fallback to most recent topic
            smart_followups.append(f"Go deeper into {topics[-1]}")

        # 2. Second option: First unexplored pillar
        if unexplored_pillars:
            smart_followups.append(unexplored_pillars[0][1])

        # 3. Third option: Second unexplored pillar OR synthesis
        if len(unexplored_pillars) > 1:
            smart_followups.append(unexplored_pillars[1][1])
        elif len(explored_pillars) >= 3:
            smart_followups.append("See how all these pieces connect end-to-end")
        elif current_topic_pillar != "noahs_background":
            smart_followups.append("Learn about Noah's technical background and projects")

        logger.debug(f"Smart follow-ups (Fix 6): depth + unexplored pillars: {smart_followups}")
        return smart_followups[:3]

    # If all pillars explored (5), offer synthesis and resume
    if len(explored_pillars) >= 5:
        logger.info("All 5 pillars explored - offering synthesis and resume")
        return [
            "Want to see how all these pieces connect end-to-end?",
            "Ready to see Noah's resume and LinkedIn?",
            "Should we go deeper into any specific area?"
        ]

    # =========================================================================
    # PRIORITY 1: Quality-based guidance (if user is stuck or answer was repetitive)
    # =========================================================================
    if quality_warnings and "answer_too_similar" in quality_warnings:
        # User is stuck - suggest unexplored pillars instead of generic topics
        if unexplored_pillars:
            return [prompt for _, prompt in unexplored_pillars[:3]]
        return [
            "Want to explore a different aspect of the system?",
            "Curious about the data layer or deployment instead?",
            "Should we look at production patterns or enterprise use cases?"
        ]

    # =========================================================================
    # PRIORITY 1.5: Context-aware follow-ups based on current topic
    # Guide back to unexplored pillars instead of going deeper into sub-topics
    # =========================================================================
    if chat_history:
        last_user_msg = None
        for msg in reversed(chat_history[-6:]):  # Check last 6 messages
            if isinstance(msg, dict):
                if msg.get("role") == "user" or msg.get("type") == "human":
                    last_user_msg = msg.get("content", "")
                    break
            elif hasattr(msg, "type") and (msg.type == "human" or getattr(msg, "role", None) == "user"):
                last_user_msg = getattr(msg, "content", "")
                break

        if last_user_msg:
            query_lower = last_user_msg.lower()

            # Check if user is going deep into enterprise - guide back to other pillars
            is_enterprise_query = any(kw in query_lower for kw in ["adapt", "adapts", "adaptation", "customer support", "enterprise", "use case"])

            if is_enterprise_query and unexplored_pillars:
                # Instead of going deeper into enterprise, offer unexplored pillars
                pillar_prompts = [prompt for _, prompt in unexplored_pillars[:2]]
                pillar_prompts.append("Go deeper into enterprise adaptation patterns")
                return pillar_prompts[:3]

    # =========================================================================
    # PRIORITY 2: Conversation guidance flags (from validate_conversation_guidance)
    # =========================================================================
    if guidance_flags:
        if "stuck_need_redirection" in guidance_flags:
            return [
                "Want to explore a different topic area?",
                "Curious about the backend architecture or data pipeline?",
                "Should we look at testing strategies or deployment patterns?"
            ]

        if "missing_enterprise_guidance" in guidance_flags:
            return [
                "Want to see how this adapts to customer support?",
                "Curious about enterprise deployment patterns?",
                "Should I show the production safeguards that make this enterprise-ready?"
            ]

        if "suggest_depth_increase" in guidance_flags:
            return [
                "Want to go deeper into the implementation details?",
                "Curious about the code-level architecture?",
                "Should I show the actual node implementations?"
            ]

        if "suggest_synthesis" in guidance_flags:
            return [
                "Want to see how these pieces connect together?",
                "Curious about the end-to-end flow across all these topics?",
                "Should I synthesize how the architecture, data, and deployment work together?"
            ]

        if "suggest_new_territory" in guidance_flags:
            return [
                "We've covered a lot! Want to explore something completely different?",
                "Curious about an area we haven't touched yet?",
                "Should we revisit any topic with a fresh perspective?"
            ]

    # PRIORITY 3: Phase-aware suggestions (when no specific guidance flags)
    if conversation_phase == "synthesis":
        return [
            "Want to see how all these pieces connect end-to-end?",
            "Ready for a synthesis of what we've covered?",
            "Should I map out how the architecture, data, and deployment work together?"
        ]

    if conversation_phase == "extended":
        return [
            "We've covered a lot! Want to explore something completely different?",
            "Curious about an area we haven't touched yet?",
            "Should we revisit any topic with fresh perspective?"
        ]

    # PRIORITY 3: Conversation flow analysis for natural progression
    # Use sliding window: only analyze last 6 messages (scalable for 100+ turns)
    if chat_history and len(chat_history) >= 2:
        # Analyze conversation pattern to infer progression (scalable version)
        flow_analysis = _analyze_conversation_flow(chat_history, session_memory)

        # Use conversation arc prediction for proactive follow-ups
        topics = session_memory.get("topics", []) if session_memory else []
        predicted_topics = _predict_next_topics(role_mode, topics[-3:], conversation_phase)

        if predicted_topics:
            # Generate follow-ups based on predicted next topics
            if "synthesis" in predicted_topics or "big_picture" in predicted_topics:
                return [
                    "Want to see how all these pieces connect end-to-end?",
                    "Ready for a synthesis of what we've covered?",
                    "Should I map out how everything works together?"
                ]
            elif "enterprise_adaptation" in predicted_topics:
                return [
                    "Want to see how this adapts to enterprise use cases?",
                    "Curious about customer support or internal documentation applications?",
                    "Should I show how the architecture scales to different domains?"
                ]
            elif "implementation" in predicted_topics or "code_walkthrough" in predicted_topics:
                return [
                    "Want to see the actual code implementation?",
                    "Curious about the code-level architecture?",
                    "Should I show the node implementations?"
                ]
            elif "cost_analysis" in predicted_topics:
                return [
                    "Want to see the cost breakdown at scale?",
                    "Curious about the pricing model?",
                    "Should I show the ROI calculations?"
                ]

        # Generate context-aware followups based on progression pattern
        if flow_analysis.get("pattern") == "orchestration_to_enterprise":
            return [
                "Want to see how orchestration scales in enterprise deployments?",
                "Curious about the enterprise patterns built on this architecture?",
                "Should I show the production safeguards that make this enterprise-ready?"
            ]
        elif flow_analysis.get("pattern") == "architecture_to_implementation":
            return [
                "Want to see the code behind this architecture?",
                "Curious about the specific implementation details?",
                "Should I walk through the technical implementation?"
            ]
        elif flow_analysis.get("pattern") == "general_to_specific":
            # User is getting more specific - offer deeper dives
            return [
                "Want to dive deeper into the technical details?",
                "Curious about how this works under the hood?",
                "Should I show you the implementation or walk through the code?"
            ]

    # Subcategory-specific followups take priority for technical queries
    if intent in {"technical", "engineering"} and active_subcats:
        return _build_subcategory_followups(active_subcats)

    # Intent-based followups
    if intent in {"technical", "engineering"}:
        return [
            "Want me to walk through the LangGraph node transitions in detail?",
            "Curious how the Supabase pgvector query works under load?",
            "Should we map this architecture to your internal stack?",
        ]
    elif intent in {"data", "analytics"}:
        return [
            "Need the retrieval accuracy metrics for last week?",
            "Want the cost-per-query breakdown?",
            "Should we compare grounding confidence across roles?",
        ]
    elif intent in {"career", "general"}:
        return [
            "Want the story behind building this assistant end to end?",
            "Should I outline Noah's production launch checklist?",
            "Curious how this adapts to your team's workflow?",
        ]

    # Variant-based fallback (original logic)
    if variant == "engineering":
        return [
            "Walk through the LangGraph node transitions",
            "Inspect the pgvector retrieval implementation",
            "Map this pattern onto your stack",
        ]
    if variant == "business":
        return [
            "Review the rollout checklist for enterprise teams",
            "Estimate cost savings for your workflow",
            "Explore adoption risks and mitigation steps",
        ]
    return [
        "See how the architecture could be adapted to customer support",
        "Peek at the analytics dashboard",
        "Ask for the testing and QA strategy",
    ]
