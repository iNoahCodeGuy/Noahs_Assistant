"""Query classification utilities.

This module detects what kind of question the user is asking so we can
route it appropriately and plan the right follow-up actions.

Types we detect:
- Technical queries (architecture, code, stack questions)
- Career queries (resume, experience, achievements)
- Data display requests (show analytics, metrics)
- MMA queries (Noah's fight history)
- Fun queries (hobbies, fun facts)

Vague query expansion:
- Detects single-word or very short queries that need context enrichment
- Expands them into fuller questions to improve retrieval quality

Design Principles Applied:
- SRP: This module only classifies queries, doesn't retrieve or generate
- Loose Coupling: Communicates via state dict, no direct node calls
- Defensibility: Fail-fast validation on required fields
- Maintainability: Pure helper functions separated from I/O
"""

import re
import logging
from typing import Dict, Any, Optional

# Import NEW TypedDict state (Phase 3A migration)
from assistant.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)

TOPIC_KEYWORDS = {
    "architecture": [
        "architecture",
        "system design",
        "diagram",
        "pipeline",
        "orchestration",
        "flow",
        "graph",
        "node",
        "frontend",
        "backend",
        "full stack",
        "infra",
        "infrastructure",
        "orchestrator",
        "deployment",
        "scale",
        "scalability",
    ],
    "data": [
        "data layer",
        "pgvector",
        "supabase",
        "vector store",
        "embeddings",
        "similarity",
        "index",
        "schema",
        "analytics",
        "metrics",
        "dataset",
        "latency",
        "performance",
        "p95",
        "p99",
        "throughput",
        "cost",
        "budget",
    ],
    "retrieval": [
        "retrieval",
        "rag",
        "search",
        "query",
        "chunk",
        "grounding",
        "context",
    ],
    "testing": [
        "testing",
        "pytest",
        "qa",
        "quality",
        "reliability",
        "monitoring",
        "observability",
    ],
    "career": [
        "career",
        "experience",
        "background",
        "resume",
        "history",
        "projects",
    ],
    "mma": [
        "mma",
        "fight",
        "ufc",
        "bout",
        "cage",
    ],
    "fun": [
        "fun fact",
        "hobby",
        "hot dog",
        "interesting fact",
    ],
}


def detect_topic_focus(query: str) -> str:
    """Detect the primary topic focus for conversational follow-ups."""
    lowered = query.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return topic
    return "general"


# Ambiguous queries that should trigger "Ask Mode" - Portfolia asks clarifying questions
# These are too broad and should prompt the user to specify what aspect they want
AMBIGUOUS_QUERIES = {
    "engineering": {
        "options": ["frontend", "backend", "data pipelines", "architecture", "qa/testing", "deployment"],
        "context": "Portfolia herself as example (frontend chat UI, backend RAG, data pipeline, architecture design, testing, Vercel deployment)"
    },
    "engineer": {
        "options": ["frontend", "backend", "data pipelines", "architecture", "qa/testing", "deployment"],
        "context": "Portfolia herself as example"
    },
    "technical": {
        "options": ["code examples", "architecture", "stack choices", "system design", "performance"],
        "context": "Portfolia's implementation"
    },
    "architecture": {
        "options": ["frontend", "backend", "data layer", "full stack overview"],
        "context": "Portfolia's architecture (Streamlit+Vercel frontend, LangGraph backend, pgvector data)"
    },
    "show me how you work": {
        "options": ["code snippets", "data flow", "system diagram", "high-level explanation"],
        "context": "How Portfolia answers questions"
    },
    "how do you work": {
        "options": ["code snippets", "data flow", "system diagram", "high-level explanation"],
        "context": "RAG pipeline and conversation flow"
    },
    "how does this work": {
        "options": ["code examples", "data flow", "architecture diagram", "plain English"],
        "context": "Portfolia's RAG system"
    },
}

# Vague query expansion mappings (for queries that are specific enough but too short)
VAGUE_QUERY_EXPANSIONS = {
    "skills": "What technical skills does Noah have in software engineering and AI?",
    "ai": "What is Noah's experience with AI, machine learning, and GenAI systems?",
    "genai": "What does Noah understand about production GenAI systems and patterns?",
    "rag": "What is Noah's experience with Retrieval-Augmented Generation and how did he implement it?",
    "python": "How strong is Noah's Python and what has he built with it?",
    "projects": "What projects has Noah built with AI and software engineering?",
    "experience": "What is Noah's career experience and technical background?",
    "background": "What is Noah's background in software engineering and career history?",
    "tesla": "What is Noah's role at Tesla and how has he contributed to AI initiatives?",
    "debugging": "How does Noah debug and troubleshoot applications?",
    "testing": "What testing and quality assurance practices does Noah follow?",
    "deployment": "What is Noah's experience with deploying applications to production?",
    "databases": "What database technologies and patterns has Noah worked with?",
    "apis": "What API integration experience does Noah have?",
}


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


def _detect_file_request(query: str) -> Optional[str]:
    """Detect if query requests a specific file to be read.

    Purpose: Enable on-demand file reading when users ask to see specific files.
    Detects patterns like "show me file X", "read X.py", "open X", etc.

    Args:
        query: User's query text

    Returns:
        File path if detected, None otherwise

    Examples:
        >>> _detect_file_request("show me assistant/core/rag_engine.py")
        "assistant/core/rag_engine.py"
        >>> _detect_file_request("read the pgvector_retriever file")
        None  # Too vague, no specific path
    """
    lowered = query.lower()

    # Patterns that indicate file requests
    file_patterns = [
        r'show\s+me\s+(?:the\s+)?(?:file\s+)?([a-zA-Z0-9_/\.\-]+\.(?:py|tsx?|jsx?|md))',
        r'read\s+(?:the\s+)?(?:file\s+)?([a-zA-Z0-9_/\.\-]+\.(?:py|tsx?|jsx?|md))',
        r'open\s+(?:the\s+)?(?:file\s+)?([a-zA-Z0-9_/\.\-]+\.(?:py|tsx?|jsx?|md))',
        r'display\s+(?:the\s+)?(?:file\s+)?([a-zA-Z0-9_/\.\-]+\.(?:py|tsx?|jsx?|md))',
        r'view\s+(?:the\s+)?(?:file\s+)?([a-zA-Z0-9_/\.\-]+\.(?:py|tsx?|jsx?|md))',
        r'see\s+(?:the\s+)?(?:file\s+)?([a-zA-Z0-9_/\.\-]+\.(?:py|tsx?|jsx?|md))',
    ]

    for pattern in file_patterns:
        match = re.search(pattern, lowered)
        if match:
            file_path = match.group(1)
            # Clean up common prefixes
            file_path = file_path.lstrip('./')
            return file_path

    # Also check for file paths that might be mentioned directly
    # Pattern: path/to/file.py or assistant/core/file.py
    direct_path_pattern = r'([a-zA-Z0-9_/]+/[a-zA-Z0-9_/]+\.(?:py|tsx?|jsx?|md))'
    match = re.search(direct_path_pattern, query)
    if match:
        file_path = match.group(1)
        # Check if it's in a context that suggests file reading
        context_keywords = ['show', 'read', 'open', 'display', 'view', 'see', 'file']
        query_before = query[:match.start()].lower()
        if any(keyword in query_before for keyword in context_keywords):
            return file_path

    return None


def _is_ambiguous_query(query: str) -> tuple[bool, dict | None]:
    """Check if query is ambiguous and should trigger Ask Mode.

    Ambiguous queries are too broad and should prompt Portfolia to ask
    clarifying questions about what specific aspect the user wants to explore.

    Args:
        query: Original user query

    Returns:
        Tuple of (is_ambiguous, ambiguity_config) where config contains
        options and context for crafting the clarifying question
    """
    # Check if query matches any ambiguous patterns
    lowered = query.lower().strip()
    clean_query = re.sub(r'[^\w\s]', '', lowered)

    if clean_query in AMBIGUOUS_QUERIES:
        return True, AMBIGUOUS_QUERIES[clean_query]

    # Also check if query is a longer phrase that matches
    for pattern, config in AMBIGUOUS_QUERIES.items():
        if pattern in lowered:
            return True, config

    return False, None


def _expand_vague_query(query: str) -> str:
    """Expand single-word or vague queries into fuller questions.

    When users ask vague questions like 'skills' or 'rag', we expand
    them into more specific queries that will retrieve better context from
    the knowledge base.

    Note: Ambiguous queries (like 'engineering', 'architecture') should
    NOT be expanded - they trigger Ask Mode instead.

    Args:
        query: Original user query

    Returns:
        Expanded query if match found, otherwise original query
    """
    # Check if query is very short (likely vague)
    if len(query.split()) <= 2:
        lowered = query.lower().strip()
        # Remove punctuation for matching
        clean_query = re.sub(r'[^\w\s]', '', lowered)

        if clean_query in VAGUE_QUERY_EXPANSIONS:
            expanded = VAGUE_QUERY_EXPANSIONS[clean_query]
            return expanded

    return query


def _should_ask_detail_preference(state: ConversationState) -> bool:
    """Determine if we should ask about detail level before responding.

    Returns False if:
    - Query has explicit detail signals ("brief", "quick", "detailed", "deep dive", etc.)
    - Role strongly indicates preference (Software Developer → detailed, Nontechnical HM → brief)
    - session_memory has established detail_preference
    - Query is menu selection

    Returns True if:
    - Query is broad ("tell me about", "explain", "how does", "what is")
    - No specific indicators in query (no "code", "sql", "implementation")
    - Early conversation (conversation_turn < 2) OR no established preference
    - Role is ambiguous ("Just Looking Around")

    Args:
        state: Current conversation state

    Returns:
        True if we should ask about detail preference, False otherwise
    """
    # Don't ask if menu selection
    if state.get("query_type") == "menu_selection" or state.get("menu_choice"):
        return False

    query = state.get("query", "").lower()

    # Check for explicit detail signals in query
    explicit_signals = [
        "brief", "quick", "summary", "overview", "short",
        "detailed", "comprehensive", "deep dive", "full", "in depth",
        "explain in detail", "walk me through", "step by step"
    ]
    if any(signal in query for signal in explicit_signals):
        return False

    # Check if role strongly indicates preference
    role_mode = state.get("role_mode", "").lower()
    if role_mode in ["software_developer", "hiring_manager_technical"]:
        # Technical roles default to detailed - don't ask
        return False
    elif role_mode in ["hiring_manager_nontechnical", "hiring_manager"]:
        # Non-technical roles default to brief - don't ask
        return False

    # Check if preference already established (in state or session memory)
    if state.get("detail_preference"):
        return False
    session_memory = state.get("session_memory", {})
    if session_memory.get("detail_preference"):
        return False

    # Check if query is broad enough to warrant asking
    broad_queries = [
        "tell me about", "explain", "how does", "what is", "what are",
        "show me", "walk me through", "describe", "help me understand"
    ]

    if not any(broad in query for broad in broad_queries):
        return False

    # Check if query has specific indicators that suggest detail level
    specific_indicators = [
        "code", "sql", "implementation", "architecture diagram",
        "show code", "display code", "source code"
    ]
    if any(indicator in query for indicator in specific_indicators):
        return False  # Query is specific enough to infer detail level

    # Check conversation turn - ask more in early conversation
    conversation_turn = state.get("conversation_turn", 0)
    if conversation_turn >= 2 and session_memory.get("detail_preference"):
        return False  # Preference should be established by now

    # Role is ambiguous or early conversation - ask about preference
    if role_mode in ["just_looking_around", "explorer", ""] or conversation_turn < 2:
        return True

    return False


def _parse_detail_preference(query: str) -> str | None:
    """Parse detail preference from user response.

    Detects patterns indicating user's detail preference:
    - "brief", "quick", "summary", "1" → "brief"
    - "moderate", "medium", "some detail", "2" → "moderate"
    - "detailed", "comprehensive", "deep dive", "full", "3" → "comprehensive"

    Handles both pure preference responses ("brief") and combined responses
    ("brief overview of RAG").

    Args:
        query: User's query text

    Returns:
        Preference string ("brief", "moderate", "comprehensive") or None if not detected
    """
    lowered = query.lower().strip()

    # Check for exact matches first (pure preference responses)
    if lowered in ["1", "brief", "quick", "summary", "overview"]:
        return "brief"
    if lowered in ["2", "moderate", "medium"]:
        return "moderate"
    if lowered in ["3", "detailed", "comprehensive", "full"]:
        return "comprehensive"

    # Check for preference patterns in longer queries
    # Brief indicators (check word boundaries to avoid false positives)
    brief_patterns = [r"\bbrief\b", r"\bquick\b", r"\bsummary\b", r"\boverview\b", r"\bshort\b", r"\bhigh level\b"]
    if any(re.search(pattern, lowered) for pattern in brief_patterns):
        return "brief"

    # Moderate indicators
    moderate_patterns = [r"\bmoderate\b", r"\bmedium\b", r"\bsome detail\b"]
    if any(re.search(pattern, lowered) for pattern in moderate_patterns):
        return "moderate"

    # Comprehensive indicators
    comprehensive_patterns = [
        r"\bdetailed\b", r"\bcomprehensive\b", r"\bdeep dive\b", r"\bfull\b",
        r"\bin depth\b", r"\bcomplete\b", r"\bthorough\b", r"\bextensive\b"
    ]
    if any(re.search(pattern, lowered) for pattern in comprehensive_patterns):
        return "comprehensive"

    return None


def classify_intent(state: ConversationState) -> Dict[str, Any]:
    """Classify the incoming query intent and return an updated state.

    This is the first node in the conversation pipeline. It looks at the
    user's question and figures out what category it falls into.

    First, it expands vague queries (like "engineering") into fuller questions
    to improve retrieval quality. Then it detects query type.

    Detects:
    - Code display requests (show/display code, how do you, implementation details)
    - Import/stack questions (why use X, what imports, explain dependencies)
    - Technical queries (architecture, retrieval, pipeline)
    - Career queries (resume, experience, achievements)
    - MMA queries (fight references)
    - Data display requests (show analytics, display data)
    - Fun queries (fun facts, hobbies)
    - Response length needs (teaching, why/how questions need longer explanations)

    Node Signature (LangGraph Pattern):
        - Accepts: ConversationState (TypedDict with all conversation data)
        - Returns: Dict[str, Any] (PARTIAL update with only modified fields)

    Design Principles:
        - Fail-Fast (Defensibility): Validates required fields before processing
        - Pure Logic Extraction (Maintainability): Business logic in helper functions
        - Loose Coupling: Returns dict update, doesn't modify input state

    Args:
        state: Current conversation state with the user's query

    Returns:
        Partial state update dict with classification results

    Raises:
        ValueError: If required fields (query, role) are missing
    """
    # Fail-fast validation (Defensibility principle)
    try:
        query = state["query"]
        role = state.get("role_mode") or state.get("role", "Developer")  # Persona already normalized upstream
    except KeyError as e:
        logger.error(f"classify_intent: Missing required field: {e}")
        return {
            "error": "classification_failed",
            "error_message": f"Missing required field for classification: {e}"
        }

    # Initialize partial update dict (only fields we're modifying)
    update: Dict[str, Any] = {}

    # PRIORITY: Check for detail preference response FIRST (user responding to clarification)
    # This handles cases where user is responding to a detail preference question
    detail_pref = _parse_detail_preference(query)
    if detail_pref:
        # User is responding with detail preference
        update["detail_preference"] = detail_pref
        update["detail_preference_needed"] = False
        # Store in session memory for persistence
        session_memory = state.get("session_memory", {})
        session_memory["detail_preference"] = detail_pref
        update["session_memory"] = session_memory
        logger.info(f"Detail preference detected: '{query}' → {detail_pref}")
        # Continue with normal classification after preference is set
        # Note: We'll skip asking for detail preference again since it's now set

    # PRIORITY: Check for menu selection (explicit navigation choice)
    # Matches: "1", "2", "3", "4", "1️⃣", "2️⃣", "3️⃣", "4️⃣", "option 1", etc.
    menu_pattern = r'^\s*(1️⃣|2️⃣|3️⃣|4️⃣|[1-4]|option\s*[1-4])\s*$'
    menu_match = re.match(menu_pattern, query.strip(), re.IGNORECASE)

    if menu_match:
        # Extract just the number (remove emoji suffix if present)
        menu_choice = query.strip().rstrip('️⃣').lower().replace('option', '').strip()
        update["query_type"] = "menu_selection"
        update["menu_choice"] = menu_choice
        update["intent_confidence"] = 1.0
        update["is_ambiguous"] = False
        update["clarification_needed"] = False  # Menu selections are explicit
        update["detail_preference_needed"] = False  # Menu selections don't need detail preference
        logger.info(f"Menu selection detected: '{query}' → option {menu_choice}")
        # DEBUG: Verify state update will propagate
        logger.info(f"✅ Menu choice SET in update dict: menu_choice={menu_choice}, query_type={update['query_type']}")
        return update  # Return partial update - LangGraph will merge into state

    chat_history = state.get("chat_history", [])
    user_turns = sum(1 for message in chat_history if message.get("role") == "user")
    if user_turns:
        update["conversation_turn"] = user_turns
        update["emotional_pacing"] = "surge" if user_turns % 2 else "reflect"
    else:
        update["conversation_turn"] = 0
        update["emotional_pacing"] = "surge"

    update["topic_focus"] = detect_topic_focus(query)

    # Check if we should ask about detail preference (before checking topic ambiguity)
    # Only check if not already responding to a detail preference question
    # Also check if preference wasn't just set in this turn
    if not detail_pref and "detail_preference" not in update:
        # Create a temporary state with the update merged to check if preference exists
        temp_state = {**state, **update}
        should_ask_detail = _should_ask_detail_preference(temp_state)
        if should_ask_detail:
            update["detail_preference_needed"] = True
            logger.info(f"Detail preference needed for query: '{query}'")

    # First, check if query is ambiguous (should trigger Ask Mode)
    is_ambiguous, ambiguity_config = _is_ambiguous_query(query)

    if is_ambiguous:
        update["topic_focus"] = detect_topic_focus(query)
        update["is_ambiguous"] = True  # Critical: downstream nodes check this flag
        update["ambiguous_query"] = True
        update["ambiguity_options"] = ambiguity_config["options"]
        update["ambiguity_context"] = ambiguity_config["context"]
        update["query_type"] = "ambiguous"
        logger.info(f"Ambiguous query detected: '{query}' → Ask Mode triggered")
        # Don't expand ambiguous queries - we want user to clarify
        return update  # Return partial update - LangGraph will merge into state

    # If not ambiguous, expand vague queries for better retrieval (pure function)
    expanded_query = _expand_vague_query(query)

    if expanded_query != query:
        update["expanded_query"] = expanded_query
        update["vague_query_expanded"] = True
        logger.info(f"Expanded vague query: '{query}' → '{expanded_query}'")

    lowered = query.lower()

    # Detect file reading requests (Layer 1: File Reading Infrastructure)
    file_path = _detect_file_request(query)
    if file_path:
        update["file_request"] = file_path
        update["query_type"] = "file_request"
        update["intent_confidence"] = 0.9
        logger.info(f"File request detected: '{query}' → {file_path}")
        # Continue with other classifications but mark as file request

    # Detect self-referential queries (Layer 5: Query Classification Enhancement)
    # Distinguish questions about Portfolia from questions about Noah
    self_referential_keywords = [
        "how do you", "how does this work", "show me your code", "how are you built",
        "your architecture", "your implementation", "how are you", "how do you work",
        "how does this system work", "show me how you", "explain how you",
        "what is your", "what are your", "your code", "your system",
        "how does your", "how do your", "your retrieval", "your pipeline",
        "how are you built", "how were you built", "explain yourself"
    ]
    is_self_referential = any(keyword in lowered for keyword in self_referential_keywords)

    if is_self_referential:
        update["is_self_referential"] = True
        update["query_type"] = "self_referential"
        logger.info(f"Self-referential query detected: '{query}'")

    # Detect when a longer teaching-focused response is needed
    # These queries require depth, explanation, and educational context
    teaching_keywords = [
        "why", "how does", "how did", "how do", "explain", "walk me through",
        "what is", "what are", "what's the difference", "compare",
        "help me understand", "break down", "teach me", "show me how",
        "architecture", "design", "pattern", "principle", "strategy",
        "trade-off", "tradeoff", "benefit", "advantage", "disadvantage",
        "when to use", "when should", "best practice", "enterprise"
    ]
    if any(keyword in lowered for keyword in teaching_keywords):
        update["needs_longer_response"] = True
        update["teaching_moment"] = True

    # Code display triggers (explicit requests to see code)
    code_display_keywords = [
        "show code", "display code", "show me code", "show the code",
        "show implementation", "display implementation",
        "how do you", "how does it", "how is it",
        "show me the", "show retrieval", "show api",
        "code snippet", "code example", "source code"
    ]
    if any(keyword in lowered for keyword in code_display_keywords):
        update["code_display_requested"] = True
        update["query_type"] = "technical"

    # PROACTIVE code detection - when code would clarify the answer for technical roles
    # Per PROJECT_REFERENCE_OVERVIEW: "proactively displays code snippets when they clarify concepts"
    # Per DATA_COLLECTION_AND_SCHEMA_REFERENCE: "If user is technical and seems unsure → proactively show code"
    proactive_code_topics = [
        # Implementation questions (even without "show me")
        "implement", "build", "create", "develop", "write",
        # Architecture/design questions that benefit from code
        "rag pipeline", "vector search", "retrieval", "embedding", "orchestration",
        "langgraph", "conversation flow", "node", "pipeline",
        # Technical concepts best shown with code
        "api route", "endpoint", "function", "class", "method",
        "pgvector", "supabase query", "database", "migration",
        "prompt engineering", "llm call", "generation",
        # Patterns that need examples
        "pattern", "approach", "technique", "strategy"
    ]

    # Only proactively show code for technical roles
    if role in ["Software Developer", "Hiring Manager (technical)"]:
        if any(topic in lowered for topic in proactive_code_topics):
            update["code_would_help"] = True
            update["query_type"] = "technical"
            logger.info(f"Proactive code detection: query '{query}' would benefit from code examples")

    # Import/stack explanation triggers (why did you choose X?)
    import_keywords = [
        "why use", "why choose", "why did you use", "why did you choose",
        "what imports", "explain imports", "your imports", "dependencies",
        "why supabase", "why openai", "why langchain", "why vercel",
        "why pgvector", "why twilio", "why resend",
        "justify", "trade-off", "alternative", "vs", "instead of",
        "enterprise", "production", "scale", "library", "libraries"
    ]

    # Specific library mentions
    library_names = [
        "supabase", "openai", "pgvector", "langchain", "langgraph",
        "vercel", "resend", "twilio", "langsmith", "streamlit"
    ]

    if any(keyword in lowered for keyword in import_keywords):
        update["import_explanation_requested"] = True
        update["query_type"] = "technical"
    elif any(lib in lowered for lib in library_names) and any(word in lowered for word in ["why", "what", "how", "explain"]):
        update["import_explanation_requested"] = True
        update["query_type"] = "technical"

    # Query type classification
    if any(re.search(r"\b" + k + r"\b", lowered) for k in ["mma", "fight", "ufc", "bout", "cage"]):
        update["query_type"] = "mma"
    elif any(term in lowered for term in ["fun fact", "hobby", "interesting fact", "hot dog"]):
        update["query_type"] = "fun"
    elif _is_data_display_request(lowered):
        update["data_display_requested"] = True
        update["query_type"] = "data"
    else:
        # PROACTIVE data detection - when analytics/metrics would clarify the answer
        # Questions about performance, usage, trends benefit from actual data
        proactive_data_topics = [
            # Performance/metrics questions
            "how many", "how much", "how often", "frequency",
            "performance", "metrics", "statistics", "stats",
            "usage", "activity", "engagement", "interactions",
            # Trend/pattern questions
            "trend", "pattern", "over time", "growth",
            "most common", "popular", "typical", "average",
            # Comparison questions that need numbers
            "compare", "difference between", "vs",
            # Success/outcome questions
            "success rate", "conversion", "effectiveness"
        ]

        if any(topic in lowered for topic in proactive_data_topics):
            update["data_would_help"] = True
            update["query_type"] = "data"
            logger.info(f"Proactive data detection: query '{query}' would benefit from analytics")

        # PROACTIVE chart detection - when a chart/graph would be clearer than a table
        # Charts are most valuable for trends, comparisons, distributions, and relationships
        # More specific than data detection - only when visualization adds clear value
        proactive_chart_topics = [
            # Time-based trends (line charts)
            "trend", "over time", "growth", "change over", "evolution", "how has",
            "increased", "decreased", "fluctuated", "varied",
            # Comparisons (bar charts)
            "compare", "vs", "versus", "difference between", "contrast",
            # Distributions (pie/donut charts, bar charts)
            "breakdown", "distribution", "percentage", "proportion", "split",
            "most common", "least common", "break down by",
            # Relationships (scatter plots, correlation)
            "correlation", "relationship between", "how does x relate to y",
            "connection between", "association"
        ]

        if any(topic in lowered for topic in proactive_chart_topics):
            update["chart_would_help"] = True
            # Don't override query_type if already set to data
            if "query_type" not in update:
                update["query_type"] = "data"
            logger.info(f"Proactive chart detection: query '{query}' would benefit from visualization")

    # Detect "how does [product/system/chatbot] work" queries as technical
    if any(term in lowered for term in ["code", "technical", "stack", "architecture", "implementation", "retrieval"]) \
         or (("how does" in lowered or "how did" in lowered or "explain how" in lowered)
             and any(word in lowered for word in ["product", "system", "chatbot", "assistant", "rag", "pipeline", "work", "built"])):
        # Override if not already set
        if "query_type" not in update:
            update["query_type"] = "technical"
    elif any(term in lowered for term in ["career", "resume", "cv", "experience", "achievement", "work"]):
        if "query_type" not in update:
            update["query_type"] = "career"
    elif "query_type" not in update:
        update["query_type"] = "general"

    update["topic_focus"] = detect_topic_focus(expanded_query)

    intent_label = update.get("query_type", "general")
    if any(keyword in lowered for keyword in ["latency", "cost", "reliability", "budget", "roi"]):
        intent_label = "business_value"

    update["query_intent"] = intent_label
    update["intent_confidence"] = 0.9

    # Return partial update - LangGraph will merge into state
    return update


def classify_query(state: ConversationState) -> Dict[str, Any]:
    """Backward-compatible alias preserving older node name."""
    return classify_intent(state)
