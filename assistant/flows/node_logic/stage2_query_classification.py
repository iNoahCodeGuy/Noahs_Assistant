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
import time
from typing import Dict, Any

# Import NEW TypedDict state (Phase 3A migration)
from assistant.state.conversation_state import ConversationState
from assistant.flows.node_logic.util_edge_case_detection import detect_edge_cases

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


def _is_ambiguous_query(query: str) -> tuple[bool, dict | None]:
    """Check if query is ambiguous and should trigger Ask Mode.

    Ambiguous queries are too broad and should prompt Portfolia to ask
    clarifying questions about what specific aspect the user wants to explore.

    Only flags queries that are:
    - Exactly the ambiguous word (e.g., "architecture")
    - Very short queries that are just the ambiguous word with minimal context
    - Does NOT flag specific questions that happen to contain the word (e.g., "architecture adapts to customer support")

    Args:
        query: Original user query

    Returns:
        Tuple of (is_ambiguous, ambiguity_config) where config contains
        options and context for crafting the clarifying question
    """
    # Check if query matches any ambiguous patterns
    lowered = query.lower().strip()
    clean_query = re.sub(r'[^\w\s]', '', lowered)

    # Split into words for more precise matching
    words = clean_query.split()
    word_count = len(words)

    # Exact match (single word or very short phrase)
    if clean_query in AMBIGUOUS_QUERIES:
        return True, AMBIGUOUS_QUERIES[clean_query]

    # Check for ambiguous patterns, but only if query is short (1-3 words)
    # This prevents "architecture adapts to customer support" from being flagged
    # while still catching "show me architecture" or "architecture overview"
    if word_count <= 3:
        for pattern, config in AMBIGUOUS_QUERIES.items():
            # Only match if pattern is the main focus (first word or standalone)
            if pattern == words[0] or pattern == clean_query:
                return True, config

    # For longer queries, only flag if it's a very generic pattern match
    # (e.g., "how does architecture work" but not "architecture adapts to customer support")
    if word_count > 3:
        # Check for very generic ambiguous patterns that indicate broad questions
        generic_patterns = ["how does this work", "show me how you work", "how do you work"]
        for pattern in generic_patterns:
            if pattern in lowered:
                return True, AMBIGUOUS_QUERIES.get(pattern)

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

    # PRIORITY: Check for menu selection FIRST (explicit navigation choice)
    # Matches: "1", "2", "3", "4", "1️⃣", "2️⃣", "3️⃣", "4️⃣", "option 1", etc.
    menu_pattern = r'^\s*(1️⃣|2️⃣|3️⃣|4️⃣|[1-4]|option\s*[1-4])\s*$'
    menu_match = re.match(menu_pattern, query.strip(), re.IGNORECASE)

    if menu_match:
        # Extract just the number (remove emoji suffix if present)
        menu_choice = query.strip().rstrip('️⃣').lower().replace('option', '').strip()

        # Detect repeated menu selection
        session_memory = state.setdefault("session_memory", {})
        last_menu_choice = session_memory.get("last_menu_choice")
        if last_menu_choice == menu_choice:
            # User selected same menu option twice
            update["is_repeated_menu_selection"] = True
            session_memory["repeated_menu_count"] = session_memory.get("repeated_menu_count", 0) + 1
            logger.info(f"Repeated menu selection detected: option {menu_choice} selected again (count: {session_memory['repeated_menu_count']})")
        else:
            session_memory["repeated_menu_count"] = 0
            update["is_repeated_menu_selection"] = False

        session_memory["last_menu_choice"] = menu_choice

        update["query_type"] = "menu_selection"
        update["menu_choice"] = menu_choice
        update["intent_confidence"] = 1.0
        update["is_ambiguous"] = False
        update["clarification_needed"] = False  # Menu selections are explicit
        logger.info(f"Menu selection detected: '{query}' → option {menu_choice}")
        # DEBUG: Verify state update will propagate
        logger.info(f"✅ Menu choice SET in update dict: menu_choice={menu_choice}, query_type={update['query_type']}")
        return update  # Return partial update - LangGraph will merge into state

    # Track query timestamp for rapid-fire detection
    session_memory = state.setdefault("session_memory", {})
    query_timestamps = session_memory.setdefault("query_timestamps", [])
    query_timestamps.append(time.time())
    # Keep only last 10 timestamps to avoid memory bloat
    if len(query_timestamps) > 10:
        query_timestamps[:] = query_timestamps[-10:]
    session_memory["query_timestamps"] = query_timestamps

    # PRIORITY: Detect edge cases BEFORE other classification
    # Edge cases need early detection to provide conversational responses
    edge_cases = detect_edge_cases(state)
    if edge_cases.get("edge_case_type"):
        # Edge case detected - flag it and return early
        update = {
            "edge_case_detected": True,
            "edge_case_type": edge_cases["edge_case_type"],
            **edge_cases  # Include all edge case flags and metadata
        }

        # Store in session memory for meta-teaching context (session_memory already initialized above)
        session_memory["last_edge_case"] = edge_cases["edge_case_type"]
        session_memory["last_edge_case_query"] = query

        logger.info(f"Edge case detected: {edge_cases['edge_case_type']} for query: '{query[:50]}...'")
        return update  # Return early, skip normal classification

    chat_history = state.get("chat_history", [])
    # Count user messages (support both dict format and LangGraph message objects)
    user_turns = 0
    for msg in chat_history:
        # Check dict format
        if isinstance(msg, dict):
            if msg.get("role") == "user" or msg.get("type") == "human":
                user_turns += 1
        # Check LangGraph message objects
        elif hasattr(msg, "type") and (msg.type == "human" or getattr(msg, "role", None) == "user"):
            user_turns += 1

    if user_turns:
        update["conversation_turn"] = user_turns
        update["emotional_pacing"] = "surge" if user_turns % 2 else "reflect"
    else:
        update["conversation_turn"] = 0
        update["emotional_pacing"] = "surge"

    # Validate conversation_turn calculation matches chat_history
    conversation_turn = update.get("conversation_turn", 0)
    if user_turns != conversation_turn:
        logger.warning(
            f"Conversation turn mismatch: calculated={user_turns}, stored={conversation_turn}. "
            f"This may indicate chat_history format inconsistency."
        )

    update["topic_focus"] = detect_topic_focus(query)

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

    # Multi-strategy confidence scoring for ambiguous queries
    confidence_scores = {
        "keyword_match": 0.0,
        "conversation_context": 0.0,
        "entity_extraction": 0.0
    }

    # Strategy 1: Keyword matching
    if any(kw in lowered for kw in ["code", "implementation", "show me"]):
        confidence_scores["keyword_match"] = 0.8
    elif any(kw in lowered for kw in ["how", "what", "explain"]):
        confidence_scores["keyword_match"] = 0.6
    else:
        confidence_scores["keyword_match"] = 0.4

    # Strategy 2: Conversation context
    session_memory = state.get("session_memory", {})
    if chat_history and len(chat_history) >= 2:
        # Extract content from messages (support both dict format and LangGraph message objects)
        recent_contents = []
        for msg in chat_history[-2:]:
            # Check dict format first
            if isinstance(msg, dict):
                content = msg.get("content", "")
            # Handle LangGraph message objects (Pydantic models)
            elif hasattr(msg, "content"):
                content = msg.content if hasattr(msg, "content") else ""
            else:
                content = ""
            recent_contents.append(content)
        recent_content = " ".join(recent_contents)
        session_topics = session_memory.get("topics", []) if session_memory else []
        if any(topic in recent_content.lower() for topic in session_topics):
            confidence_scores["conversation_context"] = 0.9  # High if previous context exists
        else:
            confidence_scores["conversation_context"] = 0.5
    else:
        confidence_scores["conversation_context"] = 0.3  # Lower for new conversations

    # Strategy 3: Entity extraction
    entities = state.get("entities", {})
    is_menu_selection = update.get("query_type") == "menu_selection"
    if entities.get("menu_context") or entities.get("menu_selection") or is_menu_selection:
        confidence_scores["entity_extraction"] = 0.7  # Menu selections are explicit
    else:
        confidence_scores["entity_extraction"] = 0.3

    # Aggregate confidence (weighted average)
    overall_confidence = (
        confidence_scores["keyword_match"] * 0.4 +
        confidence_scores["conversation_context"] * 0.4 +
        confidence_scores["entity_extraction"] * 0.2
    )
    update["intent_confidence"] = overall_confidence

    # Low confidence triggers clarification
    if overall_confidence < 0.5 and not is_menu_selection:
        update["clarification_needed"] = True
        logger.info(f"Low confidence query detected ({overall_confidence:.2f}), requesting clarification")

    # Return partial update - LangGraph will merge into state
    return update


def classify_query(state: ConversationState) -> Dict[str, Any]:
    """Backward-compatible alias preserving older node name."""
    return classify_intent(state)
