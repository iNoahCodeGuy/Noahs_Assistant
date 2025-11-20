"""ConversationState TypedDict for LangGraph nodes.

This module defines the state structure passed between LangGraph nodes in the
conversation pipeline. Uses TypedDict with total=False to allow partial state
updates while maintaining type safety.

Design Principles Applied:
- Loose Coupling: Nodes communicate only via this state dict
- Simplicity (YAGNI): No StateHelper class - Python's dict methods are sufficient
- Maintainability: Clear type hints for all fields
- Portability: Works across all LangGraph-compatible environments

Architecture:
    LangGraph nodes receive ConversationState as input and return Dict[str, Any]
    containing partial updates. LangGraph automatically merges these updates into
    the full state before passing to the next node.

Example Usage:
    ```python
    from assistant.state.conversation_state import ConversationState
    from typing import Dict, Any

    def classify_query(state: ConversationState) -> Dict[str, Any]:
        \"\"\"Classify user query type.\"\"\"
        query = state["query"]  # Direct access for required fields
        role = state.get("role", "Developer")  # .get() for optional fields

        query_type = _determine_type(query, role)
        return {"query_type": query_type}  # Partial update
    ```

References:
    - LangGraph StateGraph documentation for state management patterns
    - QA_STRATEGY.md § Design Principles for architectural rationale
    - QA_LANGGRAPH_MIGRATION.md § Standard 2 for node signature requirements
"""

from typing import Annotated, Any, Dict, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class ConversationState(TypedDict, total=False):
    """State dictionary passed between LangGraph nodes.

    Uses total=False to allow partial state updates from nodes. Each node
    returns a Dict[str, Any] containing only the fields it modifies, and
    LangGraph merges these updates into the full state.

    Design Rationale:
        - total=False: Allows nodes to return partial updates (Principle: Modularity)
        - Annotated[list, add_messages]: LangGraph auto-appends chat messages (Principle: DRY)
        - Type hints: Self-documenting, IDE-friendly (Principle: Maintainability)
        - No default values: Explicit initialization required (Principle: Defensibility)

    Field Categories:
        Core Conversation: query, role, session_id, chat_history
        Classification: query_type, is_greeting
        Retrieval: retrieved_chunks, code_snippets
        Generation: answer
        Actions: planned_actions, executed_actions
        Metadata: timestamp
        Resume Distribution: hiring_signals, resume_offered
        Error Tracking: error, error_message
    """

    # --- Core Conversation Fields (typically required) ---
    query: str
    """User's current query text. Required in most flows."""

    role: str
    """Selected user role: 'Software Developer', 'Hiring Manager (technical)', etc."""

    role_mode: str
    """Normalized persona mode after classification (tech HM, nontech HM, developer, explorer)."""

    role_confidence: float
    """Confidence score (0-1) for role classification."""

    session_id: str
    """Unique session identifier for analytics and conversation tracking."""

    chat_history: Annotated[list, add_messages]
    """Conversation history with automatic message appending.

    The add_messages annotation tells LangGraph to append new messages rather
    than replace the list. This follows the standard chat interface pattern
    used across LangChain/LangGraph applications.

    Example:
        state = {"chat_history": [{"role": "user", "content": "Hello"}]}
        update = {"chat_history": [{"role": "assistant", "content": "Hi!"}]}
        # LangGraph merges: chat_history = [user_msg, assistant_msg]
    """

    # --- Query Classification ---
    query_type: str
    """Classified query type: 'technical', 'career', 'analytics', 'greeting', 'menu_selection', 'self_referential', 'file_request', etc."""

    is_self_referential: bool
    """Flag indicating if query is about Portfolia herself (True) vs about Noah (False).

    Purpose: Route self-referential queries to codebase/documentation chunks,
    career queries to career_kb chunks.
    """

    file_request: str
    """File path requested by user (e.g., 'assistant/core/rag_engine.py').

    Purpose: Enable on-demand file reading when users explicitly request specific files.
    """

    menu_choice: str
    """User's menu selection (e.g., '1', '2', '3', '4') when query_type is 'menu_selection'."""

    query_intent: str
    """Higher level intent grouping (engineering, business, data, action)."""

    intent_confidence: float
    """Confidence score for query intent classification."""

    is_greeting: bool
    """True if query is a greeting (allows pipeline short-circuit)."""

    topic_focus: str
    """Primary topical focus of the query (architecture, data, testing, etc.)."""

    relate_to_enterprise: bool
    """True when responses should explicitly tie insights back to enterprise value."""

    show_technical_depth: bool
    """True when responses should include code, architecture details, and implementation specifics."""

    # --- Entity Extraction & Memory ---
    entities: Dict[str, Any]
    """Extracted entities (company, position, timeline, technology, contact preference)."""

    session_memory: Dict[str, Any]
    """Lightweight memory store capturing soft signals across turns."""

    clarification_needed: bool
    """Whether Portfolia should pause to ask a clarifying question."""

    clarifying_question: str
    """Targeted clarifying question to ask the user when context is ambiguous."""

    clarification_type: str
    """Type of clarification needed: 'topic', 'detail', or 'both'."""

    detail_preference_needed: bool
    """Whether Portfolia should ask about detail level preference."""

    detail_preference: str
    """User's detail preference: 'brief', 'moderate', 'comprehensive', or None."""

    composed_query: str
    """Reformulated query sent to retrieval once role and entities applied."""

    # --- Retrieval Results (RAG Pipeline) ---
    retrieved_chunks: List[Dict[str, Any]]
    """Top-k semantic search results from pgvector knowledge base.

    Each chunk contains:
        - id: UUID of kb_chunks row
        - section: KB section name
        - content: Retrieved text
        - similarity_score: Cosine similarity (0-1)
    """

    retrieval_scores: List[float]
    """Similarity scores for retrieved chunks (kept for analytics and validation)."""

    code_snippets: List[Dict[str, Any]]
    """Code examples retrieved for technical queries.

    Each snippet contains:
        - file_path: Source file location
        - code: Code content
        - language: Programming language
    """

    file_content: Dict[str, Any]
    """File content when user explicitly requests a specific file.

    Structure:
        - content: Full file content as string
        - file_path: Original file path
        - line_count: Total number of lines
        - last_modified: File modification timestamp (if available)
        - success: Boolean indicating if read was successful
    """

    file_read_success: bool
    """Flag indicating if file reading was successful (True) or failed (False)."""

    # --- Response Generation ---
    draft_answer: str
    """Pre-format answer returned by the LLM before role styling."""

    answer: str
    """Generated assistant response (post-LLM generation)."""

    grounding_status: str
    """Outcome of grounding validation (ok, insufficient, broaden_search)."""

    hallucination_safe: bool
    """Indicates whether hallucination check passed."""

    depth_level: int
    """Presentation depth chosen by depth_controller (1=overview, 2=guided, 3=deep dive)."""

    detail_strategy: str
    """Explanation of why the current depth level was selected (for analytics/QA)."""

    display_toggles: Dict[str, Any]
    """Flags from display_controller indicating which supporting artifacts to surface."""

    display_reasons: Dict[str, str]
    """Human-readable reasons for each display toggle (used in analytics/debugging)."""

    layout_variant: str
    """High-level layout variant: 'engineering', 'business', or 'mixed'."""

    followup_variant: str
    """Variant used to craft follow-up invitations (maps to layout variant)."""

    # --- Action Planning & Execution ---
    pending_actions: List[Dict[str, Any]]
    """Actions queued for execution (resume send, analytics display, etc.)."""

    planned_actions: List[Dict[str, Any]]
    """Actions to execute after response generation.

    Examples:
        - {"type": "send_analytics", "session_id": "..."}
        - {"type": "offer_resume", "email": "user@example.com"}
        - {"type": "send_sms", "message": "..."}
    """

    executed_actions: List[str]
    """List of action types that were successfully executed."""

    followup_prompts: List[str]
    """Persona-aware follow-up suggestions to continue the conversation."""

    # --- Metadata ---
    timestamp: str
    """ISO 8601 timestamp of conversation turn."""

    analytics_metadata: Dict[str, Any]
    """Collected analytics metrics for tracing and Supabase logging."""

    # --- Resume Distribution (Hiring Manager Flows) ---
    hiring_signals: List[str]
    """Detected hiring intent signals.

    Examples:
        - "keyword:hiring"
        - "company:TechCorp"
        - "position:Senior Engineer"
        - "timeline:immediate"
    """

    resume_offered: bool
    """True if resume has been offered in this session (prevents duplicate offers)."""

    hiring_signals_strength: int
    """Number of distinct hiring signals detected during this session."""

    hiring_signals_strong: bool
    """Whether hiring signals are strong enough to unlock resume prompts."""

    # --- Error Tracking (Graceful Degradation) ---
    error: str | None
    """Error type if node failed: 'classify_failed', 'retrieval_failed', etc."""

    error_message: str | None
    """Human-readable error description for logging/debugging."""


# Type alias for common node return type
NodeUpdate = Dict[str, Any]
"""Type alias for node return values (partial state updates)."""


def validate_required_fields(state: ConversationState) -> None:
    """Validate that required state fields are present (fail-fast pattern).

    This function enforces the Defensibility principle by validating inputs
    early before expensive operations (LLM calls, DB queries).

    Args:
        state: ConversationState to validate

    Raises:
        ValueError: If any required field is missing, with details about
            which fields are absent and what fields are present.

    Example:
        ```python
        def my_node(state: ConversationState) -> Dict[str, Any]:
            validate_required_fields(state)  # Fail-fast if incomplete
            # ... proceed with node logic
        ```
    """
    required = ["query", "role", "session_id"]
    missing = [field for field in required if field not in state]

    if missing:
        raise ValueError(
            f"Missing required state fields: {missing}. "
            f"Required: {required}. "
            f"Present: {list(state.keys())}"
        )


def get_safe(state: ConversationState, key: str, default: Any = None) -> Any:
    """Safely get value from state with default (convenience wrapper).

    This is a thin wrapper around dict.get() provided for consistency with
    the codebase's state access patterns. Follows the KISS principle - if
    Python's built-in works, use it.

    Args:
        state: ConversationState to read from
        key: Field name to retrieve
        default: Value to return if key is missing

    Returns:
        State value or default

    Example:
        ```python
        answer = get_safe(state, "answer", "No response yet")
        chunks = get_safe(state, "retrieved_chunks", [])
        ```
    """
    return state.get(key, default)
