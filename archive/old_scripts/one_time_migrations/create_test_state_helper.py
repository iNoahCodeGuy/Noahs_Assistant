"""Helper script to generate ConversationState TypedDict literals for tests.

This script provides a factory function that creates properly formatted
ConversationState dict literals following the TypedDict pattern, with
all required fields and sensible defaults.

Design Principles Applied:
- KISS (#8): Simple factory function, no complex abstractions
- DRY (#8): Single source of truth for test state creation
- Defensibility (#6): All required fields present, type-safe defaults
- Maintainability (#7): Easy to update when state schema changes
"""

from typing import Dict, Any, List


def create_test_state(
    role: str = "Software Developer",
    query: str = "test query",
    chat_history: List[Dict[str, str]] = None,
    hiring_signals: List[str] = None,
    resume_sent: bool = False,
    resume_explicitly_requested: bool = False,
    job_details: Dict[str, str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Create a ConversationState TypedDict literal for testing.

    Args:
        role: User role (default: "Software Developer")
        query: User query text (default: "test query")
        chat_history: Conversation history (default: [])
        hiring_signals: Detected hiring signals (default: [])
        resume_sent: Whether resume was sent (default: False)
        resume_explicitly_requested: Whether resume explicitly requested (default: False)
        job_details: Extracted job details (default: {})
        **kwargs: Any additional state fields

    Returns:
        Dict conforming to ConversationState TypedDict

    Example:
        >>> state = create_test_state(
        ...     role="hiring_manager_technical",
        ...     query="We're hiring a GenAI engineer"
        ... )
        >>> print(state["role"])
        'hiring_manager_technical'
    """
    state: Dict[str, Any] = {
        "role": role,
        "query": query,
        "chat_history": chat_history or [],
        "hiring_signals": hiring_signals or [],
        "resume_sent": resume_sent,
        "resume_explicitly_requested": resume_explicitly_requested,
        "job_details": job_details or {},
    }

    # Add any additional kwargs
    state.update(kwargs)

    return state


# Example usage patterns for tests:
if __name__ == "__main__":
    # Basic usage
    state1 = create_test_state(
        role="hiring_manager_technical",
        query="Tell me about RAG"
    )
    print("✅ Basic state:", state1)

    # With hiring signals
    state2 = create_test_state(
        role="hiring_manager_nontechnical",
        query="We're hiring",
        hiring_signals=["mentioned_hiring", "described_role"]
    )
    print("✅ With signals:", state2)

    # With chat history
    state3 = create_test_state(
        query="Follow-up question",
        chat_history=[
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"}
        ]
    )
    print("✅ With history:", state3)

    print("\n✅ All examples work correctly!")
