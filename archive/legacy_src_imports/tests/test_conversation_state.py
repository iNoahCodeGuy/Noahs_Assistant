"""Tests for ConversationState TypedDict and state management patterns.

This test suite validates that the ConversationState TypedDict follows LangGraph
patterns correctly and that state access patterns align with the 8 design principles.

Test Categories:
    1. TypedDict Structure: Validate field types and partial updates
    2. Chat History Annotation: Test add_messages behavior
    3. State Access Patterns: Validate safe access patterns for migration
    4. Error Handling: Test fail-fast validation
    5. Immutability: Ensure nodes don't mutate input state

Design Principles Tested:
    - Loose Coupling: State is the only communication mechanism
    - Defensibility (Fail-Fast): validate_required_fields catches missing data
    - Immutability: Nodes return new dicts, don't mutate input
    - Simplicity (KISS): Direct dict access is sufficient
"""

import pytest
from typing import Dict, Any
from langgraph.graph.message import add_messages

from src.state.conversation_state import (
    ConversationState,
    validate_required_fields,
    get_safe,
)


class TestConversationStateStructure:
    """Test TypedDict structure and field types."""

    def test_state_creation_with_core_fields(self):
        """Should create state with required conversation fields."""
        state: ConversationState = {
            "query": "How does RAG work?",
            "role": "Software Developer",
            "session_id": "test-session-123"
        }

        assert state["query"] == "How does RAG work?"
        assert state["role"] == "Software Developer"
        assert state["session_id"] == "test-session-123"

    def test_state_allows_partial_fields(self):
        """Should allow creating state with subset of fields (total=False)."""
        # This is valid with total=False
        state: ConversationState = {"query": "test"}

        assert "query" in state
        assert "role" not in state  # Optional field

    def test_state_supports_all_field_types(self):
        """Should support all defined field types."""
        state: ConversationState = {
            # Core
            "query": "test",
            "role": "Developer",
            "session_id": "123",
            "chat_history": [{"role": "user", "content": "Hello"}],

            # Classification
            "query_type": "technical",
            "is_greeting": False,

            # Retrieval
            "retrieved_chunks": [{"id": "1", "content": "chunk"}],
            "code_snippets": [{"file_path": "test.py", "code": "print()"}],

            # Generation
            "answer": "Generated response",

            # Actions
            "planned_actions": [{"type": "log"}],
            "executed_actions": ["log"],

            # Metadata
            "timestamp": "2025-10-17T12:00:00Z",

            # Resume distribution
            "hiring_signals": ["keyword:hiring"],
            "resume_offered": False,

            # Errors
            "error": None,
            "error_message": None,
        }

        # All fields accessible
        assert state["query"] == "test"
        assert state["query_type"] == "technical"
        assert state["is_greeting"] is False
        assert len(state["retrieved_chunks"]) == 1
        assert state["answer"] == "Generated response"


class TestPartialStateUpdates:
    """Test LangGraph's partial state update pattern."""

    def test_partial_update_merges_correctly(self):
        """Should merge partial updates without losing existing fields."""
        # Simulate initial state
        state: ConversationState = {
            "query": "test query",
            "role": "Developer",
            "session_id": "abc-123"
        }

        # Simulate node returning partial update
        update: Dict[str, Any] = {"query_type": "technical"}

        # LangGraph merges like this
        merged_state = {**state, **update}

        # Verify merge behavior
        assert merged_state["query"] == "test query"  # Original preserved
        assert merged_state["query_type"] == "technical"  # New field added
        assert merged_state["session_id"] == "abc-123"  # Original preserved

    def test_partial_update_overwrites_existing_field(self):
        """Should overwrite fields when update contains same key."""
        state: ConversationState = {
            "query": "original query",
            "answer": "old answer"
        }

        update: Dict[str, Any] = {"answer": "new answer"}
        merged_state = {**state, **update}

        assert merged_state["answer"] == "new answer"  # Overwritten
        assert merged_state["query"] == "original query"  # Unchanged

    def test_multiple_partial_updates_accumulate(self):
        """Should accumulate updates across multiple node executions."""
        state: ConversationState = {"query": "test", "role": "Developer"}

        # Node 1: Classify
        update1 = {"query_type": "technical"}
        state = {**state, **update1}

        # Node 2: Retrieve
        update2 = {"retrieved_chunks": [{"id": "1"}]}
        state = {**state, **update2}

        # Node 3: Generate
        update3 = {"answer": "Generated response"}
        state = {**state, **update3}

        # All updates accumulated
        assert state["query"] == "test"
        assert state["query_type"] == "technical"
        assert len(state["retrieved_chunks"]) == 1
        assert state["answer"] == "Generated response"


class TestChatHistoryAnnotation:
    """Test Annotated[list, add_messages] behavior."""

    def test_add_messages_appends_not_replaces(self):
        """Should append messages rather than replace the list."""
        # Initial state with one message
        state: ConversationState = {
            "query": "test",
            "chat_history": [{"role": "user", "content": "Hello"}]
        }

        # Node returns new message (simulating assistant response)
        update = {"chat_history": [{"role": "assistant", "content": "Hi there!"}]}

        # Simulate add_messages behavior (LangGraph does this automatically)
        merged_history = add_messages(
            state.get("chat_history", []),
            update["chat_history"]
        )

        # Should have both messages (appended, not replaced)
        # Note: add_messages converts dicts to LangChain Message objects
        assert len(merged_history) == 2
        assert merged_history[0].type == "human"  # Converted from role="user"
        assert merged_history[0].content == "Hello"
        assert merged_history[1].type == "ai"  # Converted from role="assistant"
        assert merged_history[1].content == "Hi there!"

    def test_add_messages_handles_empty_initial_state(self):
        """Should handle case where chat_history doesn't exist yet."""
        state: ConversationState = {"query": "test"}

        # First message
        update = {"chat_history": [{"role": "user", "content": "First message"}]}

        merged_history = add_messages(
            state.get("chat_history", []),
            update["chat_history"]
        )

        # Note: add_messages converts dicts to LangChain Message objects
        assert len(merged_history) == 1
        assert merged_history[0].type == "human"
        assert merged_history[0].content == "First message"

    def test_add_messages_preserves_message_order(self):
        """Should maintain chronological order of messages."""
        state: ConversationState = {
            "query": "test",
            "chat_history": [
                {"role": "user", "content": "Message 1"},
                {"role": "assistant", "content": "Message 2"}
            ]
        }

        # Add third message
        update = {"chat_history": [{"role": "user", "content": "Message 3"}]}

        merged_history = add_messages(
            state["chat_history"],
            update["chat_history"]
        )

        # Note: add_messages converts dicts to LangChain Message objects
        assert len(merged_history) == 3
        assert merged_history[0].type == "human"
        assert merged_history[0].content == "Message 1"
        assert merged_history[1].type == "ai"
        assert merged_history[1].content == "Message 2"
        assert merged_history[2].type == "human"
        assert merged_history[2].content == "Message 3"


class TestStateAccessPatterns:
    """Test proper state access patterns for node implementation."""

    def test_direct_access_for_required_fields(self):
        """Should use direct bracket access for required fields (fail-fast)."""
        state: ConversationState = {
            "query": "test",
            "role": "Developer",
            "session_id": "123"
        }

        # Direct access succeeds
        query = state["query"]
        role = state["role"]

        assert query == "test"
        assert role == "Developer"

    def test_direct_access_raises_on_missing_required(self):
        """Should raise KeyError on missing required field (fail-fast principle)."""
        state: ConversationState = {"query": "test"}

        # Should raise KeyError for missing required field
        with pytest.raises(KeyError):
            _ = state["role"]

    def test_get_method_for_optional_fields(self):
        """Should use .get() for optional fields with defaults."""
        state: ConversationState = {"query": "test"}

        # Using .get() with default
        answer = state.get("answer", "No answer yet")
        chunks = state.get("retrieved_chunks", [])

        assert answer == "No answer yet"
        assert chunks == []

    def test_get_safe_wrapper_function(self):
        """Should support get_safe() convenience wrapper."""
        state: ConversationState = {"query": "test"}

        # get_safe is a thin wrapper around dict.get()
        answer = get_safe(state, "answer", "Default")
        existing = get_safe(state, "query", "Default")

        assert answer == "Default"
        assert existing == "test"


class TestListFieldAppendPattern:
    """Test safe pattern for appending to list fields."""

    def test_safe_list_append_pattern(self):
        """Should safely append to list fields without mutation."""
        state: ConversationState = {
            "query": "test",
            "hiring_signals": ["keyword:hiring"]
        }

        # Node pattern: Get existing list, append, return update
        current_signals = state.get("hiring_signals", [])
        updated_signals = current_signals + ["company:TechCorp"]

        update: Dict[str, Any] = {"hiring_signals": updated_signals}

        # Verify immutability
        assert update["hiring_signals"] == ["keyword:hiring", "company:TechCorp"]
        assert state["hiring_signals"] == ["keyword:hiring"]  # Original unchanged

    def test_list_append_handles_missing_field(self):
        """Should handle case where list field doesn't exist yet."""
        state: ConversationState = {"query": "test"}

        # Safe pattern handles missing field
        current_signals = state.get("hiring_signals", [])
        updated_signals = current_signals + ["keyword:hiring"]

        assert updated_signals == ["keyword:hiring"]


class TestErrorHandlingPatterns:
    """Test defensive programming patterns."""

    def test_validate_required_fields_success(self):
        """Should pass validation when all required fields present."""
        state: ConversationState = {
            "query": "test",
            "role": "Developer",
            "session_id": "123"
        }

        # Should not raise
        validate_required_fields(state)

    def test_validate_required_fields_failure(self):
        """Should raise ValueError with details when required fields missing."""
        state: ConversationState = {"query": "test"}  # Missing role, session_id

        with pytest.raises(ValueError) as exc_info:
            validate_required_fields(state)

        error_message = str(exc_info.value)
        assert "Missing required state fields" in error_message
        assert "role" in error_message
        assert "session_id" in error_message
        assert "query" in error_message  # Shows what's present

    def test_node_error_handling_pattern(self):
        """Should demonstrate proper node error handling pattern."""
        def example_node(state: ConversationState) -> Dict[str, Any]:
            """Example node with proper error handling."""
            try:
                # Fail-fast: Validate required fields
                query = state["query"]
                role = state.get("role", "Developer")
            except KeyError as e:
                # Return error state (graceful degradation)
                return {
                    "error": "missing_field",
                    "error_message": f"Missing required field: {e}"
                }

            # Business logic
            result = f"Processed: {query} for {role}"
            return {"answer": result}

        # Test success case
        state_success: ConversationState = {
            "query": "test",
            "role": "Developer"
        }
        result = example_node(state_success)
        assert "answer" in result
        assert result["error"] is None if "error" in result else True

        # Test error case
        state_error: ConversationState = {}
        result = example_node(state_error)
        assert result["error"] == "missing_field"
        assert "Missing required field" in result["error_message"]


class TestStateImmutability:
    """Test that nodes don't mutate input state (principle: avoid side effects)."""

    def test_node_does_not_mutate_input_state(self):
        """Should return new dict, not mutate input."""
        def safe_node(state: ConversationState) -> Dict[str, Any]:
            """Example node that returns new dict."""
            query_type = "technical" if "code" in state.get("query", "") else "general"
            return {"query_type": query_type}  # New dict

        original_state: ConversationState = {
            "query": "Show me code",
            "role": "Developer"
        }

        # Make a copy to verify immutability
        state_copy = original_state.copy()

        # Execute node
        result = safe_node(original_state)

        # Original state unchanged
        assert original_state == state_copy
        assert "query_type" not in original_state

        # Result is new dict
        assert result["query_type"] == "technical"

    def test_node_avoids_in_place_modification(self):
        """Should not modify lists/dicts in place."""
        def unsafe_node(state: ConversationState) -> Dict[str, Any]:
            """Example of INCORRECT pattern (modifies in place)."""
            # ❌ BAD: Modifies input state's list in place
            signals = state.get("hiring_signals", [])
            signals.append("new_signal")  # IN-PLACE MODIFICATION!
            return {"hiring_signals": signals}

        def safe_node(state: ConversationState) -> Dict[str, Any]:
            """Example of CORRECT pattern (creates new list)."""
            # ✅ GOOD: Creates new list
            current_signals = state.get("hiring_signals", [])
            updated_signals = current_signals + ["new_signal"]
            return {"hiring_signals": updated_signals}

        # Test unsafe pattern (for demonstration)
        state_unsafe: ConversationState = {
            "query": "test",
            "hiring_signals": ["existing"]
        }
        unsafe_node(state_unsafe)
        assert len(state_unsafe["hiring_signals"]) == 2  # MUTATED!

        # Test safe pattern
        state_safe: ConversationState = {
            "query": "test",
            "hiring_signals": ["existing"]
        }
        result = safe_node(state_safe)
        assert len(state_safe["hiring_signals"]) == 1  # Unchanged
        assert len(result["hiring_signals"]) == 2  # New list


class TestNodeSignatureCompliance:
    """Test that example nodes follow LangGraph signature requirements."""

    def test_node_accepts_conversation_state(self):
        """Nodes should accept ConversationState as input."""
        def example_node(state: ConversationState) -> Dict[str, Any]:
            return {"result": "success"}

        state: ConversationState = {"query": "test"}
        result = example_node(state)

        assert isinstance(result, dict)
        assert result["result"] == "success"

    def test_node_returns_dict_not_state(self):
        """Nodes should return Dict[str, Any], not ConversationState."""
        def correct_node(state: ConversationState) -> Dict[str, Any]:
            """Returns partial update."""
            return {"query_type": "technical"}

        result = correct_node({"query": "test"})

        # Result is dict, not full ConversationState
        assert isinstance(result, dict)
        assert "query_type" in result
        assert "query" not in result  # Partial update

    def test_node_returns_partial_state_update(self):
        """Nodes should return only changed fields."""
        def classify_node(state: ConversationState) -> Dict[str, Any]:
            """Classifies query, returns only classification fields."""
            return {
                "query_type": "technical",
                "is_greeting": False
            }

        state: ConversationState = {
            "query": "Show me code",
            "role": "Developer",
            "session_id": "123"
        }

        result = classify_node(state)

        # Only new fields returned
        assert set(result.keys()) == {"query_type", "is_greeting"}
        # Original fields not included in update
        assert "query" not in result
        assert "role" not in result
