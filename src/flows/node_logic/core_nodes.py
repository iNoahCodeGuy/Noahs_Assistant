"""Core conversation nodes - backward-compatible aliases for split modules.

This module now serves as a compatibility layer, re-exporting functions from:
- retrieval_nodes: retrieve_chunks, re_rank_and_dedup, validate_grounding, handle_grounding_gap
- generation_nodes: generate_draft, hallucination_check
- formatting_nodes: format_answer
- logging_nodes: log_and_notify, suggest_followups, update_memory

All new code should import from the specific modules. This file provides
backward compatibility for tests and legacy modules.

See individual modules for detailed documentation:
- retrieval_nodes.py: Retrieval pipeline documentation
- generation_nodes.py: Generation pipeline documentation
- formatting_nodes.py: Formatting pipeline documentation
- logging_nodes.py: Logging pipeline documentation
"""

# Re-export from new modules for backward compatibility
from src.flows.node_logic.retrieval_nodes import (
    retrieve_chunks,
    re_rank_and_dedup,
    validate_grounding,
    handle_grounding_gap,
)
from src.flows.node_logic.generation_nodes import (
    generate_draft,
    hallucination_check,
)
from src.flows.node_logic.formatting_nodes import (
    format_answer,
)
from src.flows.node_logic.logging_nodes import (
    log_and_notify,
    suggest_followups,
    update_memory,
)

# Backward-compatible aliases (tests and legacy modules)
def generate_answer(state, rag_engine):
    """Backward-compatible alias for generate_draft."""
    return generate_draft(state, rag_engine)


def apply_role_context(state, rag_engine):
    """Backward-compatible alias for format_answer."""
    return format_answer(state, rag_engine)


__all__ = [
    "retrieve_chunks",
    "re_rank_and_dedup",
    "validate_grounding",
    "handle_grounding_gap",
    "generate_draft",
    "generate_answer",
    "hallucination_check",
    "format_answer",
    "apply_role_context",
    "log_and_notify",
    "suggest_followups",
    "update_memory",
]
