"""Node logic package - contains all conversation pipeline node implementations.

This package organizes node modules by responsibility:
- session_management: State initialization and session tracking
- role_routing: Role classification and routing
- query_classification: Intent detection and query analysis
- entity_extraction: Company, role, contact info extraction
- clarification: Vague query detection and clarification prompts
- query_composition: Retrieval-ready query construction
- presentation_control: Depth and display formatting control
- core_nodes: Retrieval, generation, grounding, logging
- action_planning: Role-based action decision making
- action_execution: Side effects (email, SMS, storage, analytics)
- code_validation: Code sanitization and validation
- greetings: Role-specific welcome messages
- resume_distribution: Hiring signal detection and resume delivery
- analytics_renderer: Analytics display formatting
- performance_metrics: Performance tracking and metrics

All functions are re-exported through src/flows/conversation_nodes.py
for a stable public API.
"""

from __future__ import annotations

# Re-export all node functions for clean imports (stage-prefixed for pipeline visibility)
from assistant.flows.node_logic.stage0_session_management import initialize_conversation_state
from assistant.flows.node_logic.stage2_role_routing import classify_role_mode
from assistant.flows.node_logic.stage2_query_classification import classify_intent, classify_query
from assistant.flows.node_logic.stage2_entity_extraction import extract_entities
from assistant.flows.node_logic.stage3_clarification import assess_clarification_need, ask_clarifying_question
from assistant.flows.node_logic.stage3_query_composition import compose_query
from assistant.flows.node_logic.query_preprocessing import preprocess_query
from assistant.flows.node_logic.stage3_presentation_control import (
    presentation_controller,
    depth_controller,  # Deprecated alias
    display_controller,  # Deprecated no-op
)
from assistant.flows.node_logic.stage4_retrieval_nodes import (
    retrieve_chunks,
    re_rank_and_dedup,
    validate_grounding,
    handle_grounding_gap,
)
from assistant.flows.node_logic.stage5_generation_nodes import (
    generate_draft,
    hallucination_check,
)
from assistant.flows.node_logic.stage6_formatting_nodes import (
    format_answer,
)
from assistant.flows.node_logic.stage7_logging_nodes import (
    log_and_notify,
    suggest_followups,
    update_memory,
)
from assistant.flows.node_logic.util_core_nodes import (
    generate_answer,
    apply_role_context,
)
from assistant.flows.node_logic.stage6_action_planning import plan_actions
from assistant.flows.node_logic.stage7_action_execution import execute_actions
from assistant.flows.node_logic.util_code_validation import (
    is_valid_code_snippet,
    sanitize_generated_answer
)
from assistant.flows.node_logic.stage1_greetings import should_show_greeting, is_first_turn
from assistant.flows.node_logic.util_resume_distribution import (
    detect_hiring_signals,
    handle_resume_request,
    should_add_availability_mention,
    extract_email_from_query,
    extract_name_from_query,
    should_gather_job_details,
    get_job_details_prompt,
    extract_job_details_from_query
)
from assistant.flows.node_logic.util_role_specific import (
    route_hiring_manager_technical,
    onboard_hiring_manager_technical,
    explain_enterprise_adaptation,
    show_certifications,
    show_enterprise_pattern_example,
)

__all__ = [
    "initialize_conversation_state",
    "classify_role_mode",
    "classify_intent",
    "classify_query",
    "extract_entities",
    "assess_clarification_need",
    "ask_clarifying_question",
    "compose_query",
    "preprocess_query",  # Typo correction + normalization
    "presentation_controller",  # NEW: Unified depth + display
    "depth_controller",  # DEPRECATED: alias
    "display_controller",  # DEPRECATED: no-op
    "route_hiring_manager_technical",
    "onboard_hiring_manager_technical",
    "explain_enterprise_adaptation",
    "show_certifications",
    "show_enterprise_pattern_example",
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
    "plan_actions",
    "execute_actions",
    "is_valid_code_snippet",
    "sanitize_generated_answer",
    "should_show_greeting",
    "is_first_turn",
    "detect_hiring_signals",
    "handle_resume_request",
    "should_add_availability_mention",
    "extract_email_from_query",
    "extract_name_from_query",
    "should_gather_job_details",
    "get_job_details_prompt",
    "extract_job_details_from_query",
]
