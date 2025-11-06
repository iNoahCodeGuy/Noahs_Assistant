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

# Re-export all node functions for clean imports
from src.flows.node_logic.session_management import initialize_conversation_state
from src.flows.node_logic.role_routing import classify_role_mode
from src.flows.node_logic.query_classification import classify_intent, classify_query
from src.flows.node_logic.entity_extraction import extract_entities
from src.flows.node_logic.clarification import assess_clarification_need, ask_clarifying_question
from src.flows.node_logic.query_composition import compose_query
from src.flows.node_logic.presentation_control import depth_controller, display_controller, update_enterprise_affinity
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
from src.flows.node_logic.core_nodes import (
    generate_answer,
    apply_role_context,
)
from src.flows.node_logic.action_planning import plan_actions
from src.flows.node_logic.action_execution import execute_actions
from src.flows.node_logic.code_validation import (
    is_valid_code_snippet,
    sanitize_generated_answer
)
from src.flows.node_logic.greetings import get_role_greeting, should_show_greeting, is_first_turn
from src.flows.node_logic.resume_distribution import (
    detect_hiring_signals,
    handle_resume_request,
    should_add_availability_mention,
    extract_email_from_query,
    extract_name_from_query,
    should_gather_job_details,
    get_job_details_prompt,
    extract_job_details_from_query
)
from src.flows.node_logic.role_specific import (
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
    "depth_controller",
    "display_controller",
    "update_enterprise_affinity",
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
    "get_role_greeting",
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
