# Runtime Dependency Inventory

This document catalogs the modules and services invoked during a standard Streamlit session to help identify unused code paths.

## Application Entry
- `src/main.py` – Streamlit entry-point. Depending on the `LANGGRAPH_FLOW_ENABLED` flag it will:
  - Instantiate `RagEngine`, `Memory`, `ResponseFormatter`, and `supabase_analytics`.
  - Use `run_conversation_flow` (preferred path) which in turn calls nodes in `src/flows/conversation_nodes.py`.
  - Fallback to the legacy `RoleRouter` plus response formatting when the flag is disabled.

## Core Runtime Modules
- `src/flows/conversation_state.py` – shared state container for the LangGraph-style pipeline.
- `src/flows/conversation_flow.py` – orchestrator wiring `classify_query`, `retrieve_chunks`, `generate_answer`, `plan_actions`, `apply_role_context`, `execute_actions`, and `log_and_notify`.
- `src/core/rag_engine.py` – handles retrieval and response generation.
- `src/core/memory.py` – maintains multi-turn memory for the legacy RoleRouter path.
- `src/analytics/supabase_analytics.py` – logs interactions to Supabase.
- `src/services/resend_service.py`, `src/services/twilio_service.py`, `src/services/storage_service.py` – invoked by `execute_actions` for email, SMS, and signed URLs.

## Conditional / Legacy Components
- `src/agents/role_router.py` and `src/agents/response_formatter.py` – only used when `LANGGRAPH_FLOW_ENABLED` is `false`.
- `scripts/` utilities (migrations, analytics checks) are manual operations; not part of normal Streamlit runtime.
- `examples/`, `demo_*.py`, and `archive/docs/` contain reference material with no active imports.

## Recommended Cleanup Targets
- Files now placed under `archive/docs` and `archive/sql` are not imported anywhere and can be removed from deployments.
- `demo_*.py`, `example_streamlit_integration.py`, and notebooks in `examples/` remain unused in production; keep only if needed for documentation.

Use this inventory when pruning code: anything outside the modules listed above should be considered optional or legacy before inclusion in production bundles.
