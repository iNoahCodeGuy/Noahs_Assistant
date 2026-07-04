# Code Display Testing - Test Suite Summary

## Overview
Node-level tests for the code display and import explanation path of the
universal conversation pipeline: trigger detection (`classify_query`), action
planning (`plan_actions`), answer enrichment (`format_answer`), and action
execution (`execute_actions`). All tests are hermetic — they use plain state
dicts and dummy RAG engines, and make no network calls.

## Architecture Under Test
- `ConversationState` is a TypedDict (plain dict). Nodes read/write state keys
  directly and return partial update dicts that callers merge via
  `state.update(result)`.
- `classify_query` (alias of `classify_intent` in
  `assistant/flows/node_logic/stage2_query_classification.py`) sets
  `code_display_requested` / `import_explanation_requested` flags and
  `query_type`.
- `plan_actions` (`stage6_action_planning.py`) queues
  `{"type": "include_code_reference"}` for code display and
  `{"type": "explain_imports"}` for stack justification questions. The old
  `display_code_snippet` action type no longer exists.
- `format_answer` (`stage6_formatting_nodes.py`) appends a formatted code block
  (via `rag_engine.retrieve_with_code` + `content_blocks.format_code_snippet`)
  when `include_code_reference` is queued, and renders resource links for
  `action_request` queries.
- Import explanations come from `assistant/retrieval/import_retriever.py`,
  backed by `data/imports_kb.csv` with role→tier mapping (`ROLE_TO_TIER`).

## Test Files

### `test_code_display_policy.py` (24 tests)
- **TestCodeDisplayTriggers** (5) — "show code" / "show implementation" /
  "how do you" queries set `code_display_requested`; general questions don't;
  `plan_actions` queues `include_code_reference` for a developer code request.
- **TestImportExplanationTriggers** (4) — "why use X" / "explain imports" /
  trade-off questions set `import_explanation_requested`; the developer flow
  queues the `explain_imports` action.
- **TestImportRetrieval** (5) — tier-1 vs tier-2 retrieval, import name
  detection in queries, keyword search over the imports KB, `ROLE_TO_TIER`
  mapping.
- **TestCodeFormatting** (4) — `content_blocks.format_code_snippet` metadata
  (file path, description, branch) and `code_display_guardrails` copy.
- **TestImportFormatting** (3) — `content_blocks.format_import_explanation`
  tiered output (enterprise concern / alternative / when-to-switch sections).
- **TestEndToEndFlow** (3) — developer "how do you" query yields both
  `include_code_reference` and `explain_imports` actions; technical hiring
  manager gets tier-1 supabase explanation; casual role never gets
  `include_code_reference`.

### `test_code_display_accuracy.py` (1 test)
- `test_software_developer_code_request_appends_code_block` — full node chain
  classify → plan → format with a dummy engine; asserts the final answer keeps
  the draft prose and gains a ```python code block with its file citation.

### `test_code_display_ci.py` (1 test)
- `test_linkedin_request_prompts_follow_up` — a LinkedIn request is classified
  as an `action_request`, plans `send_linkedin` + `ask_reach_out`,
  `format_answer` renders the LinkedIn URL and the
  "Would you like Noah to reach out directly?" prompt, and `execute_actions`
  records `analytics_metadata["linkedin_offer"]` without touching external
  services.

## Removed Legacy Tests
The following were deleted because they asserted behaviors removed in the
migration from the role-based architecture to the universal pipeline:
- Role-specific answer enrichments ("Architecture Snapshot", "Enterprise Fit",
  "Data Collection Overview", "Staying Current" blocks appended by
  `apply_role_context`) — these blocks no longer exist in `format_answer`.
- Unconditional resume-offer prompts for hiring managers ("Would you like me to
  email you my resume") — resume offers are now gated behind hiring signals /
  engagement scoring, with different copy.
- Resume email + "Resume dispatched" SMS dispatch driven by `user_email` /
  `user_name` stashed in state — nothing populates those state fields anymore,
  and the old `nodes.get_resend_service` monkeypatch surface is gone.
- Contact-request email/SMS notifications — direct contact requests are now
  intercepted by the stage-1 intent router (contact form + `pipeline_halt`)
  before `plan_actions` runs.
- The class-based `ConversationState` interface (`.fetch()`, `.stash()`,
  `.set_answer()`, `.pending_actions` attribute) — replaced by the TypedDict
  state with partial-update merging.

## Running Tests
```bash
OPENAI_API_KEY=sk-dummy ANTHROPIC_API_KEY=sk-ant-dummy \
SUPABASE_URL=https://dummy.supabase.co SUPABASE_SERVICE_ROLE_KEY=dummy \
.venv/bin/python -m pytest tests/test_code_display_policy.py \
  tests/test_code_display_accuracy.py tests/test_code_display_ci.py -q
```

## Current Status

| Test Suite | Tests | Coverage Area |
|------------|-------|---------------|
| `test_code_display_policy.py` | 24 | Trigger detection, action planning, import retrieval, formatting helpers |
| `test_code_display_accuracy.py` | 1 | classify → plan → format code block enrichment |
| `test_code_display_ci.py` | 1 | action_request planning, formatting, execution |
| **TOTAL** | **26** | |

## Maintenance Notes
- Tests are hermetic: no LLM, Supabase, Resend, or Twilio calls. Dummy env
  vars are sufficient; the real `.env` must not be required.
- `test_get_import_explanation_*` and `test_search_import_explanations` read
  the real `data/imports_kb.csv` from disk (local file only).
- The legacy role strings ("Software Developer", "Hiring Manager (technical)",
  etc.) are still branched on inside `plan_actions` and `ROLE_TO_TIER`, so the
  tests use them as fixture data even though the production UI no longer
  assigns these role values.
