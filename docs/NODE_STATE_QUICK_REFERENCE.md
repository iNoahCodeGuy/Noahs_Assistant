# Node → State Quick Reference

**Purpose:** Fast lookup showing which nodes read/write which state fields.

**How to use:** Need to debug `retrieval_scores`? Search this table to find which nodes touch it.

---

## Complete Node-State Matrix

| Node | Stage | Primary Fields Set | Primary Fields Read | Short-Circuit? | Exit Condition |
|------|-------|-------------------|---------------------|----------------|----------------|
| **initialize_conversation_state** | 0 | All 46 fields (defaults) | None | No | N/A |
| **prompt_for_role_selection** | 1 | `answer`, `pipeline_halt`, `is_greeting`, `session_memory.persona_hints` | `role`, `session_memory` | Yes | First turn + no role |
| **handle_greeting** | 1 | `answer`, `is_greeting` | `query`, `chat_history` | Yes | Simple greeting detected |
| **classify_role_mode** | 2 | `role`, `role_mode`, `analytics_metadata.role_confidence` | `query`, `chat_history`, `session_memory` | No | N/A |
| **classify_intent** | 2 | `query_type`, `query_intent`, `topic_focus`, `relate_to_enterprise`, `analytics_metadata.technical_subcategories` | `query`, `role` | No | N/A |
| **extract_entities** | 2 | `entities` (company, email, phone, name, role_title) | `query` | No | N/A |
| **assess_clarification_need** | 3 | `clarification_needed` | `query`, `chat_history`, `entities` | No | N/A |
| **ask_clarifying_question** | 3 | `answer`, `clarifying_question`, `pipeline_halt` | `clarification_needed`, `query` | Yes | Vague query detected |
| **presentation_controller** | 3 | `depth_level`, `display_toggles`, `display_reasons` | `role`, `query_type`, `query_intent` | No | N/A |
| **compose_query** | 3 | `composed_query` | `query`, `role`, `entities`, `topic_focus` | No | N/A |
| **retrieve_chunks** | 4 | `retrieved_chunks`, `retrieval_scores`, `code_snippets` (if developer) | `composed_query`, `role`, `query_type` | No | N/A |
| **validate_grounding** | 4 | `grounding_status`, `grounding_gap_reason` | `retrieval_scores`, `retrieved_chunks` | No | N/A |
| **handle_grounding_gap** | 4 | `answer`, `pipeline_halt`, `followup_prompts` | `grounding_status`, `query`, `topic_focus` | Yes | Low retrieval scores |
| **generate_draft** | 5 | `draft_answer` | `retrieved_chunks`, `composed_query`, `role`, `depth_level`, `chat_history` | No | N/A |
| **hallucination_check** | 5 | `answer`, `hallucination_safe`, `citations` | `draft_answer`, `retrieved_chunks` | No | N/A |
| **plan_actions** | 6 | `planned_actions`, `hiring_signals`, `hiring_signals_strength`, `resume_offered` | `entities`, `query_intent`, `role`, `chat_history` | No | N/A |
| **format_answer** | 6 | `answer`, `followup_prompts` | `draft_answer`, `role`, `query_type`, `topic_focus`, `display_toggles`, `depth_level` | No | N/A |
| **execute_actions** | 7 | `executed_actions` | `planned_actions`, `entities` | No | N/A |
| **update_memory** | 7 | `session_memory` (affinity_scores, query_count, last_topic) | `query`, `query_type`, `role`, `entities` | No | N/A |
| **log_and_notify** | 7 | `analytics_metadata` (latency_ms, timestamp) | All fields | No | Always runs |

---

## State Field Ownership (Primary Writer)

| Field | Primary Node | Stage | Also Modified By | Notes |
|-------|--------------|-------|------------------|-------|
| **query** | (User input) | 0 | Never | Immutable after initialization |
| **role** | classify_role_mode | 2 | Never | Set once, used everywhere |
| **session_id** | (User input) | 0 | Never | Immutable |
| **chat_history** | (User input) | 0 | Never | Read-only in pipeline |
| **role_mode** | classify_role_mode | 2 | Never | "hiring" vs "exploration" |
| **query_type** | classify_intent | 2 | Never | "technical", "career", "general", "mma" |
| **query_intent** | classify_intent | 2 | Never | "INFORM_MODE", "ASK_MODE", "SHOW_ME_MODE" |
| **topic_focus** | classify_intent | 2 | Never | "technical_skills", "leadership", etc. |
| **relate_to_enterprise** | classify_intent | 2 | Never | Boolean flag |
| **entities** | extract_entities | 2 | Never | Dict with company, email, phone, etc. |
| **is_greeting** | handle_greeting | 1 | prompt_for_role_selection | Triggers short-circuit |
| **pipeline_halt** | Multiple | 1/3/4 | Cleared by prompt_for_role_selection | Stops pipeline execution |
| **clarification_needed** | assess_clarification_need | 3 | Never | Boolean flag |
| **clarifying_question** | ask_clarifying_question | 3 | Never | String question |
| **depth_level** | presentation_controller | 3 | Never | 1 (brief), 2 (moderate), 3 (deep) |
| **display_toggles** | presentation_controller | 3 | Never | Dict of show_code, show_bullets, etc. |
| **display_reasons** | presentation_controller | 3 | Never | Why toggles set this way |
| **composed_query** | compose_query | 3 | Never | Retrieval-ready query |
| **retrieved_chunks** | retrieve_chunks | 4 | Never | List of dicts |
| **retrieval_scores** | retrieve_chunks | 4 | Never | List of floats |
| **code_snippets** | retrieve_chunks | 4 | Never | List (for developers only) |
| **grounding_status** | validate_grounding | 4 | Never | "sufficient", "insufficient", "unknown" |
| **grounding_gap_reason** | validate_grounding | 4 | Never | Why grounding failed |
| **draft_answer** | generate_draft | 5 | Never | Before citations added |
| **answer** | generate_draft | 5 | hallucination_check, format_answer, handle_grounding_gap, ask_clarifying_question | Final user-facing response |
| **hallucination_safe** | hallucination_check | 5 | Never | Boolean flag |
| **citations** | hallucination_check | 5 | Never | List of source refs |
| **planned_actions** | plan_actions | 6 | Never | List of action dicts |
| **hiring_signals** | plan_actions | 6 | Never | List of detected signals |
| **hiring_signals_strength** | plan_actions | 6 | Never | "strong", "weak", "none" |
| **resume_offered** | plan_actions | 6 | Never | Boolean flag |
| **followup_prompts** | format_answer | 6 | handle_grounding_gap | List of suggested questions |
| **executed_actions** | execute_actions | 7 | Never | List of completed actions |
| **session_memory** | update_memory | 7 | prompt_for_role_selection | Persists across turns |
| **analytics_metadata** | classify_intent | 2 | log_and_notify | Accumulated metrics |
| **timestamp** | log_and_notify | 7 | Never | When logged |

---

## Fields by Read Frequency (Most Used)

| Field | Read By (# of Nodes) | Nodes |
|-------|----------------------|-------|
| **role** | 10 nodes | classify_intent, presentation_controller, compose_query, retrieve_chunks, generate_draft, format_answer, plan_actions, update_memory, log_and_notify, handle_greeting |
| **query** | 8 nodes | handle_greeting, classify_role_mode, classify_intent, extract_entities, assess_clarification_need, ask_clarifying_question, compose_query, update_memory |
| **query_type** | 6 nodes | presentation_controller, retrieve_chunks, format_answer, handle_grounding_gap, update_memory, log_and_notify |
| **chat_history** | 5 nodes | handle_greeting, classify_role_mode, assess_clarification_need, generate_draft, plan_actions |
| **entities** | 4 nodes | assess_clarification_need, compose_query, plan_actions, execute_actions |

---

## Fields by Write Frequency (Most Modified)

| Field | Written By (# of Nodes) | Nodes |
|-------|-------------------------|-------|
| **answer** | 5 nodes | prompt_for_role_selection, handle_greeting, ask_clarifying_question, handle_grounding_gap, hallucination_check, format_answer |
| **pipeline_halt** | 4 nodes | prompt_for_role_selection, handle_greeting, ask_clarifying_question, handle_grounding_gap |
| **analytics_metadata** | 2 nodes | classify_intent, log_and_notify |
| **session_memory** | 2 nodes | prompt_for_role_selection, update_memory |
| All others | 1 node | (See ownership table above) |

---

## Short-Circuit Nodes (Can Exit Pipeline Early)

| Node | Exit Condition | Sets Before Exit |
|------|----------------|------------------|
| **prompt_for_role_selection** | First turn, no role set | `answer=_INITIAL_GREETING`, `pipeline_halt=True`, `is_greeting=True` |
| **handle_greeting** | Simple greeting detected | `answer="Hey there! What would you like to explore?"`, `is_greeting=True` |
| **ask_clarifying_question** | Vague query | `answer=<clarifying question>`, `pipeline_halt=True`, `clarifying_question=<text>` |
| **handle_grounding_gap** | Low retrieval scores | `answer=<fallback suggestions>`, `pipeline_halt=True`, `followup_prompts=[...]` |

**Note:** When short-circuit occurs, remaining stages (5-7) are skipped EXCEPT `log_and_notify`, which always runs.

---

## State Mutation Patterns

### Immutable Fields (Never Modified After Init)
- `query` (user input)
- `session_id` (UUID)
- `chat_history` (read-only snapshot)

### Write-Once Fields (Set Once, Never Changed)
- `role` (Stage 2)
- `query_type` (Stage 2)
- `query_intent` (Stage 2)
- `depth_level` (Stage 3)
- `composed_query` (Stage 3)
- `draft_answer` (Stage 5)

### Accumulative Fields (Appended To)
- `retrieved_chunks` (Stage 4: append from pgvector)
- `planned_actions` (Stage 6: append from plan_actions)
- `executed_actions` (Stage 7: append from execute_actions)
- `followup_prompts` (Stage 6: append from format_answer or handle_grounding_gap)

### Overwrite Fields (Replaced Multiple Times)
- `answer` (Stages 1, 3, 4, 5, 6: different nodes update based on flow)
- `pipeline_halt` (Stages 1, 3, 4: set to True, then cleared)
- `analytics_metadata` (Stages 2, 7: incremental enrichment)
- `session_memory` (Stages 1, 7: merged dictionaries)

---

## Debugging Checklist

**Problem:** Wrong answer tone/depth  
**Suspect Fields:** `role`, `depth_level`, `display_toggles`  
**Check Nodes:** classify_role_mode (Stage 2), presentation_controller (Stage 3)

**Problem:** Irrelevant retrieval results  
**Suspect Fields:** `composed_query`, `retrieval_scores`, `retrieved_chunks`  
**Check Nodes:** compose_query (Stage 3), retrieve_chunks (Stage 4)

**Problem:** Missing code examples  
**Suspect Fields:** `display_toggles.show_code`, `code_snippets`  
**Check Nodes:** presentation_controller (Stage 3), retrieve_chunks (Stage 4)

**Problem:** No followup questions  
**Suspect Fields:** `followup_prompts`, `query_type`, `topic_focus`  
**Check Nodes:** format_answer (Stage 6)

**Problem:** Resume not sent  
**Suspect Fields:** `hiring_signals`, `planned_actions`, `executed_actions`, `entities`  
**Check Nodes:** extract_entities (Stage 2), plan_actions (Stage 6), execute_actions (Stage 7)

**Problem:** Session memory not persisting  
**Suspect Fields:** `session_memory`, `session_id`  
**Check Nodes:** update_memory (Stage 7), check if session_id matches across turns

**Problem:** Pipeline exits early unexpectedly  
**Suspect Fields:** `pipeline_halt`, `is_greeting`, `clarification_needed`, `grounding_status`  
**Check Nodes:** prompt_for_role_selection, handle_greeting, ask_clarifying_question, handle_grounding_gap

---

## Related Documentation

- **CONVERSATION_JOURNEY_MAP.md** - Full walkthrough of state transformations per stage
- **conversation_state.py** - TypedDict definition of all 46 fields
- **conversation_flow.py** - Pipeline implementation with stage comments
- **docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md** - High-level architecture
