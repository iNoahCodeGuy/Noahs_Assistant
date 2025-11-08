# How to Follow Node & State Logic as Conversation Progresses

**Quick Start:** Open these 3 files side-by-side to trace any conversation.

## The 3-File System

### 1. **CONVERSATION_JOURNEY_MAP.md** (Your Roadmap)
**Location:** `docs/CONVERSATION_JOURNEY_MAP.md`

**Use When:** Understanding overall flow, debugging state issues

**How to Read:**
```
1. Find the stage you're interested in (0-7)
2. Look at "State After" section
3. See exactly what fields exist at that point
```

**Example - Debugging "Why is role empty?":**
- Go to Stage 2 (Classification)
- See: `"role": "Just looking around"` ← Set here by classify_role_mode
- Check Stage 1: `"role": ""` ← Still empty
- **Conclusion:** Role doesn't exist until after Stage 2

### 2. **NODE_STATE_QUICK_REFERENCE.md** (Your Lookup Table)
**Location:** `docs/NODE_STATE_QUICK_REFERENCE.md`

**Use When:** Finding which node modifies a specific field

**How to Search:**
```bash
# Find who sets 'retrieval_scores'
# Search in file for "retrieval_scores"
# Find: retrieve_chunks (Stage 4) sets it
```

**Example - Debugging "Where does hiring_signals get set?":**
- Search table for "hiring_signals"
- Find row: `plan_actions | 6 | hiring_signals | entities, query_intent`
- **Answer:** Stage 6 (plan_actions node) sets it, reads from entities

### 3. **conversation_flow.py** (Your Pipeline Definition)
**Location:** `src/flows/conversation_flow.py` lines 125-195

**Use When:** Seeing node execution order, understanding short-circuits

**How to Read:**
```python
# Each stage has visual separator:
# ═══ STAGE 2: CLASSIFICATION ═══
# Lists nodes in execution order:
classify_role_mode,    # Node 1
classify_intent,       # Node 2  
extract_entities,      # Node 3
```

**Example - Understanding "Why did pipeline exit early?":**
- Look for `# Short-Circuit:` comments
- Stage 1: Exits if `is_greeting=True`
- Stage 3: Exits if `clarification_needed=True`
- Stage 4: Exits if `grounding_status="insufficient"`

---

## Step-by-Step: Tracing a Conversation

### Scenario: User asks "Tell me about Noah's Python experience"

**Step 1: Find the Query in JOURNEY_MAP**
- Open `docs/CONVERSATION_JOURNEY_MAP.md`
- See example query matches exactly
- Read through Stages 0-7 to see full state evolution

**Step 2: Check Specific Field**
- Want to know when `depth_level` is set?
- Look at Stage 3 (Query Preparation)
- See: `"depth_level": 2` ← Set by presentation_controller

**Step 3: Find the Node Implementation**
- Search `NODE_STATE_QUICK_REFERENCE.md` for "depth_level"
- Find: `presentation_controller | 3 | depth_level | role, query_type`
- Go to: `src/flows/node_logic/presentation_control.py`
- Read `presentation_controller()` function

**Step 4: Understand Why Value Was Chosen**
- See in code: `if role == "Just looking around": depth_level = 2`
- Cross-reference JOURNEY_MAP Stage 2: `"role": "Just looking around"`
- **Conclusion:** Casual visitors get medium depth (not 1=brief, not 3=deep)

---

## Quick Debugging Patterns

### Problem: Wrong answer tone/depth

**Steps:**
1. Open `NODE_STATE_QUICK_REFERENCE.md`
2. Search "Debugging Checklist" → "Wrong answer tone"
3. Find: Check `role`, `depth_level`, `display_toggles`
4. Open `CONVERSATION_JOURNEY_MAP.md` → Stage 2 & 3
5. Verify role classification and presentation settings

### Problem: No code examples shown

**Steps:**
1. Search `NODE_STATE_QUICK_REFERENCE.md` for "Missing code examples"
2. Find: Check `display_toggles.show_code`, `code_snippets`
3. Open `conversation_flow.py` → Stage 3 (presentation_controller)
4. Check if `role == "Software Developer"` (required for show_code=True)

### Problem: Pipeline exits before generation

**Steps:**
1. Open `conversation_flow.py`
2. Look for `# Short-Circuit:` comments in Stages 1-4
3. Open `CONVERSATION_JOURNEY_MAP.md` → "Short-Circuit Paths" section
4. Identify which condition triggered (is_greeting, clarification_needed, etc.)

### Problem: Resume not sent to recruiter

**Steps:**
1. Open `NODE_STATE_QUICK_REFERENCE.md` → Search "Resume not sent"
2. Find: Check entities, hiring_signals, planned_actions
3. Open `CONVERSATION_JOURNEY_MAP.md` → Stage 2 (extract_entities)
4. Verify email/company extracted correctly
5. Check Stage 6 (plan_actions) → Are hiring_signals detected?

---

## File Locations Quick Reference

| What You Need | File | Lines |
|---------------|------|-------|
| Full state evolution example | `docs/CONVERSATION_JOURNEY_MAP.md` | All |
| Which node sets which field | `docs/NODE_STATE_QUICK_REFERENCE.md` | Table at top |
| Node execution order | `src/flows/conversation_flow.py` | 125-195 |
| Stage 0 implementation | `src/flows/node_logic/session_management.py` | initialize_conversation_state |
| Stage 1 implementation | `src/flows/node_logic/session_management.py` | prompt_for_role_selection |
| Stage 2: Role | `src/flows/node_logic/role_routing.py` | classify_role_mode |
| Stage 2: Intent | `src/flows/node_logic/query_classification.py` | classify_intent |
| Stage 2: Entities | `src/flows/node_logic/entity_extraction.py` | extract_entities |
| Stage 3: Clarification | `src/flows/node_logic/clarification.py` | assess + ask |
| Stage 3: Presentation | `src/flows/node_logic/presentation_control.py` | presentation_controller |
| Stage 3: Query | `src/flows/node_logic/query_composition.py` | compose_query |
| Stage 4: Retrieval | `src/flows/node_logic/retrieval_nodes.py` | retrieve_chunks |
| Stage 4: Grounding | `src/flows/node_logic/retrieval_nodes.py` | validate + handle_gap |
| Stage 5: Generation | `src/flows/node_logic/generation_nodes.py` | generate_draft |
| Stage 5: Safety | `src/flows/node_logic/generation_nodes.py` | hallucination_check |
| Stage 6: Actions | `src/flows/node_logic/action_planning.py` | plan_actions |
| Stage 6: Formatting | `src/flows/node_logic/formatting_nodes.py` | format_answer |
| Stage 7: Execution | `src/flows/node_logic/action_execution.py` | execute_actions |
| Stage 7: Memory | `src/flows/node_logic/logging_nodes.py` | update_memory |
| Stage 7: Analytics | `src/flows/node_logic/logging_nodes.py` | log_and_notify |

---

## Pro Tips

### 1. Use VS Code's "Go to Definition" (F12)
```python
# In conversation_flow.py, click on any node:
classify_role_mode,  # ← Press F12 here
# Jumps directly to implementation
```

### 2. Search Across Files (Cmd+Shift+F)
```
Query: "depth_level"
Results:
- conversation_state.py (definition)
- presentation_control.py (sets it)
- generation_nodes.py (reads it)
- JOURNEY_MAP.md (example values)
```

### 3. Use Grep for Field Usage
```bash
# Find all nodes that read 'role'
grep -n "state.get(\"role\")" src/flows/node_logic/*.py

# Find all nodes that set 'answer'
grep -n "state\[\"answer\"\]" src/flows/node_logic/*.py
```

### 4. Follow the Stage Comments in conversation_flow.py
- Each stage has `# ═══ STAGE X: NAME ═══` separator
- Lists: Purpose, State Modified, Performance, Short-Circuit conditions
- Read top-to-bottom to understand execution order

---

## Learning Path for New Developers

### Day 1: Understanding State
1. Read `src/state/conversation_state.py` (46 field definitions)
2. Read `docs/CONVERSATION_JOURNEY_MAP.md` Stages 0-2
3. Trace example query through first 3 stages

### Day 2: Following Retrieval
1. Read `docs/CONVERSATION_JOURNEY_MAP.md` Stages 3-4
2. Open `src/flows/node_logic/retrieval_nodes.py`
3. Understand: composed_query → pgvector → retrieved_chunks

### Day 3: Generation & Formatting
1. Read `docs/CONVERSATION_JOURNEY_MAP.md` Stages 5-6
2. Open `src/flows/node_logic/generation_nodes.py`
3. Open `src/flows/node_logic/formatting_nodes.py`
4. Understand: retrieved_chunks → draft_answer → answer (with followups)

### Day 4: Actions & Side Effects
1. Read `docs/CONVERSATION_JOURNEY_MAP.md` Stage 7
2. Open `src/flows/node_logic/action_planning.py`
3. Open `src/flows/node_logic/action_execution.py`
4. Understand: hiring_signals → planned_actions → executed_actions

### Day 5: Practice
1. Pick a field (e.g., `display_toggles`)
2. Search `NODE_STATE_QUICK_REFERENCE.md` to find writer node
3. Read node implementation
4. Trace through JOURNEY_MAP to see value evolution

---

## Summary: The Workflow

```
┌─────────────────────────────────────────────────┐
│ 1. START: User asks "Tell me about Python"     │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│ 2. Open: CONVERSATION_JOURNEY_MAP.md           │
│    → Find similar example query                 │
│    → Read Stages 0-7 state snapshots            │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│ 3. Question: "When is role set?"               │
│    → Check Stage 2 in JOURNEY_MAP               │
│    → See: classify_role_mode sets it            │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│ 4. Open: NODE_STATE_QUICK_REFERENCE.md         │
│    → Search for "role" in ownership table       │
│    → Find: classify_role_mode (Stage 2)         │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│ 5. Open: src/flows/node_logic/role_routing.py  │
│    → Read classify_role_mode() implementation   │
│    → Understand classification logic            │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│ 6. DONE: You now understand the full flow      │
│    from user query → role classification        │
└─────────────────────────────────────────────────┘
```

---

## TL;DR

**3 files, 3 purposes:**

1. **CONVERSATION_JOURNEY_MAP.md** = See state evolution over time
2. **NODE_STATE_QUICK_REFERENCE.md** = Find which node owns which field
3. **conversation_flow.py** (lines 125-195) = See execution order with stage comments

**Workflow:**
1. Open JOURNEY_MAP → understand overall flow
2. Open QUICK_REFERENCE → find specific node
3. Open node file → read implementation
4. Use F12 (Go to Definition) to jump between files

**That's it!** The stage-based thinking is documented, even though files are still in `node_logic/` directory.
