# Conversation Journey Map: User Query → Response

**Purpose:** Trace a single conversation turn through all 18 nodes, showing exactly what state fields exist at each stage.

**How to use this:** Read top-to-bottom to follow a user's query through the pipeline. Check "State After" sections to see what data is available at each checkpoint.

---

## Example Query: "Tell me about Noah's Python experience"

### Stage 0: INITIALIZATION

**Nodes:** `initialize_conversation_state`

**Purpose:** Set up all 46 state fields with safe defaults so downstream nodes never encounter missing keys.

**State After:**
```python
{
    # Core fields (user-provided)
    "query": "Tell me about Noah's Python experience",
    "role": "",  # Empty - not inferred yet
    "session_id": "abc-123",
    "chat_history": [],
    
    # Structured containers (initialized empty)
    "session_memory": {},
    "entities": {},
    "analytics_metadata": {},
    "retrieved_chunks": [],
    "retrieval_scores": [],
    "code_snippets": [],
    "planned_actions": [],
    "executed_actions": [],
    "hiring_signals": [],
    "followup_prompts": [],
    
    # Status flags (safe defaults)
    "is_greeting": False,
    "pipeline_halt": False,
    "clarification_needed": False,
    "hallucination_safe": True,
    "grounding_status": "unknown",
    "topic_focus": "general",
    "relate_to_enterprise": False,
    
    # Answer fields (empty until generation)
    "answer": "",
    "draft_answer": "",
    "clarifying_question": "",
    "composed_query": "",
    
    # ... (22 more fields with defaults)
}
```

**Key Insight:** After this stage, every node can safely call `state.get("field_name")` without None checks.

---

### Stage 1: GREETING

**Nodes:** `prompt_for_role_selection`, `handle_greeting`

**Purpose:** 
- `prompt_for_role_selection`: Show Portfolia's first message if no role set
- `handle_greeting`: Detect simple greetings ("hi", "hello") and respond warmly

**State After:**
```python
{
    # Query analysis
    "query": "Tell me about Noah's Python experience",  # Not a greeting
    "role": "",  # Still empty (will be inferred in Stage 2)
    
    # Greeting detection
    "is_greeting": False,  # ✅ Not "hi/hello", this is a substantive question
    "pipeline_halt": False,  # ✅ Continue to next stage
    
    # Session tracking
    "session_memory": {
        "persona_hints": {
            "initial_greeting_shown": True  # ✅ Portfolia already messaged first
        }
    },
    
    # ... all other fields unchanged from Stage 0
}
```

**Short-Circuit Check:** If `is_greeting=True`, pipeline exits here with warm welcome message.

**Key Insight:** Non-greeting queries skip answer generation here and proceed to classification.

---

### Stage 2: CLASSIFICATION

**Nodes:** `classify_role_mode`, `classify_intent`, `extract_entities`

**Purpose:** Understand WHO the user is, WHAT they want, and WHO/WHAT they're asking about.

**State After:**
```python
{
    # User identity (WHO)
    "role": "Just looking around",  # ✅ Inferred from casual tone
    "role_mode": "exploration",     # ✅ Not hiring-focused
    
    # Query intent (WHAT)
    "query_type": "career",           # ✅ Career/background question
    "query_intent": "INFORM_MODE",    # ✅ Wants factual information (not "show me code")
    "topic_focus": "technical_skills", # ✅ Python = technical domain
    "relate_to_enterprise": False,    # ✅ No enterprise/business keywords
    
    # Extracted entities (WHO/WHAT they mention)
    "entities": {},  # No company name, email, or contact info detected
    
    # Analytics metadata (tracked for debugging)
    "analytics_metadata": {
        "role_confidence": 0.75,  # Medium confidence in role inference
        "technical_subcategories": ["programming_languages", "backend"]
    },
    
    # ... all other fields unchanged from Stage 1
}
```

**Key Insight:** After this stage, we know the user is a casual visitor asking about technical skills (not hiring, not code review).

---

### Stage 3: QUERY PREPARATION

**Nodes:** `assess_clarification_need`, `ask_clarifying_question`, `presentation_controller`, `compose_query`

**Purpose:** Clarify vague queries, set presentation depth, build retrieval-ready query.

**State After:**
```python
{
    # Clarification check
    "clarification_needed": False,  # ✅ Query is clear enough
    "clarifying_question": "",      # No need to ask followup
    
    # Presentation settings (depth & display)
    "depth_level": 2,  # Medium depth (1=brief, 2=moderate, 3=deep)
    "display_toggles": {
        "show_code": False,      # ✅ Not a developer, hide code
        "show_architecture": False,  # Not technical HM
        "show_bullets": True,    # Structured format
        "show_metrics": False,   # No enterprise focus
        "show_certifications": False
    },
    "display_reasons": {
        "show_code": "Role is not developer/technical",
        "show_bullets": "Improves readability for general queries"
    },
    
    # Retrieval-ready query
    "composed_query": "[Just looking around] Tell me about Noah's Python experience in backend systems and data engineering",
    # ☝️ Includes role context + expanded keywords for better retrieval
    
    # ... all other fields unchanged from Stage 2
}
```

**Short-Circuit Check:** If `clarification_needed=True`, pipeline exits here with clarifying question.

**Key Insight:** Presentation settings are decided BEFORE retrieval so formatting is consistent.

---

### Stage 4: RETRIEVAL

**Nodes:** `retrieve_chunks`, `validate_grounding`, `handle_grounding_gap`

**Purpose:** Search pgvector for relevant knowledge, validate sufficiency, handle gaps.

**State After:**
```python
{
    # Retrieved knowledge
    "retrieved_chunks": [
        {
            "content": "Noah has 5+ years of Python experience across backend APIs, data pipelines, and AI applications...",
            "doc_id": "career_kb.csv",
            "section": "technical_skills_python",
            "similarity": 0.89
        },
        {
            "content": "Backend stack: Flask, FastAPI, PostgreSQL, Redis. Built RESTful APIs handling 10K+ req/min...",
            "doc_id": "career_kb.csv",
            "section": "technical_stack",
            "similarity": 0.85
        },
        {
            "content": "Database experience: PostgreSQL (5 years), MongoDB, Redis. Designed schemas for multi-tenant SaaS...",
            "doc_id": "career_kb.csv",
            "section": "database_expertise",
            "similarity": 0.78
        }
    ],
    
    # Retrieval metadata
    "retrieval_scores": [0.89, 0.85, 0.78],
    "code_snippets": [],  # Empty - no code retrieval for non-developers
    
    # Grounding validation
    "grounding_status": "sufficient",  # ✅ Avg score 0.84 > threshold 0.3
    "grounding_gap_reason": "",        # No gap detected
    
    # ... all other fields unchanged from Stage 3
}
```

**Short-Circuit Check:** If `grounding_status="insufficient"`, pipeline exits with fallback suggestions.

**Key Insight:** High similarity scores (0.78-0.89) mean confident retrieval → safe to generate answer.

---

### Stage 5: GENERATION

**Nodes:** `generate_draft`, `hallucination_check`

**Purpose:** Generate LLM answer from retrieved context, validate with citations.

**State After:**
```python
{
    # Draft answer (before safety checks)
    "draft_answer": """Noah has extensive Python experience spanning over 5 years, focused on backend development and data engineering.

**Backend Development:**
- Built production APIs using Flask and FastAPI
- Handled high-throughput systems (10K+ requests/min)
- Designed RESTful endpoints with proper authentication

**Database Expertise:**
- PostgreSQL (5 years): schema design, query optimization
- MongoDB and Redis for caching and real-time features
- Multi-tenant SaaS architecture

His Python work emphasizes production-ready systems with proper testing, monitoring, and scalability patterns.""",

    # Final answer (after citation injection)
    "answer": """Noah has extensive Python experience spanning over 5 years, focused on backend development and data engineering.

**Backend Development:**
- Built production APIs using Flask and FastAPI [1]
- Handled high-throughput systems (10K+ requests/min) [1]
- Designed RESTful endpoints with proper authentication

**Database Expertise:**
- PostgreSQL (5 years): schema design, query optimization [2]
- MongoDB and Redis for caching and real-time features [2]
- Multi-tenant SaaS architecture [2]

His Python work emphasizes production-ready systems with proper testing, monitoring, and scalability patterns.

**Sources:**
[1] career_kb.csv - technical_stack
[2] career_kb.csv - database_expertise""",
    
    # Safety validation
    "hallucination_safe": True,  # ✅ All claims backed by retrieved chunks
    "citations": [
        {"chunk_index": 0, "section": "technical_skills_python"},
        {"chunk_index": 1, "section": "technical_stack"},
        {"chunk_index": 2, "section": "database_expertise"}
    ],
    
    # ... all other fields unchanged from Stage 4
}
```

**Key Insight:** Answer is grounded in retrieved context with explicit source citations.

---

### Stage 6: ENRICHMENT

**Nodes:** `plan_actions`, `format_answer`

**Purpose:** Decide on actions (email/resume), format answer with followups/toggles.

**State After:**
```python
{
    # Action planning
    "planned_actions": [],  # No actions - no hiring signals detected
    "hiring_signals": [],   # No email/company mentioned
    "hiring_signals_strength": "none",  # Not a recruiting conversation
    
    # Followup generation (based on query_type="career" + topic="technical_skills")
    "followup_prompts": [
        "What backend frameworks has Noah used in production?",
        "Tell me about Noah's database architecture experience",
        "What's Noah's experience with API design?"
    ],
    
    # Final formatted answer (with enrichments)
    "answer": """Noah has extensive Python experience spanning over 5 years, focused on backend development and data engineering.

**Backend Development:**
- Built production APIs using Flask and FastAPI [1]
- Handled high-throughput systems (10K+ requests/min) [1]
- Designed RESTful endpoints with proper authentication

**Database Expertise:**
- PostgreSQL (5 years): schema design, query optimization [2]
- MongoDB and Redis for caching and real-time features [2]
- Multi-tenant SaaS architecture [2]

His Python work emphasizes production-ready systems with proper testing, monitoring, and scalability patterns.

---

**Want to explore more?**
- What backend frameworks has Noah used in production?
- Tell me about Noah's database architecture experience
- What's Noah's experience with API design?

**Sources:**
[1] career_kb.csv - technical_stack
[2] career_kb.csv - database_expertise""",
    
    # ... all other fields unchanged from Stage 5
}
```

**Key Insight:** Followups are auto-generated based on query context (career + technical_skills → related questions).

---

### Stage 7: FINALIZATION

**Nodes:** `execute_actions`, `update_memory`, `log_and_notify`

**Purpose:** Fire side-effects (email/SMS/storage), update session memory, log analytics.

**State After:**
```python
{
    # Action execution
    "executed_actions": [],  # No actions to execute (planned_actions was empty)
    
    # Session memory (tracks user patterns across turns)
    "session_memory": {
        "persona_hints": {
            "initial_greeting_shown": True
        },
        "career_queries": 1,      # ✅ First career question this session
        "technical_queries": 0,   # No code/architecture questions yet
        "last_topic": "technical_skills",
        "query_count": 1,
        "affinity_scores": {
            "enterprise": 0.0,    # No business keywords
            "technical": 0.4      # Moderate technical interest
        }
    },
    
    # Analytics metadata (logged to Supabase)
    "analytics_metadata": {
        "session_id": "abc-123",
        "role": "Just looking around",
        "query_type": "career",
        "latency_ms": 1200,  # ✅ Total pipeline execution time
        "retrieval_count": 3,
        "retrieval_avg_score": 0.84,
        "tokens_prompt": 320,
        "tokens_completion": 180,
        "model_used": "gpt-4o-mini",
        "hallucination_safe": True,
        "actions_taken": [],
        "timestamp": "2025-11-07T20:15:30Z"
    },
    
    # Final answer (unchanged - this is what user sees)
    "answer": "[Same as Stage 6 - formatted with followups]",
    
    # ... all other fields unchanged from Stage 6
}
```

**Key Insight:** Session memory persists across turns, allowing personalization in future queries ("You asked about Python earlier...").

---

## Next Turn: User asks "What about FastAPI?"

**Stages repeat with updated context:**

- **Stage 0:** State initialized with chat_history containing previous turn
- **Stage 1:** Not a greeting (skip)
- **Stage 2:** 
  - `role` already set: "Just looking around" (from session_memory)
  - `query_type`: "technical" (more specific this time)
  - `topic_focus`: "frameworks"
  - `session_memory.career_queries`: 2 (incremented)
- **Stage 3:** 
  - `composed_query`: "[Just looking around] Tell me about Noah's FastAPI experience, especially in relation to his previous Python backend work"
  - `depth_level`: 2 (consistent with first turn)
- **Stage 4:** Retrieves FastAPI-specific chunks
- **Stage 5:** Generates answer referencing previous context ("As mentioned earlier...")
- **Stage 6:** Followups suggest: "Compare Flask vs FastAPI", "Show me FastAPI code"
- **Stage 7:** `session_memory.last_topic`: "frameworks", affinity_scores.technical increases

---

## State Field Lifecycle Summary

| Field | Initialized | First Set | Modified By | Final Use |
|-------|-------------|-----------|-------------|-----------|
| `query` | Stage 0 | Stage 0 (user input) | Never | Stage 5 (generation) |
| `role` | Stage 0 (empty) | Stage 2 (classification) | Never | All stages |
| `query_type` | Stage 0 | Stage 2 (classification) | Never | Stage 4 (retrieval) |
| `depth_level` | Stage 0 | Stage 3 (presentation) | Never | Stage 6 (formatting) |
| `retrieved_chunks` | Stage 0 (empty) | Stage 4 (retrieval) | Never | Stage 5 (generation) |
| `answer` | Stage 0 (empty) | Stage 5 (generation) | Stage 6 (enrichment) | Return to user |
| `session_memory` | Stage 0 (empty) | Stage 1 (greeting) | Stage 7 (memory) | Persist for next turn |
| `analytics_metadata` | Stage 0 (empty) | Stage 2 (classification) | Stage 7 (logging) | Logged to Supabase |

---

## Short-Circuit Paths

**Path 1: First-turn greeting**
```
Stage 0 (init) → Stage 1 (greeting shows, halt) → Stage 7 (log only)
```

**Path 2: Simple greeting mid-conversation**
```
Stage 0 → Stage 1 (greeting detected, halt) → Stage 7
```

**Path 3: Vague query**
```
Stage 0 → Stage 1 → Stage 2 → Stage 3 (clarification needed, halt) → Stage 7
```

**Path 4: Insufficient retrieval**
```
Stage 0 → Stage 1 → Stage 2 → Stage 3 → Stage 4 (grounding gap, halt) → Stage 7
```

**Path 5: Full pipeline (normal case)**
```
Stage 0 → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 6 → Stage 7
```

**Note:** Stage 7 (`log_and_notify`) ALWAYS executes, even on short-circuits.

---

## Debugging Tips

**Problem:** User gets irrelevant answer  
**Check:** Stage 4 `retrieval_scores` - Are they > 0.5? If not, KB may be missing data.

**Problem:** Answer has wrong tone/depth  
**Check:** Stage 2 `role` and Stage 3 `depth_level` - Is role correctly inferred?

**Problem:** Missing followup questions  
**Check:** Stage 6 `followup_prompts` - Does `query_type` match intent?

**Problem:** Actions not executing  
**Check:** Stage 6 `planned_actions` - Are hiring signals detected in Stage 2 `entities`?

**Problem:** Session memory not persisting  
**Check:** Stage 7 `session_memory` - Is `session_id` consistent across turns?

---

## Related Documentation

- **NODE_STATE_QUICK_REFERENCE.md** - Which nodes touch which fields
- **SYSTEM_ARCHITECTURE_SUMMARY.md** - High-level pipeline overview
- **conversation_state.py** - TypedDict definition of all 46 fields
- **conversation_flow.py** - Actual pipeline implementation
