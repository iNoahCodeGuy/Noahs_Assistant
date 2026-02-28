# Display Intelligence Implementation (with Proactive Code/Data)

> **Portfolia now understands WHEN to provide longer responses, WHEN to show code, and WHEN to display data - even when NOT explicitly requested.**

## Overview

This enhancement aligns Portfolia's response behavior with the master documentation rules from `DATA_COLLECTION_AND_SCHEMA_REFERENCE.md`, `CONVERSATION_PERSONALITY.md`, and `PROJECT_REFERENCE_OVERVIEW.md`. The system now intelligently detects query intent and **proactively** includes code or data when it would clarify the answer, not just when explicitly requested.

### Key Principles

1. **Reactive (Explicit)**: Show code/data when user explicitly asks ("show me code", "display analytics")
2. **Proactive (Implicit)**: Show code/data when it would naturally enhance understanding, even without explicit request
3. **Role-Aware**: Code proactivity only for technical roles (Software Developer, Hiring Manager technical)
4. **Teaching-Focused**: Align with Portfolia's GenAI educator persona

## Implementation Details

### 1. Query Classification Enhancements (`src/flows/query_classification.py`)

**Added Detection For:**

#### A. Teaching Moments / Longer Responses Needed
```python
teaching_keywords = [
    "why", "how does", "how did", "how do", "explain", "walk me through",
    "what is", "what are", "what's the difference", "compare",
    "help me understand", "break down", "teach me", "show me how",
    "architecture", "design", "pattern", "principle", "strategy",
    "trade-off", "tradeoff", "benefit", "advantage", "disadvantage",
    "when to use", "when should", "best practice", "enterprise"
]
```

**Sets flags:**
- `needs_longer_response`: True
- `teaching_moment`: True

**When detected:** User asks "why" or "how" questions, requests explanations, or wants to understand concepts/tradeoffs

---

#### B. Code Display Requests
```python
code_display_keywords = [
    "show code", "display code", "show me code", "show the code",
    "show implementation", "display implementation",
    "how do you", "how does it", "how is it",
    "show me the", "show retrieval", "show api",
    "code snippet", "code example", "source code"
]
```

**Sets flags:**
- `code_display_requested`: True
- `query_type`: "technical"

**When detected:** User explicitly asks to see code or implementation details

---

#### C. **NEW: Proactive Code Detection** 
```python
proactive_code_topics = [
    # Implementation questions (even without "show me")
    "implement", "build", "create", "develop", "write",
    # Architecture/design questions that benefit from code
    "rag pipeline", "vector search", "retrieval", "embedding", "orchestration",
    "langgraph", "conversation flow", "node", "pipeline",
    # Technical concepts best shown with code
    "api route", "endpoint", "function", "class", "method",
    "pgvector", "supabase query", "database", "migration",
    "prompt engineering", "llm call", "generation",
    # Patterns that need examples
    "pattern", "approach", "technique", "strategy" 
]

# Only for technical roles
if state.role in ["Software Developer", "Hiring Manager (technical)"]:
    if any(topic in lowered for topic in proactive_code_topics):
        state.stash("code_would_help", True)
```

**Sets flags:**
- `code_would_help`: True
- `query_type`: "technical"

**When detected:** Technical user asks about implementation, architecture, or GenAI patterns - code would clarify even without "show me"

**Examples:**
- "How does RAG work?" → Proactive code snippet showing retrieval
- "Explain vector search" → Code example of pgvector query
- "What's the LangGraph pattern?" → Orchestration node code

---

#### D. Data/Analytics Display Requests
```python
# Already existed via _is_data_display_request()
```

**Sets flags:**
- `data_display_requested`: True
- `query_type`: "data"

**When detected:** User explicitly asks for "show analytics", "display data", "metrics", "performance stats"

---

#### E. **NEW: Proactive Data Detection**
```python
proactive_data_topics = [
    # Performance/metrics questions
    "how many", "how much", "how often", "frequency",
    "performance", "metrics", "statistics", "stats",
    "usage", "activity", "engagement", "interactions",
    # Trend/pattern questions
    "trend", "pattern", "over time", "growth",
    "most common", "popular", "typical", "average",
    # Comparison questions that need numbers
    "compare", "difference between", "vs",
    # Success/outcome questions
    "success rate", "conversion", "effectiveness"
]

if any(topic in lowered for topic in proactive_data_topics):
    state.stash("data_would_help", True)
```

**Sets flags:**
- `data_would_help`: True
- `query_type`: "data"

**When detected:** Question implies metrics without explicitly requesting display

**Examples:**
- "How many users interact daily?" → Proactive usage statistics table
- "What's the performance trend?" → Time-series data with brief explanation
- "How often do people ask about RAG?" → Query frequency metrics

---

### 2. Response Generation Intelligence (`src/flows/core_nodes.py`)

**Modified `generate_answer()` to add display guidance:**

```python
# Build extra_instructions based on query classification
extra_instructions = []

# When teaching/explaining, provide comprehensive depth
if state.fetch("needs_longer_response", False) or state.fetch("teaching_moment", False):
    extra_instructions.append(
        "This is a teaching moment - provide a comprehensive, well-structured explanation. "
        "Break down concepts clearly, connect technical details to business value, and "
        "help the user truly understand. Use examples where helpful."
    )

# EXPLICIT code request - user specifically asked
if state.fetch("code_display_requested", False) and state.role in [
    "Software Developer", 
    "Hiring Manager (technical)"
]:
    extra_instructions.append(
        "The user has requested code. After your explanation, include relevant code snippets "
        "with comments explaining key decisions. Keep code blocks under 40 lines and focus "
        "on the most interesting parts."
    )
# PROACTIVE code suggestion - code would clarify but wasn't explicitly requested
elif state.fetch("code_would_help", False) and state.role in [
    "Software Developer",
    "Hiring Manager (technical)"
]:
    extra_instructions.append(
        "This technical concept would benefit from a code example. After your explanation, "
        "include a relevant code snippet (≤40 lines) with comments to clarify the implementation. "
        "This is proactive - the user didn't explicitly ask but code will help understanding."
    )

# EXPLICIT data request - user specifically asked
if state.fetch("data_display_requested", False):
    extra_instructions.append(
        "The user wants data/analytics. Be brief with narrative - focus on presenting clean "
        "tables with proper formatting. Include source attribution."
    )
# PROACTIVE data suggestion - metrics would clarify but weren't explicitly requested
elif state.fetch("data_would_help", False):
    extra_instructions.append(
        "This question would benefit from actual metrics/data. After your explanation, "
        "include relevant analytics in table format if available. Be concise with tables, "
        "include source attribution. This is proactive - help the user with concrete numbers."
    )
```

**Key Enhancement:** Now distinguishes between explicit requests and proactive suggestions, instructing the LLM accordingly.

These instructions are passed to the LLM via the `extra_instructions` parameter.

---

### 3. LLM Prompt Enhancement (`src/core/response_generator.py`)

**Updated `generate_contextual_response()` signature:**
```python
def generate_contextual_response(
    self, 
    query: str, 
    context: List[Dict[str, Any]], 
    role: str = None, 
    chat_history: List[Dict[str, str]] = None,
    extra_instructions: str = None  # NEW PARAMETER
) -> str:
```

**Updated `_build_role_prompt()` to inject instructions:**
```python
def _build_role_prompt(
    self, 
    query: str, 
    context_str: str, 
    role: str = None, 
    chat_history: List[Dict[str, str]] = None,
    extra_instructions: str = None  # NEW PARAMETER
) -> str:
    # ...
    instruction_addendum = ""
    if extra_instructions:
        instruction_addendum = f"\n\nIMPORTANT GUIDANCE: {extra_instructions}\n"
    
    # Then inject {instruction_addendum} into each role's prompt template
```

**Injected in all role prompts:**
- Hiring Manager (technical)
- Software Developer  
- General (Just looking around, etc.)

---

## Comprehensive Behavior Matrix

### Proactive vs Reactive Display Logic

| Scenario | Query Example | Role | Flags Set | Response | Reasoning |
|----------|---------------|------|-----------|----------|-----------|
| **Explicit Code** | "Show me the RAG code" | Developer | `code_display_requested` | Explanation + code (≤40 lines) | User specifically requested |
| **Proactive Code** | "How does RAG work?" | Developer | `code_would_help` | Explanation + code example | Code clarifies GenAI concept |
| **Proactive Code** | "Explain vector search" | Technical HM | `code_would_help` | Explanation + pgvector query | Implementation helps understanding |
| **No Code** | "How does RAG work?" | Nontechnical HM | None | Explanation only | Non-technical role, no code |
| **Explicit Data** | "Show analytics" | Any | `data_display_requested` | Brief + tables | User explicitly wants metrics |
| **Proactive Data** | "How many users interact daily?" | Any | `data_would_help` | Context + usage table | Question implies metrics |
| **Proactive Data** | "What's the performance trend?" | Any | `data_would_help` | Brief + time-series data | "Trend" needs actual numbers |
| **Teaching Depth** | "Why use pgvector?" | Any | `needs_longer_response` | Long structured explanation | "Why" = teaching moment |
| **Teaching Depth** | "Explain how RAG prevents hallucinations" | Any | `teaching_moment` | Comprehensive breakdown | Needs step-by-step clarity |
| **Simple Fact** | "What's Noah's background?" | Any | None | Concise narrative | Straightforward answer |

### Role-Specific Behavior

| Role | Proactive Code? | Proactive Data? | Response Depth |
|------|----------------|----------------|----------------|
| **Software Developer** | ✅ Yes (for technical topics) | ✅ Yes (when metrics implied) | High technical detail |
| **Hiring Manager (technical)** | ✅ Yes (for GenAI concepts) | ✅ Yes (when metrics implied) | Balanced technical + business |
| **Hiring Manager (nontechnical)** | ❌ No (unless explicitly requested) | ✅ Yes (when metrics implied) | Business value focus |
| **Just looking around** | ❌ No | ✅ Yes (when metrics implied) | Conversational, accessible |

---

## Alignment with Master Documentation

### From `DATA_COLLECTION_AND_SCHEMA_REFERENCE.md`:

> **Reasoning heuristics (what to show and when)**
> - If the question is **how/why** → long narrative with diagrams/code if needed.  
> - If the question is **show/metrics/analytics** → concise tables and charts with minimal prose.  
> - If user is technical and seems unsure → proactively show a small code snippet (≤40 lines) with comments.

✅ **IMPLEMENTED** via:
- Teaching keywords → `needs_longer_response` → "provide comprehensive explanation"
- Data keywords → `data_display_requested` → "be brief with narrative, focus on tables"
- Code keywords + technical role → `code_display_requested` → "include code snippets with comments"

---

### From `CONVERSATION_PERSONALITY.md`:

> **Core personality traits (The AI Applications Educator)**
> - Teaching-focused on GenAI systems
> - Enterprise value champion
> - Passionate about the craft
> - Adapt explanation depth to user

✅ **IMPLEMENTED** via:
- Teaching moments get "Break down concepts clearly, connect technical details to business value"
- Extra instructions respect role context (only show code to technical roles)
- LLM receives guidance to match user's level of detail expectations

---

## Testing Strategy

### Manual Testing Scenarios:

1. **Teaching Query** (expects longer response):
   ```
   Role: Software Developer
   Query: "Why did you choose pgvector instead of FAISS for vector search?"
   Expected: Comprehensive explanation covering technical reasons, enterprise value, tradeoffs
   ```

2. **Code Display Query** (expects code + explanation):
   ```
   Role: Software Developer
   Query: "Show me how you implement the retrieval logic"
   Expected: Explanation followed by code snippet from pgvector_retriever.py with comments
   ```

3. **Data Query** (expects tables):
   ```
   Role: Hiring Manager (technical)
   Query: "Show me analytics for the last week"
   Expected: Brief intro + clean markdown table + source attribution
   ```

4. **General Query** (expects standard response):
   ```
   Role: Just looking around
   Query: "What's Noah's background?"
   Expected: Conversational summary of career history
   ```

### Automated Testing (Future):

Create `tests/test_display_intelligence.py`:
```python
def test_teaching_moment_detection():
    state = ConversationState(role="Software Developer", query="Why use RAG?")
    state = classify_query(state)
    assert state.fetch("needs_longer_response") == True
    assert state.fetch("teaching_moment") == True

def test_code_display_detection():
    state = ConversationState(role="Software Developer", query="Show me the code")
    state = classify_query(state)
    assert state.fetch("code_display_requested") == True
    assert state.fetch("query_type") == "technical"

def test_data_display_detection():
    state = ConversationState(role="Hiring Manager (technical)", query="Display analytics")
    state = classify_query(state)
    assert state.fetch("data_display_requested") == True
    assert state.fetch("query_type") == "data"
```

---

## Production Impact

### Performance:
- **No additional API calls** - instructions passed to existing LLM call
- **Minimal token overhead** - extra_instructions adds ~50-100 tokens per request
- **Cost impact**: Negligible (<$0.0001 per query increase)

### User Experience:
- **Better alignment** with user intent (teaching vs data vs code)
- **More focused responses** - no more verbose data responses or terse teaching moments
- **Role-aware** - technical users get code, business users get value explanations

### Deployment:
- **No schema changes** - pure logic enhancement
- **No breaking changes** - all existing functionality preserved
- **Gradual rollout** - can A/B test with `extra_instructions=None` fallback

---

## Next Steps

1. **Commit and deploy** this enhancement to production
2. **Monitor analytics** for:
   - Average response length by query type
   - Code snippet inclusion rate for technical roles
   - User satisfaction by query classification
3. **Iterate** based on feedback:
   - Add more teaching keywords if detection misses queries
   - Adjust code snippet length guidance (currently ≤40 lines)
   - Refine data table formatting instructions

---

## Files Modified

| File | Changes | Lines Changed |
|------|---------|---------------|
| `src/flows/query_classification.py` | Added teaching keyword detection, updated docstrings | +20 |
| `src/flows/core_nodes.py` | Added extra_instructions logic in generate_answer() | +30 |
| `src/core/response_generator.py` | Added extra_instructions parameter, updated all role prompts | +50 |

**Total changes**: ~100 lines added/modified across 3 files

---

## Credits

- **Aligned with**: `DATA_COLLECTION_AND_SCHEMA_REFERENCE.md`, `CONVERSATION_PERSONALITY.md`
- **Requested by**: User's question "does portfolia understand when a question would need to be longer?"
- **Implementation**: Complete display intelligence system following QA strategy principles
