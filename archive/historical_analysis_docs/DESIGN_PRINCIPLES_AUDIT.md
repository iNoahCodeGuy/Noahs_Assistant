# Design Principles Compliance Audit

**Date**: October 19, 2025
**Purpose**: Check Portfolia codebase against 8 design principles
**Scope**: All source files in `src/` directory

---

## Executive Summary

**Overall Grade**: 🟢 **B+ (85/100)**

Portfolia follows most design principles well, with some areas for improvement:

| Principle | Score | Status | Priority Fixes |
|-----------|-------|--------|----------------|
| 1. SRP (Single Responsibility) | 9/10 | 🟢 Excellent | None |
| 2. Encapsulation | 7/10 | 🟡 Good | Add property decorators for state access |
| 3. Loose Coupling | 8/10 | 🟢 Very Good | Extract hardcoded model names |
| 4. Reusability | 9/10 | 🟢 Excellent | None |
| 5. Portability | 10/10 | 🟢 Perfect | None |
| 6. Defensibility | 8/10 | 🟢 Very Good | Add more input validation |
| 7. Maintainability | 9/10 | 🟢 Excellent | None |
| 8. Simplicity (KISS/DRY/YAGNI) | 8/10 | 🟢 Very Good | Remove unused state fields |

---

## 1. Single Responsibility Principle (SRP)

**Score**: 9/10 🟢 **Excellent**

### ✅ Strengths

**Excellent Module Separation**:
- `query_classification.py` - ONLY classifies queries (no retrieval/generation)
- `content_blocks.py` - ONLY returns enterprise messaging strings
- `data_reporting.py` - ONLY formats analytics displays
- `action_execution.py` - ONLY handles side effects
- `greetings.py` - ONLY generates greeting messages

**Example** (query_classification.py):
```python
# ✅ GOOD: Pure classification logic, no I/O
def _is_ambiguous_query(query: str) -> tuple[bool, dict | None]:
    """Check if query is ambiguous - pure function, no side effects"""
    lowered = query.lower().strip()
    if lowered in AMBIGUOUS_QUERIES:
        return True, AMBIGUOUS_QUERIES[lowered]
    return False, None
```

### ⚠️ Minor Issues

**File**: `src/flows/conversation_nodes.py` (Line 133)
```python
def generate_answer(state, rag_engine):
    """This function does THREE things:
    1. Checks if ambiguous → generates clarifying question
    2. Calls RAG engine for normal answers
    3. Sanitizes SQL artifacts from output
    """
```

**Fix**: Extract clarifying question generation to separate node
```python
# ✅ Better: Split responsibilities
def check_ambiguous_query(state):
    """ONLY check for ambiguous queries and generate clarifying questions"""
    if state.get("is_ambiguous"):
        return {"answer": _generate_clarifying_question(state)}
    return state

def generate_answer(state, rag_engine):
    """ONLY generate answers from RAG engine"""
    answer = rag_engine.generate_response(...)
    return {"answer": answer}

def sanitize_answer(state):
    """ONLY sanitize output"""
    clean_answer = _remove_sql_artifacts(state["answer"])
    return {"answer": clean_answer}
```

**Priority**: 🟡 Low (current design works, but refactor would improve clarity)

---

## 2. Encapsulation

**Score**: 7/10 🟡 **Good**

### ✅ Strengths

**Good Service Encapsulation**:
```python
# src/services/resend_service.py
class ResendService:
    def __init__(self):
        self._api_key = os.getenv("RESEND_API_KEY")  # ✅ Private attribute
        self._client = resend.Client(self._api_key)

    def send_email(self, to, subject, html):  # ✅ Public interface
        """Clean public API, hides implementation"""
```

### ⚠️ Issues

**Problem 1**: State dict directly accessed everywhere (no encapsulation)

**File**: Multiple files
```python
# ❌ BAD: Direct state manipulation
state["answer"] = "Some value"
state["query_type"] = "technical"

if state.get("is_ambiguous"):
    # ...
```

**Why it's bad**:
- No validation when setting values
- Typos in keys not caught until runtime
- No single source of truth for valid state keys

**Fix**: Add state accessor methods
```python
# ✅ Better: Encapsulated state access
class ConversationStateManager:
    def __init__(self, state_dict: ConversationState):
        self._state = state_dict

    @property
    def answer(self) -> Optional[str]:
        return self._state.get("answer")

    @answer.setter
    def answer(self, value: str):
        if not value or not isinstance(value, str):
            raise ValueError("Answer must be non-empty string")
        self._state["answer"] = value

    @property
    def query_type(self) -> Optional[str]:
        return self._state.get("query_type")

    @query_type.setter
    def query_type(self, value: str):
        valid_types = ["technical", "career", "data", "mma", "fun", "general", "ambiguous"]
        if value not in valid_types:
            raise ValueError(f"Invalid query_type: {value}")
        self._state["query_type"] = value
```

**Problem 2**: RAG engine internals not fully encapsulated

**File**: `src/core/rag_engine.py` (Lines 45-60)
```python
# ❌ BAD: Direct access to retriever internals
results = self.retriever.search(query, top_k=4)
chunks = [r["chunk"] for r in results["matches"]]  # Exposes internal structure
```

**Fix**: Add abstraction layer
```python
# ✅ Better: Hide retriever implementation details
def retrieve(self, query: str, top_k: int = 4) -> List[str]:
    """Returns list of text chunks, hiding internal structure"""
    return self.retriever.get_relevant_chunks(query, limit=top_k)
```

**Priority**: 🟡 Medium (would improve type safety and catch bugs earlier)

---

## 3. Loose Coupling

**Score**: 8/10 🟢 **Very Good**

### ✅ Strengths

**Excellent Dependency Injection**:
```python
# src/flows/conversation_flow.py
def run_conversation_flow(state, rag_engine, session_id):
    """✅ Dependencies injected, not created inside function"""
    result = handle_greeting(state, rag_engine)  # Passed in
    result = classify_query(result)  # No dependencies
    result = retrieve_chunks(result, rag_engine)  # Injected
    return result
```

**Factory Pattern for Services**:
```python
# src/services/resend_service.py
def get_resend_service() -> Optional[ResendService]:
    """✅ Factory returns None if dependencies missing (graceful degradation)"""
    if not os.getenv("RESEND_API_KEY"):
        return None
    return ResendService()
```

### ⚠️ Issues

**Problem**: Hardcoded OpenAI model names

**File**: `src/config/supabase_config.py` (Line 78)
```python
# ❌ BAD: Hardcoded model name
self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
```

**File**: `src/retrieval/pgvector_retriever.py` (Line 53)
```python
# ❌ BAD: Hardcoded embedding model
embedding_model = "text-embedding-3-small"
```

**Why it's bad**:
- Can't easily swap models for testing
- Tight coupling to OpenAI (hard to switch providers)

**Fix**: Use strategy pattern
```python
# ✅ Better: Model provider interface
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, model="gpt-4o-mini"):
        self.model = model

    def generate(self, prompt: str) -> str:
        return openai.ChatCompletion.create(model=self.model, ...)

# Easy to swap providers
llm = OpenAIProvider(model="gpt-4o-mini")  # Or AnthropicProvider, GeminiProvider, etc.
```

**Priority**: 🟢 Low (current design works, but strategy pattern would improve testability)

---

## 4. Reusability

**Score**: 9/10 🟢 **Excellent**

### ✅ Strengths

**Excellent Content Block Pattern**:
```python
# src/flows/content_blocks.py - Reusable messaging building blocks
def enterprise_value_block() -> str:
    """✅ Reusable across multiple conversation flows"""
    return "- Enterprises care about production-ready systems..."

def resume_offer_block() -> str:
    """✅ Called from multiple nodes when appropriate"""
    return "If it would help, I can share Noah's full resume..."

# Used in multiple places
answer += content_blocks.enterprise_value_block()
answer += content_blocks.resume_offer_block()
```

**Reusable Helper Functions**:
```python
# src/flows/query_classification.py
def _is_ambiguous_query(query: str) -> tuple[bool, dict | None]:
    """✅ Pure function, reusable in tests and production"""
    # No side effects, easy to test, composable

def _expand_vague_query(query: str) -> str:
    """✅ Pure function, reusable"""
    # ...
```

### ⚠️ Minor Issue

**File**: `src/flows/conversation_nodes.py` (Lines 200-250)
```python
# ⚠️ Long inline code for clarifying question generation
# Could be extracted to reusable function
if state.get("is_ambiguous"):
    options = state.get("ambiguity_options", [])
    context = state.get("ambiguity_context", "")

    options_text = ", ".join([f"**{opt}**" for opt in options[:-1]])
    options_text += f", or **{options[-1]}**"

    clarifying_answer = f"""Oh I love this question! ..."""
    # 40+ lines of template code
```

**Fix**: Extract to helper function (already done in content_blocks.py)
```python
# ✅ Better: Move to content_blocks.py
def generate_clarifying_question(query: str, options: List[str], context: str) -> str:
    """Reusable clarifying question generator"""
    # ...
```

**Priority**: 🟢 Low (minor improvement)

---

## 5. Portability

**Score**: 10/10 🟢 **Perfect**

### ✅ Strengths

**Excellent Cross-Platform Support**:
```python
# ✅ Uses pathlib everywhere (Windows/Mac/Linux compatible)
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
career_kb_path = DATA_DIR / "career_kb.csv"
```

**Environment Detection**:
```python
# src/config/supabase_config.py
@property
def is_vercel(self) -> bool:
    """✅ Detects deployment environment"""
    return os.getenv("VERCEL") == "1"

@property
def is_production(self) -> bool:
    """✅ Environment-agnostic checks"""
    return self.is_vercel or os.getenv("ENV") == "production"
```

**No Platform-Specific Code**: ✅ All file I/O uses pathlib, all config from env vars

---

## 6. Defensibility (Fail-Fast)

**Score**: 8/10 🟢 **Very Good**

### ✅ Strengths

**Good Input Validation**:
```python
# src/flows/query_classification.py (Line 193)
def classify_query(state: ConversationState) -> Dict[str, Any]:
    """✅ Fail-fast validation"""
    try:
        query = state["query"]
        role = state.get("role", "Developer")
    except KeyError as e:
        logger.error(f"classify_query: Missing required field: {e}")
        return {
            "error": "classification_failed",
            "error_message": f"Missing required field: {e}"
        }
```

**Graceful Degradation**:
```python
# src/services/resend_service.py
def get_resend_service() -> Optional[ResendService]:
    """✅ Returns None if API key missing (doesn't crash)"""
    if not os.getenv("RESEND_API_KEY"):
        logger.warning("Resend API key not set, email disabled")
        return None
    return ResendService()
```

### ⚠️ Issues

**Problem 1**: State dict allows any key (no runtime validation)

**File**: Multiple files
```python
# ❌ BAD: Typo not caught until runtime
state["querry_type"] = "technical"  # Typo! Should be "query_type"

# Later...
if state.get("query_type") == "technical":  # Always False, bug hidden
```

**Fix**: Add runtime validation
```python
# ✅ Better: Validate keys when setting
VALID_STATE_KEYS = {
    "query", "role", "answer", "query_type", "is_ambiguous", ...
}

def set_state_value(state: dict, key: str, value: Any):
    if key not in VALID_STATE_KEYS:
        raise ValueError(f"Invalid state key: {key}")
    state[key] = value
```

**Problem 2**: No validation of LLM responses

**File**: `src/core/rag_engine.py` (Line 120)
```python
# ❌ BAD: No check if LLM returned empty/invalid response
answer = llm.generate(prompt)
return answer  # Could be empty, None, or error message
```

**Fix**: Validate LLM output
```python
# ✅ Better: Fail-fast if LLM returns invalid response
answer = llm.generate(prompt)
if not answer or len(answer.strip()) < 10:
    raise ValueError("LLM returned empty or too-short response")
return answer
```

**Priority**: 🟡 Medium (would catch bugs earlier in development)

---

## 7. Maintainability

**Score**: 9/10 🟢 **Excellent**

### ✅ Strengths

**Excellent Documentation**:
```python
# ✅ Every module has clear docstring explaining purpose
"""Query classification utilities.

This module detects what kind of question the user is asking so we can
route it appropriately and plan the right follow-up actions.

Types we detect:
- Technical queries (architecture, code, stack questions)
- Career queries (resume, experience, achievements)
...

Design Principles Applied:
- SRP: This module only classifies queries, doesn't retrieve or generate
- Loose Coupling: Communicates via state dict, no direct node calls
- Defensibility: Fail-fast validation on required fields
- Maintainability: Pure helper functions separated from I/O
"""
```

**Clear Naming Conventions**:
```python
# ✅ Function names clearly describe what they do
def _is_ambiguous_query(query: str) -> tuple[bool, dict | None]:
def _expand_vague_query(query: str) -> str:
def should_show_greeting(query: str, chat_history: list) -> bool:
```

**Small, Focused Functions**:
```python
# ✅ Most functions < 50 lines, single responsibility
def _is_data_display_request(lowered_query: str) -> bool:
    """12 lines, one job, easy to understand"""
    return any(keyword in lowered_query for keyword in DATA_DISPLAY_KEYWORDS)
```

### ⚠️ Minor Issue

**File**: `src/flows/conversation_nodes.py` (Line 133-250)
```python
# ⚠️ generate_answer() function is 117 lines (too long)
def generate_answer(state, rag_engine):
    """This function does too many things:
    - Check ambiguous queries
    - Generate clarifying questions
    - Call RAG engine
    - Handle MMA queries
    - Sanitize SQL artifacts
    - Format responses
    """
    # 117 lines of code
```

**Fix**: Extract sub-functions
```python
# ✅ Better: Break into smaller functions
def generate_answer(state, rag_engine):
    """Main orchestrator (20 lines)"""
    if state.get("is_ambiguous"):
        return _generate_clarifying_question(state)

    if state.get("query_type") == "mma":
        return _generate_mma_response(state)

    answer = _call_rag_engine(state, rag_engine)
    answer = _sanitize_answer(answer)
    return {"answer": answer}
```

**Priority**: 🟡 Low (current code works, but refactor would improve readability)

---

## 8. Simplicity (KISS, DRY, YAGNI)

**Score**: 8/10 🟢 **Very Good**

### ✅ Strengths

**DRY Principle** - No code duplication:
```python
# ✅ Good: Content blocks reused across conversation flows
answer += content_blocks.enterprise_value_block()
answer += content_blocks.resume_offer_block()
```

**KISS Principle** - Simple solutions:
```python
# ✅ Simple greeting detection (no complex NLP)
def should_show_greeting(query: str, chat_history: list) -> bool:
    if not is_first_turn(chat_history):
        return False

    greeting_patterns = ["hello", "hi", "hey", ...]
    return any(pattern in query.lower() for pattern in greeting_patterns)
```

### ⚠️ Issues

**Problem 1**: State dict has too many fields (30+ fields)

**File**: `src/state/conversation_state.py`
```python
# ❌ YAGNI violation: Some fields never used or rarely used
class ConversationState(TypedDict, total=False):
    query: str
    role: str
    answer: str
    # ... 30+ more fields
    # Some like "vague_query_expanded", "teaching_moment" rarely used
```

**Fix**: Remove unused fields
```python
# ✅ Better: Only fields actually used in production
class ConversationState(TypedDict, total=False):
    # Core fields (always used)
    query: str
    role: str
    answer: str
    chat_history: List[Dict[str, str]]

    # Query classification
    query_type: str
    is_ambiguous: bool
    ambiguity_options: List[str]

    # RAG results
    chunks: List[str]
    matches: int

    # Actions
    actions: List[Dict[str, Any]]
```

**Problem 2**: Redundant state fields

**File**: `src/state/conversation_state.py`
```python
# ❌ DRY violation: Redundant fields
ambiguous_query: bool  # Same as is_ambiguous
data_display_requested: bool  # Redundant with query_type == "data"
code_display_requested: bool  # Redundant with query_type == "technical"
```

**Fix**: Use single source of truth
```python
# ✅ Better: Derive from query_type instead of storing separately
def is_data_display(state):
    return state.get("query_type") == "data"

def is_code_display(state):
    return state.get("query_type") == "technical" and state.get("code_would_help")
```

**Priority**: 🟡 Medium (cleanup would simplify state management)

---

## Summary of Violations

### 🔴 High Priority (Fix Soon)
None - all issues are minor

### 🟡 Medium Priority (Fix When Convenient)
1. **Encapsulation**: Add state accessor methods to validate keys/values
2. **Defensibility**: Add runtime validation for state keys to catch typos
3. **Simplicity**: Remove unused state fields (YAGNI violation)

### 🟢 Low Priority (Nice to Have)
1. **SRP**: Split `generate_answer()` into smaller functions
2. **Loose Coupling**: Extract OpenAI model names to strategy pattern
3. **Reusability**: Move clarifying question template to content_blocks.py
4. **Maintainability**: Break large functions (>100 lines) into smaller ones

---

## Recommendations

1. **Keep current architecture** - It's well-designed overall (85/100 score)
2. **Focus on quick wins**:
   - Remove unused state fields (30 min)
   - Add state key validation (1 hour)
   - Extract long functions (2 hours)
3. **Defer larger refactors**:
   - State accessor classes (4 hours)
   - Strategy pattern for LLM providers (6 hours)
   - Full SRP compliance (8 hours)

These can wait until we have more bandwidth or hit actual pain points.

---

## Next Steps

1. ✅ **Complete this audit** (DONE)
2. ⏳ **Continue with Todo #4**: Cleanup audit (unused imports, dead code)
3. ⏳ **Apply quick wins** if time permits
4. ⏳ **Deploy fixes** to production
