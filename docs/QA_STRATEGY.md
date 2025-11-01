# Quality Assurance Strategy

**Purpose**: Ensure conversation quality, code quality, and **documentation-code alignment** remain intact as the codebase evolves.

**Last Updated**: October 16, 2025

---

## 📑 Table of Contents

### Quick Navigation
- [🎯 For New Developers](#for-new-developers-start-here) - Start here if onboarding
- [🧪 Running Tests](#quick-test-commands) - How to run tests locally
- [📊 Current Status](#current-test-status) - See what's passing/failing
- [🚀 For Feature Development](#feature-development-checklist) - Adding new features

### Core Sections

#### 1. [Quality Standards & Testing](#current-quality-standards)
   - 1.1 [Conversation Quality Tests (19 tests)](#1-conversation-quality-19-tests---current-status-19-passing)
   - 1.2 [Content Storage vs User Presentation](#content-storage-vs-user-presentation-standards)
   - 1.3 [Test Coverage Map](#test-coverage-map)

#### 2. [Automated Testing](#automated-testing)
   - 2.1 [Running Tests Locally](#running-tests-locally)
   - 2.2 [Test Organization](#test-organization)

#### 3. [Documentation Alignment Testing](#documentation-alignment-testing)
   - 3.1 [The Problem We're Solving](#the-problem-were-solving)
   - 3.2 [Test Suite Overview](#test-suite-documentation-alignment)
   - 3.3 [Running Alignment Tests](#running-documentation-alignment-tests)

#### 4. [Feature Development Workflow](#feature-development-documentation-workflow)
   - 4.1 [Decision Tree: Where to Document](#decision-tree-where-to-document-changes)
   - 4.2 [Scenario 1: Adding New Features](#-scenario-1-adding-a-new-feature)
   - 4.3 [Scenario 2: Changing Implementation](#-scenario-2-changing-existing-feature-implementation)
   - 4.4 [Scenario 3: Architecture Changes](#-scenario-3-changing-system-behaviorarchitecture)
   - 4.5 [Scenario 4: Adding New Roles](#-scenario-4-adding-new-role-or-query-type)
   - 4.6 [Documentation Anti-Patterns](#documentation-anti-patterns-dont-do-this)

#### 5. [Pre-Commit Hooks](#pre-commit-hooks)

#### 6. [CI/CD Pipeline](#cicd-pipeline)

#### 7. [Documentation Quality](#documentation-quality-standards)
   - 7.1 [Single Source of Truth Principle](#1-single-source-of-truth-ssot-principle)
   - 7.2 [Code-First Updates](#2-code-first-documentation-updates)
   - 7.3 [Documentation Hygiene](#3-documentation-hygiene-checklist)
   - 7.4 [Master Doc Updates](#4-master-documentation-update-process)
   - 7.5 [93% Alignment Philosophy](#5-the-93-alignment-philosophy-when-is-it-acceptable)

#### 8. [Quarterly Audit](#quarterly-documentation-audit)

#### 9. [Testing Best Practices](#testing-best-practices--common-issues)
   - 9.1 [Core Testing Principles](#core-testing-principles)
   - 9.2 [Common Test Failures](#common-test-failures--how-to-fix)
   - 9.3 [Adding New Quality Standards](#adding-new-quality-standards)

#### 10. [Manual Testing](#manual-testing-procedures)
   - 10.1 [Testing Pyramid](#testing-pyramid-manual-vs-automated)
   - 10.2 [Role Functionality Checklists](#role-functionality-checklists)
   - 10.3 [Cross-Role Consistency Tests](#cross-role-consistency-tests)
   - 10.4 [Pre-Release Protocol](#pre-release-testing-protocol)

#### 11. [Anti-Drift Protection](#preventing-documentation-file-misalignment)
   - 11.1 [The Problem: New .md Files](#the-problem-new-md-files-create-drift)
   - 11.2 [Solution: 3-Layer Protection](#solution-automated-documentation-registration)
   - 11.3 [Pre-Commit Hook Implementation](#step-1-add-pre-commit-hook-for-new-md-files)

#### 12. [Design Principles](#design-principles-for-langgraph-migration) 🆕
   - 12.1 [Cohesion & SRP](#1-cohesion--single-responsibility-principle-srp)
   - 12.2 [Encapsulation & Abstraction](#2-encapsulation--abstraction)
   - 12.3 [Loose Coupling & Modularity](#3-loose-coupling--modularity)
   - 12.4 [Reusability & Extensibility](#4-reusability--extensibility)
   - 12.5 [Portability](#5-portability)
   - 12.6 [Defensibility](#6-defensibility-fail-fast-fail-safe-fail-loud)
   - 12.7 [Maintainability & Testability](#7-maintainability--testability)
   - 12.8 [Simplicity (KISS, DRY, YAGNI)](#8-simplicity-kiss-dry-yagni)

#### 13. [Phase 2: Production Monitoring](#phase-2-production-monitoring-with-langsmith) 🆕
   - 13.1 [The Hybrid QA Approach](#the-hybrid-qa-approach)
   - 13.2 [What LangSmith Monitors](#what-well-monitor)
   - 13.3 [Implementation Plan](#implementation-plan)
   - 13.4 [Alert Thresholds](#alert-thresholds)
   - 13.5 [Daily Reports](#daily-report-example)
   - 13.6 [Cost Analysis](#cost-analysis)

---

## For New Developers (Start Here)

### What This Document Covers
This QA Strategy ensures:
- ✅ **Conversation quality** remains professional and helpful (18 automated tests)
- ✅ **Documentation stays in sync** with code (12 alignment tests)
- ✅ **New features don't break existing behavior** (regression protection)
- ✅ **Team alignment** on quality standards

### Quick Onboarding Path

**Step 1: Understand the Testing Philosophy** (5 min)
- Read [Content Storage vs User Presentation](#content-storage-vs-user-presentation-standards)
- Key principle: KB can have rich formatting, user responses must be professional

**Step 2: Run the Tests** (3 min)
```bash
# Run all tests (30 total: 18 conversation + 12 alignment)
pytest tests/ -v

# Should see: 28/30 passing (93% overall)
```

**Step 3: Before Adding Features** (2 min)
- Check [Feature Development Decision Tree](#decision-tree-where-to-document-changes)
- Follow [Documentation Anti-Patterns](#documentation-anti-patterns-dont-do-this)

**Step 4: Read Your Role-Specific Section**
- Backend dev? → [Testing Best Practices](#testing-best-practices--common-issues)
- Documentation? → [Documentation Quality Standards](#documentation-quality-standards)
- QA engineer? → [Manual Testing Procedures](#manual-testing-procedures)

---

## Quick Test Commands

```bash
# Run everything (recommended before committing)
pytest tests/ -v

# Run only conversation quality tests (19 tests)
pytest tests/test_conversation_quality.py -v

# Run only documentation alignment tests (12 tests)
pytest tests/test_documentation_alignment.py -v

# Run specific test
pytest tests/test_conversation_quality.py::test_no_emoji_headers -v

# Run with detailed output (useful for debugging)
pytest tests/ -vv

# Run tests matching pattern
pytest tests/ -k "emoji" -v
```

---

## Current Test Status

### Test Suite Overview

**Last Updated**: October 17, 2025
**Overall Status**: ✅ **99% pass rate** (76/77 active tests passing, 1 intentionally skipped)
**Target**: ✅ ACHIEVED - Maintain 99% pass rate through LangGraph migration

| Test Suite | Tests | Passing | Status | Test File |
|------------|-------|---------|--------|-----------|
| **Conversation Quality** | 19 | 19 | ✅ 100% | `tests/test_conversation_quality.py` (512 lines) |
| **Documentation Alignment** | 15 | 14 | ✅ 93% (1 skipped) | `tests/test_documentation_alignment.py` |
| **Resume Distribution** | 37 | 37 | ✅ 100% | `tests/test_resume_distribution.py` |
| **Error Handling** | 6 | 6 | ✅ 100% | `tests/test_error_handling.py` (~450 lines) |
| **TOTAL** | **77** | **76** | **✅ 99% pass rate** | 4 test suites |

### Suite Descriptions

#### 1. Conversation Quality (19 tests) - 100% ✅
**Purpose**: Ensure conversation quality remains professional and helpful
**Key Policies Tested**:
- KB content can use `###` headers/emojis (teaching structure)
- LLM responses must strip to professional `**Bold**` format only
- No Q&A verbatim responses (synthesis required)
- No pushy resume offers (subtle availability mentions only)
- Single follow-up prompts (no duplicate CTAs)

**Recent Updates**:
- ✅ Oct 16: Added `test_no_pushy_resume_offers` for hybrid resume distribution
- ✅ Oct 16: Updated `test_no_emoji_headers` to check LLM responses (not KB files)
- ✅ Oct 15: Added `test_no_qa_verbatim_responses` and `test_response_synthesis_in_prompts`

#### 2. Documentation Alignment (15 tests) - 93% ✅
**Purpose**: Ensure documentation matches code implementation (no phantom functions, no outdated paths)
**Key Checks**:
- Conversation flow documented correctly
- File references valid (no 404s)
- Role names match implementation
- Temperature/model settings accurate
- Master docs exist and non-empty

**Note**: 1 test intentionally skipped (`test_count_documented_correctly`) - changes too frequently during active development

#### 3. Resume Distribution (37 tests) - 100% ✅
**Purpose**: Validate hybrid resume distribution system (education mode + hiring signals mode)
**Key Features Tested**:
- Hiring signal detection (keywords, company, position, timeline)
- Mode switching (education vs availability mention)
- Resume offering after 2 turns (not immediately)
- Email sending integration
- Storage service integration

**New**: Oct 16, 2025 - Entire suite created for intelligent resume distribution feature

#### 4. Error Handling (6 tests) - 100% ✅
**Purpose**: Validate production-grade error handling and resilience
**Key Patterns Tested**:
- Service degradation (Twilio, Resend fail gracefully)
- LLM failure handling (OpenAI rate limits)
- Input sanitization (XSS, SQL injection rejected)
- API error handling (malformed JSON → 400/500 responses)
- RAG pipeline resilience (low-quality retrieval → fallback suggestions)
- Observability (all errors logged to LangSmith)

**Recent Updates**:
- ✅ Oct 17: Added `test_low_quality_retrieval_fallback` for RAG resilience
- ✅ Oct 17: All 6 tests passing after Phase 1.5 implementation

### Running Tests

```bash
# Run all tests (recommended before committing)
pytest tests/ -v

# Run specific suite
pytest tests/test_conversation_quality.py -v      # 19 tests
pytest tests/test_documentation_alignment.py -v   # 15 tests
pytest tests/test_resume_distribution.py -v       # 37 tests
pytest tests/test_error_handling.py -v            # 6 tests

# Run specific test
pytest tests/test_conversation_quality.py::test_no_emoji_headers -v

# Run with detailed output
pytest tests/ -vv

# Run tests matching pattern
pytest tests/ -k "emoji" -v
```

### Expected Output

```
============================== test session starts ==============================
collected 77 items

tests/test_conversation_quality.py::test_no_emoji_headers PASSED        [  1%]
tests/test_conversation_quality.py::test_no_qa_verbatim_responses PASSED [  2%]
...
tests/test_error_handling.py::test_low_quality_retrieval_fallback PASSED [100%]

============================== 76 passed, 1 skipped in 2.34s ==============================
```

**Success Criteria**: ✅ 76 passing, 1 skipped (99% active test pass rate)

---

## Feature Development Checklist

**Before starting work:**
- [ ] Read relevant test file (`test_conversation_quality.py` or alignment tests)
- [ ] Understand current quality standards
- [ ] Check if feature requires new documentation

**During development:**
- [ ] Write tests first (TDD) or alongside feature
- [ ] Run tests frequently: `pytest tests/ -v`
- [ ] Update documentation if behavior changes

**Before committing:**
- [ ] All tests passing: `pytest tests/ -v`
- [ ] Documentation updated (see [Workflow](#feature-development-documentation-workflow))
- [ ] No anti-patterns (see [Documentation Anti-Patterns](#documentation-anti-patterns-dont-do-this))

**In pull request:**
- [ ] Include test results in PR description
- [ ] Reference related documentation updates
- [ ] Note any intentional test changes

---

## Design Principles for LangGraph Migration

**Purpose:** Foundational principles for building production-grade LangGraph applications
**Source:** [QUICK_REFERENCE.md](https://github.com/iNoahCodeGuy/NoahsAIAssistant-/blob/main/QUICK_REFERENCE.md)
**Context:** These principles guide the migration from functional pipeline to actual LangGraph library

---

### 1. Cohesion & Single Responsibility Principle (SRP)

**Rule:** One node, one job - Each node should have only one reason to change

**Why This Matters for LangGraph:**
- LangGraph visualizes your flow as a DAG (Directed Acyclic Graph)
- Each node appears as a box in the visual debugger
- If a node does multiple things, debugging becomes harder
- High cohesion = related functionality grouped, unrelated functionality separated

**LangGraph Example:**

```python
# ✅ GOOD: Single responsibility per node
def classify_query(state: ConversationState) -> Dict[str, Any]:
    """ONLY classifies query type, nothing else."""
    query_type = _determine_type(state["query"])
    return {"query_type": query_type}

def retrieve_chunks(state: ConversationState) -> Dict[str, Any]:
    """ONLY retrieves relevant chunks, nothing else."""
    chunks = rag_engine.retrieve(state["query"], top_k=4)
    return {"retrieved_chunks": chunks.get("chunks", [])}

# Graph construction
graph_builder.add_node("classify", classify_query)
graph_builder.add_node("retrieve", retrieve_chunks)
graph_builder.add_edge("classify", "retrieve")  # Clear separation

# ❌ BAD: Multiple responsibilities in one node
def classify_and_retrieve(state: ConversationState) -> Dict[str, Any]:
    """Does TWO things - violates SRP, hard to debug."""
    query_type = _determine_type(state["query"])
    chunks = rag_engine.retrieve(state["query"])  # Should be separate node!
    return {"query_type": query_type, "retrieved_chunks": chunks}
```

**Ask yourself:** "Can I describe this node's purpose in one sentence without using 'and'?"

**Test Standard:**
```python
def test_node_single_responsibility():
    """Ensure each node has one clear purpose."""
    # Each node should update only its designated state keys
    state = {"query": "test"}

    result = classify_query(state)
    assert set(result.keys()) == {"query_type"}, "Node should only set query_type"

    result = retrieve_chunks(state)
    assert set(result.keys()) == {"retrieved_chunks"}, "Node should only set chunks"
```

---

### 2. Encapsulation & Abstraction

**Rule:** Hide the details, show the interface - Keep internal state private, expose behavior

**Why This Matters for LangGraph:**
- Nodes are the public interface to your graph
- Internal implementation (helpers, utils) should be hidden
- Abstracts away complexity from the graph visualization
- Makes nodes easier to test in isolation

**LangGraph Example:**

```python
# ✅ GOOD: Encapsulated node with hidden implementation
def retrieve_chunks(state: ConversationState) -> Dict[str, Any]:
    """Public interface - clean and simple."""
    try:
        chunks = _fetch_from_pgvector(state["query"])  # Private helper
        return {"retrieved_chunks": chunks}
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return {
            "retrieved_chunks": [],
            "error": "retrieval_failed",
            "error_message": "I'm having trouble accessing information right now."
        }

def _fetch_from_pgvector(query: str) -> List[Dict]:
    """Private implementation - users don't see this in graph."""
    # Complex pgvector logic, embedding generation, similarity search
    embeddings = _generate_embeddings(query)
    results = supabase.rpc("search_kb_chunks", {
        "query_embedding": embeddings,
        "match_count": 4
    })
    return _parse_results(results)

# ❌ BAD: Implementation details exposed in node
def retrieve_chunks(state: ConversationState) -> Dict[str, Any]:
    """Too much implementation detail - hard to test."""
    embeddings = openai.embeddings.create(
        input=state["query"],
        model="text-embedding-3-small"
    ).data[0].embedding

    results = supabase.rpc("search_kb_chunks", {
        "query_embedding": embeddings,
        "match_count": 4
    }).execute()

    chunks = [{"content": r["content"], "score": r["similarity"]} for r in results.data]
    return {"retrieved_chunks": chunks}
```

**Ask yourself:** "If I change this internal implementation, will client code (the graph) break?"

**Test Standard:**
```python
def test_node_encapsulation():
    """Ensure nodes hide implementation details."""
    with patch('src.retrieval.pgvector_retriever.get_retriever') as mock:
        # Should be able to swap implementation without changing node interface
        mock.return_value.retrieve.return_value = {"chunks": [{"content": "test"}]}

        state = {"query": "test"}
        result = retrieve_chunks(state)

        assert "retrieved_chunks" in result, "Interface maintained despite implementation change"
```

---

### 3. Loose Coupling & Modularity

**Rule:** Nodes communicate ONLY via state, not direct function calls

**Why This Matters for LangGraph:**
- **CRITICAL** - This is how LangGraph works!
- Nodes are vertices in a graph, edges define data flow
- Direct function calls bypass the graph orchestration
- Loose coupling = nodes can be reordered, removed, or replaced without breaking others

**LangGraph Example:**

```python
# ✅ GOOD: Loose coupling via state
def classify_query(state: ConversationState) -> Dict[str, Any]:
    """Outputs to state only."""
    return {"query_type": "technical"}

def retrieve_chunks(state: ConversationState) -> Dict[str, Any]:
    """Reads from state only."""
    query_type = state.get("query_type")  # Input from previous node via state
    if query_type == "greeting":
        return {"retrieved_chunks": []}  # Skip retrieval

    chunks = rag_engine.retrieve(state["query"], top_k=4)
    return {"retrieved_chunks": chunks}

# Graph construction enforces loose coupling
graph_builder.add_node("classify", classify_query)
graph_builder.add_node("retrieve", retrieve_chunks)
graph_builder.add_edge("classify", "retrieve")  # Data flows via state

# ❌ BAD: Tight coupling via direct calls
def classify_query(state: ConversationState) -> Dict[str, Any]:
    result = {"query_type": "technical"}
    # DON'T DO THIS! Bypasses graph orchestration
    chunks = retrieve_chunks(result)  # Direct call breaks LangGraph!
    return {**result, **chunks}
```

**Ask yourself:** "Can I test this node without instantiating half my system?"

**Why This is Critical:**
- LangGraph's `.compile()` method builds the execution plan
- Direct function calls skip this plan, breaking conditional edges, retries, and observability
- Every arrow in the LangSmith trace represents a state transition - direct calls are invisible

**Test Standard:**
```python
def test_nodes_loosely_coupled():
    """Ensure nodes don't call each other directly."""
    import inspect

    source = inspect.getsource(classify_query)
    assert "retrieve_chunks(" not in source, "Nodes must not call other nodes directly"

    # Nodes should accept state and return dict only
    state = {"query": "test"}
    result = classify_query(state)
    assert isinstance(result, dict), "Node must return dict for state update"
```

---

### 4. Reusability & Extensibility

**Rule:** Open for extension, closed for modification - Use composition over inheritance

**Why This Matters for LangGraph:**
- Graphs should be extensible without editing existing nodes
- Conditional edges enable branching without modifying nodes
- New nodes can be added to handle new cases

**LangGraph Example:**

```python
# ✅ GOOD: Extensible via conditional edges
def should_retrieve(state: ConversationState) -> str:
    """Router function - easily extended for new cases."""
    if state.get("is_greeting"):
        return "skip_retrieval"
    if state.get("resume_explicitly_requested"):
        return "handle_resume"
    # Easy to add new conditions without modifying nodes
    return "retrieve"

graph_builder.add_conditional_edges(
    "classify",
    should_retrieve,
    {
        "retrieve": "retrieve_chunks",
        "skip_retrieval": "generate_answer",
        "handle_resume": "handle_resume_request"
    }
)

# Adding new functionality = add new route, don't modify existing nodes
graph_builder.add_node("handle_resume", handle_resume_request)  # NEW NODE

# ❌ BAD: Hard-coded behavior in node (closed for extension)
def classify_query(state: ConversationState) -> Dict[str, Any]:
    if "hello" in state["query"].lower():
        return {"query_type": "greeting"}
    elif "resume" in state["query"].lower():
        return {"query_type": "resume"}
    # Adding new type requires editing this function!
    else:
        return {"query_type": "general"}
```

**Ask yourself:** "Can I add new functionality without editing existing code?"

**Test Standard:**
```python
def test_graph_extensibility():
    """Ensure graph can be extended without modifying existing nodes."""
    # Original graph
    graph1 = graph_builder.compile()

    # Extended graph (add new node without modifying existing ones)
    graph_builder.add_node("new_feature", new_feature_node)
    graph_builder.add_conditional_edges("classify", router, {
        "existing": "retrieve",
        "new": "new_feature"  # New route
    })
    graph2 = graph_builder.compile()

    # Original nodes should work identically
    state1 = graph1.invoke({"query": "test"})
    state2 = graph2.invoke({"query": "test"})
    assert state1["query_type"] == state2["query_type"], "Extension doesn't break existing behavior"
```

---

### 5. Portability

**Rule:** Write once, run anywhere - Use cross-platform libraries and environment variables

**Why This Matters for LangGraph:**
- Graphs should run on local dev (Mac/Windows), CI/CD (Linux), and production (Vercel serverless)
- Configuration should be environment-aware
- No hardcoded paths or platform-specific assumptions

**LangGraph Example:**

```python
# ✅ GOOD: Portable configuration
from pathlib import Path
import os

def load_configuration(state: ConversationState) -> Dict[str, Any]:
    """Portable - works on any platform."""
    # Use pathlib for cross-platform paths
    data_dir = Path(__file__).parent.parent / "data"
    kb_path = data_dir / "career_kb.csv"

    # Use environment variables for config
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("TEMPERATURE", "0.7"))

    return {
        "kb_path": str(kb_path),
        "model": model,
        "temperature": temperature
    }

# ❌ BAD: Platform-specific hardcoded paths
def load_configuration(state: ConversationState) -> Dict[str, Any]:
    """Won't work on Windows or in Vercel."""
    kb_path = "/Users/noah/project/data/career_kb.csv"  # Mac-specific!
    model = "gpt-4o-mini"  # Hardcoded - can't change in production
    return {"kb_path": kb_path, "model": model}
```

**Ask yourself:** "Will this work on Linux, Windows, and Mac?"

**Test Standard:**
```python
def test_node_portability():
    """Ensure nodes use portable paths and configs."""
    import platform

    state = {"query": "test"}
    result = load_configuration(state)

    # Paths should use os.sep or pathlib
    kb_path = result["kb_path"]
    assert os.path.exists(kb_path), f"Path should exist on {platform.system()}"

    # No hardcoded /Users/ or C:\\ paths
    assert "/Users/" not in kb_path or platform.system() == "Darwin"
    assert "C:\\" not in kb_path or platform.system() == "Windows"
```

---

### 6. Defensibility (Fail-Fast, Fail-Safe, Fail-Loud)

**Rule:** Validate input immediately, degrade gracefully, log failures observably

**Why This Matters for LangGraph:**
- **Critical for production** - One failing node shouldn't crash the entire graph
- Graphs should continue executing even when individual nodes fail
- Errors should be visible in LangSmith traces

**LangGraph Example:**

```python
# ✅ GOOD: Defensive node with fail-fast, fail-safe, fail-loud
from src.observability import trace_generation

@trace_generation(name="retrieve_chunks")  # Fail-loud: Observable in LangSmith
def retrieve_chunks(state: ConversationState) -> Dict[str, Any]:
    """Defensive node with comprehensive error handling."""

    # FAIL-FAST: Validate input immediately
    if not state.get("query"):
        logger.warning("Empty query in retrieve_chunks")
        return {
            "retrieved_chunks": [],
            "error": "empty_query",
            "error_message": "I need a question to search for information."
        }

    if len(state["query"]) > 10000:
        raise ValueError("Query too long (>10k chars) - possible attack")  # Fail-fast on abuse

    try:
        # Attempt primary operation
        chunks = rag_engine.retrieve(state["query"], top_k=4)

        if not chunks or len(chunks) == 0:
            # FAIL-SAFE: Provide fallback even on empty results
            logger.info(f"No chunks found for query: {state['query'][:100]}")
            return {
                "retrieved_chunks": [],
                "error": "no_results",
                "error_message": "I don't have specific information about that, but I can try to help based on general knowledge."
            }

        return {"retrieved_chunks": chunks}

    except Exception as e:
        # FAIL-LOUD: Log to LangSmith with full context
        logger.error(f"Retrieval failed: {e}", extra={
            "query": state.get("query"),
            "role": state.get("role"),
            "error_type": type(e).__name__
        })

        # FAIL-SAFE: Return graceful degradation instead of crashing
        return {
            "retrieved_chunks": [],
            "error": "retrieval_failed",
            "error_message": "I'm having trouble accessing my knowledge base right now. Let me try to help anyway."
        }

# ❌ BAD: No error handling - graph crashes on failure
def retrieve_chunks(state: ConversationState) -> Dict[str, Any]:
    """One error crashes the entire graph!"""
    chunks = rag_engine.retrieve(state["query"], top_k=4)  # Can throw exception
    return {"retrieved_chunks": chunks.get("chunks")}  # Can return None and crash next node
```

**Ask yourself:** "What's the worst that could happen with bad input?"

**The Three Levels of Defense:**

1. **Fail-Fast (Input Validation):**
   - Validate at the earliest possible point
   - Raise exceptions for invalid/malicious input
   - Prevents wasted computation on bad data

2. **Fail-Safe (Graceful Degradation):**
   - Never let one node crash the graph
   - Return error states instead of raising exceptions
   - Provide fallback responses

3. **Fail-Loud (Observability):**
   - Log ALL errors to LangSmith with context
   - Include query, role, error type, stack trace
   - Makes production debugging possible

**Test Standard:**
```python
def test_node_defensibility():
    """Ensure nodes handle errors gracefully and observably."""

    # Test 1: Fail-fast on invalid input
    with pytest.raises(ValueError):
        retrieve_chunks({"query": "x" * 10001})  # Too long

    # Test 2: Fail-safe on exceptions
    with patch('src.core.rag_engine.RagEngine.retrieve', side_effect=Exception("DB down")):
        state = {"query": "test"}
        result = retrieve_chunks(state)

        assert "error" in result, "Node must set error flag"
        assert result["retrieved_chunks"] == [], "Node must provide fallback"
        assert "error_message" in result, "Node must explain error to user"

    # Test 3: Fail-loud (observability)
    with patch('langsmith.Client.create_run') as mock_trace:
        retrieve_chunks({"query": "test"})
        assert mock_trace.called, "Node must create LangSmith trace"
```

---

### 7. Maintainability & Testability

**Rule:** Future you will thank present you - Write clear, testable code with pure functions

**Why This Matters for LangGraph:**
- Nodes should be easy to test in isolation
- Pure functions (no side effects) make testing deterministic
- Separate business logic from I/O for unit testing

**LangGraph Example:**

```python
# ✅ GOOD: Pure business logic extracted, easy to test
def _calculate_hiring_signals(query: str, role: str) -> List[str]:
    """Pure function - deterministic, no side effects, easy to test."""
    signals = []

    query_lower = query.lower()
    if "hiring" in query_lower or "position" in query_lower:
        signals.append("mentioned_hiring")
    if "team" in query_lower or "join us" in query_lower:
        signals.append("described_team")
    if role in ["Hiring Manager (technical)", "Hiring Manager (nontechnical)"]:
        signals.append("hiring_manager_role")

    return signals

# Node wraps pure logic with I/O
@trace_generation(name="detect_hiring_signals")  # I/O: Observability
def detect_hiring_signals(state: ConversationState) -> Dict[str, Any]:
    """Node handles I/O, delegates to pure function."""
    signals = _calculate_hiring_signals(state["query"], state["role"])

    # I/O: Analytics logging
    if signals:
        logger.info(f"Hiring signals detected: {signals}")

    return {"hiring_signals": signals}

# Easy to unit test pure function
def test_hiring_signal_detection():
    """Pure function = easy, fast, deterministic tests."""
    assert _calculate_hiring_signals("We're hiring engineers", "Developer") == ["mentioned_hiring"]
    assert _calculate_hiring_signals("Join our team!", "Hiring Manager (technical)") == [
        "described_team",
        "hiring_manager_role"
    ]

# ❌ BAD: Logic mixed with I/O - hard to test
@trace_generation(name="detect_hiring_signals")
def detect_hiring_signals(state: ConversationState) -> Dict[str, Any]:
    """Mixed logic and I/O - requires mocking logger, analytics, etc."""
    signals = []

    logger.info(f"Checking query: {state['query']}")  # I/O mixed in

    if "hiring" in state["query"].lower():
        signals.append("mentioned_hiring")
        supabase_analytics.log_event("hiring_signal_detected")  # I/O mixed in

    if state["role"].startswith("Hiring Manager"):
        signals.append("hiring_manager_role")
        send_slack_notification("Hiring manager detected!")  # I/O mixed in

    return {"hiring_signals": signals}
```

**Ask yourself:** "Can I write a unit test for this without mocking 5 things?"

**The Pattern:**
1. **Extract pure logic** to private functions (prefix with `_`)
2. **Node handles I/O** (logging, analytics, external calls)
3. **Test pure functions** without mocks (fast, deterministic)
4. **Integration test nodes** with mocks (slower, but comprehensive)

**Test Standard:**
```python
def test_node_testability():
    """Ensure nodes separate logic from I/O."""
    # Pure functions should have no external dependencies
    result = _calculate_hiring_signals("hiring engineers", "Hiring Manager (technical)")
    assert isinstance(result, list)
    assert all(isinstance(s, str) for s in result)

    # Node tests can use mocks for I/O only
    with patch('src.analytics.supabase_analytics.log_event') as mock_log:
        state = {"query": "hiring", "role": "Hiring Manager (technical)"}
        result = detect_hiring_signals(state)

        # Business logic result
        assert "hiring_signals" in result

        # I/O was called
        assert mock_log.called
```

---

### 8. Simplicity (KISS, DRY, YAGNI)

**Rule:** Keep it simple, don't repeat yourself, you aren't gonna need it

**Why This Matters for LangGraph:**
- Simple graphs are easier to debug in LangSmith
- DRY prevents maintenance burden across nodes
- YAGNI prevents over-engineering with unnecessary nodes

**LangGraph Example - KISS (Keep It Simple):**

```python
# ✅ GOOD: Simple, readable router
def should_retrieve(state: ConversationState) -> str:
    """Simple decision logic - easy to understand."""
    if state.get("is_greeting"):
        return "skip_retrieval"
    return "retrieve"

graph_builder.add_conditional_edges(
    "classify",
    should_retrieve,
    {
        "skip_retrieval": "generate_answer",
        "retrieve": "retrieve_chunks"
    }
)

# ❌ BAD: Over-engineered router
def should_retrieve(state: ConversationState) -> str:
    """Unnecessarily complex for simple decision."""
    # Decision matrix for 2 cases!?
    decision_matrix = {
        ("greeting", True, "casual"): "skip_retrieval",
        ("greeting", True, "formal"): "skip_retrieval",
        ("greeting", False, "casual"): "retrieve",
        ("greeting", False, "formal"): "retrieve",
        # ... 20 more combinations
    }

    key = (
        state.get("query_type"),
        state.get("is_greeting"),
        state.get("tone", "casual")
    )

    return decision_matrix.get(key, "retrieve")
```

**LangGraph Example - DRY (Don't Repeat Yourself):**

```python
# ✅ GOOD: Centralized error handler
def _handle_node_error(node_name: str, error: Exception, state: ConversationState) -> Dict:
    """Reusable error handling for all nodes."""
    logger.error(f"{node_name} failed: {error}", extra={
        "query": state.get("query"),
        "role": state.get("role"),
        "error_type": type(error).__name__
    })

    return {
        "error": f"{node_name}_failed",
        "error_message": "I encountered an issue. Let me try to help anyway."
    }

# Use in all nodes
def classify_query(state: ConversationState) -> Dict:
    try:
        return {"query_type": _determine_type(state["query"])}
    except Exception as e:
        return _handle_node_error("classify_query", e, state)

def retrieve_chunks(state: ConversationState) -> Dict:
    try:
        chunks = rag_engine.retrieve(state["query"])
        return {"retrieved_chunks": chunks}
    except Exception as e:
        return _handle_node_error("retrieve_chunks", e, state)

# ❌ BAD: Repeated error handling in every node
def classify_query(state: ConversationState) -> Dict:
    try:
        return {"query_type": _determine_type(state["query"])}
    except Exception as e:
        logger.error(f"classify_query failed: {e}")  # Repeated!
        return {"error": "classify_query_failed"}

def retrieve_chunks(state: ConversationState) -> Dict:
    try:
        chunks = rag_engine.retrieve(state["query"])
        return {"retrieved_chunks": chunks}
    except Exception as e:
        logger.error(f"retrieve_chunks failed: {e}")  # Repeated!
        return {"error": "retrieve_chunks_failed"}
```

**LangGraph Example - YAGNI (You Aren't Gonna Need It):**

```python
# ✅ GOOD: Only the nodes we actually use
graph_builder = StateGraph(ConversationState)
graph_builder.add_node("classify", classify_query)
graph_builder.add_node("retrieve", retrieve_chunks)
graph_builder.add_node("generate", generate_answer)
graph = graph_builder.compile()

# ❌ BAD: Building for hypothetical future features
graph_builder = StateGraph(ConversationState)
graph_builder.add_node("classify", classify_query)
graph_builder.add_node("retrieve", retrieve_chunks)
graph_builder.add_node("sentiment_analysis", analyze_sentiment)  # Not used yet!
graph_builder.add_node("language_detection", detect_language)  # Might need later?
graph_builder.add_node("spam_filter", filter_spam)  # Just in case?
graph_builder.add_node("profanity_check", check_profanity)  # Future-proofing?
graph = graph_builder.compile()  # Graph is now complex for no reason
```

**Ask yourself:**
- KISS: "Am I making this more complex than it needs to be?"
- DRY: "Have I written this exact logic elsewhere?"
- YAGNI: "Will I really need this feature right now?"

**Test Standard:**
```python
def test_graph_simplicity():
    """Ensure graph doesn't have unnecessary complexity."""
    # Count nodes
    nodes = list(graph.nodes.keys())

    # Should have only essential nodes (8 for our flow)
    assert len(nodes) <= 10, f"Graph has {len(nodes)} nodes - might be over-engineered"

    # Each node should be used (no dead code)
    for node_name in nodes:
        if node_name not in ["__start__", "__end__"]:
            # Trace execution - node should be hit
            state = graph.invoke({"query": "test", "role": "Developer"})
            # (Check execution trace to ensure node was called)
```

---

## Phase 2: Production Monitoring with LangSmith

**Status**: 📅 Planned (after LangGraph migration completes)
**Purpose**: Complement pytest testing with production observability
**Reference**: Originally documented in `QA_LANGSMITH_INTEGRATION.md` (now archived)

---

### The Hybrid QA Approach

**Philosophy**: Testing catches bugs before deployment, monitoring catches edge cases after deployment.

| Tool | Purpose | When | What It Catches |
|------|---------|------|-----------------|
| **pytest** | Pre-deployment testing | Before code merges | 90% of bugs (logic errors, policy violations) |
| **LangSmith** | Post-deployment monitoring | After production deploy | 10% of bugs (edge cases, performance, real LLM behavior) |

**Why Both?**
- **pytest** = Fast, deterministic, blocks bad code from merging
- **LangSmith** = Real behavior, edge cases from actual user queries, performance metrics
- **Together** = Comprehensive QA (test what you can control, monitor what you can't)

---

### What LangSmith IS (Production Observability)

✅ **Production observability tool**
- Traces every LLM call in real-time
- Shows actual prompts, responses, latency, costs
- Detects patterns across thousands of queries
- Root cause analysis for production issues

### What LangSmith is NOT

❌ **Not a testing framework**
- Can't mock dependencies
- Can't control inputs (depends on real user queries)
- Not deterministic (LLM responses vary)
- Not a pytest replacement (complements it)

---

### What We'll Monitor

| Quality Standard | pytest Test | LangSmith Check | Why Both? |
|-----------------|-------------|-----------------|-----------|
| **No emoji headers** | `test_no_emoji_headers()` | ✅ Scan responses for `###` | Test uses mocks, LangSmith sees real LLM output |
| **Response length** | `test_no_information_overload()` | ✅ Alert if >15k chars | Test checks logic, LangSmith catches prompt engineering bugs |
| **Duplicate prompts** | `test_no_duplicate_prompts()` | ✅ Count "would you like" | Test validates code, LangSmith finds edge cases |
| **Latency** | ❌ No test | ✅ Track p50, p95, p99 | Requires real production traffic |
| **Error rate** | ❌ No test | ✅ Track exceptions | Runtime errors only appear in production |
| **Token costs** | ❌ No test | ✅ Cost per query | Budget monitoring requires production data |

---

### Implementation Plan

#### Step 1: Setup (Week 2 - After Phase 1 Complete)
```bash
# Get LangSmith API key
# Visit: https://smith.langchain.com/

# Add to .env
echo "LANGCHAIN_API_KEY=lsv2_pt_..." >> .env
echo "LANGCHAIN_TRACING_V2=true" >> .env
echo "LANGCHAIN_PROJECT=noahs-ai-assistant" >> .env

# Deploy to Vercel (auto-detects env vars)
vercel --prod
```

#### Step 2: Enhanced Quality Monitor
**File**: `scripts/quality_monitor.py`

```python
from langsmith import Client
from src.observability import get_langsmith_client

def check_langsmith_traces():
    """Check production LLM traces for quality violations."""
    client = get_langsmith_client()

    if not client:
        print("⚠️  LangSmith not configured, skipping trace checks")
        return []

    # Get last 24 hours
    runs = client.list_runs(
        project_name="noahs-ai-assistant",
        start_time=datetime.now() - timedelta(hours=24)
    )

    violations = []

    for run in runs:
        if not run.outputs or "answer" not in run.outputs:
            continue

        answer = run.outputs["answer"]

        # Policy 1: No markdown headers
        if re.search(r'#{1,6}\s', answer):
            violations.append(f"🔴 CRITICAL: Markdown headers in trace {run.id}")

        # Policy 2: No information overload
        if len(answer) > 15000:
            violations.append(f"⚠️  WARNING: Response {len(answer)} chars in trace {run.id}")

        # Policy 3: Performance
        if run.total_time and run.total_time > 3000:
            violations.append(f"⚠️  WARNING: Slow query {run.total_time}ms in trace {run.id}")

    return violations

def main():
    """Run all quality checks."""
    all_violations = []

    # Check 1: Supabase metrics (existing)
    supabase_violations = check_supabase_metrics()
    all_violations.extend(supabase_violations)

    # Check 2: LangSmith traces (NEW - Phase 2)
    langsmith_violations = check_langsmith_traces()
    all_violations.extend(langsmith_violations)

    # Report results
    if all_violations:
        print(f"\n❌ {len(all_violations)} quality violations found:\n")
        for violation in all_violations:
            print(f"  {violation}")
        sys.exit(1)
    else:
        print("\n✅ All quality checks passed!")
        sys.exit(0)
```

#### Step 3: Alert Thresholds

```python
ALERT_RULES = {
    "emoji_headers": {
        "severity": "CRITICAL",
        "threshold": "> 0 occurrences in 24h",
        "action": "Email + Slack + Create GitHub issue"
    },
    "response_length": {
        "severity": "WARNING",
        "threshold": "> 5 responses >15k chars in 24h",
        "action": "Email summary"
    },
    "error_rate": {
        "severity": "CRITICAL",
        "threshold": "> 1% error rate",
        "action": "Email + Slack + Page on-call"
    },
    "latency_p95": {
        "severity": "WARNING",
        "threshold": "> 3s p95 latency",
        "action": "Email summary"
    },
    "daily_cost": {
        "severity": "INFO",
        "threshold": "> $5/day token costs",
        "action": "Email summary"
    }
}
```

---

### Daily Report Example

```
📊 Quality Report - October 17, 2025

✅ Overall Status: HEALTHY

Metrics (24h):
  - 234 queries processed
  - 1.2s avg latency (p95: 2.1s)
  - $0.45 total cost ($0.0019/query)
  - 0 errors (0%)

Policy Compliance:
  ✅ No markdown headers detected
  ✅ All responses <15k chars
  ✅ Single follow-up prompts only
  ⚠️  3 queries >3s latency (1.3%)

Top Queries:
  1. "explain conversation nodes" - 45 times
  2. "show me code examples" - 32 times
  3. "what are noah's skills" - 28 times

Slowest Query:
  - Query: "explain the full technical architecture"
  - Latency: 4.2s
  - Trace: https://smith.langchain.com/...
  - Action: Optimize prompt length
```

---

### Cost Analysis

**LangSmith Pricing**:
- Free tier: 5,000 traces/month (good for development)
- Team tier: $39/month for 100k traces (recommended for production)

**Break-Even Analysis**:
- If LangSmith catches **1 production bug/month** → Saves 2-4 hours debugging → Worth $39
- If it prevents **1 user complaint** → Maintains reputation → Priceless

**Current Usage** (estimated):
- ~234 queries/day × 30 days = 7,020 traces/month
- **Recommendation**: Start with Team tier ($39/month)

---

### Why This Matters for KB vs Response Policy

**The Problem**: Our policy separates KB formatting (rich markdown) from user responses (professional bold only).

**pytest validates this with mocks** ✅
```python
def test_no_emoji_headers():
    mock_engine.generate_response.return_value = "**Bold Header**\n\nContent..."
    assert "###" not in state.answer  # ✅ Test passes
```

**But what if in production...**
- ❌ LLM ignores the prompt instruction?
- ❌ New OpenAI model behaves differently?
- ❌ Edge case query triggers unexpected formatting?
- ❌ Prompt injection bypasses sanitization?

**LangSmith catches these** 🔍
```python
# Scenario: New GPT-4-turbo ignores our "strip ###" instruction
# pytest: ✅ PASSES (mocks return clean output)
# Production: 🔴 Users see ### headers!

# LangSmith alert:
{
    "severity": "CRITICAL",
    "violation": "markdown_headers_in_production",
    "trace_id": "abc-123",
    "query": "explain conversation nodes",
    "response": "### Node 1: handle_greeting...",  # Uh oh!
    "model": "gpt-4-turbo-2024-10-15",  # New model version
    "action": "Rollback to gpt-4 or update prompt"
}
```

---

### Implementation Timeline

| Week | Phase | Tasks | Status |
|------|-------|-------|--------|
| **Week 1** | Phase 1 Testing | 77 automated tests, 99% pass rate | ✅ COMPLETE |
| **Week 2-3** | LangGraph Migration | Convert to StateGraph, update tests | 🚧 IN PROGRESS |
| **Week 4** | Phase 2 Monitoring | LangSmith setup, quality monitor | 📅 PLANNED |
| **Ongoing** | Refinement | Adjust thresholds, add new checks | 📅 PLANNED |

---

### Summary: The Winning Formula

```
Phase 1 (Testing):  pytest (77 tests, 99% pass) ← Prevents 90% of bugs
                           ↓ Deploy
Phase 2 (Monitoring): LangSmith ← Catches the other 10%
                           ↓ Learn
Phase 3 (Improve):  Add new pytest tests for patterns found in production
                           ↓ Repeat
```

**Key Takeaways**:
1. **pytest** = Pre-deployment quality gate (fast, deterministic, blocks bad code)
2. **LangSmith** = Post-deployment safety net (real behavior, edge cases, performance)
3. **Together** = Comprehensive QA (test what you can control, monitor what you can't)
4. **Cost**: $39/month is worth it if it catches 1 production bug/month

---

## Current Quality Standards

### 1. Conversation Quality (19 Tests - Current Status: 19 Passing ✅)
**File**: `tests/test_conversation_quality.py`
**Last Updated**: October 16, 2025

#### Content Storage vs User Presentation Standards

**CRITICAL PRINCIPLE**: Internal KB format ≠ User-facing responses

| Layer | Headers Allowed | Emojis Allowed | Format |
|-------|----------------|----------------|---------|
| **KB Storage** (`data/*.csv`) | ✅ Yes (`###`, `##`) | ✅ Yes (teaching structure) | Rich markdown for semantic search |
| **LLM Response** (user sees) | ❌ No (`###`) | ❌ No in headers | Professional `**Bold**` only |

#### Test Coverage Map

| Standard | Test | Current Status |
|----------|------|---------------|
| KB aggregated (not 245 rows) | `test_kb_coverage_aggregated_not_detailed` | ✅ PASSING |
| KPIs calculated | `test_kpi_metrics_calculated` | ✅ PASSING |
| Recent activity limited | `test_recent_activity_limited` | ✅ PASSING |
| Confessions private | `test_confessions_privacy_protected` | ✅ PASSING |
| Single follow-up prompt | `test_no_duplicate_prompts_in_full_flow` | ✅ PASSING |
| **No pushy resume offers** | `test_no_pushy_resume_offers` | ✅ PASSING (NEW - Oct 16, 2025) |
| **No emoji headers IN RESPONSES** | `test_no_emoji_headers` | ✅ PASSING |
| LLM no self-prompts | `test_llm_no_self_generated_prompts` | ✅ PASSING |
| Data display canned intro | `test_display_data_uses_canned_intro` | ✅ PASSING |
| SQL artifact sanitization | `test_generated_answer_sanitizes_sql_artifacts` | ✅ PASSING |
| Code display graceful | `test_empty_code_index_shows_helpful_message` | ✅ PASSING |
| Code validation logic | `test_code_content_validation_logic` | ✅ PASSING |
| No information overload | `test_no_information_overload` | ✅ PASSING |
| Consistent formatting | `test_consistent_formatting_across_roles` | ✅ PASSING |
| No section iteration | `test_analytics_no_section_iteration` | ✅ PASSING |
| Prompts deprecated | `test_response_generator_no_prompts` | ✅ PASSING |
| Single prompt location | `test_conversation_nodes_single_prompt_location` | ✅ PASSING |
| **Q&A synthesis** | `test_no_qa_verbatim_responses`, `test_response_synthesis_in_prompts` | ✅ PASSING (2 tests) |

**Current Pass Rate**: 19/19 tests passing (100%) ✅ *(Updated Oct 16, 2025 - added pushy resume offers test)*
**Target**: 19/19 tests passing (100%) ✅

**Run**: `pytest tests/test_conversation_quality.py -v`

---

### 1.1 Intelligent Resume Distribution Exception (NEW - Oct 16, 2025)

**Feature**: Intelligent Resume Distribution System (Hybrid Approach)
**Documentation**: `docs/features/INTELLIGENT_RESUME_DISTRIBUTION.md`

#### The Exception: Subtle Availability Mentions

**Standard Behavior**: Single follow-up prompt only, no pushy CTAs

**Exception**: When hiring signals detected (user mentions hiring, describes role, discusses team), **ONE subtle availability mention is allowed**:

✅ **Acceptable (Mode 2 - Hiring Signals Detected)**:
```
[Educational response about RAG systems - 80% of content]

By the way, Noah's available for roles like this if you'd like to
learn more about his experience building production RAG systems.
```

❌ **Not Acceptable (Too Pushy)**:
```
[Educational response]

INTERESTED IN HIRING NOAH?
Send me your email and I'll forward his resume right away!
Click here to schedule a call!
```

#### Quality Standards for Subtle Mentions

| Standard | Requirement | Why |
|----------|-------------|-----|
| **User-initiated interest** | Only after user mentions hiring | Respects user autonomy |
| **Once per conversation** | Max 1 subtle mention | Prevents spam feeling |
| **Education remains primary** | ≥50% of response is educational | Stays true to primary purpose |
| **No aggressive CTAs** | No "send email", "click here", "sign up" | Maintains professional tone |
| **Natural placement** | At end of educational response | Feels like afterthought, not pitch |

#### Test: `test_no_pushy_resume_offers()`

**Validates**:
1. **Mode 1 (Pure Education)**: ZERO resume mentions ✅
2. **Mode 2 (Hiring Signals)**: ONE subtle mention allowed ✅
3. **No pushy phrases**: No aggressive CTAs ✅
4. **Education-focused**: ≥50% educational content even with mention ✅

**Example Passing Response** (Mode 2):
```
RAG systems work by combining retrieval with generation. First, relevant
documents are retrieved from a vector database based on semantic similarity.
Then, these documents are passed to the LLM as context, allowing it to
generate informed responses grounded in your data.

The key advantage is that you get current, domain-specific answers without
retraining the entire model. Would you like me to walk through Noah's
implementation with code examples?

By the way, Noah's available for roles like this if you'd like to learn more.
```

**Why This Works**:
- ✅ Primary content is education (3 paragraphs)
- ✅ Availability mention is subtle (1 sentence)
- ✅ No pressure to act ("if you'd like")
- ✅ User mentioned hiring first (triggered Mode 2)

---

### 1.2 Resume Distribution Test Suite (NEW - Oct 16, 2025)

**File**: `tests/test_resume_distribution.py` (551 lines, 37 tests)
**Pass Rate**: 100% (37/37 passing in 0.04s)

#### Test Coverage Breakdown

| Test Class | Tests | Purpose | Pass Rate |
|------------|-------|---------|-----------|
| **TestHiringSignalDetection** | 8 | Validates passive signal tracking, no false positives | ✅ 8/8 (100%) |
| **TestExplicitResumeRequestHandling** | 6 | Validates Mode 3 immediate distribution, no qualification | ✅ 6/6 (100%) |
| **TestSubtleAvailabilityMentions** | 5 | Validates Mode 2 conditions (≥2 signals, HM role) | ✅ 5/5 (100%) |
| **TestJobDetailsGathering** | 6 | Validates post-interest gathering, extraction accuracy | ✅ 6/6 (100%) |
| **TestOncePerSessionEnforcement** | 2 | Validates resume_sent flag prevents duplicates | ✅ 2/2 (100%) |
| **TestEmailNameExtraction** | 5 | Validates contact info parsing, graceful fallbacks | ✅ 5/5 (100%) |
| **TestHybridApproachIntegration** | 5 | Validates full Mode 1/2/3 workflows | ✅ 5/5 (100%) |
| **TOTAL** | **37** | **Comprehensive hybrid approach validation** | **✅ 37/37 (100%)** |

#### Quality Standards Enforced

**1. Education-First Principle** (Mode 1):
- Pure education queries receive ZERO resume mentions
- Test: `test_education_mode_zero_mentions()` ✅
- Example: "How do RAG systems work?" → Educational answer only

**2. Explicit Request Priority** (Mode 3):
- Direct requests trigger immediate email collection
- No qualification checks, no delay tactics
- Test: `test_explicit_request_immediate_response()` ✅
- Example: "Can I get your resume?" → "I'd be happy to send that. What's your email?"

**3. Passive Signal Tracking** (Mode 2 enabler):
- System tracks hiring indicators (mentioned_hiring, described_role, team_context)
- Does NOT proactively offer resume
- Enables subtle availability mention when ≥2 signals detected
- Test: `test_detects_all_signal_types()` ✅

**4. Subtle Availability Mentions** (Mode 2):
- Only when ≥2 hiring signals + hiring manager role + not sent yet
- ONE sentence at end of educational response
- No aggressive CTAs, no pressure language
- Test: `test_subtle_mention_with_hiring_signals()` ✅

**5. Job Details Gathering** (Post-Interest):
- Only AFTER resume sent
- Conversational tone ("Just curious — what company are you with?")
- Regex extraction for company, position, timeline
- Test: `test_post_interest_job_details()` ✅

**6. Once-Per-Session Enforcement**:
- `resume_sent` flag prevents duplicate distributions
- Polite response: "I've already sent my resume to your email"
- Test: `test_resume_sent_flag_prevents_duplicate()` ✅

#### Key Test Patterns

**Regex Pattern Validation** (Optimized via TDD):
```python
# Timeline detection: "available", "when.*start"
assert detect_hiring_signals(state).fetch("hiring_signals") == ["timeline_urgency"]

# Company extraction: "I'm with Acme Corp, hiring for..."
assert extract_job_details_from_query(state).fetch("job_details")["company"] == "Acme Corp"

# Position extraction: "Hiring for Senior Engineer" (case-insensitive)
assert details["position"] == "Senior Engineer"
```

**State Isolation** (No Side Effects):
```python
# Each test creates fresh ConversationState
state = ConversationState(query="test query", role="Hiring Manager (technical)")

# Assertions check state mutations only
assert state.fetch("resume_explicitly_requested") is True
```

**Edge Case Coverage**:
- Empty queries → Graceful fallback ✅
- Malformed input → No crashes ✅
- Missing optional fields → Default values ✅
- Once-per-session enforcement → Duplicate prevention ✅

#### Running Resume Distribution Tests

```bash
# Run all 37 tests (ultra-fast execution)
pytest tests/test_resume_distribution.py -v

# Run specific test class
pytest tests/test_resume_distribution.py::TestJobDetailsGathering -v

# Run with detailed failure output
pytest tests/test_resume_distribution.py -vv

# Expected output:
# ============================== 37 passed in 0.04s ==============================
```

---

## **NEW: Documentation Alignment Testing**

### The Problem We're Solving

**Before**: Documentation could claim function names or flows that didn't match reality. Example:
- Docs said: `classify_intent` → Code actually used: `classify_query`
- Result: Developers waste time searching for functions that don't exist

**Solution**: Automated tests that verify documentation matches code.

---

### Test Suite: Documentation Alignment

**File**: `tests/test_documentation_alignment.py` (NEW)

#### Test 1: Conversation Pipeline Flow Matches Code
```python
def test_conversation_flow_documented_correctly():
    """Verify SYSTEM_ARCHITECTURE_SUMMARY describes actual pipeline."""

    # Read master documentation
    with open("docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md") as f:
        doc_content = f.read()

    # Extract documented node names from code section
    import re
    code_section = re.search(r"```python\n# Pipeline.*?\n(.*?)```", doc_content, re.DOTALL)
    if not code_section:
        pytest.fail("No code pipeline found in SYSTEM_ARCHITECTURE_SUMMARY.md")

    documented_nodes = []
    for line in code_section.group(1).split('\n'):
        if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('→'):
            # Extract node name (first word before whitespace or →)
            node = line.strip().split()[0].split('→')[0]
            if node and not node.startswith('Source:'):
                documented_nodes.append(node)

    # Get actual pipeline from code
    from src.flows.conversation_flow import run_conversation_flow
    import inspect
    source = inspect.getsource(run_conversation_flow)

    # Verify documented nodes appear in actual code
    actual_nodes = [
        "initialize_conversation_state",
        "handle_greeting",
        "classify_role_mode",
        "classify_intent",
        "detect_hiring_signals",
        "handle_resume_request",
        "extract_entities",
        "assess_clarification_need",
        "ask_clarifying_question",
        "compose_query",
        "retrieve_chunks",
        "re_rank_and_dedup",
        "validate_grounding",
        "handle_grounding_gap",
        "generate_draft",
        "hallucination_check",
        "plan_actions",
        "format_answer",
        "execute_actions",
        "suggest_followups",
        "update_memory",
        "log_and_notify",
    ]

    for node in actual_nodes:
        assert node in documented_nodes, (
            f"Node '{node}' exists in code but not documented in "
            f"SYSTEM_ARCHITECTURE_SUMMARY.md. Update docs to match implementation."
        )

    # Verify documented nodes actually exist in code
    for node in documented_nodes:
        if node in actual_nodes:  # Skip conceptual descriptions
            assert node in source, (
                f"Node '{node}' documented but doesn't exist in conversation_flow.py. "
                f"Remove from docs or implement in code."
            )
```

**What it catches**: Function name mismatches, missing nodes, phantom nodes

---

#### Test 2: Code References Are Valid File Paths
```python
def test_documentation_file_references_valid():
    """Ensure all file paths mentioned in docs actually exist."""
    import os
    import re

    doc_files = [
        "docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md",
        "docs/context/DATA_COLLECTION_AND_SCHEMA_REFERENCE.md",
        "docs/RAG_ENGINE.md",
        "docs/CONVERSATION_PIPELINE_MODULES.md",
    ]

    invalid_references = []

    for doc_file in doc_files:
        with open(doc_file) as f:
            content = f.read()

        # Find references like "src/flows/core_nodes.py" or "Source: src/..."
        file_refs = re.findall(r'(?:src/[\w/]+\.py)|(?:tests/[\w/]+\.py)', content)

        for ref in file_refs:
            if not os.path.exists(ref):
                invalid_references.append({
                    "doc": doc_file,
                    "reference": ref,
                    "line": content[:content.find(ref)].count('\n') + 1
                })

    assert len(invalid_references) == 0, (
        f"Found {len(invalid_references)} invalid file references:\\n" +
        "\\n".join([
            f"  {inv['doc']} line {inv['line']}: {inv['reference']}"
            for inv in invalid_references
        ]) +
        "\\nUpdate documentation to reference correct files."
    )
```

**What it catches**: Outdated file paths, typos, files that were moved/deleted

---

#### Test 3: Role Names Match Between Docs and Code
```python
def test_role_names_consistent():
    """Verify role names in docs match actual role definitions."""

    # Get documented roles from PROJECT_REFERENCE_OVERVIEW
    with open("docs/context/PROJECT_REFERENCE_OVERVIEW.md") as f:
        doc_content = f.read()

    doc_roles = set()
    if "Software Developer" in doc_content:
        doc_roles.add("Software Developer")
    if "Hiring Manager (technical)" in doc_content:
        doc_roles.add("Hiring Manager (technical)")
    if "Hiring Manager (nontechnical)" in doc_content or "Hiring Manager (non-technical)" in doc_content:
        doc_roles.add("Hiring Manager (nontechnical)")
    if "Just Exploring" in doc_content or "Just looking" in doc_content:
        doc_roles.add("Just looking around")
    if "Confess" in doc_content:
        doc_roles.add("Looking to confess crush")

    # Get actual roles from code
    from src.agents.roles import AVAILABLE_ROLES
    code_roles = set(AVAILABLE_ROLES)

    # Check for mismatches
    missing_in_docs = code_roles - doc_roles
    extra_in_docs = doc_roles - code_roles

    assert len(missing_in_docs) == 0, (
        f"Roles in code but not documented: {missing_in_docs}. "
        f"Add to docs/context/PROJECT_REFERENCE_OVERVIEW.md"
    )

    assert len(extra_in_docs) == 0, (
        f"Roles documented but not in code: {extra_in_docs}. "
        f"Remove from docs or implement in src/agents/roles.py"
    )
```

**What it catches**: New roles added without documentation, renamed roles, deprecated roles still in docs

---

#### Test 4: Temperature Settings Match Documentation
```python
def test_temperature_settings_documented_correctly():
    """Verify temperature value in docs matches actual code."""

    # Get documented temperature
    with open("docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md") as f:
        doc_content = f.read()

    import re
    temp_match = re.search(r'temperature[:\s]+(\d+\.?\d*)', doc_content)
    if not temp_match:
        pytest.fail("Temperature setting not documented in SYSTEM_ARCHITECTURE_SUMMARY.md")

    documented_temp = float(temp_match.group(1))

    # Get actual temperature from code
    from src.core.rag_factory import RagFactory
    import inspect
    source = inspect.getsource(RagFactory.create_llm)

    code_temp_match = re.search(r'temperature=(\d+\.?\d*)', source)
    assert code_temp_match, "Temperature not found in RagFactory.create_llm"

    actual_temp = float(code_temp_match.group(1))

    assert documented_temp == actual_temp, (
        f"Temperature mismatch: docs say {documented_temp}, code uses {actual_temp}. "
        f"Update docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md to match code."
    )
```

**What it catches**: Outdated configuration values in documentation

---

#### Test 5: Master Docs Cross-Reference Integrity
```python
def test_master_docs_cross_references_valid():
    """Ensure cross-references between master docs point to existing sections."""

    import os
    import re

    master_docs = {
        "PROJECT_REFERENCE_OVERVIEW.md": None,
        "SYSTEM_ARCHITECTURE_SUMMARY.md": None,
        "DATA_COLLECTION_AND_SCHEMA_REFERENCE.md": None,
        "CONVERSATION_PERSONALITY.md": None,
    }

    # Read all master docs and extract headers
    for doc_name in master_docs:
        path = f"docs/context/{doc_name}"
        with open(path) as f:
            content = f.read()
            # Extract all markdown headers
            headers = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
            master_docs[doc_name] = {
                "content": content,
                "headers": headers,
                "path": path
            }

    # Find cross-references (See FILENAME.md, reference to FILENAME, etc.)
    invalid_refs = []
    for doc_name, doc_data in master_docs.items():
        # Find references to other master docs
        for other_doc in master_docs:
            if other_doc != doc_name and other_doc in doc_data["content"]:
                # Valid reference, but check if it points to existing content
                pass

    # Check for references to sections that don't exist
    # Example: "See CONVERSATION_PERSONALITY.md section X" where X doesn't exist

    assert len(invalid_refs) == 0, (
        f"Found {len(invalid_refs)} broken cross-references in master docs"
    )
```

**What it catches**: Broken links between master documentation files

---

### Running Documentation Alignment Tests

```bash
# Run all documentation alignment tests
pytest tests/test_documentation_alignment.py -v

# Run specific test
pytest tests/test_documentation_alignment.py::test_conversation_flow_documented_correctly -v

# Run with detailed output
pytest tests/test_documentation_alignment.py -vv
```

---

## **Feature Development Documentation Workflow**

### When Adding New Features or Changing Behavior

This section answers: **"Should I create a new .md file or update an existing one?"**

---

### Decision Tree: Where to Document Changes

```
┌─ Is this a NEW feature (adds capability)?
│  ├─ YES → Create feature doc in docs/features/
│  │         Example: DISPLAY_INTELLIGENCE_IMPLEMENTATION.md
│  │
│  └─ NO (existing feature change)
│      └─ Is this an IMPLEMENTATION detail change?
│          ├─ YES (e.g., refactored code, new helper function)
│          │  └─ Update existing feature doc in docs/features/
│          │
│          └─ NO (BEHAVIOR or ARCHITECTURE change)
│              └─ Update MASTER docs in docs/context/
│                  - SYSTEM_ARCHITECTURE_SUMMARY.md
│                  - PROJECT_REFERENCE_OVERVIEW.md
│                  - DATA_COLLECTION_AND_SCHEMA_REFERENCE.md
│                  - CONVERSATION_PERSONALITY.md
```

---

### Feature Documentation Checklist

#### ✅ **Scenario 1: Adding a New Feature**

**Example**: Adding sentiment analysis to user queries

**Required Documentation**:
1. **Create new feature doc**: `docs/features/SENTIMENT_ANALYSIS_IMPLEMENTATION.md`
   - Include: Problem statement, implementation approach, code files, examples
   - Template available in `docs/features/README.md`

2. **Update CHANGELOG.md**:
   ```markdown
   ## [Unreleased]
   ### Added
   - Sentiment analysis for query classification
   ```

3. **Update master docs IF behavior changes**:
   - `SYSTEM_ARCHITECTURE_SUMMARY.md`: Add sentiment node to pipeline
   - `PROJECT_REFERENCE_OVERVIEW.md`: Mention sentiment capability
   - `CONVERSATION_PERSONALITY.md`: IF sentiment affects tone

4. **Add tests**:
   - Conversation quality test (if user-facing)
   - Alignment test (if documented functions/flow)

**Files Created/Modified** (example):
```
✅ NEW: docs/features/SENTIMENT_ANALYSIS_IMPLEMENTATION.md
✅ MODIFIED: CHANGELOG.md
✅ MODIFIED: docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md (if pipeline changes)
✅ NEW: tests/test_[feature_name].py
```

---

#### ✅ **Scenario 2: Changing Existing Feature Implementation**

**Example**: Refactoring code display logic without changing behavior

**Required Documentation**:
1. **Update existing feature doc**: `docs/features/DISPLAY_INTELLIGENCE_IMPLEMENTATION.md`
   - Update code file references
   - Add "Refactoring Notes" section if architecture changed
   - Keep behavior description unchanged

2. **Update CHANGELOG.md**:
   ```markdown
   ## [Unreleased]
   ### Changed
   - Refactored code display logic for maintainability
   ```

3. **Master docs**: NO update needed (behavior unchanged)

4. **Run alignment tests**: Ensure function references still valid

**Files Modified**:
```
✅ MODIFIED: docs/features/DISPLAY_INTELLIGENCE_IMPLEMENTATION.md
✅ MODIFIED: CHANGELOG.md
✅ RUN: pytest tests/test_documentation_alignment.py -v
```

---

#### ✅ **Scenario 3: Changing System Behavior/Architecture**

**Example**: Changing conversation pipeline from 8 nodes to 10 nodes

**Required Documentation**:
1. **Update MASTER doc**: `docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md`
   - Update pipeline diagram
   - Update function list with source file references
   - Explain why architecture changed

2. **Update related feature docs** (if affected):
   - Example: If greeting node split into two, update greeting implementation doc

3. **Update CHANGELOG.md**:
   ```markdown
   ## [Unreleased]
   ### Changed
   - Conversation pipeline: Split greeting logic into separate validation and response nodes
   ```

4. **Update alignment tests**:
   ```python
   # In test_documentation_alignment.py
   actual_nodes = [
       "handle_greeting", "validate_greeting", "respond_greeting",  # ← Added new node
       "classify_query", "retrieve_chunks", ...
   ]
   ```

**Files Modified**:
```
✅ MODIFIED: docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md
✅ MODIFIED: docs/features/GREETING_INTELLIGENCE.md (example)
✅ MODIFIED: CHANGELOG.md
✅ MODIFIED: tests/test_documentation_alignment.py
✅ RUN: pytest tests/ -v
```

---

#### ✅ **Scenario 4: Adding New Role or Query Type**

**Example**: Adding "Recruiter" role with specialized retrieval

**Required Documentation**:
1. **Update MASTER docs**:
   - `PROJECT_REFERENCE_OVERVIEW.md`: Add role to list with description
   - `SYSTEM_ARCHITECTURE_SUMMARY.md`: Explain retrieval strategy
   - `CONVERSATION_PERSONALITY.md`: Define tone/enthusiasm level

2. **Create role-specific doc** (optional, if complex):
   - `docs/features/RECRUITER_ROLE_IMPLEMENTATION.md`

3. **Update CHANGELOG.md**:
   ```markdown
   ## [Unreleased]
   ### Added
   - Recruiter role with specialized hiring pipeline knowledge
   ```

4. **Add alignment tests**:
   ```python
   # In test_documentation_alignment.py
   EXPECTED_ROLES = [
       "hiring_manager_nontechnical",
       "hiring_manager_technical",
       "software_developer",
       "just_looking_around",
       "looking_to_confess_crush",
       "recruiter",  # ← New role
   ]
   ```

**Files Modified**:
```
✅ MODIFIED: docs/context/PROJECT_REFERENCE_OVERVIEW.md
✅ MODIFIED: docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md
✅ MODIFIED: docs/context/CONVERSATION_PERSONALITY.md
✅ MODIFIED: CHANGELOG.md
✅ MODIFIED: tests/test_documentation_alignment.py
✅ OPTIONAL: docs/features/RECRUITER_ROLE_IMPLEMENTATION.md
```

---

### Quick Reference: Documentation Types

| Documentation Type | Purpose | When to Use | Location |
|-------------------|---------|-------------|----------|
| **Master Docs** | Define system behavior, architecture, personality | Behavior/architecture changes | `docs/context/` |
| **Feature Docs** | Implementation details, code walkthrough | New features, refactors | `docs/features/` |
| **Setup Guides** | Installation, configuration, deployment | Adding new services | `docs/setup/` |
| **Analysis Docs** | Technical decisions, performance analysis | Major architectural decisions | `docs/analysis/` |
| **Implementation Reports** | Completion summaries for large changes | End of feature development cycle | `docs/implementation/` |
| **CHANGELOG** | User-facing changes | Every code change | Root: `CHANGELOG.md` |

---

### Documentation Anti-Patterns (Don't Do This)

❌ **Wrong: Creating duplicate behavior documentation**
```markdown
# docs/features/MY_FEATURE.md
The system uses a temperature of 0.4 for LLM calls...
```
→ This duplicates `SYSTEM_ARCHITECTURE_SUMMARY.md`

✅ **Right: Reference master docs**
```markdown
# docs/features/MY_FEATURE.md
This feature uses the standard LLM configuration (see
[SYSTEM_ARCHITECTURE_SUMMARY](../context/SYSTEM_ARCHITECTURE_SUMMARY.md#llm-configuration))...
```

---

❌ **Wrong: Using conceptual names**
```markdown
The classify_intent function determines query type...
```
→ Function doesn't exist with that name

✅ **Right: Use actual code names**
```markdown
The `classify_query` function (in `src/flows/conversation_nodes.py`, line 45)
determines query type...
```

---

❌ **Wrong: Documenting without testing**
```markdown
# Add new feature documentation
# No alignment test created
```
→ Documentation will drift from code

✅ **Right: Add alignment test**
```python
# tests/test_documentation_alignment.py
def test_my_new_feature_documented():
    """Verify my_feature function appears in feature docs."""
    with open("docs/features/MY_FEATURE.md") as f:
        assert "my_feature_function" in f.read()
```

---

### Pull Request Checklist for Feature Changes

```markdown
## Documentation Updates (Required)
- [ ] CHANGELOG.md updated with user-facing changes
- [ ] Master docs updated (if behavior/architecture changed)
- [ ] Feature doc created/updated (implementation details)
- [ ] Alignment test added (if new documented functions/flow)
- [ ] All tests passing: `pytest tests/ -v`

## Documentation Type Decision
- [ ] I understand when to create new docs vs update existing
- [ ] I used actual function/file names (not conceptual terms)
- [ ] I cross-referenced master docs (not duplicated content)
- [ ] I added code file references with line numbers
```

---

## Pre-Commit Hooks

**Status**: ✅ **IMPLEMENTED** (October 16, 2025)

**File**: `.pre-commit-config.yaml` (118 lines)

### What Gets Checked

Pre-commit hooks run **automatically before every commit** and validate:

#### 1. Quality Tests (30 tests)
- ✅ **Conversation Quality Tests** (18 tests) - Professional formatting, no emoji headers, response length limits
- ✅ **Documentation Alignment Tests** (12 tests) - Function names match code, valid file paths, correct config values

#### 2. Documentation Drift Prevention
- ✅ **New .md file validation** (`scripts/check_new_docs.py`) - Ensures new docs are registered in master docs, follow naming conventions

#### 3. Code Hygiene (Auto-Fix)
- ✅ **Trailing whitespace** - Automatically removed
- ✅ **End-of-file fixes** - Ensures single newline at EOF
- ✅ **YAML syntax** - Validates `.yml` and `.yaml` files

### Installation

```bash
# One-time setup (required for contributors)
pip install pre-commit
pre-commit install

# Optional: Run hooks on all files
pre-commit run --all-files
```

### Execution Time

~2-3 seconds total for all hooks (fast feedback loop)

### Bypassing Hooks (Emergency Only)

**⚠️ Not recommended** - CI/CD will still run tests on push

```bash
git commit --no-verify -m "emergency hotfix"
```

### Configuration

See `.pre-commit-config.yaml` for complete hook definitions:
- Lines 1-23: Quality test hooks
- Lines 28-32: Documentation drift checker
- Lines 37-53: Code quality checks (commented out - requires cleanup first)
- Lines 58-97: Standard hygiene hooks

---

## CI/CD Pipeline

**Status**: ✅ **IMPLEMENTED** (October 16, 2025)

**File**: `.github/workflows/qa-tests.yml` (126 lines)

### What Gets Tested

GitHub Actions runs **automatically on every push and PR** to `main` and `develop` branches.

#### Test Jobs

**Job 1: Conversation Quality Tests**
- Runs 18 tests validating professional formatting, response quality, and conversation flow
- Includes code coverage reporting
- Timeout: 10 minutes
- Status: **BLOCKING** (PR merge blocked if fails)

**Job 2: Documentation Alignment Tests**
- Runs 12 tests verifying docs match code implementation
- Validates function names, file paths, configuration values
- Timeout: 10 minutes
- Status: **BLOCKING** (PR merge blocked if fails)

**Job 3: Test Summary**
- Aggregates results from both test suites
- Posts summary to GitHub Actions UI
- Provides clear pass/fail status for reviewers

### Execution Flow

```
Developer pushes code
     ↓
GitHub Actions triggered automatically
     ↓
Job 1: Conversation Quality Tests (18 tests)
Job 2: Documentation Alignment Tests (12 tests)
     ↓
Both jobs must pass ✅
     ↓
Job 3: Summary posted to PR
     ↓
Tests pass → PR can be merged ✅
Tests fail → PR blocked, fix required ❌
```

### Branch Protection Rules

**Recommended setup** (configure in GitHub → Settings → Branches):
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Required checks: `conversation-quality`, `documentation-alignment`

### Local Testing Before Push

Run the same tests locally to catch issues before CI/CD:

```bash
# Run all tests (same as CI/CD)
pytest tests/test_conversation_quality.py tests/test_documentation_alignment.py -v

# Or use pre-commit hooks (faster feedback)
git commit -m "your message"  # Hooks run automatically
```

### Performance Optimizations

- **Dependency caching**: pip cache reused across runs (saves ~30 seconds)
- **Parallel execution**: Both test suites run simultaneously (saves ~5 minutes)
- **Targeted tests**: Only runs tests relevant to changed files
- **Fast feedback**: Average execution time 2-3 minutes for full suite

### Monitoring & Alerts

- ✅ **GitHub UI**: View test results in PR "Checks" tab
- ✅ **Slack/Email**: Configure GitHub notifications for failed builds
- ✅ **Status badges**: README.md shows current build status

### Branch Protection Rules (Recommended Setup)

**Why needed:** CI/CD tests are useless without branch protection—tests can pass but PRs still merge without them.

**Setup (5 minutes):**

1. Go to GitHub → **Settings** → **Branches**
2. Click **Add branch protection rule**
3. **Branch name pattern:** `main`
4. Enable these settings:

```yaml
Required Settings:
  ✅ Require a pull request before merging
     - Required approvals: 1 (adjust for team size)

  ✅ Require status checks to pass before merging
     - Require branches to be up to date before merging ✅
     - Status checks that are required:
       - conversation-quality  ← CRITICAL (18 tests)
       - documentation-alignment  ← CRITICAL (12 tests)

  ✅ Do not allow bypassing the above settings
     - Ensures even admins must pass tests

Optional (Recommended):
  ✅ Require conversation resolution before merging
  ✅ Require linear history (prevents messy merges)
  ⬜ Include administrators (enable after team is comfortable)
```

5. **Save changes**

**Result:**
- ✅ PRs blocked if any of 30 tests fail
- ✅ "Merge" button disabled until all checks pass
- ✅ Status visible in PR UI (green checkmark = ready to merge)

**Testing the setup:**
```bash
# Create test branch with intentional failure
git checkout -b test-branch-protection
echo "print('test')" >> src/main.py  # Will fail strict checks (when enabled)
git add -A && git commit -m "test: verify branch protection"
git push origin test-branch-protection

# Create PR in GitHub UI
# Expected: "Checks have failed" message, merge blocked
```

**Common Issues:**

| Issue | Cause | Fix |
|-------|-------|-----|
| "Required checks not found" | Workflow hasn't run yet | Push a commit to trigger workflow first |
| "Merge anyway" button visible | "Do not allow bypassing" not enabled | Re-check protection rule settings |
| Checks not blocking | Wrong check names in protection rule | Use exact names: `conversation-quality`, `documentation-alignment` |

### Configuration Details

See `.github/workflows/qa-tests.yml` for complete workflow:
- Lines 1-8: Trigger configuration (push/PR to main/develop)
- Lines 10-48: Conversation quality job with caching
- Lines 50-82: Documentation alignment job
- Lines 84-126: Summary job with pass/fail logic

---

## Documentation Quality Standards

### 1. Single Source of Truth (SSOT) Principle

**Rule**: Master documentation in `docs/context/` is authoritative. All other docs MUST:
- Cross-reference master docs, never duplicate
- Describe implementation details ("how we built it"), not behavior ("what it does")
- Use actual function/file names from code, not conceptual terms

**Example - ❌ Wrong**:
```markdown
# Some Feature Doc
The system uses classify_intent to understand user queries.
```

**Example - ✅ Right**:
```markdown
# Some Feature Doc
The system uses `classify_query()` (see src/flows/query_classification.py)
to understand user intent, as described in docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md.
```

---

### 2. Code-First Documentation Updates

**Rule**: When code changes, documentation MUST be updated in the same commit.

**Process**:
1. Developer changes function name: `classify_intent` → `classify_query`
2. Same commit updates:
   - Master docs: `docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md`
   - Implementation docs: Any doc that references the function
   - Tests: Update `test_conversation_flow_documented_correctly()`
3. CI fails if docs not updated (caught by alignment tests)

---

### 3. Documentation Hygiene Checklist

**Before adding new documentation**, verify:

- [ ] Can this information be added to existing doc instead of creating new file?
- [ ] Does this doc cross-reference master docs for behavior/concepts?
- [ ] Have I used actual function names (not conceptual placeholders)?
- [ ] Have I included file paths to referenced code?
- [ ] Have I added this doc to the README structure guide?
- [ ] If creating feature doc, does CHANGELOG.md reference it?

**Before updating existing documentation**:

- [ ] Am I updating master docs? (Requires extra review - these are source of truth)
- [ ] Do code references still point to correct files/lines?
- [ ] Do cross-references to other docs still work?
- [ ] Have I tested that code examples still run?

---

### 4. Master Documentation Update Process

**Special care for `docs/context/` files** (these guide Copilot and developers):

1. **Propose change**: Create issue explaining what's outdated and why
2. **Review actual code**: Verify current implementation before documenting
3. **Update master doc**: Use actual function names, file paths, current behavior
4. **Update alignment tests**: If structure changed, update test expectations
5. **PR review**: Requires 2 approvals for master doc changes
6. **Copilot verification**: Test that Copilot references updated content correctly

---

### 5. The 93% Alignment Philosophy: When Is It Acceptable?

**Question**: "Why is 93% documentation alignment considered excellent when we could aim for 100%?"

**Answer**: Because not all documentation serves the same purpose, and enforcing 100% alignment creates unnecessary overhead for historical/retrospective content.

---

#### Documentation Alignment Tiers

| Doc Category | Required Alignment | Why | Examples |
|--------------|-------------------|-----|----------|
| **Tier 1: Operational** | 🔴 **100%** (Strict) | Guides daily development decisions | `docs/context/`, `docs/features/`, `docs/setup/` |
| **Tier 2: Historical** | 🟡 **80%** (Light) | Informational only, doesn't affect future code | `docs/analysis/`, `docs/implementation/` |

---

#### Tier 1: Operational Docs (100% Alignment Required)

**What they are**:
- **Master context docs** (`docs/context/`): Define system behavior, architecture, personality
- **Feature implementation docs** (`docs/features/`): Explain how features work, what code to modify
- **Setup guides** (`docs/setup/`): Installation, configuration, deployment instructions

**Why strict alignment matters**:
- Developers follow these to write code → Must match reality
- AI (Copilot) references these for suggestions → Must be accurate
- New team members onboard from these → Outdated info wastes time
- Feature modifications rely on these → Wrong info breaks things

**Examples of critical alignment**:
```markdown
✅ GOOD: "The `classify_query()` function (line 45) determines intent"
❌ BAD: "The classify_intent() function determines intent" (function doesn't exist)

✅ GOOD: "System uses temperature=0.4 for balanced responses"
❌ BAD: "System uses temperature=0.7" (outdated config)

✅ GOOD: "Pipeline: classify → retrieve → generate → plan → execute"
❌ BAD: "Pipeline: classify → generate" (missing 3 nodes)
```

**QA Enforcement**:
- ✅ 12 alignment tests verify these docs match code
- ✅ CI/CD blocks merges if alignment tests fail
- ✅ Quarterly audits check for drift
- ✅ Pre-commit hooks warn on changes without doc updates

---

#### Tier 2: Historical Docs (80% Alignment Acceptable)

**What they are**:
- **Analysis docs** (`docs/analysis/`): "Why we chose Vercel over AWS Lambda"
- **Implementation reports** (`docs/implementation/`): "What we shipped in October 2025"
- **Retrospectives**: "Lessons learned from performance refactor"

**Why light alignment is OK**:
- These are **retrospective** (describe past decisions, not current behavior)
- These are **informational** (provide context, don't guide future work)
- These are **snapshots** (frozen in time, intentionally don't evolve)
- Maintaining 100% alignment creates overhead with little value

**Examples of acceptable drift**:
```markdown
✅ ACCEPTABLE: "STREAMLIT_VS_VERCEL_ANALYSIS.md mentions old Streamlit setup"
   → Document explains historical decision, not current implementation

✅ ACCEPTABLE: "CODE_READABILITY_COMPARISON.md references old file structure"
   → Document is retrospective analysis, not setup guide

✅ ACCEPTABLE: "SYSTEM_COMPLETION_REPORT_2025-10.md lists features from October"
   → Document is historical record, intentionally doesn't update monthly
```

**QA Approach**:
- ⚠️ Mentioned in QA_STRATEGY.md but no detailed checklists
- ⚠️ No alignment tests (changes won't break anything)
- ⚠️ No CI/CD enforcement (retrospectives don't guide new code)
- ✅ Quarterly audit reviews for context (but doesn't require updates)

---

#### Decision Matrix: When to Enforce Alignment

**Use this when deciding if a new doc needs strict alignment**:

| Question | Yes → Tier 1 (100%) | No → Tier 2 (80%) |
|----------|---------------------|-------------------|
| Will developers follow this to write code? | Master docs, feature docs, setup guides | Analysis docs, completion reports |
| Does it describe current behavior? | Yes → Strict alignment | No → Historical snapshot |
| Will AI reference this for suggestions? | Yes → Must be accurate | No → Context only |
| Does it include code file paths/function names? | Yes → Must match reality | Maybe → Outdated OK |
| Will outdated info break something? | Yes → Strict enforcement | No → Informational |

---

#### The 93% Sweet Spot

**Current Alignment Status** (as of Oct 16, 2025):
- ✅ `docs/context/`: 100% (5/5 files verified)
- ✅ `docs/features/`: 100% (6/6 files verified)
- ✅ `docs/setup/`: 100% (3/3 files verified)
- ⚠️ `docs/analysis/`: 80% (5 files mentioned, not detailed)
- ⚠️ `docs/implementation/`: 80% (2 files mentioned, not detailed)

**= 93% overall alignment** ✅

**Why 93% is excellent**:
- **Everything that guides future work is 100% aligned**
- **Everything that's historical context is loosely mentioned**
- **Zero broken functional links** (all references work)
- **Minimal maintenance overhead** (don't update retrospectives)

**When to push for 100%**:
- If Tier 2 docs are being referenced as operational guides → Upgrade to Tier 1
- If analysis docs contain setup instructions → Extract to setup guide
- If implementation reports duplicate current behavior → Remove duplication

---

#### Real-World Example: Analytics Consolidation

**Scenario**: Consolidated `LIVE_ANALYTICS_IMPLEMENTATION.md` + `DATA_ANALYTICS_ENHANCEMENT.md` → `ANALYTICS_IMPLEMENTATION.md`

**Tier 1 Action Required** ✅:
1. Update all operational doc references (ENTERPRISE_ADAPTATION_GUIDE.md, LEARNING_GUIDE_COMPLETE_SYSTEM.md)
2. Fix broken links (93% → 95%)
3. Verify alignment tests still pass
4. Commit with QA-compliant message

**Tier 2 Action NOT Required** ⏭️:
- Don't update old retrospectives that mentioned the old doc name
- Don't create alignment tests for historical analysis docs
- Don't enforce 100% coverage of every mention in blog posts/notes

**Result**: 95% functional alignment with minimal overhead ✅

---

#### Summary: When Is 93% Acceptable?

| Alignment % | What It Means | When Acceptable | When Not |
|-------------|---------------|-----------------|----------|
| **100%** | All docs match code perfectly | Tier 1 operational docs only | Not needed for retrospectives |
| **93-95%** | Core docs aligned, historical loosely mentioned | ✅ **Current state - excellent!** | If Tier 1 docs have drift |
| **<90%** | Functional links broken or Tier 1 drift | ❌ Not acceptable | Fix immediately |

**Action Items**:
- ✅ Maintain 100% alignment for `docs/context/`, `docs/features/`, `docs/setup/`
- ✅ Keep functional links working (no 404s)
- ⏭️ Don't enforce strict alignment for `docs/analysis/`, `docs/implementation/`
- ⏭️ Don't create alignment tests for historical docs

**Philosophy**: *Enforce what matters, document what happened.*

---

## Code Quality Standards

**Last Updated**: October 16, 2025
**Status**: ✅ Standards defined, cleanup in progress (Phase 1.5)

### Production Code Requirements

These standards ensure code is production-ready for serverless deployment (Vercel), observable by monitoring tools (LangSmith), and maintainable by the team.

---

### 1. Logging over Print Statements

**Standard**: Use structured logging (`logger.info()`) not console prints (`print()`).

**Why This Matters**:
- ✅ **Vercel deployment**: `print()` statements may not appear in serverless function logs
- ✅ **Log levels**: Can filter by severity (INFO, WARNING, ERROR, DEBUG)
- ✅ **Observability**: LangSmith and monitoring tools expect structured logs
- ✅ **Performance**: Print is synchronous and can block in high-traffic scenarios
- ✅ **Production debugging**: Can dynamically change log levels without redeployment

**✅ Correct Usage**:
```python
import logging

logger = logging.getLogger(__name__)

def retrieve_chunks(query: str):
    logger.info(f"Retrieving chunks for query: {query[:50]}...")
    chunks = search_pgvector(query)
    logger.info(f"Retrieved {len(chunks)} chunks in {latency_ms}ms")
    return chunks
```

**❌ Avoid**:
```python
def retrieve_chunks(query: str):
    print(f"Found {len(chunks)} chunks")  # Won't appear in Vercel logs
    return chunks
```

**Exception**: CLI scripts and one-time migration tools can use `print()` for user feedback.

---

### 2. Configuration over Hardcoding

**Standard**: Use `supabase_settings` for all paths and configuration values.

**Why This Matters**:
- ✅ **Environment flexibility**: Different paths for dev/staging/production
- ✅ **Docker compatibility**: Paths change in containerized environments
- ✅ **Testing**: Can use temporary directories in tests
- ✅ **Configuration**: Single source of truth in `.env` file
- ✅ **Security**: Sensitive values in environment, not committed to git

**✅ Correct Usage**:
```python
from src.config.supabase_config import supabase_settings

def save_confession(confession: str):
    path = supabase_settings.confessions_path  # Configurable via env
    with open(path, 'a') as f:
        f.write(confession)
```

**❌ Avoid**:
```python
def save_confession(confession: str):
    path = "data/confessions.csv"  # Hardcoded, breaks in Docker
    with open(path, 'a') as f:
        f.write(confession)
```

**Exception**: Default parameters with override capability are acceptable:
```python
def __init__(self, persistence_file: str = "data/session_memory.json"):
    self.file = persistence_file  # Caller can override
```

---

### 3. Environment Awareness

**Standard**: Check deployment environment before making assumptions about filesystem, resources, or behavior.

**Why This Matters**:
- ✅ **Vercel limitations**: No persistent filesystem, 10s function timeout, 50MB memory
- ✅ **Local development**: Different paths, unlimited time, full filesystem access
- ✅ **Production safety**: Avoid resource-intensive operations in serverless
- ✅ **Cost optimization**: Use different models or cache strategies per environment

**✅ Correct Usage**:
```python
from src.config.supabase_config import supabase_settings

def expensive_analytics_query():
    if supabase_settings.is_production:
        # Use cached results in production (fast, cheap)
        return get_cached_analytics()
    else:
        # Run full query in dev (accurate, slower)
        return run_full_analytics_query()
```

**❌ Avoid**:
```python
def expensive_analytics_query():
    # Always runs 5-minute query, times out in Vercel
    return run_full_analytics_query()
```

**Environment Detection**:
```python
# In src/config/supabase_config.py
class SupabaseSettings:
    def __init__(self):
        self.is_vercel = os.getenv("VERCEL") is not None
        self.is_production = os.getenv("ENVIRONMENT") == "production"
        self.is_development = not self.is_production
```

---

### Current Code Quality Status (Phase 1.5)

**Audit Completed**: October 16, 2025

#### Print Statements Identified (8 instances in 6 files)

**Priority 1 - Core Production Code (HIGH IMPACT)**:
- [ ] `src/core/retrieval/pgvector_adapter.py:48` - Retrieval pipeline (every query hits this)
- [ ] `src/utils/embeddings.py:16` - Embedding generation (core RAG)

**Priority 2 - Service Layer (MEDIUM IMPACT)**:
- [ ] `src/services/twilio_service.py:315` - SMS delivery status
- [ ] `src/services/storage_service.py:261, 312` - Storage operations

**Priority 3 - Analytics (LOW IMPACT)**:
- [x] `src/analytics/feedback_test_generator.py` - Archived (unused, replaced by LangSmith evaluation pipeline)
- [ ] `src/analytics/code_display_monitor.py:233, 236` - Monitoring scripts

**Implementation Plan**:
1. Fix Priority 1 first (core retrieval)
2. Add logging configuration to `supabase_config.py`
3. Add tests using pytest `caplog` fixture
4. Fix Priority 2 and 3 in batch PRs
5. Enable strict pre-commit hook (currently commented out in `.pre-commit-config.yaml` lines 37-53)

#### Hardcoded Paths Identified (1 requires fix)

- [ ] `src/main.py:273` - Confession storage path (should use `supabase_settings.confessions_path`)

**Note**: Other findings were false positives (docstring examples, config file itself, or prompt text).

---

### Enforcement Strategy

#### Pre-Commit Hook (Planned - After Cleanup)

**File**: `.pre-commit-config.yaml` (currently commented out, lines 37-53)

```yaml
# Code quality checks (strict mode - enable after Phase 1.5 cleanup)
- repo: local
  hooks:
    - id: check-print-statements
      name: Check for print() in production code
      entry: bash -c 'grep -rn "^[^#]*print(" src/ --exclude-dir=__pycache__ && exit 1 || exit 0'
      language: system
      pass_filenames: false
```

**When to Enable**: After all print() statements migrated to logger.

---

#### Automated Test (Planned - After Cleanup)

**File**: `tests/test_code_quality.py` (to be created)

```python
"""Test code quality standards (production readiness)."""

def test_no_print_statements_in_production_code():
    """Verify no print() statements in src/ directory."""
    # Scans src/ for print() calls, fails if found
    # See full implementation in code cleanup planning docs
```

**Test Count Impact**: 30 tests → 32 tests (adds 2 code quality tests)

---

### Migration Guide

**For Contributors**: If you need to log output:

```python
# ❌ DON'T DO THIS
print(f"Processing {count} items...")

# ✅ DO THIS INSTEAD
import logging
logger = logging.getLogger(__name__)
logger.info(f"Processing {count} items...")

# For debugging during development (remove before commit):
logger.debug(f"Debug info: {variable}")  # Won't appear in production logs (level=INFO)
```

**For Reviewers**: In code review, check for:
- [ ] No `print()` statements in `src/` directory
- [ ] Paths use `supabase_settings` not hardcoded strings
- [ ] Environment-specific logic checks `is_production` or `is_vercel`
- [ ] Tests use `caplog` fixture to verify logger output

---

### Related Documentation

- **Cleanup Progress**: See [Current Test Status](#current-test-status) → Error Handling (6 tests, 100%)
- **Testing Standards**: See [Testing Best Practices & Common Issues](#testing-best-practices--common-issues)
- **Pre-Commit Hooks**: See [Pre-Commit Hooks](#pre-commit-hooks) section above

---

## Error Handling & Resilience Standards

**Last Updated**: October 17, 2025
**Status**: ✅ Standards defined, test suite implemented (5 core tests)

### Overview

**Purpose**: Ensure Portfolia degrades gracefully under failure conditions, maintaining conversation quality even when external services fail.

**Philosophy**: **Never crash on user** — better to continue conversation with reduced features than show technical errors.

**Audit Context**: Comprehensive audit (October 17, 2025) found:
- ✅ **Code quality excellent**: Comprehensive try/except coverage across services, RAG engine, API endpoints
- ⚠️ **Documentation gap**: Error handling patterns not formalized in QA policy
- 📊 **Test coverage**: 6 core error handling tests added (76/77 passing, 99%)

**Related Documentation**: See `docs/archive/analysis/QA_AUDIT_FINDINGS_ERROR_HANDLING.md` for full audit report

---

### Core Principles

#### 1. Never Crash on User
**Standard**: Conversation flow MUST continue even when services fail.

**Bad** ❌:
```python
def send_sms(phone, message):
    client = Twilio(api_key)  # Crashes if API key missing
    client.messages.create(...)
```

**Good** ✅:
```python
def send_sms(phone, message):
    twilio = get_twilio_service()  # Returns None if unavailable
    if twilio:
        try:
            twilio.send_sms(phone, message)
            logger.info(f"SMS sent to {phone}")
        except TwilioRestException as e:
            logger.error(f"SMS failed: {e}")
            # Conversation continues, just no SMS
    else:
        logger.warning("Twilio unavailable, skipping SMS")
```

**Test**: `test_conversation_without_twilio()` ✅

---

#### 2. Graceful Degradation
**Standard**: Return polite user-facing errors, not technical stack traces.

**Bad** ❌:
```python
# API returns raw exception to user
{
    "success": false,
    "error": "NoneType object has no attribute 'execute'"
}
```

**Good** ✅:
```python
# API returns helpful message
{
    "success": false,
    "error": "Unable to retrieve data at this time. Please try again in a moment."
}
```

**Implementation Pattern**:
```python
try:
    result = database.query(...)
except Exception as e:
    logger.error(f"Database error: {e}", exc_info=True)  # Log full trace
    return {"success": False, "error": "Service temporarily unavailable"}
```

**Test**: `test_email_validation()` ✅

---

#### 3. Observable Failures
**Standard**: All errors MUST be logged with sufficient context for debugging.

**Required Context**:
- **What** failed (service name, function, operation)
- **Why** it failed (error type, message)
- **When** it failed (timestamp, trace_id)
- **Impact** (user affected? data lost? conversation degraded?)

**Example**:
```python
try:
    chunks = retriever.retrieve(query)
except Exception as e:
    logger.error(
        f"Retrieval failed",
        extra={
            "query": query[:100],
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "trace_id": trace_id,
            "session_id": session_id
        },
        exc_info=True
    )
```

**Observability Tools**:
- **LangSmith**: Traces all LLM calls, embeddings, retrievals
- **Vercel Logs**: Captures structured logs from API endpoints
- **Supabase Analytics**: Tracks conversation success/failure rates

---

#### 4. Fail-Fast on Startup
**Standard**: Invalid configuration MUST prevent deployment, not silently fail at runtime.

**Example** (Supabase Config Validation):
```python
# src/config/supabase_config.py
def validate_supabase(self) -> None:
    """Validate Supabase config on initialization."""
    if not self.supabase_config.url:
        raise ValueError(
            "SUPABASE_URL not set. Add to .env:\n"
            "SUPABASE_URL=https://xxx.supabase.co"
        )

    if '\n' in self.supabase_config.service_role_key:
        raise ValueError(
            "SUPABASE_SERVICE_ROLE_KEY contains newlines. "
            "Remove trailing newlines in .env file."
        )
```

**Why**: Prevents "works locally, fails in production" scenarios.

**Test**: Manual verification during CI/CD (config validation runs on import)

---

#### 5. Defensive Coding
**Standard**: Validate inputs, check for None, handle edge cases proactively.

**Checklist**:
- [ ] User inputs sanitized (XSS, SQL injection, path traversal)
- [ ] API responses validated before processing
- [ ] None checks before attribute access
- [ ] Empty collection handling (empty list, empty string)
- [ ] Length limits enforced (queries, emails, file uploads)

**Example**:
```python
def extract_email(query: str) -> Optional[str]:
    """Extract email from query with validation."""
    if not query or len(query) > 10000:
        return None

    match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', query)
    if not match:
        return None

    email = match.group(0)

    # Sanitize for XSS
    if '<' in email or '>' in email or 'script' in email.lower():
        logger.warning(f"Suspicious email rejected: {email}")
        return None

    return email
```

**Test**: `test_email_validation()` ✅

---

### Service Layer Standards

#### Factory Pattern for Service Initialization

**Standard**: All external service clients MUST use factory functions that return `None` on failure.

**Pattern**:
```python
# src/services/[service]_service.py
def get_[service]_service() -> Optional[ServiceClient]:
    """Factory returns None if credentials missing."""
    api_key = os.getenv("SERVICE_API_KEY")

    if not api_key:
        logger.warning("SERVICE_API_KEY not set, service disabled")
        return None

    try:
        return ServiceClient(api_key=api_key)
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        return None
```

**Usage**:
```python
# api/feedback.py
service = get_service()
if service:
    service.perform_action()
else:
    return {"success": False, "error": "Service unavailable"}
```

**Services Using This Pattern** ✅:
- `get_twilio_service()` - SMS alerts
- `get_resend_service()` - Email distribution
- `get_storage_service()` - Resume uploads
- `get_supabase_client()` - Database operations

**Test**: `test_conversation_without_twilio()`, `test_conversation_without_resend()` ✅

---

#### Service Error Handling Checklist

For each external service integration:
- [ ] Factory function returns None if credentials missing
- [ ] Specific exception handling (e.g., `TwilioRestException`, `ResendAPIError`)
- [ ] Errors logged with service name and operation
- [ ] Conversation continues if service fails (degraded mode)
- [ ] User receives polite error message (no technical details)

**Example** (Twilio Service):
```python
# src/services/twilio_service.py
class TwilioService:
    def send_sms(self, to: str, message: str) -> bool:
        """Send SMS, return False on failure."""
        try:
            self.client.messages.create(to=to, body=message, from_=self.from_number)
            return True
        except TwilioRestException as e:
            logger.error(f"Twilio SMS failed: {e.msg}", extra={
                "to": to[:3] + "***",  # Privacy: mask phone number
                "error_code": e.code,
                "status": e.status
            })
            return False
```

---

### API Endpoint Standards

#### Structured Error Responses

**Standard**: All API endpoints MUST return consistent error structure with HTTP status codes.

**Response Schema**:
```python
# Success
{
    "success": true,
    "data": {...},
    "timestamp": "2025-10-17T10:30:00Z"
}

# Error
{
    "success": false,
    "error": "User-friendly error message",
    "error_code": "INVALID_INPUT",  # Optional: machine-readable code
    "timestamp": "2025-10-17T10:30:00Z"
}
```

**HTTP Status Codes**:
| Status | When | Example |
|--------|------|---------|
| **200** | Success | Query processed, response generated |
| **400** | Client error | Invalid JSON, missing required fields |
| **401** | Unauthorized | Invalid API key (future) |
| **429** | Rate limited | Too many requests (future) |
| **500** | Server error | Unexpected exception, service down |

**Implementation Pattern**:
```python
# api/chat.py
class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Parse request
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)

            # Validate required fields
            if 'query' not in data:
                return self._send_error(400, "Missing required field: query")

            # Process request
            result = process_query(data)
            return self._send_json(200, {"success": True, "data": result})

        except json.JSONDecodeError:
            return self._send_error(400, "Invalid JSON in request body")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return self._send_error(500, "Internal server error")
```

**Test**: `test_invalid_json_in_api()` ✅

---

#### API Validation Checklist

For each API endpoint:
- [ ] JSON parsing errors → 400 error with "Invalid JSON" message
- [ ] Missing required fields → 400 error with field name
- [ ] Invalid field values → 400 with validation error
- [ ] Service unavailable → 500 with "Service temporarily unavailable"
- [ ] Unexpected exceptions → 500 with generic message + full log
- [ ] CORS preflight handling (OPTIONS requests)

---

### Conversation Flow Standards

#### Node Error Handling

**Standard**: Conversation nodes MUST NOT raise exceptions that crash the pipeline.

**Bad** ❌:
```python
def retrieve_chunks(state: ConversationState, rag_engine: RagEngine):
    """Retrieve chunks - CRASHES if rag_engine is None."""
    chunks = rag_engine.retrieve(state.query)  # No None check
    return state.set_chunks(chunks)
```

**Good** ✅:
```python
def retrieve_chunks(state: ConversationState, rag_engine: RagEngine):
    """Retrieve chunks with error handling."""
    try:
        if not rag_engine:
            logger.error("RAG engine not initialized")
            return state.set_error("retrieval_failed")

        chunks = rag_engine.retrieve(state.query)
        logger.info(f"Retrieved {len(chunks)} chunks")
        return state.set_chunks(chunks)

    except Exception as e:
        logger.error(f"Retrieval error: {e}", exc_info=True)
        return state.set_error("retrieval_failed")
```

**Pipeline Behavior**:
```python
# src/flows/conversation_flow.py
def run_conversation_flow(state, rag_engine, session_id):
    """Run pipeline with error handling."""
    for node_fn in pipeline:
        try:
            state = node_fn(state, rag_engine)

            # Check for error flag (graceful degradation)
            if state.has_error:
                logger.warning(f"Node {node_fn.__name__} set error flag")
                # Continue pipeline with degraded state

        except Exception as e:
            logger.error(f"Node {node_fn.__name__} crashed: {e}", exc_info=True)
            # Set error state, continue to next node
            state = state.set_error(node_fn.__name__)

    return state
```

**Test**: `test_conversation_without_supabase()` (Phase 2)

---

### Input Validation Standards

#### User Input Sanitization

**Standard**: All user inputs MUST be validated and sanitized before processing.

**Validation Rules**:

| Input Type | Max Length | Allowed Characters | Sanitization |
|------------|-----------|-------------------|--------------|
| **Query** | 10,000 chars | Any UTF-8 | Truncate if exceeded |
| **Email** | 320 chars | Email regex | Reject if XSS detected |
| **Phone** | 20 chars | Digits + `+()-` | Reject if invalid format |
| **Name** | 100 chars | Letters, spaces, hyphens | Strip special chars |

**Example** (Email Validation):
```python
def validate_email(email: str) -> Optional[str]:
    """Validate and sanitize email."""
    if not email or len(email) > 320:
        return None

    # XSS check
    if any(char in email for char in ['<', '>', 'script', 'javascript']):
        logger.warning(f"Suspicious email rejected: {email}")
        return None

    # Format validation
    if not re.match(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$', email):
        return None

    return email.strip().lower()
```

**Security Threats Prevented**:
- ✅ **XSS attacks**: `<script>alert('xss')</script>` rejected
- ✅ **SQL injection**: `test'; DROP TABLE users; --` rejected
- ✅ **Path traversal**: `../../etc/passwd` rejected
- ✅ **Buffer overflow**: 50,000 char queries truncated

**Test**: `test_email_validation()` ✅

---

#### Edge Case Handling

**Standard**: System MUST handle edge cases gracefully.

**Common Edge Cases**:
```python
# Empty inputs
query = ""  # → Return helpful "Ask me anything" message, not error
chunks = []  # → Generate response from chat history only

# Very long inputs
query = "a" * 50000  # → Truncate to 10,000 chars, log warning

# Special characters
query = "💻🚀🔥"  # → Process normally (UTF-8 supported)

# Malformed data
email = "notanemail"  # → Reject politely: "Invalid email format"

# Null values
user_name = None  # → Use "there" as fallback in greetings
```

**Test**: `tests/test_code_display_edge_cases.py` (6 tests covering XSS, SQL injection, path traversal) ✅

---

### RAG Pipeline Resilience

#### Embedding Failures

**Standard**: Empty embedding returns MUST NOT crash retrieval.

**Implementation** (`src/retrieval/pgvector_retriever.py`):
```python
def embed(self, text: str) -> List[float]:
    """Generate embedding, return empty list on failure."""
    try:
        if not text or not text.strip():
            logger.warning("Empty text for embedding")
            return []

        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return []  # Never raise
```

**Fallback Behavior**:
- Empty embedding → Skip retrieval → LLM generates from chat history only
- User sees response (may be less accurate), not error message

**Test**: `test_openai_rate_limit_handling()` ✅

---

#### Retrieval Failures

**Standard**: Retrieval errors MUST return empty list, not raise.

**Implementation**:
```python
def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """Retrieve chunks, return empty list on failure."""
    try:
        embedding = self.embed(query)
        if not embedding:
            logger.warning("Empty embedding, returning no results")
            return []

        response = self.supabase.rpc('match_documents', {
            'query_embedding': embedding,
            'match_count': top_k
        }).execute()

        return response.data or []

    except Exception as e:
        logger.error(f"Retrieval failed: {e}", exc_info=True)
        return []  # Graceful degradation
```

**Client-Side Fallback**:
```python
# src/core/rag_engine.py
def generate_response(self, query: str, chat_history: List):
    """Generate response with fallback if retrieval fails."""
    chunks = self.retrieve(query)

    if not chunks:
        logger.warning("No chunks retrieved, using chat history only")
        # LLM still generates response, just without KB context

    return self.llm.generate(query=query, context=chunks, history=chat_history)
```

---

#### Low-Quality Retrieval Fallback 🆕

**Standard**: When ALL retrieval scores < 0.4, provide helpful fallback message instead of low-quality results.

**Threshold Explanation** (Cosine similarity, 0.0-1.0 scale):
- **1.0** = Perfect match (identical vectors)
- **0.7-1.0** = Good match (use normally)
- **0.4-0.7** = Moderate match (use with caution, may lack relevance)
- **< 0.4** = Poor match (trigger fallback)

**Common Triggers**:
- Typos/misspellings ("buisness" → "business")
- Out-of-domain queries (not in knowledge base)
- Overly generic queries ("tell me everything")

**Implementation** (`src/flows/core_nodes.py:135-152`):
```python
def generate_answer(state: ConversationState, rag_engine: RagEngine) -> ConversationState:
    """Generate answer, with fallback for low-quality retrieval."""

    # Check for very low retrieval quality (all scores below threshold)
    retrieval_scores = state.fetch("retrieval_scores", [])
    if retrieval_scores and all(score < 0.4 for score in retrieval_scores):
        fallback_answer = f"""I'm not finding great matches for "{state.query}" in my knowledge base, but I'd love to help!

Here are some things I can tell you about:
- **Noah's engineering skills and experience** - "What are your software engineering skills?"
- **Production GenAI systems** - "What do you understand about production GenAI systems?"
- **System architecture** - "How do you approach system architecture?"
- **Specific projects** - "What projects have you built?"
- **Technical stack and tools** - "What technologies do you use?"
- **Career background** - "Tell me about your career journey"

Or ask me to explain how I work - I love teaching about RAG, vector search, and LLM orchestration! What sounds interesting?"""

        state.set_answer(fallback_answer)
        state.stash("fallback_used", True)
        logger.info(f"Used fallback for low-quality retrieval (scores: {retrieval_scores})")
        return state

    # Normal response generation
    answer = rag_engine.generate_response(state.query, state.chat_history)
    state.set_answer(answer)
    return state
```

**User Experience**:
- ✅ No error message shown
- ✅ Helpful alternative suggestions provided (role-specific)
- ✅ Maintains conversational tone
- ✅ Encourages rephrasing
- ✅ Acknowledges user's query (echoes back)

**Monitoring**:
- `fallback_used=True` flag in conversation state
- LangSmith trace includes retrieval scores
- Logged to application logs for analysis

**Production Example** (Real screenshot from October 17, 2025):
```
User: "buisness"
System: "I'm not finding great matches for 'buisness' in my knowledge base, but I'd love to help!..."
Scores: [0.35, 0.28]
Result: ✅ User redirected to relevant topics
```

**Test**: `test_low_quality_retrieval_fallback()` ✅

**Why This Matters**:
- Prevents showing irrelevant/confusing responses
- Improves user experience on edge cases
- Guides users to high-quality content
- Builds trust (transparency about limitations)

---

#### LLM Generation Failures

**Standard**: LLM errors MUST return polite fallback message.

**Implementation** (`src/core/response_generator.py`):
```python
def generate(self, query: str, context: List, history: List) -> str:
    """Generate response with fallback."""
    try:
        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=build_messages(query, context, history),
            temperature=0.7
        )
        return response.choices[0].message.content

    except RateLimitError:
        logger.error("OpenAI rate limit hit")
        return (
            "I'm experiencing high volume right now. "
            "Please try again in a moment, or email noah@example.com directly."
        )
    except Exception as e:
        logger.error(f"LLM generation failed: {e}", exc_info=True)
        return (
            "I'm having trouble generating a response. "
            "Please rephrase your question or try again."
        )
```

**Test**: `test_openai_rate_limit_handling()` ✅

---

### Error Handling Test Suite

**File**: `tests/test_error_handling.py` (400 lines, 15 tests)

**Current Status**: ✅ 5/5 core tests passing (100%)

#### Required Tests (Priority 1 - Implemented)

| Test | Purpose | Pass Criteria | Status |
|------|---------|---------------|--------|
| `test_conversation_without_twilio` | Service degradation | Conversation continues, no crash | ✅ PASSING |
| `test_conversation_without_resend` | Service degradation | Polite error message to user | ✅ PASSING |
| `test_openai_rate_limit_handling` | LLM failure | Fallback response provided | ✅ PASSING |
| `test_email_validation` | Input sanitization | Malicious input rejected politely | ✅ PASSING |
| `test_invalid_json_in_api` | API validation | 400 error with helpful message | ✅ PASSING |

#### Additional Tests (Priority 2 - Phase 2)

| Test | Purpose | Pass Criteria | Status |
|------|---------|---------------|--------|
| `test_conversation_without_supabase` | Database failure | Fallback response provided | ⬜ TODO |
| `test_query_length_limits` | Input validation | Very long queries truncated | ⬜ TODO |
| `test_missing_required_fields` | API validation | 400 error with field name | ⬜ TODO |
| `test_unauthorized_access` | Security | 401 error returned | ⬜ TODO |
| `test_rate_limiting` | Abuse prevention | 429 error after threshold | ⬜ TODO |
| `test_concurrent_requests` | Thread safety | No race conditions | ⬜ TODO |
| `test_memory_leak_prevention` | Resource management | No memory growth over time | ⬜ TODO |
| `test_timeout_handling` | Performance limits | Slow queries interrupted | ⬜ TODO |
| `test_cors_preflight` | CORS handling | OPTIONS request handled | ⬜ TODO |
| `test_logging_on_errors` | Observability | Errors logged with context | ⬜ TODO |

**Target**: 15/15 tests passing (100%) by Phase 2 completion

**Run**: `pytest tests/test_error_handling.py -v`

---

### Production Error Monitoring (Phase 2)

**Tool**: LangSmith + Custom Monitoring

**Status**: 🔜 Planned for Phase 2 (Q1 2025)

#### Alert Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| **Error rate** | >5% of requests | Email + Slack alert |
| **Service unavailable** | >10% of requests | Page on-call |
| **Response latency** | p95 >5s | Email summary |
| **OpenAI rate limit** | >10 hits/day | Email warning |
| **Embedding failures** | >5% of retrievals | Slack alert |
| **Daily cost** | >$10 | Email budget alert |

#### Daily Automated Report

**Delivered**: Email to engineering team (8am PT)

**Contents**:
- Total queries processed
- Error rate and top error types
- Service availability metrics
- Performance metrics (p50, p95, p99 latency)
- Cost analysis (tokens, $ per query)
- Recommended actions

**Example Report**:
```
📊 Error Handling Report - October 17, 2025

✅ Overall Status: HEALTHY

Metrics (24h):
  - 234 queries processed
  - 1.2s avg latency (p95: 2.1s)
  - $0.45 total cost ($0.0019/query)
  - 2 errors (0.8%)

Service Availability:
  ✅ Twilio: 100% (12 SMS sent)
  ✅ Resend: 100% (5 emails sent)
  ✅ OpenAI: 99.2% (2 rate limit hits)
  ✅ Supabase: 100%

Error Breakdown:
  - OpenAI RateLimitError: 2 (0.8%)
    → Action: Implement exponential backoff

Top Queries:
  1. "explain conversation nodes" - 45 times
  2. "show me code examples" - 32 times
  3. "what are noah's skills" - 28 times

Recommendations:
  - OpenAI rate limits detected → Consider caching strategy
  - No service outages → Continue current approach
```

#### LangSmith Integration

**Configuration** (`.env`):
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=noahs-ai-assistant
```

**What Gets Traced**:
- All LLM calls (prompt, response, latency, tokens)
- All embedding calls (text, vector, latency)
- All retrieval calls (query, chunks, scores)
- All errors (exception type, message, stack trace)

**Quality Checks** (via `scripts/quality_monitor.py`):
```python
def check_langsmith_traces():
    """Check production traces for quality violations."""
    client = get_langsmith_client()
    runs = client.list_runs(project_name="noahs-ai-assistant", start_time=datetime.now() - timedelta(hours=24))

    violations = []
    for run in runs:
        answer = run.outputs.get("answer", "")

        # Check for markdown headers (our policy!)
        if re.search(r'#{1,6}\s', answer):
            violations.append({
                "type": "emoji_headers",
                "trace_id": run.id,
                "query": run.inputs.get("query", ""),
                "response": answer[:200]
            })

    return violations
```

**Cost**: $39/month for 100k traces (Team tier)

---

### Migration Guide for Developers

#### Adding Error Handling to Existing Code

**Step 1: Identify Failure Points**
```python
# Before (crashes on failure)
def send_notification(user_email: str):
    client = EmailService()  # Crashes if API key missing
    client.send(user_email, "Your resume has been sent!")
```

**Step 2: Add Factory Pattern**
```python
# After (graceful degradation)
def send_notification(user_email: str):
    service = get_email_service()  # Returns None if unavailable
    if service:
        try:
            service.send(user_email, "Your resume has been sent!")
            logger.info(f"Notification sent to {user_email}")
        except Exception as e:
            logger.error(f"Notification failed: {e}")
            # User doesn't get email, but conversation continues
    else:
        logger.warning("Email service unavailable, skipping notification")
```

**Step 3: Add Test**
```python
# tests/test_error_handling.py
def test_notification_without_email_service():
    """Test conversation works even if email service down."""
    with patch('src.services.email_service.get_email_service', return_value=None):
        # Simulate user requesting resume
        state = ConversationState(query="send me your resume")
        result = run_conversation_flow(state, rag_engine)

        # Conversation continues
        assert result.answer
        assert "resume" in result.answer.lower()
        # User gets response, just no email notification
```

---

### Checklist for Code Reviewers

When reviewing code that involves external services or user input:

**Service Integration**:
- [ ] Factory function returns None if credentials missing
- [ ] Specific exception types caught (not bare `except:`)
- [ ] Errors logged with service name and context
- [ ] Conversation continues if service fails (no crash)
- [ ] User receives polite error message (no technical details)

**Input Validation**:
- [ ] User input sanitized for XSS, SQL injection, path traversal
- [ ] Length limits enforced (query, email, file uploads)
- [ ] None checks before attribute access
- [ ] Empty string/collection handling

**API Endpoints**:
- [ ] JSON parsing errors return 400 status
- [ ] Missing required fields return 400 with field name
- [ ] Unexpected exceptions return 500 with generic message
- [ ] Full stack traces logged (not exposed to user)
- [ ] CORS preflight handling (OPTIONS requests)

**Testing**:
- [ ] Error handling test added to `tests/test_error_handling.py`
- [ ] Test covers failure scenario (mocked service returns None/raises)
- [ ] Test verifies conversation continues (no crash)
- [ ] Test runs in CI/CD pipeline

---

### Related Documentation

- **Error Handling Audit**: `docs/archive/analysis/QA_AUDIT_FINDINGS_ERROR_HANDLING.md` (856 lines, comprehensive analysis)
- **Test Implementation**: `tests/test_error_handling.py` (400 lines, 15 tests)
- **Service Patterns**: See individual services in `src/services/` (Twilio, Resend, Storage)
- **Production Monitoring (Phase 2)**: See [Phase 2: Production Monitoring with LangSmith](#phase-2-production-monitoring-with-langsmith)
- **Observability Guide**: `docs/platform_operations.md` (current), `docs/archive/legacy/OBSERVABILITY_LEGACY.md` (historical)

---

## Quarterly Documentation Audit

**Schedule**: Every 3 months (January, April, July, October)

**Checklist**:

### 1. Run Full Alignment Test Suite
```bash
pytest tests/test_documentation_alignment.py -v
```
Fix any failures before proceeding.

### 2. Manual Cross-Check
- [ ] Open `docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md`
- [ ] For each function mentioned, CMD+click to verify it exists in codebase
- [ ] For each file path, verify file still exists at that location
- [ ] For each configuration value (temperature, model, etc.), verify matches code

### 3. Master Docs Review
- [ ] PROJECT_REFERENCE_OVERVIEW: Do roles, stack, behavior match reality?
- [ ] SYSTEM_ARCHITECTURE_SUMMARY: Does conversation flow match code?
- [ ] DATA_COLLECTION_AND_SCHEMA_REFERENCE: Do tables, queries match Supabase?
- [ ] CONVERSATION_PERSONALITY: Do greetings match src/flows/greetings.py?

### 4. Feature Docs Review
- [ ] Check `docs/features/` for outdated implementation notes
- [ ] Archive docs for deprecated features to `docs/archive/`
- [ ] Update CHANGELOG.md with any undocumented changes

### 5. Code Reference Validation
```bash
# Find all file references in docs
grep -r "src/.*\.py" docs/ | while read line; do
  file=$(echo "$line" | grep -o "src/[^ ]*\.py")
  if [ ! -f "$file" ]; then
    echo "BROKEN REFERENCE: $line"
  fi
done
```

### 6. Redundancy Check
- [ ] Look for duplicate content (same concept explained in multiple docs)
- [ ] Consolidate or add cross-references
- [ ] Update DOCUMENTATION_CONSOLIDATION_ANALYSIS.md if structure changes

---

## Success Metrics

### Documentation Alignment Metrics

| Metric | Target | Current | How to Measure |
|--------|--------|---------|----------------|
| File reference validity | 100% | - | `test_documentation_file_references_valid` |
| Function name accuracy | 100% | - | `test_conversation_flow_documented_correctly` |
| Role name consistency | 100% | - | `test_role_names_consistent` |
| Config value accuracy | 100% | - | `test_temperature_settings_documented_correctly` |
| Quarterly audit completion | 4/year | 0 | Manual tracking |
| Doc alignment test coverage | 10+ tests | 5 | Count tests in `test_documentation_alignment.py` |

### Conversation Quality Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Automated tests passing | 100% | ✅ 15/15 |
| Manual test checklist items | 100% | TBD |
| User-reported quality issues | <2/month | Track in issues |

---

## Adding Documentation Alignment Tests

### When to Add New Test

Add documentation alignment test when:
1. **New feature added**: Test that feature is documented
2. **Architecture changes**: Test that master docs reflect new structure
3. **Bug from misalignment**: Add test that would have caught it

### Test Template

```python
def test_YOUR_ALIGNMENT_CHECK():
    """Verify [WHAT] matches between docs and code."""

    # 1. Read documentation
    with open("docs/PATH/TO/DOC.md") as f:
        doc_content = f.read()

    # 2. Extract expected value from docs
    import re
    expected = re.search(r'PATTERN', doc_content).group(1)

    # 3. Get actual value from code
    from src.module import function
    import inspect
    actual = inspect.getsource(function)

    # 4. Assert they match
    assert expected in actual, (
        f"Documentation says '{expected}' but code doesn't match. "
        f"Update docs/PATH/TO/DOC.md to reflect current implementation."
    )
```

---

## Preventing Documentation File Misalignment

### The Problem: New .md Files Create Drift

**Every time you add a new .md file**, you risk:
- ❌ Outdated file reference tests (test expects 5 docs, now there are 6)
- ❌ Missing cross-references (new doc not linked from QA_STRATEGY)
- ❌ Documentation fragmentation (content should be in existing doc)
- ❌ Alignment tests don't know about new file

**Example Failure Scenario**:
```bash
# Developer adds docs/features/NEW_FEATURE.md
git add docs/features/NEW_FEATURE.md
git commit -m "Add new feature doc"

# CI runs tests
pytest tests/test_documentation_alignment.py
# ❌ FAILS: test_documentation_file_references_valid
#    doesn't check docs/features/

# ❌ FAILS: New doc never referenced in QA_STRATEGY.md
# ❌ FAILS: No alignment test for new doc's code references
```

---

### Solution: Automated Documentation Registration

#### Step 1: Add Pre-Commit Hook for New .md Files

**File**: `.pre-commit-config.yaml` (create this)

```yaml
repos:
  - repo: local
    hooks:
      # Existing hooks...

      - id: check-new-docs
        name: Check new .md files are registered
        entry: python scripts/check_new_docs.py
        language: system
        files: '^docs/.*\.md$'
        pass_filenames: false
```

**File**: `scripts/check_new_docs.py` (create this)

```python
#!/usr/bin/env python3
"""Pre-commit hook: Check new .md files are properly registered.

Runs when any .md file in docs/ is added/modified.
Ensures:
1. Master docs (docs/context/) are referenced in QA_STRATEGY.md
2. Feature docs (docs/features/) follow naming convention
3. New docs are added to appropriate README.md
4. Alignment tests updated if needed
"""

import os
import sys
import subprocess
from pathlib import Path

def get_staged_md_files():
    """Get list of .md files staged for commit."""
    result = subprocess.run(
        ['git', 'diff', '--cached', '--name-only', '--diff-filter=A'],
        capture_output=True, text=True
    )

    md_files = [
        line for line in result.stdout.split('\n')
        if line.startswith('docs/') and line.endswith('.md')
    ]

    return md_files

def check_master_doc_registration(filepath):
    """Check if new master doc is referenced in QA_STRATEGY.md"""
    if not filepath.startswith('docs/context/'):
        return True  # Not a master doc

    doc_name = Path(filepath).name

    # Check if mentioned in QA_STRATEGY.md
    with open('docs/QA_STRATEGY.md') as f:
        qa_content = f.read()

    if doc_name not in qa_content:
        print(f"""
❌ ERROR: New master doc not registered in QA_STRATEGY.md

You added: {filepath}

ACTION REQUIRED:
1. Add reference to \`docs/QA_STRATEGY.md\` §7 "Master Documentation Update Process"
2. Add to "Quick Reference: Documentation Types" table in QA_STRATEGY.md
3. Consider adding alignment test in tests/test_documentation_alignment.py

Example:
    # In QA_STRATEGY.md
    - {doc_name}: [Description of purpose]

    # In tests/test_documentation_alignment.py
    def test_{doc_name.replace('.md', '').lower()}_exists():
        assert Path("docs/context/{doc_name}").exists()
""")
        return False

    return True

def check_feature_doc_convention(filepath):
    """Check if feature doc follows naming convention."""
    if not filepath.startswith('docs/features/'):
        return True  # Not a feature doc

    doc_name = Path(filepath).stem  # Without .md

    # Convention: FEATURE_NAME_IMPLEMENTATION.md or FEATURE_NAME_SUMMARY.md
    valid_suffixes = ['_IMPLEMENTATION', '_SUMMARY', '_GUIDE']

    if not any(doc_name.endswith(suffix) for suffix in valid_suffixes):
        print(f"""
⚠️  WARNING: Feature doc doesn't follow naming convention

You added: {filepath}

RECOMMENDED:
Feature docs should end with:
- _IMPLEMENTATION.md (implementation details)
- _SUMMARY.md (overview/summary)
- _GUIDE.md (how-to guide)

Example: {doc_name}_IMPLEMENTATION.md

This helps maintain consistency. Continue anyway? (y/n)
""")

        response = input().strip().lower()
        return response == 'y'

    return True

def check_readme_registration(filepath):
    """Check if new doc is mentioned in appropriate README.md"""
    dir_path = Path(filepath).parent
    readme_path = dir_path / 'README.md'

    if not readme_path.exists():
        return True  # No README to update

    doc_name = Path(filepath).name

    with open(readme_path) as f:
        readme_content = f.read()

    if doc_name not in readme_content:
        print(f"""
⚠️  WARNING: New doc not listed in {readme_path}

You added: {filepath}

RECOMMENDED ACTION:
Add to {readme_path} with brief description.

Example:
    - **{doc_name}**: [Brief description of what this doc covers]

Continue without updating README? (y/n)
""")

        response = input().strip().lower()
        return response == 'y'

    return True

def suggest_alignment_test(filepath):
    """Suggest alignment test if doc references code."""
    with open(filepath) as f:
        content = f.read()

    # Check if doc references code files
    has_code_refs = (
        'src/' in content or
        '.py' in content or
        '```python' in content
    )

    if has_code_refs:
        print(f"""
💡 SUGGESTION: Consider adding alignment test

Your new doc ({filepath}) references code files.

OPTIONAL ACTION:
Add alignment test to tests/test_documentation_alignment.py

Example:
    def test_{Path(filepath).stem.lower()}_code_references_valid():
        \"\"\"Verify code references in {Path(filepath).name} are valid.\"\"\"
        with open("{filepath}") as f:
            content = f.read()

        # Extract file paths (e.g., src/module/file.py)
        import re
        file_refs = re.findall(r'`(src/[^`]+\.py)`', content)

        for ref in file_refs:
            assert Path(ref).exists(), f"{{ref}} referenced but doesn't exist"

This prevents broken file references as code evolves.
""")

def main():
    """Main pre-commit check."""
    staged_files = get_staged_md_files()

    if not staged_files:
        sys.exit(0)  # No .md files staged

    print(f"📄 Checking {len(staged_files)} new/modified .md file(s)...")

    all_passed = True

    for filepath in staged_files:
        print(f"\n  Checking {filepath}...")

        # Required checks (block commit if fail)
        if not check_master_doc_registration(filepath):
            all_passed = False

        # Optional checks (warn but allow commit)
        check_feature_doc_convention(filepath)
        check_readme_registration(filepath)
        suggest_alignment_test(filepath)

    if not all_passed:
        print("\n❌ Pre-commit checks failed. Fix issues above and try again.\n")
        sys.exit(1)

    print("\n✅ Documentation checks passed!\n")
    sys.exit(0)

if __name__ == '__main__':
    main()
```

---

#### Step 2: Add Alignment Test for Documentation Structure

**File**: `tests/test_documentation_alignment.py` (add to existing)

```python
def test_all_docs_have_purpose_header():
    """Ensure every .md file has a clear purpose statement."""
    docs_dir = Path("docs")

    # Skip README files and archives
    md_files = [
        f for f in docs_dir.rglob("*.md")
        if 'archive' not in str(f) and f.name != 'README.md'
    ]

    missing_purpose = []

    for doc_path in md_files:
        with open(doc_path) as f:
            content = f.read()

        # Check for purpose indicators (flexible matching)
        has_purpose = any([
            '**Purpose**:' in content,
            '## Purpose' in content,
            '## What' in content,
            '## Overview' in content,
        ])

        if not has_purpose:
            missing_purpose.append(str(doc_path))

    assert not missing_purpose, (
        f"The following docs lack clear purpose statements:\n"
        f"{chr(10).join(missing_purpose)}\n\n"
        f"Add one of: **Purpose**:, ## Purpose, ## What, ## Overview"
    )


def test_feature_docs_follow_naming_convention():
    """Ensure feature docs use consistent naming."""
    feature_docs = list(Path("docs/features").glob("*.md"))

    invalid_names = []
    valid_suffixes = ['_IMPLEMENTATION.md', '_SUMMARY.md', '_GUIDE.md']

    for doc in feature_docs:
        if doc.name == 'README.md':
            continue

        if not any(doc.name.endswith(suffix) for suffix in valid_suffixes):
            invalid_names.append(doc.name)

    assert not invalid_names, (
        f"Feature docs should end with _IMPLEMENTATION.md, _SUMMARY.md, or _GUIDE.md:\n"
        f"{chr(10).join(invalid_names)}"
    )


def test_new_master_docs_referenced_in_qa():
    """Ensure all master docs are referenced in QA_STRATEGY.md"""
    master_docs = [
        f.name for f in Path("docs/context").glob("*.md")
        if f.name != 'README.md'
    ]

    with open("docs/QA_STRATEGY.md") as f:
        qa_content = f.read()

    missing_refs = []

    for doc in master_docs:
        if doc not in qa_content:
            missing_refs.append(doc)

    assert not missing_refs, (
        f"Master docs not referenced in QA_STRATEGY.md:\n"
        f"{chr(10).join(missing_refs)}\n\n"
        f"Add references to QA_STRATEGY.md §7 'Master Documentation Update Process'"
    )


def test_docs_subdirectory_integrity():
    """Ensure docs/ subdirectories match expected structure."""
    expected_dirs = {
        'context': 'Master documentation (SSOT)',
        'features': 'Feature implementation details',
        'setup': 'Installation and configuration',
        'implementation': 'Milestone completion reports',
        'testing': 'Testing strategies and checklists',
        'archive': 'Historical/deprecated docs',
        'analysis': 'Technical decisions and analysis',
    }

    docs_dir = Path("docs")
    actual_dirs = {
        d.name for d in docs_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    }

    # Check for unexpected directories
    unexpected = actual_dirs - set(expected_dirs.keys())

    if unexpected:
        pytest.fail(
            f"Unexpected subdirectories in docs/: {unexpected}\n\n"
            f"Expected structure:\n" +
            '\n'.join(f"  - {k}: {v}" for k, v in expected_dirs.items()) +
            "\n\nIf adding new subdirectory:\n"
            "1. Update this test with new directory and purpose\n"
            "2. Add to QA_STRATEGY.md §4 'Quick Reference: Documentation Types'\n"
            "3. Create README.md in new subdirectory explaining purpose"
        )
```

---

#### Step 3: GitHub Actions Workflow for Doc Changes

**File**: `.github/workflows/doc-alignment-check.yml` (create this)

```yaml
name: Documentation Alignment Check

on:
  pull_request:
    paths:
      - 'docs/**/*.md'

jobs:
  check-doc-alignment:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install pytest

      - name: Check for new .md files
        id: check_new_files
        run: |
          # Get changed files
          git fetch origin main
          NEW_DOCS=$(git diff --name-only --diff-filter=A origin/main HEAD | grep 'docs/.*\.md$' || true)

          if [ -n "$NEW_DOCS" ]; then
            echo "new_docs_found=true" >> $GITHUB_OUTPUT
            echo "New documentation files detected:"
            echo "$NEW_DOCS"
          fi

      - name: Run documentation alignment tests
        run: |
          pytest tests/test_documentation_alignment.py -v

      - name: Comment on PR if new master docs found
        if: steps.check_new_files.outputs.new_docs_found == 'true'
        uses: actions/github-script@v6
        with:
          script: |
            const newDocs = `${{ steps.check_new_files.outputs.NEW_DOCS }}`;

            if (newDocs.includes('docs/context/')) {
              github.rest.issues.createComment({
                issue_number: context.issue.number,
                owner: context.repo.owner,
                repo: context.repo.name,
                body: `## ⚠️ New Master Documentation Detected

                You've added new files to \`docs/context/\` (master docs).

                **Required Actions**:
                - [ ] Add reference to \`docs/QA_STRATEGY.md\` §7 "Master Documentation Update Process"
                - [ ] Add to "Quick Reference: Documentation Types" table in QA_STRATEGY.md
                - [ ] Consider adding alignment test in \`tests/test_documentation_alignment.py\`

                **Why?** Master docs are the single source of truth. They must be tracked in QA.`
              })
            }
```

---

### Quick Reference: Documentation Addition Checklist

**Before creating new .md file**, ask:

```
┌─ Can this content fit in EXISTING doc?
│  ├─ YES → Update existing doc (no new file needed)
│  └─ NO → Proceed to create new doc
│      └─ What type of content?
│          ├─ System behavior/architecture → docs/context/ (MASTER)
│          ├─ Feature implementation → docs/features/
│          ├─ Setup/configuration → docs/setup/
│          ├─ Milestone completion → docs/implementation/
│          └─ Testing procedures → Consolidate into QA_STRATEGY.md
```

**After creating new .md file**:

1. ✅ **Add to appropriate README.md** (docs/context/README.md, docs/features/README.md, etc.)
2. ✅ **Cross-reference in QA_STRATEGY.md** (if master doc or testing-related)
3. ✅ **Update CHANGELOG.md** (if user-facing content)
4. ✅ **Run alignment tests**: `pytest tests/test_documentation_alignment.py -v`
5. ✅ **Consider alignment test** (if doc references code files)

---

### Auto-Detection: Uncommitted .md Files

Add to daily maintenance script:

**File**: `scripts/daily_maintenance.py` (enhance existing)

```python
def check_uncommitted_docs():
    """Check for .md files not tracked in git."""
    result = subprocess.run(
        ['git', 'ls-files', '--others', '--exclude-standard', 'docs/'],
        capture_output=True, text=True
    )

    untracked_docs = [
        line for line in result.stdout.split('\n')
        if line.endswith('.md')
    ]

    if untracked_docs:
        print("\n⚠️  WARNING: Untracked .md files found:")
        for doc in untracked_docs:
            print(f"  - {doc}")

        print("\nACTION: Review these files and either:")
        print("  1. Add to git (if they should be tracked)")
        print("  2. Delete (if they're scratch notes)")
        print("  3. Add to .gitignore (if they're local-only)")
```

---

## Preventing Future Misalignment

### 1. Editor Integration

**VS Code**: Add to `.vscode/settings.json`:
```json
{
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "tests",
    "--doctest-modules"
  ],
  "python.linting.enabled": true,
  "grammarly.files.include": ["**/*.md"],
  "grammarly.selectors": [
    {
      "language": "markdown",
      "scheme": "file"
    }
  ]
}
```

### 2. Pull Request Template

**File**: `.github/pull_request_template.md`

**See**: [Feature Development Documentation Workflow](#feature-development-documentation-workflow) for detailed guidance.

```markdown
## Changes Made
- [ ] Code changes described
- [ ] If changing function names/signatures: Updated docs
- [ ] If adding new feature: Added to CHANGELOG.md
- [ ] If modifying conversation flow: Updated SYSTEM_ARCHITECTURE_SUMMARY.md

## Documentation Updates (See QA_STRATEGY.md §4 for Workflow)
- [ ] Master docs updated (if behavior/architecture changed)
- [ ] Feature doc created/updated (implementation details)
- [ ] I understand when to create new docs vs update existing
- [ ] Code references use actual function/file names (not conceptual terms)
- [ ] Alignment test added (if new documented functions/flow)
- [ ] Tests pass: `pytest tests/test_documentation_alignment.py -v`

## Testing
- [ ] All conversation quality tests pass
- [ ] All documentation alignment tests pass
- [ ] Manual testing completed
```

### 3. Commit Message Convention

When updating docs, use these prefixes:
- `docs(master): Update SYSTEM_ARCHITECTURE_SUMMARY` - Master doc change
- `docs(feature): Update DISPLAY_INTELLIGENCE` - Feature doc change
- `docs(fix): Fix broken file reference in RAG_ENGINE.md` - Doc bug fix
- `docs(align): Update flow to match code refactor` - Alignment fix

---

## Testing Best Practices & Common Issues

### Overview

This section documents testing principles learned from fixing test failures and establishing QA standards. These patterns ensure tests remain maintainable, reliable, and actually validate what users see.

---

### Principle 1: "Test What Users See"

**Rule**: Tests must validate user-facing output, not internal storage or intermediate states.

**Why This Matters**:
- KB content uses `###` headers and emojis (teaching structure)
- LLM responses must strip these to **Bold** (professional presentation)
- Testing KB storage doesn't validate what users actually see

**Example - ❌ Wrong**:
```python
def test_no_emoji_headers(self):
    """Check if KB files contain markdown headers."""
    with open('data/career_kb.csv', 'r') as f:
        content = f.read()
        assert '###' not in content  # Wrong! Tests storage, not output
```

**Example - ✅ Right**:
```python
def test_no_emoji_headers(self):
    """Ensure LLM strips markdown headers from responses."""
    mock_engine = MagicMock()
    mock_engine.generate_response.return_value = "**Bold Header**\n\nContent..."

    state = ConversationState(role="...", query="...")
    state = run_conversation_flow(state, mock_engine)

    # Test actual user-facing output
    assert '###' not in state.answer  # Right! Tests what user sees
    assert re.search(r'\*\*[\w\s]+\*\*', state.answer)  # Validates Bold format
```

**Impact**: Fixed `test_no_emoji_headers` on Oct 16, 2025 using this principle. Test now validates actual conversation quality instead of KB structure.

---

### Principle 2: "No @patch for Non-Existent Attributes"

**Rule**: Only use `@patch()` decorator for attributes that are actually imported in the target module. Otherwise, create mocks directly.

**Why This Matters**:
- Bad `@patch` causes `AttributeError` that masks whether code or test is broken
- Direct mocks are clearer, more maintainable, and easier to debug
- Failing tests block development and erode confidence in QA

**Example - ❌ Wrong**:
```python
@patch('src.flows.conversation_nodes.RagEngine')  # RagEngine NOT imported there!
def test_my_feature(self, mock_rag_engine):
    mock_engine = MagicMock()
    mock_rag_engine.return_value = mock_engine
    # Result: AttributeError: module does not have attribute 'RagEngine'
```

**Example - ✅ Right**:
```python
def test_my_feature(self):
    # Create mock directly - no @patch needed
    mock_engine = MagicMock()
    mock_engine.retrieve.return_value = {"chunks": []}

    state = ConversationState(role="...", query="...")
    state = classify_query(state)
    state = apply_role_context(state, mock_engine)

    assert len(state.answer) > 0
```

**When @patch IS Appropriate**:
```python
# RagEngine IS imported in rag_engine.py, so @patch works
@patch('src.core.rag_engine.OpenAIEmbeddings')
def test_rag_engine_initialization(self, mock_embeddings):
    # This works because OpenAIEmbeddings is actually imported
    pass
```

**Impact**: Fixed 4 tests on Oct 16, 2025 by removing bad `@patch` decorators. Tests now pass reliably.

---

### Principle 3: "Update Tests When Behavior Intentionally Changes"

**Rule**: When code behavior changes intentionally, update test expectations to match. Don't change code to match outdated tests.

**Example - Test Failure After Code Change**:
```python
# Code was refactored, now uses new intro text
def generate_data_intro():
    return "Fetching live analytics data from Supabase..."  # NEW

# Test still expects old text
def test_display_data_uses_canned_intro(self):
    assert state.answer.startswith("Here's the live analytics snapshot")  # OLD
    # Result: AssertionError - test expectations outdated
```

**Fix**:
```python
def test_display_data_uses_canned_intro(self):
    # UPDATED: Expectations match current implementation
    assert state.answer.startswith("Fetching live analytics data from Supabase")
```

**QA Compliance**: ✅ Update test expectations, add comment explaining why changed.

**When NOT to Update Tests**: If test fails due to unintended behavior change (bug), fix the code, not the test.

---

### Common Test Failures & Diagnostic Guide

#### Issue 1: "Expected text doesn't match actual output"

**Symptom**:
```
AssertionError: assert False
 +  where False = <built-in method startswith of str object>.startswith("Expected text")
 +  where "Actual text different from expected" = state.answer
```

**Diagnostic Steps**:
1. Check if behavior intentionally changed
2. Review git history: `git log -p -- path/to/changed/file.py`
3. Ask: Should code match test, or test match code?

**Fix Pattern**:
- If code is correct → Update test expectations
- If code is wrong → Fix code, keep test

---

#### Issue 2: "@patch() AttributeError"

**Symptom**:
```
AttributeError: <module 'src.flows.conversation_nodes'> does not have the attribute 'RagEngine'
```

**Root Cause**: Trying to patch a class that isn't imported in the target module.

**Diagnostic Command**:
```bash
# Check if class is actually imported
grep -n "from.*import.*RagEngine" src/flows/conversation_nodes.py
grep -n "import.*rag_engine" src/flows/conversation_nodes.py
```

**Fix Pattern**:
1. Remove `@patch()` decorator
2. Remove mock parameter from function signature
3. Create mock directly with `MagicMock()`

**Before**:
```python
@patch('src.flows.conversation_nodes.RagEngine')
def test_my_feature(self, mock_rag_engine):
    mock_engine = MagicMock()
    mock_rag_engine.return_value = mock_engine
```

**After**:
```python
def test_my_feature(self):
    mock_engine = MagicMock()
    mock_engine.retrieve.return_value = {"chunks": []}
```

---

#### Issue 3: "Test passes locally but fails in CI"

**Common Causes**:
1. **Environment differences**: Local has cached data, CI doesn't
2. **Timing issues**: Race conditions in async code
3. **File paths**: Hardcoded absolute paths instead of relative

**Diagnostic Steps**:
```bash
# Run tests with same environment as CI
python3 -m pytest tests/ -v --tb=short

# Check for hardcoded paths
grep -r "Users/noah" tests/

# Check for timing assumptions
grep -r "sleep\|wait\|timeout" tests/
```

**Prevention**:
- Use `pathlib.Path(__file__).parent` for file paths
- Mock time-dependent operations
- Use `pytest-timeout` to catch hangs

---

### When to Add New Tests

Add regression tests in these scenarios:

1. **Bug found in production** → Add test that would have caught it
   ```python
   def test_bug_123_no_malformed_code_output(self):
       """Regression test for issue #123: Malformed code in responses."""
       # Test that would have caught the bug
   ```

2. **New quality standard defined** → Add test to enforce it
   ```python
   def test_responses_under_15k_chars(self):
       """Enforce new quality standard: No information overload."""
   ```

3. **Feature changes behavior** → Update existing tests + add edge cases
   ```python
   def test_new_feature_happy_path(self):
       """Test primary use case for new feature."""

   def test_new_feature_edge_case_empty_input(self):
       """Test edge case: What happens with empty input?"""
   ```

4. **Personality requirement added** → Add test checking for trait
   ```python
   def test_responses_include_follow_up_questions(self):
       """Portfolia personality: Should offer to go deeper."""
       assert "would you like" in answer.lower() or "want to see" in answer.lower()
   ```

---

### Test Template

Use this template when adding new quality tests:

```python
def test_NEW_QUALITY_STANDARD(self):
    """Brief description of what quality issue this prevents.

    Context: Why this test exists (reference bug/feature/requirement).
    """
    # Setup: Create mocks
    mock_engine = MagicMock()
    mock_engine.retrieve.return_value = {"chunks": [], "matches": []}
    mock_engine.generate_response.return_value = "Expected output"

    # Execute: Run conversation flow
    state = ConversationState(
        role="Hiring Manager (technical)",
        query="Test query"
    )
    state.set_answer("Expected output")
    state = classify_query(state)
    state = apply_role_context(state, mock_engine)

    # Assert: Check quality standard
    answer = state.answer
    assert QUALITY_CHECK, "Error message explaining what quality standard was violated"

    # Example assertions:
    # assert len(answer) < 15000, f"Response too long: {len(answer)} chars"
    # assert answer.count("would you like") <= 1, "Too many follow-up prompts"
    # assert '###' not in answer, "Markdown headers in user-facing output"
```

---

### Test Maintenance Best Practices

1. **Run tests before committing**:
   ```bash
   pytest tests/test_conversation_quality.py -v
   pytest tests/test_documentation_alignment.py -v
   ```

2. **Fix failures immediately**: Don't commit code with failing tests (blocks CI/CD)

3. **Update tests with code changes**: Same commit updates both (prevents drift)

4. **Document why tests were updated**: Add comments explaining behavioral changes

5. **Review test coverage**: Ensure new features have tests

---

### Quick Reference: Test Fixes Applied (Oct 16, 2025)

These tests were fixed using the principles above:

| Test | Issue | Fix Applied | Principle |
|------|-------|-------------|-----------|
| `test_no_emoji_headers` | Tested KB storage, not output | Check `state.answer` in full flow | #1: Test What Users See |
| `test_no_duplicate_prompts_in_full_flow` | Bad `@patch` decorator | Remove @patch, create mock directly | #2: No @patch for Non-Existent |
| `test_display_data_uses_canned_intro` | Outdated expected text | Update to match new intro | #3: Update Tests When Behavior Changes |
| `test_empty_code_index_shows_helpful_message` | Bad `@patch` decorator | Remove @patch, create mock directly | #2: No @patch for Non-Existent |
| `test_no_information_overload` | Bad `@patch` decorator | Remove @patch, create mock directly | #2: No @patch for Non-Existent |
| `test_consistent_formatting_across_roles` | Bad `@patch` decorator | Remove @patch, create mock directly | #2: No @patch for Non-Existent |

**Result**: 14/18 → 18/18 conversation tests passing (78% → 100%) ✅

**Archived Details**: See `docs/archive/bugfixes/PHASE_1_TEST_FIXES_OCT_16_2025.md` for full context.

---

## Manual Testing Procedures

**Source**: Consolidated from `docs/testing/ROLE_FUNCTIONALITY_CHECKLIST.md` (Oct 16, 2025)

### Purpose

Manual testing complements automated testing by validating user experience, edge cases, and cross-functional behavior that's difficult to automate.

---

### Testing Pyramid

```
           /\
          /  \
         / E2E \         Manual Testing (this section)
        /------\         - User workflows
       /  Integ \        - Cross-role behavior
      /----------\       - Visual/UX validation
     /  Automated \      Automated Testing (§1-2)
    /--------------\     - pytest suite (30 tests)
   /   Unit Tests   \    - CI/CD validation
```

**Philosophy**:
- **Automated tests** → Fast, repeatable, catches 90% of bugs
- **Manual tests** → Slow, thorough, catches the other 10% (UX, edge cases)

---

### When to Use Manual vs Automated Testing

| Scenario | Use Automated | Use Manual | Why? |
|----------|--------------|------------|------|
| **New feature validation** | ✅ Primary | ✅ Once | Automated catches regressions, manual validates UX |
| **Regression testing** | ✅ Always | ❌ Rarely | Automated is faster and more reliable |
| **User workflow testing** | ⚠️ Difficult | ✅ Better | Manual better captures real user experience |
| **Edge case discovery** | ❌ Can't predict | ✅ Good for exploration | Manual testing finds unexpected issues |
| **Cross-role consistency** | ⚠️ Partial | ✅ Comprehensive | Manual can compare side-by-side |
| **Visual/formatting issues** | ❌ Hard to automate | ✅ Easy to spot | Human eyes catch formatting problems |
| **Pre-release validation** | ✅ Required | ✅ Recommended | Both for confidence |

---

### Role Functionality Checklist

**How to Use**:
1. Pick a role to test
2. Follow checklist systematically
3. Note any failures or unexpected behavior
4. File bugs with checklist item reference
5. Re-test after fixes

---

#### 🧑‍💼 Hiring Manager (Non-Technical) - Manual Test

| Feature | Test Procedure | Expected Result | Status |
|---------|---------------|-----------------|--------|
| **Career history** | Ask: "Tell me about Noah's work experience" | Returns career highlights (no code details) | ⬜ |
| **Email resume** | Say: "Send me his resume" | Triggers resume email + confirmation message | ⬜ |
| **Email LinkedIn** | Say: "Send me his LinkedIn" | Adds LinkedIn URL to response | ⬜ |
| **Email both** | Say: "Send me both his resume and LinkedIn" | Both actions trigger | ⬜ |
| **Proactive offer** | Ask 2+ questions WITHOUT mentioning resume | System offers resume after turn 2 | ⬜ |
| **Reach out request** | Say: "Yes, have Noah reach out" | Confirmation + notification logged | ⬜ |
| **SMS notification (resume)** | Send resume request | Noah receives SMS within 30s | ⬜ |
| **SMS notification (contact)** | Request reach out | Noah receives SMS with contact info | ⬜ |

**Implementation References**:
- Resume email: `execute_actions` node → Resend service
- Proactive offer: `plan_actions` checks `user_turns >= 2`
- SMS: Twilio integration in `action_execution.py`

---

#### 🧑‍💻 Hiring Manager (Technical) - Manual Test

| Feature | Test Procedure | Expected Result | Status |
|---------|---------------|-----------------|--------|
| **Project info** | Ask: "What projects has Noah built?" | Returns technical project details | ⬜ |
| **AI/ML history** | Ask: "What AI experience does he have?" | Returns GenAI + RAG implementation history | ⬜ |
| **Enterprise fit** | Ask: "How would this work for my company?" | Explains role router + Acme Corp example | ⬜ |
| **Stack currency** | Ask: "Is this using the latest tech?" | Explains LangGraph, Supabase, version strategy | ⬜ |
| **Data strategy** | Ask: "What data do you collect?" | Shows data collection table (markdown format) | ⬜ |
| **Staying current** | Ask: "How do you keep this updated?" | Mentions LangSmith traces + KB updates | ⬜ |
| **Code display** | Ask: "Show me the conversation nodes code" | Returns code snippet with syntax highlighting | ⬜ |
| **Resume/LinkedIn** | Say: "Send resume" | Same as non-technical HM | ⬜ |
| **Proactive offer** | Ask 2+ technical questions | Offers resume after turn 2 | ⬜ |

**Implementation References**:
- Technical KB: `data/technical_kb.csv` + `data/architecture_kb.csv`
- Code display: `retrieve_with_code` from RAG engine
- Data tables: `_data_collection_table()` in `content_blocks.py`

---

#### 👨‍💻 Software Developer - Manual Test

| Feature | Test Procedure | Expected Result | Status |
|---------|---------------|-----------------|--------|
| **Architecture deep-dive** | Ask: "Explain the system architecture" | Returns technical architecture details | ⬜ |
| **Code examples** | Ask: "Show me how conversation nodes work" | Returns actual code from `conversation_nodes.py` | ⬜ |
| **Stack details** | Ask: "What's the tech stack?" | Lists LangGraph, Supabase, OpenAI, Streamlit | ⬜ |
| **Data collection** | Ask: "What analytics do you track?" | Shows data collection table | ⬜ |
| **RAG implementation** | Ask: "How does the RAG pipeline work?" | Explains pgvector retrieval flow | ⬜ |
| **Code freshness** | Ask: "Show me the latest code" | Returns current codebase (from pgvector index) | ⬜ |
| **No resume push** | Have technical conversation | System does NOT offer resume (dev audience) | ⬜ |

**Implementation References**:
- Code retrieval: `retrieve_with_code` → indexed codebase
- Technical KB: Same as technical HM
- No resume offer: Conditional in `plan_actions` based on role

---

#### 😎 Just Exploring - Manual Test

| Feature | Test Procedure | Expected Result | Status |
|---------|---------------|-----------------|--------|
| **High-level explanation** | Ask: "What is this project?" | Returns simplified, non-technical explanation | ⬜ |
| **Fun facts** | Ask: "Tell me something interesting about Noah" | Returns fun facts (MMA, hot dogs, etc.) | ⬜ |
| **MMA query** | Ask: "Did Noah really fight MMA?" | Returns fun fact + YouTube fight link | ⬜ |
| **Light tone** | General questions | Responses are friendly, not overly technical | ⬜ |
| **No data tables** | Ask about data | Explains simply, no markdown tables | ⬜ |

**Implementation References**:
- Fun facts: `_fun_facts_block()` in `content_blocks.py`
- MMA link: `share_mma_link` action uses Supabase settings
- Fun facts content: Lines 126-131 in `conversation_nodes.py`

---

#### ❤️ Confess Crush - Manual Test

| Feature | Test Procedure | Expected Result | Status |
|---------|---------------|-----------------|--------|
| **Anonymous confession** | Submit confession with "anonymous" checked | No personal info collected | ⬜ |
| **Named confession** | Submit with name/phone/email | All fields stored in DB | ⬜ |
| **SMS notification** | Submit confession | Noah receives SMS with confession text | ⬜ |
| **Secure storage** | Check Supabase `confessions` table | Data encrypted, privacy protected | ⬜ |
| **No analytics leak** | Ask about confessions in chat | System says "confessions are private" | ⬜ |

**Implementation References**:
- API endpoint: `api/confess.py`
- SMS: Twilio integration (lines 65-77 in confess handler)
- Privacy: Analytics query excludes confessions table

---

### Cross-Role Consistency Tests

These tests verify consistent behavior across ALL roles:

| Test | Procedure | Expected Result | Status |
|------|-----------|-----------------|--------|
| **Formatting consistency** | Ask same question to all 5 roles | All responses use **Bold**, no ### headers | ⬜ |
| **No duplicate prompts** | Have 5-turn conversation in each role | Only ONE "Would you like..." per response | ⬜ |
| **Response length** | Ask complex question in each role | All responses <15k characters | ⬜ |
| **Greeting consistency** | Start conversation in each role | All greetings professional, role-appropriate | ⬜ |
| **Error handling** | Ask gibberish in each role | All respond gracefully ("I don't understand...") | ⬜ |

---

### Manual Test Execution Log

**Date**: __________
**Tester**: __________
**Version/Commit**: __________

**Results Summary**:
- Total Tests: ____
- Passed: ____
- Failed: ____
- Blocked: ____

**Failures**:
1. [Test name] - [What failed] - [Bug ID]
2. [Test name] - [What failed] - [Bug ID]

**Notes**:
- [Any observations, edge cases found, improvements suggested]

---

### Automated vs Manual Test Mapping

| Automated Test (pytest) | Manual Test (checklist) | Why Both? |
|------------------------|------------------------|-----------|
| `test_no_emoji_headers` | Formatting consistency test | pytest checks LLM output, manual verifies across all roles |
| `test_no_duplicate_prompts` | Cross-role duplicate prompts | pytest catches regression, manual validates UX |
| `test_kb_coverage_aggregated` | Data strategy test | pytest validates table format, manual checks readability |
| `test_empty_code_index_shows_helpful_message` | Code examples test | pytest checks error handling, manual validates message quality |
| (No automated test) | Proactive offer after 2 turns | Difficult to automate user turn counting |
| (No automated test) | SMS notification timing | Requires real Twilio, manual validates delivery |
| (No automated test) | Cross-role tone consistency | Subjective UX validation |

**Principle**: If it's **deterministic and fast** → automate it. If it's **subjective or requires external services** → manual test.

---

### Adding New Manual Tests

**When to add**:
1. **New role added** → Add role section with feature checklist
2. **New external service** → Add service validation test (SMS, email, etc.)
3. **New user workflow** → Add cross-role consistency test
4. **Bug found in production** → Add regression test (try automated first, manual if needed)

**Template**:

```markdown
#### New Feature - Manual Test

| Feature | Test Procedure | Expected Result | Status |
|---------|---------------|-----------------|--------|
| **[Feature name]** | [How to test] | [What should happen] | ⬜ |

**Implementation Reference**: [File path, line numbers, function names]
```

---

### Pre-Release Manual Testing Protocol

**Before deploying to production**:

1. ✅ **Run automated tests**: `pytest tests/ -v` (must be 100% passing)
2. ✅ **Pick 2 roles randomly**: Use dice roll or random.org
3. ✅ **Execute full checklist** for those 2 roles
4. ✅ **Test cross-role consistency** (all 5 items)
5. ✅ **Document results** in execution log
6. ✅ **File bugs** for any failures (block deploy if critical)
7. ✅ **Re-test after fixes**

**Time estimate**: 30-45 minutes for full pre-release manual test

---

### Related Automation

**Automated tests complement manual tests**:
- See §1 "Current Quality Standards" for automated test inventory
- See §2 "Automated Testing" for pytest execution
- See §9 "Testing Best Practices" for when to update tests

**Archive**: Original `docs/testing/ROLE_FUNCTIONALITY_CHECKLIST.md` moved to `docs/archive/testing/` with completion header (Oct 16, 2025)

---

## 12. Documentation Consolidation Policy

**Purpose**: Prevent proliferation of redundant QA documentation files.

**Last Updated**: October 16, 2025 (after consolidating 5 QA docs → 1)

---

### The Problem: Documentation Sprawl

**What Happened** (Oct 2025):
- Started with 1 master doc: `docs/QA_STRATEGY.md`
- Over 2 weeks, created 4 additional QA docs for specific topics
- Result: 4,620 lines across 5 files with **~1,400 lines of pure duplication**
- Developers confused about which doc was authoritative

**Files Involved**:
1. `docs/QA_STRATEGY.md` (2,807 lines) - Master SSOT
2. ~~`docs/QA_IMPLEMENTATION_SUMMARY.md` (456 lines)~~ - **ARCHIVED Oct 17, 2025** → Content in [Current Test Status](#current-test-status)
3. ~~`docs/QA_LANGSMITH_INTEGRATION.md` (535 lines)~~ - **ARCHIVED Oct 17, 2025** → Content in [Phase 2: Production Monitoring](#phase-2-production-monitoring-with-langsmith)
4. `QA_POLICY_KB_VS_RESPONSE_SEPARATION.md` (319 lines) - Policy update
5. `QA_COMPLIANCE_AND_VERCEL_DEPLOYMENT.md` (503 lines) - Task 11 report

**Root Cause**: Created separate docs for temporary topics instead of adding sections to master doc.

---

### The Solution: Consolidation + Ongoing Policy

#### ✅ What We Did (Oct 16, 2025)

**Consolidated to Single SSOT**:
- Merged LangSmith content into QA_STRATEGY.md Section 9
- Archived historical policy docs to `docs/archive/policies/`
- Archived task-specific reports to `docs/archive/deployments/`
- Deleted redundant summary doc (content already in master)
- Extracted deployment steps to `docs/setup/VERCEL_DEPLOYMENT.md`

**Result**: 1 master QA doc (3,200 lines) + properly categorized archives

**Historical Context**: See `docs/archive/analysis/QA_DOCUMENTATION_CONSOLIDATION_PLAN_OCT16.md` for full analysis and migration steps.

---

### Ongoing Policy: When to Create New QA Docs

Use this decision tree **before creating any new .md file related to QA**:

```
New QA content identified?
  ↓
  Is it a quality standard or test?
    Yes → Add to QA_STRATEGY.md Section 1
    No → Continue
  ↓
  Is it about testing workflow/CI/CD?
    Yes → Add to QA_STRATEGY.md Section 2-6
    No → Continue
  ↓
  Is it about documentation alignment?
    Yes → Add to QA_STRATEGY.md Section 3 or 7
    No → Continue
  ↓
  Is it about production monitoring?
    Yes → Add to QA_STRATEGY.md Section 9 (LangSmith/observability)
    No → Continue
  ↓
  Is it deployment/operational instructions?
    Yes → Add to docs/setup/VERCEL_DEPLOYMENT.md or similar
    No → Continue
  ↓
  Is it a historical policy change explanation?
    Yes → Document change in QA_STRATEGY.md, archive explanation in docs/archive/policies/ with date
    No → Continue
  ↓
  Is it a one-time cleanup/migration plan?
    Yes → Create in root, execute, then archive to docs/archive/analysis/ with date
    No → Reconsider if it's actually a QA topic
```

**Golden Rule**: **When in doubt, add to QA_STRATEGY.md, NOT a new file.**

---

### File Categorization Rules

| File Type | Location | Naming | When to Create |
|-----------|----------|--------|----------------|
| **Master QA Standards** | `docs/QA_STRATEGY.md` | N/A (single file) | ✅ Already exists - ADD sections, don't duplicate |
| **Deployment Guides** | `docs/setup/` | `VERCEL_DEPLOYMENT.md`, `LOCAL_TESTING.md` | Only if operational steps, not QA standards |
| **Policy Change History** | `docs/archive/policies/` | `QA_POLICY_[TOPIC]_[MMDDYYYY].md` | After updating QA_STRATEGY.md to explain "why changed" |
| **Task-Specific Reports** | `docs/archive/deployments/` | `TASK[N]_[TOPIC]_[MMDDYYYY].md` | For compliance reports, deployment summaries |
| **One-Time Analysis** | `docs/archive/analysis/` | `[TOPIC]_ANALYSIS_[MMDDYYYY].md` | For audits, consolidation plans, investigations |

**Archive Naming Convention**: Always include `_[MMDDYYYY]` suffix (e.g., `_OCT16` for Oct 16, 2025) so archived docs don't get confused with active ones.

---

### Examples: Correct vs Incorrect Handling

#### ✅ Example 1: New Quality Standard (Correct)

**Scenario**: Discovered LLM sometimes returns Q&A format verbatim instead of synthesizing.

**Wrong Approach** ❌:
```
Create QA_POLICY_UPDATE_NO_QA_VERBATIM.md (319 lines)
Explain problem, solution, test results
```

**Right Approach** ✅:
```
1. Add test: test_no_qa_verbatim_responses() to tests/test_conversation_quality.py
2. Update QA_STRATEGY.md Section 1 test coverage table
3. Document synthesis requirement in Section 1.2 (if doesn't exist)
4. Archive historical explanation in docs/archive/policies/QA_POLICY_NO_QA_VERBATIM_OCT15.md
```

**Justification**: Standard belongs in master doc. Historical "why" belongs in archive.

---

#### ✅ Example 2: Phase 2 Monitoring Plan (Correct)

**Scenario**: Want to add LangSmith production monitoring to QA process.

**Wrong Approach** ❌:
```
Create docs/QA_LANGSMITH_INTEGRATION.md (535 lines)
Document entire Phase 2 plan separately
```

**Right Approach** ✅:
```
1. Add new Section 9 to QA_STRATEGY.md: "Phase 2: Production Monitoring with LangSmith"
2. Include integration steps, code examples, cost analysis
3. Update Phase roadmap in Section 8 to reference Section 9
4. No separate file needed - it's an ongoing QA process
```

**Justification**: Ongoing monitoring is a permanent part of QA strategy, not a temporary topic.

---

#### ✅ Example 3: Task-Specific Compliance Report (Correct)

**Scenario**: Task 11 requires QA compliance verification before Vercel deployment.

**Wrong Approach** ❌:
```
Create QA_COMPLIANCE_AND_VERCEL_DEPLOYMENT.md (503 lines) in root
Leave it there indefinitely as reference
```

**Right Approach** ✅:
```
1. Create QA_COMPLIANCE_AND_VERCEL_DEPLOYMENT.md in root (temporary)
2. Use for Task 11 execution (validates 71 tests, documents deployment steps)
3. After Task 11 complete, archive to docs/archive/deployments/TASK11_QA_COMPLIANCE_OCT16.md
4. Extract ongoing deployment steps to docs/setup/VERCEL_DEPLOYMENT.md
```

**Justification**: Task reports are historical artifacts, not ongoing standards. Operational steps belong in docs/setup/.

---

### Enforcement: Pre-Commit Hook (Future)

**Phase 3 Enhancement** (after current cleanup):

Add pre-commit hook to detect new QA-related .md files:

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Check for new QA-related .md files
new_qa_docs=$(git diff --cached --name-only --diff-filter=A | grep -E "(QA_|QUALITY_|TEST_).*\.md")

if [ -n "$new_qa_docs" ]; then
    echo "⚠️  WARNING: New QA documentation file detected:"
    echo "$new_qa_docs"
    echo ""
    echo "Before creating new QA docs, ask:"
    echo "1. Can this be added to docs/QA_STRATEGY.md instead?"
    echo "2. If not, does it belong in docs/setup/, docs/archive/policies/, or docs/archive/deployments/?"
    echo ""
    echo "See docs/QA_STRATEGY.md Section 12 for consolidation policy."
    echo ""
    read -p "Proceed anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
```

**Implementation**: After Phase 1 cleanup complete and stable.

---

### Quarterly Review Process

**Every 3 months** (Jan 15, Apr 15, Jul 15, Oct 15):

1. **Audit for new QA files**:
   ```bash
   find . -name "*QA*.md" -o -name "*QUALITY*.md" -o -name "*TEST*.md" | grep -v "docs/QA_STRATEGY.md"
   ```

2. **For each file found**:
   - Is content still relevant? → Merge into QA_STRATEGY.md or docs/setup/
   - Is content historical? → Archive to appropriate docs/archive/ subdirectory
   - Is content redundant? → Delete after confirming content in master doc

3. **Update QA_STRATEGY.md**:
   - Add any new standards discovered
   - Update test counts, pass rates
   - Verify all cross-references valid

4. **Document review**:
   - Update "Last Review" date (bottom of this doc)
   - Commit: `git commit -m "docs: Quarterly QA documentation audit (Q[N] YYYY)"`

---

### Success Metrics

**Goal**: Maintain single source of truth for QA standards.

**Track Quarterly**:
- Number of QA-related .md files outside docs/QA_STRATEGY.md: **Target ≤2** (1 master + 1 setup guide)
- Lines of duplicated QA content: **Target <100 lines** (some overlap acceptable for context)
- Developer confusion incidents (asking "which doc is correct?"): **Target 0**

**Current Status** (Oct 16, 2025):
- QA files: 1 master (docs/QA_STRATEGY.md) + 1 setup (docs/setup/VERCEL_DEPLOYMENT.md) ✅
- Duplication: ~0 lines (all consolidated) ✅
- Confusion incidents: 0 (just cleaned up) ✅

---

**Historical Context**: Full consolidation analysis and migration steps documented in `docs/archive/analysis/QA_DOCUMENTATION_CONSOLIDATION_PLAN_OCT16.md`.

---

## Related Documentation

- **Test Inventory**: See "Current Quality Standards" section above
- **Conversation Quality**: `tests/test_conversation_quality.py`
- **Master Documentation**: `docs/context/` directory
- **Setup Guides**: `docs/setup/` directory
- **Historical Context**: `docs/archive/` directory

---

**Last Review**: October 16, 2025 (Added Documentation Consolidation Policy §12)
**Next Review**: January 16, 2026
**Owner**: Engineering Team
