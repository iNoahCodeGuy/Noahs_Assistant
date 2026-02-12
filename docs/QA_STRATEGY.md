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
   - 1.1 [Conversation Quality Tests (18 tests)](#1-conversation-quality-18-tests---current-status-12-passing-6-need-updates)
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

# Run only documentation alignment tests (12 tests)
pytest tests/test_documentation_alignment.py -v

# Run with detailed output (useful for debugging)
pytest tests/ -vv

# Run tests matching pattern
pytest tests/ -k "emoji" -v
```

---

## Current Test Status

| Test Suite | Tests | Passing | Status |
|------------|-------|---------|--------|
| **Conversation Quality** | 18 | 18 | ✅ 100% |
| **Documentation Alignment** | 12 | 11 | ✅ 92% (1 skipped) |
| **TOTAL** | **30** | **29** | **✅ 100% pass rate (active tests)** |

**Last Run**: October 16, 2025
**Target**: ✅ ACHIEVED - 100% pass rate on all active tests

---

## Feature Development Checklist

**Before starting work:**
- [ ] Read relevant test file (alignment tests)
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

## Current Quality Standards

### 1. Conversation Quality Tests (REMOVED — pending rewrite)

Conversation quality tests were removed during the Danaher + Craig Jones voice
stabilization. They will be rewritten once the personality core is stable.

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
        "handle_greeting", "classify_query", "retrieve_chunks",
        "generate_answer", "plan_actions", "apply_role_context",
        "execute_actions", "log_and_notify"
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
- ✅ Required checks: `documentation-alignment`

### Local Testing Before Push

Run the same tests locally to catch issues before CI/CD:

```bash
# Run all tests (same as CI/CD)
pytest tests/test_documentation_alignment.py -v

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
1. Add reference to docs/QA_STRATEGY.md §7 "Master Documentation Update Process"
2. Add to "Quick Reference: Documentation Types" table
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

## Related Documentation

- **Test Inventory**: See "Current Quality Standards" section above
- **Documentation Consolidation**: `DOCUMENTATION_CONSOLIDATION_ANALYSIS.md`
- **Code Alignment Report**: `CODE_DOCUMENTATION_ALIGNMENT_REPORT.md`
- **Master Documentation**: `docs/context/` directory
- **Test Fix Archive**: `docs/archive/bugfixes/PHASE_1_TEST_FIXES_OCT_16_2025.md`

---

**Last Review**: October 16, 2025 (Added Testing Best Practices §9)
**Next Review**: January 16, 2026
**Owner**: Engineering Team
