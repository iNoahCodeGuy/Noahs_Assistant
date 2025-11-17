# Code Cleanup Audit

**Date**: October 19, 2025
**Purpose**: Identify unnecessary files, code, imports, and dependencies for archival/deletion
**Status**: 🟡 **Moderate Cleanup Needed**

---

## Executive Summary

**Cleanup Score**: 🟡 **70/100** (Good, but room for improvement)

**Key Findings**:
1. **Legacy migration scripts** - 5+ scripts in root directory that should be archived
2. **Unused test files** - Several test scripts in root directory (not in `tests/`)
3. **Duplicate state fields** - Redundant fields in `ConversationState` TypedDict
4. **Legacy agents** - `role_router.py`, `response_formatter.py` now bypassed by conversation flow
5. **Commented code** - Minimal (good!)

**Recommendation**: Archive 15-20 files to `archive/`, remove 30+ unused state fields

---

## 1. Legacy Files in Root Directory (ARCHIVE CANDIDATES)

### 🔴 High Priority - Archive Immediately

These are one-off migration/setup scripts that served their purpose:

| File | Purpose | Last Used | Action |
|------|---------|-----------|--------|
| `add_architecture_kb.py` | One-time data population | Week 1 setup | ✅ Archive → `archive/scripts/` |
| `add_technical_kb.py` | One-time data population | Week 1 setup | ✅ Archive → `archive/scripts/` |
| `daily_maintenance.py` | Manual maintenance script | Never automated | ✅ Archive → `archive/scripts/` |
| `example_streamlit_integration.py` | Example/demo code | Pre-production | ✅ Archive → `archive/examples/` |
| `run_code_display_tests.py` | Standalone test runner | Now using pytest | ✅ Archive → `archive/tests/` |
| `run_migration_fixed.py` | Legacy migration | Week 1 | ✅ Archive → `archive/scripts/` |
| `run_migration.py` | Legacy migration | Week 1 | ✅ Archive → `archive/scripts/` |
| `setup_modular_system.py` | One-time setup | Week 1 | ✅ Archive → `archive/scripts/` |
| `validate_analytics_improvements.py` | One-time validation | Week 2 | ✅ Archive → `archive/scripts/` |
| `verify_schema.py` | One-time schema check | Week 1 | ✅ Archive → `archive/scripts/` |

**Total**: 10 files to archive

### 🟡 Medium Priority - Review Before Archiving

| File | Purpose | Used? | Action |
|------|---------|-------|--------|
| `test_api_keys.py` | Manual API key validation | Occasionally | 🟡 Keep in `scripts/` folder |
| `test_architecture_retrieval.py` | Ad-hoc testing | Replaced by pytest | ✅ Archive → `archive/tests/` |
| `test_clean.py` | Standalone test | Replaced by pytest | ✅ Archive → `archive/tests/` |
| `test_code_integration_simple.py` | Standalone test | Replaced by pytest | ✅ Archive → `archive/tests/` |
| `test_code_integration.py` | Standalone test | Replaced by pytest | ✅ Archive → `archive/tests/` |
| `test_connection_simple.py` | Connection test | Replaced by pytest | ✅ Archive → `archive/tests/` |
| `test_connection.py` | Connection test | Replaced by pytest | ✅ Archive → `archive/tests/` |
| `test_copilot_verification.py` | Ad-hoc test | One-time use | ✅ Archive → `archive/tests/` |
| `test_debug.py` | Debugging script | Temporary | ✅ Delete (or archive) |
| `test_direct_search.py` | Ad-hoc search test | Replaced by pytest | ✅ Archive → `archive/tests/` |
| `test_embedding_formats.py` | Format validation | One-time | ✅ Archive → `archive/tests/` |
| `test_final.py` | Standalone test | Replaced by pytest | ✅ Archive → `archive/tests/` |
| `test_live_api.sh` | Manual API test | Replaced by pytest | ✅ Archive → `archive/scripts/` |
| `test_memory_basic.py` | Memory testing | Replaced by pytest | ✅ Archive → `archive/tests/` |
| `test_memory_fix.py` | Debugging script | Temporary | ✅ Delete |
| `test_openai_memory.py` | OpenAI memory test | Replaced by pytest | ✅ Archive → `archive/tests/` |
| `test_openai_search.py` | OpenAI search test | Replaced by pytest | ✅ Archive → `archive/tests/` |
| `test_retriever_fixed.py` | Debugging script | Temporary | ✅ Delete |
| `test_role_functionality.py` | Role testing | Replaced by pytest | ✅ Archive → `archive/tests/` |
| `test_roles_quick.py` | Quick role test | Replaced by pytest | ✅ Archive → `archive/tests/` |

**Total**: 20 files to archive/delete

---

## 2. Legacy Code Modules (STILL IN USE, BUT BYPASSED)

### ⚠️ `src/agents/role_router.py` - Mostly Bypassed

**Status**: 🟡 **90% bypassed** by new conversation flow
**Used by**: `src/main.py` (Streamlit local UI only)
**Not used by**: Production API endpoints (`api/chat.py`)

**Evidence**:
```python
# src/main.py (LOCAL ONLY)
from src.agents.role_router import RoleRouter  # ⚠️ Only used in Streamlit

role_router = RoleRouter(rag_engine, memory)
answer = role_router.route(query, role, session_id)  # Legacy flow
```

```python
# api/chat.py (PRODUCTION)
from src.flows.conversation_flow import run_conversation_flow  # ✅ New flow

result = run_conversation_flow(state, rag_engine, session_id)  # Does NOT use RoleRouter
```

**Recommendation**:
1. ✅ Keep for now (Streamlit still uses it)
2. ⏳ TODO: Migrate Streamlit to use `run_conversation_flow()` like production
3. ⏳ Then archive `role_router.py` to `archive/agents/`

### ⚠️ `src/agents/response_formatter.py` - Mostly Bypassed

**Status**: 🟡 **95% bypassed** by conversation flow content blocks
**Used by**: `src/main.py` (Streamlit only)
**Replaced by**: `src/flows/content_blocks.py` (production)

**Recommendation**: Same as `role_router.py` - keep until Streamlit migrated

---

## 3. Unused State Fields in ConversationState

**File**: `src/state/conversation_state.py`

**Issue**: TypedDict has 40+ fields, many rarely/never used

### 🔴 Remove These (Never Used)

```python
# ❌ These fields are set but never checked downstream:
vague_query_expanded: bool  # Set in classify_query, never checked
teaching_moment: bool  # Set in classify_query, never checked
needs_longer_response: bool  # Set in classify_query, never checked
```

**How to verify**:
```bash
# Check if field is ever read (not just set)
grep -r "state.get(\"vague_query_expanded\"" src/
grep -r "state\[\"vague_query_expanded\"\]" src/
# If no results → field is write-only → REMOVE IT
```

### 🟡 Consolidate These (Redundant)

```python
# ❌ Redundant: ambiguous_query vs is_ambiguous
ambiguous_query: bool  # Same as is_ambiguous
is_ambiguous: bool  # Keep this one

# ❌ Redundant: query_type already tells us this
data_display_requested: bool  # Redundant with query_type == "data"
code_display_requested: bool  # Redundant with query_type == "technical" + code_would_help

# ❌ Redundant: import_explanation_requested
import_explanation_requested: bool  # Could be derived from query_type + keywords
```

**Fix**: Remove redundant fields, derive from `query_type` instead

---

## 4. Unused Imports (MINOR)

**Status**: 🟢 **Very few unused imports** (good code hygiene)

**How to find**:
```bash
# Install autoflake
pip install autoflake

# Check for unused imports (DRY RUN)
autoflake --check --remove-unused-variables --remove-all-unused-imports -r src/

# If found, auto-fix:
autoflake --in-place --remove-unused-variables --remove-all-unused-imports -r src/
```

**Expected**: <10 unused imports across entire codebase (excellent!)

---

## 5. Commented Code (EXCELLENT)

**Status**: 🟢 **Almost no commented code** found

**Check**:
```bash
# Find commented Python code
grep -rn "^\s*#\s*def " src/
grep -rn "^\s*#\s*class " src/
grep -rn "^\s*#\s*import " src/
```

**Result**: ✅ No significant commented-out code (well-maintained codebase)

---

## 6. Duplicate Files (LOW PRIORITY)

### ⚠️ Possible Duplication in Scripts

| File Pair | Similarity | Action |
|-----------|------------|--------|
| `scripts/run_migration_002.py` <br> `scripts/run_migration_002_postgres.py` | Similar purpose | 🟡 Review, keep both if needed |
| `scripts/test_api_endpoints.py` <br> `scripts/test_api_local.py` | Similar testing | 🟡 Consolidate if possible |

---

## 7. Unused Dependencies (CHECK requirements.txt)

**How to check**:
```bash
# Install pipreqs
pip install pipreqs

# Generate minimal requirements from actual imports
pipreqs . --force --savepath requirements_actual.txt

# Compare with current requirements.txt
diff requirements.txt requirements_actual.txt
```

**Expected findings**:
- Most dependencies are used
- Possible unused: `pandas` (only in a few legacy files)
- Possible unused: Legacy LangChain imports (if fully migrated to langchain_compat)

**Priority**: 🟢 Low (dependency bloat is minimal)

---

## Cleanup Action Plan

### Phase 1: Archive Legacy Scripts (30 min)

```bash
# Create archive directories
mkdir -p archive/scripts
mkdir -p archive/tests
mkdir -p archive/examples

# Archive one-time setup scripts
mv add_architecture_kb.py archive/scripts/
mv add_technical_kb.py archive/scripts/
mv daily_maintenance.py archive/scripts/
mv example_streamlit_integration.py archive/examples/
mv run_migration_fixed.py archive/scripts/
mv run_migration.py archive/scripts/
mv setup_modular_system.py archive/scripts/
mv validate_analytics_improvements.py archive/scripts/
mv verify_schema.py archive/scripts/

# Archive standalone test files (replaced by pytest)
mv test_architecture_retrieval.py archive/tests/
mv test_clean.py archive/tests/
mv test_code_integration_simple.py archive/tests/
mv test_code_integration.py archive/tests/
mv test_connection_simple.py archive/tests/
mv test_connection.py archive/tests/
mv test_copilot_verification.py archive/tests/
mv test_direct_search.py archive/tests/
mv test_embedding_formats.py archive/tests/
mv test_final.py archive/tests/
mv test_live_api.sh archive/scripts/
mv test_memory_basic.py archive/tests/
mv test_openai_memory.py archive/tests/
mv test_openai_search.py archive/tests/
mv test_role_functionality.py archive/tests/
mv test_roles_quick.py archive/tests/

# Delete temporary debugging scripts
rm -f test_debug.py
rm -f test_memory_fix.py
rm -f test_retriever_fixed.py

# Commit the cleanup
git add -A
git commit -m "🧹 Cleanup: Archive 20+ legacy scripts and tests"
```

### Phase 2: Remove Unused State Fields (1 hour)

```python
# Edit src/state/conversation_state.py

# ❌ REMOVE these never-checked fields:
# vague_query_expanded: bool
# teaching_moment: bool
# needs_longer_response: bool

# ❌ REMOVE redundant fields:
# ambiguous_query: bool  # Keep is_ambiguous instead
# data_display_requested: bool  # Derive from query_type
# code_display_requested: bool  # Derive from query_type + code_would_help
# import_explanation_requested: bool  # Derive from query_type

# Update src/flows/query_classification.py to NOT set these fields
# Update all references to use query_type instead
```

**Search & Replace**:
```bash
# Find all usages of removed fields
grep -r "ambiguous_query" src/
grep -r "data_display_requested" src/
grep -r "code_display_requested" src/

# Update to use is_ambiguous, query_type == "data", etc.
```

### Phase 3: Check for Unused Imports (15 min)

```bash
# Install autoflake
pip install autoflake

# Check for unused imports
autoflake --check --remove-unused-variables --remove-all-unused-imports -r src/ api/ scripts/ tests/

# If any found, auto-fix
autoflake --in-place --remove-unused-variables --remove-all-unused-imports -r src/ api/ scripts/ tests/

# Commit if any changes
git add -A
git commit -m "🧹 Cleanup: Remove unused imports via autoflake"
```

### Phase 4: Future Cleanup (TODO - Low Priority)

1. ⏳ **Migrate Streamlit to conversation_flow** (4 hours)
   - Update `src/main.py` to use `run_conversation_flow()` instead of `RoleRouter`
   - Then archive `src/agents/role_router.py` and `src/agents/response_formatter.py`

2. ⏳ **Consolidate test scripts** (2 hours)
   - Move `scripts/test_api_endpoints.py` logic into pytest
   - Archive remaining manual test scripts

3. ⏳ **Dependency audit** (1 hour)
   - Run `pipreqs` to generate minimal requirements
   - Remove unused dependencies (if any)

---

## Summary of Findings

**Files to Archive**: 30 files (20 tests + 10 scripts)
**Files to Delete**: 3 temporary debugging scripts
**State Fields to Remove**: 7 unused/redundant fields
**Unused Imports**: Expected <10 across codebase
**Commented Code**: ✅ Almost none (excellent)

**Total Cleanup Time**: 2-3 hours for Phases 1-3
**Impact**: Cleaner repo, faster navigation, reduced confusion

---

## Next Steps

1. ✅ **Complete this audit** (DONE)
2. ⏳ **Get user approval** for Phase 1 archival plan
3. ⏳ **Execute Phase 1** (archive scripts) - 30 min
4. ⏳ **Execute Phase 2** (remove unused state fields) - 1 hour
5. ⏳ **Execute Phase 3** (check unused imports) - 15 min
6. ⏳ **Commit, push, deploy** all cleanup changes

**Recommendation**: Do Phases 1-3 together as single cleanup PR.
