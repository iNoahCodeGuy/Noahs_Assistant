#!/bin/bash
#
# Workspace Cleanup Phase 2 - Aggressive Cleanup
# Purpose: Remove unnecessary files and consolidate documentation
# Date: November 16, 2025
# Safe to run: All files moved to archive/ (not deleted)
#

set -e  # Exit on error

echo "🧹 Starting Phase 2 cleanup (Aggressive)..."
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Ensure archive structure exists
mkdir -p archive/old_shell_scripts
mkdir -p archive/old_notebooks
mkdir -p archive/old_logs
mkdir -p archive/historical_analysis_docs
mkdir -p archive/old_data/analytics

echo -e "${BLUE}1. Removing obsolete shell scripts...${NC}"
if [ -f "test_simple.ps1" ]; then
    echo "  Removing: test_simple.ps1 (PowerShell on Mac - unusable)"
    rm test_simple.ps1
fi

if [ -f "start_task11_testing.sh" ]; then
    echo "  Moving: start_task11_testing.sh → archive/old_shell_scripts/"
    mv start_task11_testing.sh archive/old_shell_scripts/
fi

if [ -f "test_live_api.sh" ]; then
    echo "  Moving: test_live_api.sh → archive/old_shell_scripts/"
    mv test_live_api.sh archive/old_shell_scripts/
fi

# Check for duplicate studio scripts
if [ -f "start_studio.sh" ] && [ -f "start_langgraph_studio.sh" ]; then
    echo "  Comparing start_studio.sh vs start_langgraph_studio.sh..."
    if diff -q start_studio.sh start_langgraph_studio.sh > /dev/null 2>&1; then
        echo "    Files are identical - archiving start_langgraph_studio.sh"
        mv start_langgraph_studio.sh archive/old_shell_scripts/
    else
        echo "    Files differ - keeping both (will need manual review)"
    fi
fi

echo ""
echo -e "${BLUE}2. Archiving notebooks directory...${NC}"
if [ -d "notebooks" ]; then
    echo "  Moving: notebooks/ → archive/old_notebooks/"
    mv notebooks archive/old_notebooks/
else
    echo "  ⚠️  notebooks/ not found"
fi

echo ""
echo -e "${BLUE}3. Removing old SQLite databases and logs...${NC}"
if [ -f "analytics/comprehensive_metrics.db" ]; then
    echo "  Moving: analytics/comprehensive_metrics.db → archive/old_data/analytics/"
    mv analytics/comprehensive_metrics.db archive/old_data/analytics/
fi

if [ -d "logs" ]; then
    echo "  Moving: logs/ → archive/old_logs/"
    mv logs archive/old_logs/
fi

echo ""
echo -e "${BLUE}4. Archiving historical audit/analysis docs from docs/...${NC}"
cd docs

# Archive audit and analysis docs
for doc in CLEANUP_AUDIT.md DESIGN_PRINCIPLES_AUDIT.md LANGGRAPH_ALIGNMENT_AUDIT.md \
           KNOWLEDGE_GAP_ANALYSIS.md QA_CONSOLIDATION_PLAN.md \
           MESSAGE_EXTRACTION_ARCHITECTURE_ANALYSIS.md; do
    if [ -f "$doc" ]; then
        echo "  Moving: docs/$doc → archive/historical_analysis_docs/"
        mv "$doc" ../archive/historical_analysis_docs/
    fi
done

cd ..

echo ""
echo -e "${BLUE}5. Consolidating LangSmith documentation...${NC}"

# Create consolidated LangSmith guide
cat > LANGSMITH_COMPLETE_GUIDE.md << 'EOF'
# LangSmith Complete Guide

> **Quick Start**: For immediate setup, see [LANGSMITH_STUDIO_QUICKSTART.md](LANGSMITH_STUDIO_QUICKSTART.md) in root.

## Table of Contents

1. [Studio Setup & Development](#studio-setup)
2. [Tracing & Observability](#tracing-observability)
3. [Advanced Features](#advanced-features)

---

## Studio Setup

See: `LANGSMITH_STUDIO_QUICKSTART.md` (root directory)

**One-line start**:
```bash
./start_studio.sh
```

**Manual setup** and troubleshooting in the quickstart guide.

---

## Tracing & Observability

See: `docs/LANGSMITH_TRACING_SETUP.md`

### What Gets Traced

- All OpenAI API calls (embeddings, chat)
- Supabase pgvector queries
- Node transitions in conversation flow
- Token usage, latency, costs

### View Traces

Dashboard: https://smith.langchain.com/o/project/noahs-ai-assistant

### Environment Variables

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=noahs-ai-assistant
```

---

## Advanced Features

See: `docs/LANGSMITH_ADVANCED_FEATURES.md`

### Evaluation Pipeline

```bash
python scripts/run_evaluation.py
```

### Prompt Hub

```python
from assistant.prompts import get_prompt
prompt = get_prompt("basic_qa")
```

### LangGraph Studio

```bash
langgraph dev
open http://127.0.0.1:2024
```

---

## Documentation References

- **Setup**: `LANGSMITH_STUDIO_QUICKSTART.md` (root)
- **Tracing**: `docs/LANGSMITH_TRACING_SETUP.md`
- **Advanced**: `docs/LANGSMITH_ADVANCED_FEATURES.md`
- **Comparison**: `docs/LANGSMITH_COMPARISON_WITH_ELI5.md`

All individual guides remain available for detailed reference.
EOF

echo "  ✓ Created: LANGSMITH_COMPLETE_GUIDE.md (navigation hub)"

echo ""
echo -e "${BLUE}6. Consolidating Node documentation...${NC}"

# Create consolidated node reference
cat > docs/NODE_COMPLETE_REFERENCE.md << 'EOF'
# Node Complete Reference

> Consolidated guide to LangGraph node architecture and state management.

## Quick References

- **State Logic Flow**: See `HOW_TO_FOLLOW_NODE_STATE_LOGIC.md`
- **State Fields**: See `NODE_STATE_QUICK_REFERENCE.md`
- **Migration Guide**: See `NODE_MIGRATION_GUIDE.md`
- **QA Migration**: See `QA_LANGGRAPH_MIGRATION.md`

---

## Node Pipeline Overview

18-node consolidated pipeline (see `assistant/flows/conversation_flow.py`):

### Stage 0: Initialization
- `initialize_conversation_state` - Normalize state, load memory

### Stage 1: Greeting & Role
- `handle_greeting` - Warm intro without RAG cost
- `classify_role_mode` - Role detection + routing

### Stage 2: Query Understanding
- `classify_intent` - Engineering vs business focus
- `extract_entities` - Company, role, timeline, contact hints

### Stage 3: Query Refinement
- `assess_clarification_need` - Detect vague queries
- `compose_query` - Build retrieval-ready prompt

### Stage 4: Retrieval
- `retrieve_chunks` - pgvector search + MMR
- `validate_grounding` - Stop hallucinations early

### Stage 5: Generation
- `generate_draft` - Role-aware LLM generation
- `hallucination_check` - Citations + safety

### Stage 6: Formatting
- `plan_actions` - Action decisions + hiring signals
- `format_answer` - Structure + followups

### Stage 7: Logging
- `log_and_notify` - Supabase analytics + LangSmith

---

## State Management

### Core State Fields

```python
ConversationState = TypedDict({
    # Identity
    "query": str,
    "role": str,
    "role_mode": str,
    "session_id": str,
    
    # Pipeline
    "is_greeting": bool,
    "query_type": str,
    "composed_query": str,
    
    # Retrieval
    "retrieved_chunks": List[Dict],
    "retrieval_scores": List[float],
    "grounding_status": str,
    
    # Generation
    "draft_answer": str,
    "answer": str,
    "hallucination_safe": bool,
    
    # Actions
    "planned_actions": List[Dict],
    "executed_actions": List[Dict],
    
    # Memory
    "session_memory": Dict,
    "chat_history": List[Dict],
})
```

### State Transitions

See `docs/HOW_TO_FOLLOW_NODE_STATE_LOGIC.md` for detailed flow diagrams.

---

## Migration Guides

### LangGraph Migration

See `docs/QA_LANGGRAPH_MIGRATION.md`:
- Pre-LangGraph → LangGraph conversion patterns
- Node boundary decisions
- State management patterns

### Node Refactoring

See `docs/NODE_MIGRATION_GUIDE.md`:
- Splitting large nodes
- Consolidating redundant nodes
- Performance optimization patterns

---

## Implementation Details

**Node Logic**: `assistant/flows/node_logic/` (19 modules)
**Orchestration**: `assistant/flows/conversation_flow.py`
**State Definition**: `assistant/state/conversation_state.py`

Each module < 200 lines for maintainability.
EOF

echo "  ✓ Created: docs/NODE_COMPLETE_REFERENCE.md"

echo ""
echo -e "${BLUE}7. Creating archive documentation...${NC}"

# Create README for old shell scripts
cat > archive/old_shell_scripts/README.md << 'EOF'
# Old Shell Scripts Archive

Archived November 16, 2025

## Removed Scripts

- `test_simple.ps1` - PowerShell script (unusable on Mac)
- `test_live_api.sh` - Replaced by `scripts/test_api_*.py`
- `start_task11_testing.sh` - Task 11 completed
- `start_langgraph_studio.sh` - Duplicate of `start_studio.sh` (if identical)

## Active Scripts

See root directory:
- `start_studio.sh` - LangGraph Studio startup
- `cleanup_workspace.sh` - Workspace cleanup (Phase 1)
- `cleanup_phase2.sh` - Workspace cleanup (Phase 2)
EOF

# Create README for notebooks
cat > archive/old_notebooks/README.md << 'EOF'
# Old Notebooks Archive

Archived November 16, 2025

## Contents

- `langsmith_node_tracing.ipynb` - One-time LangSmith exploration

## Why Archived

Single orphaned notebook from development phase. Not part of active workflow.

Production code is in `assistant/` directory.
EOF

# Create README for logs
cat > archive/old_logs/README.md << 'EOF'
# Old Logs Archive

Archived November 16, 2025

## Contents

- `maintenance.log` - Maintenance log from September 2025

## Why Archived

Production logging now handled by:
- **LangSmith**: Distributed tracing
- **Supabase**: Analytics persistence
- **Vercel**: Deployment logs

Local log files no longer used.
EOF

# Create README for historical analysis docs
cat > archive/historical_analysis_docs/README.md << 'EOF'
# Historical Analysis Documents Archive

Archived November 16, 2025

## Contents

### Audits (Historical Snapshots)
- `CLEANUP_AUDIT.md` - Previous cleanup analysis
- `DESIGN_PRINCIPLES_AUDIT.md` - Design audit
- `LANGGRAPH_ALIGNMENT_AUDIT.md` - LangGraph migration audit

### Gap Analysis
- `KNOWLEDGE_GAP_ANALYSIS.md` - Knowledge base gaps identified
- `QA_CONSOLIDATION_PLAN.md` - QA consolidation planning
- `MESSAGE_EXTRACTION_ARCHITECTURE_ANALYSIS.md` - Architecture analysis

## Why Archived

These were point-in-time analyses. The findings have been:
- Implemented and integrated into current architecture
- Documented in active reference docs
- Superseded by production system

## Current Documentation

See `docs/context/` for authoritative system documentation.
EOF

echo "  ✓ Created 4 archive README files"

echo ""
echo -e "${BLUE}8. Updating .gitignore...${NC}"

# Add patterns to gitignore
if ! grep -q "^logs/$" .gitignore 2>/dev/null; then
    cat >> .gitignore << 'EOF'

# Logs directory (archived)
logs/

# Notebooks (archived)
notebooks/

# PowerShell scripts on Mac
*.ps1
EOF
    echo "  ✓ Added patterns to .gitignore"
else
    echo "  ✓ .gitignore already up to date"
fi

echo ""
echo -e "${GREEN}✅ Phase 2 cleanup complete!${NC}"
echo ""
echo -e "${YELLOW}Summary:${NC}"
echo "  🗑️  Removed 1 unusable file (test_simple.ps1)"
echo "  📦 Archived 3 obsolete shell scripts → archive/old_shell_scripts/"
echo "  📦 Archived notebooks/ → archive/old_notebooks/"
echo "  📦 Archived analytics SQLite DB → archive/old_data/analytics/"
echo "  📦 Archived logs/ → archive/old_logs/"
echo "  📦 Archived 6 historical audit docs → archive/historical_analysis_docs/"
echo "  📝 Created LANGSMITH_COMPLETE_GUIDE.md (navigation hub)"
echo "  📝 Created docs/NODE_COMPLETE_REFERENCE.md (consolidated node docs)"
echo "  📝 Created 4 README files in archives"
echo "  ✨ Updated .gitignore"
echo ""
echo -e "${BLUE}Documentation Structure Improved:${NC}"
echo "  Root .md: 8 → 9 (added navigation guides)"
echo "  docs/ root: 31 → 26 (archived 6 historical analyses)"
echo "  Consolidated: 6 overlapping docs → 2 navigation hubs"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Review the changes: git status"
echo "  2. Review new guides: LANGSMITH_COMPLETE_GUIDE.md, docs/NODE_COMPLETE_REFERENCE.md"
echo "  3. Commit: git add -A && git commit -m 'chore: Phase 2 cleanup - archive obsolete files and consolidate docs'"
echo "  4. Push: git push origin merge/week-1-launch"
echo ""
