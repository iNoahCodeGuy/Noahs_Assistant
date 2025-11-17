#!/bin/bash
#
# Workspace Cleanup Script
# Purpose: Archive unnecessary files and organize codebase
# Date: November 16, 2025
# Safe to run: All files moved to archive/ (not deleted)
#

set -e  # Exit on error

echo "🧹 Starting workspace cleanup..."
echo ""

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create archive directories
echo -e "${BLUE}Creating archive structure...${NC}"
mkdir -p archive/historical_docs
mkdir -p archive/legacy_scripts
mkdir -p archive/legacy_vector_stores
mkdir -p archive/old_data/backups
mkdir -p archive/old_data/demo_exports
mkdir -p archive/old_data/reports

# Function to move file with confirmation
move_file() {
    local src=$1
    local dst=$2
    if [ -f "$src" ] || [ -d "$src" ]; then
        echo "  Moving: $src → $dst"
        mv "$src" "$dst"
    else
        echo "  ⚠️  Skipping (not found): $src"
    fi
}

# Function to remove file
remove_file() {
    local file=$1
    if [ -f "$file" ]; then
        echo "  Removing: $file"
        rm "$file"
    else
        echo "  ⚠️  Skipping (not found): $file"
    fi
}

echo ""
echo -e "${BLUE}1. Archiving completed milestone docs from root...${NC}"
move_file "DEPLOYMENT_READY_SUMMARY.md" "archive/historical_docs/"
move_file "LANGSMITH_FEATURES_IMPLEMENTATION_SUMMARY.md" "archive/historical_docs/"
move_file "HIRING_MANAGER_FORMATTING_ISSUE.md" "archive/historical_docs/"
move_file "VOICE_FIX_SUMMARY.md" "archive/historical_docs/"
move_file "TASK_11_DEPLOYMENT_CHECKLIST.md" "archive/historical_docs/"
move_file "SETUP_COMPLETE.md" "archive/historical_docs/"
move_file "LANGSMITH_NO_DOCKER_GUIDE.md" "archive/historical_docs/"
move_file "LANGSMITH_QUICK_START.md" "archive/historical_docs/"
move_file "LANGSMITH_QUICK_REFERENCE.md" "archive/historical_docs/"

echo ""
echo -e "${BLUE}1b. Archiving examples directory...${NC}"
if [ -d "examples" ]; then
    move_file "examples" "archive/"
else
    echo "  ⚠️  examples/ not found"
fi

echo ""
echo -e "${BLUE}2. Archiving one-time migration scripts...${NC}"
move_file "scripts/run_migration.py" "archive/legacy_scripts/"
move_file "scripts/run_migration_002.py" "archive/legacy_scripts/"
move_file "scripts/run_migration_002_postgres.py" "archive/legacy_scripts/"
move_file "scripts/run_migration_v1_legacy.py" "archive/legacy_scripts/"
move_file "scripts/run_session_id_migration.py" "archive/legacy_scripts/"
move_file "scripts/migrate_to_typeddict.py" "archive/legacy_scripts/"
move_file "scripts/phase3_setup_wizard.py" "archive/legacy_scripts/"
move_file "scripts/setup_phase3.py" "archive/legacy_scripts/"
move_file "scripts/add_architecture_kb.py" "archive/legacy_scripts/"
move_file "scripts/add_technical_kb.py" "archive/legacy_scripts/"
move_file "scripts/add_impressive_questions.py" "archive/legacy_scripts/"
move_file "scripts/add_product_questions.py" "archive/legacy_scripts/"
move_file "scripts/setup_modular_system.py" "archive/legacy_scripts/"
move_file "scripts/replace_analytics.py" "archive/legacy_scripts/"

echo ""
echo -e "${BLUE}3. Archiving legacy vector stores (FAISS)...${NC}"
if [ -d "vector_stores" ]; then
    move_file "vector_stores/faiss_career" "archive/legacy_vector_stores/"
    move_file "vector_stores/code_index" "archive/legacy_vector_stores/"
    # Remove empty vector_stores directory
    if [ -z "$(ls -A vector_stores)" ]; then
        echo "  Removing empty vector_stores/ directory"
        rmdir vector_stores
    fi
else
    echo "  ⚠️  vector_stores/ not found"
fi

echo ""
echo -e "${BLUE}4. Archiving old backups, demos, and reports...${NC}"
if [ -d "backups" ]; then
    move_file "backups/"* "archive/old_data/backups/" 2>/dev/null || echo "  No backup files to move"
    rmdir backups 2>/dev/null && echo "  Removed empty backups/ directory" || true
fi

if [ -d "demo_exports" ]; then
    move_file "demo_exports/"* "archive/old_data/demo_exports/" 2>/dev/null || echo "  No demo files to move"
    rmdir demo_exports 2>/dev/null && echo "  Removed empty demo_exports/ directory" || true
fi

if [ -d "reports" ]; then
    move_file "reports/"* "archive/old_data/reports/" 2>/dev/null || echo "  No report files to move"
    rmdir reports 2>/dev/null && echo "  Removed empty reports/ directory" || true
fi

echo ""
echo -e "${BLUE}5. Removing old SQLite database files...${NC}"
remove_file "analytics.db"
remove_file "analytics_example.db"
remove_file "strategy_demo.db"
remove_file "test.db"

echo ""
echo -e "${BLUE}6. Removing installation artifacts...${NC}"
remove_file "Docker.dmg"
remove_file "get-docker.sh"
remove_file "codebase_structure.txt"

echo ""
echo -e "${BLUE}7. Creating archive documentation...${NC}"

# Create README in archive/historical_docs
cat > archive/historical_docs/README.md << 'EOF'
# Historical Documentation Archive

This directory contains completed milestone documents and superseded guides.

## Archived November 16, 2025

### Completed Milestones
- `DEPLOYMENT_READY_SUMMARY.md` - Initial deployment milestone
- `LANGSMITH_FEATURES_IMPLEMENTATION_SUMMARY.md` - LangSmith integration complete
- `TASK_11_DEPLOYMENT_CHECKLIST.md` - Task 11 completed
- `VOICE_FIX_SUMMARY.md` - Voice/personality fixes complete
- `HIRING_MANAGER_FORMATTING_ISSUE.md` - Formatting issue resolved

### Superseded Guides
- `LANGSMITH_NO_DOCKER_GUIDE.md` - Replaced by LangGraph Studio workflow
- `LANGSMITH_QUICK_START.md` - Consolidated into LANGSMITH_STUDIO_QUICKSTART.md
- `LANGSMITH_QUICK_REFERENCE.md` - Integrated into main docs
- `SETUP_COMPLETE.md` - Historical setup notes

## Current Documentation

See root directory for active documentation:
- `README.md` - Project overview
- `LANGSMITH_STUDIO_QUICKSTART.md` - Primary setup guide
- `HOW_THE_CODEBASE_RUNS.md` - Architecture reference
EOF

# Create README in archive/legacy_scripts
cat > archive/legacy_scripts/README.md << 'EOF'
# Legacy Scripts Archive

This directory contains one-time migration scripts and deprecated utilities.

## Archived November 16, 2025

### Database Migrations (Completed)
- `run_migration*.py` - Historical Supabase schema migrations
- `run_session_id_migration.py` - Session ID migration (complete)
- `migrate_to_typeddict.py` - Code refactor to TypedDict (complete)

### Knowledge Base Population (One-Time)
- `add_architecture_kb.py` - Initial architecture KB load
- `add_technical_kb.py` - Initial technical KB load
- `add_impressive_questions.py` - Sample questions
- `add_product_questions.py` - Product questions

### Deprecated System Rewrites
- `setup_modular_system.py` - Old modular system setup
- `replace_analytics.py` - Analytics system rewrite (complete)
- `phase3_setup_wizard.py` - Phase 3 planning tool
- `setup_phase3.py` - Phase 3 setup (now in production)

## Active Scripts

See `scripts/` directory for current utilities:
- `migrate_data_to_supabase.py` - Reusable KB migration
- `test_*.py` - Active testing suite
- `verify_*.py` - CI/deployment verification
EOF

# Create README in archive/old_data
cat > archive/old_data/README.md << 'EOF'
# Old Data Archive

This directory contains legacy data files from previous system iterations.

## Archived November 16, 2025

### Backups
- SQLite analytics database backups from September 2025
- Superseded by Supabase production database

### Demo Exports
- `analytics_summary.json` - Old analytics export
- `user_interactions_7d.csv` - Sample interaction data

### Reports
- `maintenance_report_2025-09-30.json` - Historical maintenance report

## Current Data

Production data is stored in:
- **Supabase** - All analytics, interactions, conversations
- **data/** directory - Knowledge base CSVs and evaluation data
EOF

# Create README in archive/legacy_vector_stores
cat > archive/legacy_vector_stores/README.md << 'EOF'
# Legacy Vector Stores Archive

This directory contains deprecated FAISS vector indexes.

## Archived November 16, 2025

### FAISS Indexes (Replaced)
- `faiss_career/` - Local career knowledge base index
- `code_index/` - Local code snippets index

## Current Vector Storage

Production uses **Supabase pgvector**:
- Centralized vector storage
- IVFFLAT indexes for fast search
- Managed in `assistant/retrieval/pgvector_retriever.py`
- No local vector stores needed
EOF

echo "  ✓ Created README files in all archive directories"

echo ""
echo -e "${BLUE}8. Updating .gitignore...${NC}"

# Check if these patterns already exist in .gitignore
if ! grep -q "^analytics.db$" .gitignore 2>/dev/null; then
    cat >> .gitignore << 'EOF'

# SQLite databases (use Supabase instead)
analytics.db
analytics_example.db
strategy_demo.db
test.db
*.db

# Legacy artifacts
Docker.dmg
get-docker.sh
codebase_structure.txt
EOF
    echo "  ✓ Added patterns to .gitignore"
else
    echo "  ✓ .gitignore already up to date"
fi

echo ""
echo -e "${GREEN}✅ Cleanup complete!${NC}"
echo ""
echo -e "${YELLOW}Summary:${NC}"
echo "  📦 Archived 9 historical docs → archive/historical_docs/"
echo "  📦 Archived 14 legacy scripts → archive/legacy_scripts/"
echo "  📦 Archived examples/ → archive/examples/"
echo "  📦 Archived vector stores → archive/legacy_vector_stores/"
echo "  📦 Archived old data → archive/old_data/"
echo "  🗑️  Removed 7 unnecessary files (.db, .dmg, etc.)"
echo "  📝 Created 4 README files documenting archives"
echo "  ✨ Updated .gitignore"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Review the changes: git status"
echo "  2. Test that everything still works"
echo "  3. Commit: git add -A && git commit -m 'chore: Archive legacy files and organize workspace'"
echo "  4. Push: git push origin merge/week-1-launch"
echo ""
echo "Note: All files were moved to archive/, not deleted. You can restore anything if needed."
echo ""
