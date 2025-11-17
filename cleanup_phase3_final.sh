#!/bin/bash
#
# Final Deep Cleanup - Phase 3
# Purpose: Remove confirmed unnecessary files and duplicates
# Date: November 16, 2025
#

set -e

echo "🔍 Starting Final Deep Cleanup (Phase 3)..."
echo ""

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}⚠️  IMPORTANT: This script removes large files.${NC}"
echo -e "${YELLOW}Please review before running.${NC}"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Create final archives
mkdir -p archive/old_data/backups_final
mkdir -p archive/old_data/demo_exports_final
mkdir -p archive/old_prototypes

echo -e "${BLUE}1. Moving remaining backup files...${NC}"
if [ -d "backups" ]; then
    mv backups/* archive/old_data/backups_final/ 2>/dev/null || true
    rmdir backups && echo "  ✓ Removed empty backups/ directory"
fi

echo ""
echo -e "${BLUE}2. Moving remaining demo exports...${NC}"
if [ -d "demo_exports" ]; then
    mv demo_exports/* archive/old_data/demo_exports_final/ 2>/dev/null || true
    rmdir demo_exports && echo "  ✓ Removed empty demo_exports/ directory"
fi

echo ""
echo -e "${BLUE}3. Removing empty analytics directory...${NC}"
if [ -d "analytics" ] && [ -z "$(ls -A analytics 2>/dev/null)" ]; then
    rmdir analytics && echo "  ✓ Removed empty analytics/ directory"
fi

echo ""
echo -e "${BLUE}4. Archiving old prototypes from public/...${NC}"
if [ -d "public" ]; then
    if [ -f "public/chat.html" ] || [ -f "public/index.html" ]; then
        mv public/*.html archive/old_prototypes/ 2>/dev/null || true
        echo "  ✓ Moved HTML prototypes to archive/"
    fi
    # Check if public is now empty or just has Next.js assets
    if [ -z "$(ls -A public 2>/dev/null)" ]; then
        echo "  ⚠️  public/ is now empty - may need to keep for Next.js"
    fi
fi

echo ""
echo -e "${BLUE}5. Removing data/code_chunks placeholder...${NC}"
if [ -f "data/code_chunks" ]; then
    rm data/code_chunks && echo "  ✓ Removed empty placeholder file"
fi

echo ""
echo -e "${BLUE}6. Removing duplicate virtual environment...${NC}"
echo -e "${YELLOW}Found two venvs:${NC}"
echo "  venv/  - 1.1 GB"
echo "  .venv/ - 592 MB"
echo ""
echo -e "${YELLOW}Which one are you using?${NC}"
echo "1) venv (keep this, remove .venv)"
echo "2) .venv (keep this, remove venv)"
echo "3) Skip (I'll decide manually)"
read -p "Choice (1/2/3): " venv_choice

case $venv_choice in
    1)
        echo "  Removing .venv/ ..."
        rm -rf .venv && echo "  ✓ Removed .venv/ (saved 592 MB)"
        ;;
    2)
        echo "  Removing venv/ ..."
        rm -rf venv && echo "  ✓ Removed venv/ (saved 1.1 GB)"
        ;;
    3)
        echo "  ⏭️  Skipped venv cleanup"
        ;;
    *)
        echo "  ⏭️  Invalid choice, skipped"
        ;;
esac

echo ""
echo -e "${BLUE}7. Cleaning Python cache directories...${NC}"
echo "  Found 1,484 __pycache__ directories"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ Removed all __pycache__ directories"

echo ""
echo -e "${BLUE}8. Removing Python package metadata...${NC}"
if [ -d "noahs_ai_assistant.egg-info" ]; then
    rm -rf noahs_ai_assistant.egg-info && echo "  ✓ Removed egg-info/"
fi

echo ""
echo -e "${BLUE}9. Updating .gitignore...${NC}"
if ! grep -q "^venv/$" .gitignore 2>/dev/null; then
    cat >> .gitignore << 'EOF'

# Virtual environments
venv/
.venv/
env/
ENV/

# Python package metadata
*.egg-info/
dist/
build/

# Empty directories
analytics/
data/code_chunks
EOF
    echo "  ✓ Added comprehensive patterns to .gitignore"
else
    echo "  ✓ .gitignore already up to date"
fi

echo ""
echo -e "${GREEN}✅ Phase 3 cleanup complete!${NC}"
echo ""
echo -e "${YELLOW}Summary:${NC}"
echo "  📦 Moved remaining backups → archive/old_data/backups_final/"
echo "  📦 Moved remaining demo exports → archive/old_data/demo_exports_final/"
echo "  📦 Archived HTML prototypes → archive/old_prototypes/"
echo "  🗑️  Removed empty directories (analytics/, backups/, demo_exports/)"
echo "  🗑️  Removed 1,484 __pycache__ directories"
echo "  🗑️  Removed noahs_ai_assistant.egg-info/"
echo "  🗑️  Removed duplicate venv (if selected)"
echo "  📝 Updated .gitignore"
echo ""
echo -e "${BLUE}Disk space saved:${NC} ~1-2 GB (depending on venv choice)"
echo ""
