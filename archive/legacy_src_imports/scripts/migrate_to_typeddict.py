#!/usr/bin/env python3
"""
Migration Script: Delete Old Dataclass & Update All Imports

This script:
1. Updates all imports from old dataclass to new TypedDict
2. Verifies the old file can be safely deleted
3. Provides a report of changes

Design Principles Applied:
- SRP (#1): Single ConversationState definition
- YAGNI (#8): No dual implementations
- Alignment: All files use same source of truth
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

# Old and new import paths
OLD_IMPORT = "from src.state.conversation_state import"
NEW_IMPORT = "from src.state.conversation_state import"

# Files to update (found via grep)
FILES_TO_UPDATE = [
    "src/flows/action_execution.py",
    "src/flows/action_planning.py",
    "src/flows/conversation_flow.py",
    "src/flows/resume_distribution.py",
    "src/main.py",
    "api/chat.py",
    "tests/test_code_display_policy.py",
    "tests/test_code_display_accuracy.py",
    "tests/test_code_display_ci.py",
    "tests/test_conversation_flow.py",
    "tests/test_resume_distribution.py",
    "tests/test_error_handling.py",
    "verify_deployment.py",
    "test_vague_query.py",
    "test_data_display.py",
    "scripts/test_api_logic.py",
]

# Documentation files (update references only)
DOC_FILES = [
    "docs/NODE_MIGRATION_GUIDE.md",
]


def update_file(filepath):
    """Update imports in a single file. Returns (success, message) tuple."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if file has old import
        if OLD_IMPORT not in content:
            return (False, f"❌ No old import found in {filepath}")

        # Replace old import with new
        new_content = content.replace(OLD_IMPORT, NEW_IMPORT)

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return (True, f"✅ Updated {filepath}")

    except FileNotFoundError:
        return (False, f"⚠️  File not found: {filepath}")
    except Exception as e:
        return (False, f"❌ Error updating {filepath}: {e}")


def main():
    """Execute migration."""
    print("🚀 Starting ConversationState Migration")
    print("=" * 60)
    print(f"Old import: {OLD_IMPORT}")
    print(f"New import: {NEW_IMPORT}")
    print("=" * 60)
    print()

    # Update production files
    print("📁 Updating production files...")
    prod_updated = []
    prod_failed = []

    for filepath in FILES_TO_UPDATE:
        success, message = update_file(filepath)
        print(f"  {message}")
        if success:
            prod_updated.append(filepath)
        else:
            prod_failed.append(filepath)

    print()

    # Update documentation files
    print("📄 Updating documentation files...")
    doc_updated = []
    doc_failed = []

    for filepath in DOC_FILES:
        success, message = update_file(filepath)
        print(f"  {message}")
        if success:
            doc_updated.append(filepath)
        else:
            doc_failed.append(filepath)

    print()
    print("=" * 60)
    print("📊 Migration Summary")
    print("=" * 60)
    print(f"✅ Production files updated: {len(prod_updated)}/{len(FILES_TO_UPDATE)}")
    print(f"✅ Documentation files updated: {len(doc_updated)}/{len(DOC_FILES)}")

    if prod_failed:
        print(f"⚠️  Production files with issues: {len(prod_failed)}")
        for f in prod_failed:
            print(f"   - {f}")

    if doc_failed:
        print(f"⚠️  Documentation files with issues: {len(doc_failed)}")
        for f in doc_failed:
            print(f"   - {f}")

    print()
    print("=" * 60)
    print("🎯 Next Steps")
    print("=" * 60)
    print("1. ✅ Run this script to update imports")
    print("2. 🧪 Run tests: pytest tests/test_conversation_quality.py -v")
    print("3. 🗑️  Delete old file: rm src/flows/conversation_state.py")
    print("4. 🧪 Run full test suite: pytest tests/ -v")
    print("5. 📝 Commit changes: git add . && git commit -m 'Migrate to TypedDict'")
    print()

    # Calculate success rate
    total = len(FILES_TO_UPDATE) + len(DOC_FILES)
    successful = len(prod_updated) + len(doc_updated)
    success_rate = (successful / total) * 100 if total > 0 else 0

    print(f"🎉 Migration {success_rate:.0f}% complete!")

    if success_rate == 100:
        print("✅ All files updated successfully!")
        print("   Ready to delete old dataclass file.")
    else:
        print("⚠️  Some files need manual review.")
        print("   Fix issues before deleting old file.")


if __name__ == "__main__":
    main()
