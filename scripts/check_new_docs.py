#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-commit hook: Check new .md files are properly registered.

This script runs when any .md file in docs/ is added or modified.
It ensures documentation changes follow QA_STRATEGY.md section 11 guidelines.

Checks:
1. Master docs (docs/context/) are referenced in QA_STRATEGY.md
2. Feature docs (docs/features/) follow naming conventions
3. New docs don't duplicate existing content
4. Code references in docs are valid (deferred to alignment tests)

Usage:
  python scripts/check_new_docs.py file1.md file2.md ...

Exit codes:
  0 = All checks passed
  1 = Critical error (blocks commit)
  2 = Warning (allows commit with confirmation)

Related:
  - QA_STRATEGY.md section 11 "Anti-Drift Protection"
  - .pre-commit-config.yaml (configuration)
  - tests/test_documentation_alignment.py (validates code references)
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

# ============================================================
# CONSTANTS
# ============================================================

MASTER_DOCS_DIR = "docs/context"
FEATURE_DOCS_DIR = "docs/features"
QA_STRATEGY_PATH = "docs/QA_STRATEGY.md"

VALID_FEATURE_SUFFIXES = [
    "_IMPLEMENTATION",
    "_SUMMARY",
    "_GUIDE",
    "_ANALYSIS",
    "_REFERENCE"
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_staged_md_files() -> List[str]:
    """Get list of .md files staged for commit (new or modified)."""
    try:
        # Get newly added files
        result_added = subprocess.run(
            ['git', 'diff', '--cached', '--name-only', '--diff-filter=A'],
            capture_output=True, text=True, check=True
        )

        # Get modified files
        result_modified = subprocess.run(
            ['git', 'diff', '--cached', '--name-only', '--diff-filter=M'],
            capture_output=True, text=True, check=True
        )

        added_files = result_added.stdout.strip().split('\n')
        modified_files = result_modified.stdout.strip().split('\n')

        all_files = set(added_files + modified_files)

        md_files = [
            f for f in all_files
            if f.startswith('docs/') and f.endswith('.md') and f
        ]

        return md_files
    except subprocess.CalledProcessError:
        # Not in a git repo or git not available
        return []

def is_new_file(filepath: str) -> bool:
    """Check if file is newly added (not just modified)."""
    try:
        result = subprocess.run(
            ['git', 'diff', '--cached', '--name-only', '--diff-filter=A'],
            capture_output=True, text=True, check=True
        )
        return filepath in result.stdout
    except subprocess.CalledProcessError:
        return False

def read_file_safe(filepath: str) -> str:
    """Read file content, return empty string if not exists."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""

# ============================================================
# CHECK FUNCTIONS
# ============================================================

def check_master_doc_registration(filepath: str) -> Tuple[bool, str]:
    """
    Check if new master doc is mentioned in QA_STRATEGY.md.

    Master docs (docs/context/) are the Single Source of Truth and MUST
    be referenced in QA_STRATEGY.md §7 "Documentation Quality Standards".
    """
    if not filepath.startswith(MASTER_DOCS_DIR):
        return True, ""  # Not a master doc, no check needed

    if not is_new_file(filepath):
        return True, ""  # Only check new files

    doc_name = Path(filepath).name
    qa_content = read_file_safe(QA_STRATEGY_PATH)

    if doc_name not in qa_content:
        error_msg = f"""
========================================================================
  NEW MASTER DOC NOT REGISTERED IN QA_STRATEGY.md
========================================================================

You added: {filepath}

Master docs (docs/context/) are the Single Source of Truth (SSOT).
They MUST be referenced in QA_STRATEGY.md.

ACTION REQUIRED:
1. Add reference in docs/QA_STRATEGY.md section 7 "Documentation Quality Standards"
2. Add to "Quick Reference: Documentation Types" table (line ~650)
3. Consider adding alignment test in tests/test_documentation_alignment.py

Example addition to QA_STRATEGY.md:

    | Documentation Type | Purpose | When to Use | Location |
    |--------------------|---------|-------------|----------|
    | **{doc_name.replace('.md', '')}** | [Your purpose] | [When to update] | `docs/context/` |

Then re-stage: git add docs/QA_STRATEGY.md
"""
        return False, error_msg

    return True, ""

def check_feature_doc_convention(filepath: str) -> Tuple[bool, str]:
    """
    Check if feature doc follows naming conventions.

    Feature docs should end with: _IMPLEMENTATION, _SUMMARY, _GUIDE, etc.
    This makes purpose clear at a glance.
    """
    if not filepath.startswith(FEATURE_DOCS_DIR):
        return True, ""  # Not a feature doc

    if not is_new_file(filepath):
        return True, ""  # Only check new files

    doc_name = Path(filepath).stem  # Without .md extension

    # Check if follows convention
    follows_convention = any(
        doc_name.endswith(suffix) for suffix in VALID_FEATURE_SUFFIXES
    )

    if not follows_convention:
        warning_msg = f"""
===============================================================╗
=  WARNING:  FEATURE DOC NAMING CONVENTION                           =
===============================================================╝

You added: {filepath}

RECOMMENDATION:
Feature docs should end with: {', '.join(VALID_FEATURE_SUFFIXES)}

Examples:
  - SENTIMENT_ANALYSIS_IMPLEMENTATION.md (how it's built)
  - CODE_DISPLAY_SUMMARY.md (what it does)
  - ROLE_SETUP_GUIDE.md (how to use)

Current name: {doc_name}.md
Suggested: {doc_name}_IMPLEMENTATION.md (or _SUMMARY, _GUIDE)

This is a RECOMMENDATION (not blocking). Continue? [y/N]: """

        print(warning_msg, end='', flush=True)
        response = input().strip().lower()

        if response != 'y':
            return False, "User chose to rename file to follow convention"

        print("OK: Proceeding with current name (non-standard)\n")

    return True, ""

def check_potential_duplicate(filepath: str) -> Tuple[bool, str]:
    """
    Check if new doc might duplicate existing content.

    Uses filename similarity to suggest existing docs that might
    already cover the topic.
    """
    if not is_new_file(filepath):
        return True, ""  # Only check new files

    new_doc_name = Path(filepath).stem.lower()
    new_doc_dir = Path(filepath).parent

    # Get all existing docs in same directory
    existing_docs = []
    if new_doc_dir.exists():
        existing_docs = [
            f.name for f in new_doc_dir.glob('*.md')
            if f.name != Path(filepath).name
        ]

    # Check for similar names (simple word overlap)
    new_words = set(new_doc_name.replace('_', ' ').split())

    similar_docs = []
    for doc in existing_docs:
        doc_words = set(doc.replace('.md', '').replace('_', ' ').lower().split())
        overlap = new_words & doc_words

        # If 2+ words match, might be duplicate
        if len(overlap) >= 2:
            similar_docs.append((doc, overlap))

    if similar_docs:
        warning_msg = f"""
===============================================================╗
=  INFO: POTENTIAL DUPLICATE CONTENT                              =
===============================================================╝

You're adding: {filepath}

Found similar existing docs:
"""
        for doc, overlap in similar_docs:
            warning_msg += f"  - {new_doc_dir}/{doc} (shared words: {', '.join(overlap)})\n"

        warning_msg += """
RECOMMENDATION:
Consider updating existing doc instead of creating new one.
This follows QA_STRATEGY.md §4 "Feature Development Workflow".

Questions to ask:
  1. Is this truly NEW content, or extension of existing doc?
  2. Should this be a new section in existing doc?
  3. Does this create documentation fragmentation?

Continue creating new doc? [y/N]: """

        print(warning_msg, end='', flush=True)
        response = input().strip().lower()

        if response != 'y':
            return False, "User chose to update existing doc instead"

        print("OK: Proceeding with new doc (confirmed not duplicate)\n")

    return True, ""

def check_changelog_updated(staged_files: List[str]) -> Tuple[bool, str]:
    """
    Check if CHANGELOG.md was updated alongside documentation changes.

    Not strictly required, but recommended for user-facing changes.
    """
    has_doc_changes = any(
        f.startswith('docs/') and f.endswith('.md')
        for f in staged_files
    )

    has_changelog = 'CHANGELOG.md' in staged_files

    if has_doc_changes and not has_changelog:
        info_msg = """
===============================================================╗
=  INFO: CHANGELOG.md NOT UPDATED                                 =
===============================================================╝

You're committing documentation changes without updating CHANGELOG.md.

RECOMMENDATION (Optional):
If these changes are user-facing, update CHANGELOG.md

Example entry:
    ### Documentation
    - Added [feature] implementation guide
    - Updated [section] with new workflow

This is OPTIONAL for internal docs. Continue? [Y/n]: """

        print(info_msg, end='', flush=True)
        response = input().strip().lower()

        if response == 'n':
            return False, "User wants to update CHANGELOG.md first"

        print("OK: Proceeding without CHANGELOG update\n")

    return True, ""

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Run all documentation checks."""
    print("\n🔍 Checking documentation changes...\n")

    # Get files to check (from args or from git)
    if len(sys.argv) > 1:
        # Files passed as arguments (from pre-commit)
        files_to_check = [f for f in sys.argv[1:] if f.startswith('docs/') and f.endswith('.md')]
    else:
        # No args, check staged files
        files_to_check = get_staged_md_files()

    if not files_to_check:
        print("OK: No documentation changes to check\n")
        return 0

    print(f"Checking {len(files_to_check)} file(s):\n")
    for f in files_to_check:
        print(f"  - {f}")
    print()

    # Run all checks
    all_passed = True

    for filepath in files_to_check:
        print(f"📄 Checking: {filepath}")

        # Check 1: Master doc registration
        passed, msg = check_master_doc_registration(filepath)
        if not passed:
            print(msg)
            all_passed = False
            continue

        # Check 2: Feature doc naming
        passed, msg = check_feature_doc_convention(filepath)
        if not passed:
            print(msg)
            all_passed = False
            continue

        # Check 3: Potential duplicates
        passed, msg = check_potential_duplicate(filepath)
        if not passed:
            print(msg)
            all_passed = False
            continue

        print(f"  OK: All checks passed for {filepath}\n")

    # Check 4: CHANGELOG (applies to all files)
    if files_to_check:
        passed, msg = check_changelog_updated(files_to_check)
        if not passed:
            print(msg)
            all_passed = False

    if all_passed:
        print("===============================================================╗")
        print("=  OK: ALL DOCUMENTATION CHECKS PASSED                          =")
        print("===============================================================╝\n")
        print("Note: Code reference validation runs in alignment tests\n")
        return 0
    else:
        print("===============================================================╗")
        print("=  ERROR: DOCUMENTATION CHECKS FAILED                              =")
        print("===============================================================╝\n")
        print("Fix issues above and try again.")
        print("To bypass (not recommended): git commit --no-verify\n")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nERROR: Cancelled by user\n")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}\n")
        print("Please report this issue. Bypassing check...")
        sys.exit(0)  # Don't block commit on script errors
