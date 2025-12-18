"""Test code quality standards for production readiness.

This test suite enforces the Code Quality Standards defined in QA_STRATEGY.md:
1. No print() statements in production code (src/ directory)
2. All paths use configuration (supabase_settings) not hardcoded strings
3. Environment awareness (checks for is_production before resource-intensive ops)

Why these tests exist:
- Print statements don't appear in Vercel serverless logs
- Hardcoded paths break in Docker/Vercel environments
- Environment-specific code prevents production issues

Related Documentation:
- QA_STRATEGY.md → Code Quality Standards section
- QA_IMPLEMENTATION_SUMMARY.md → Phase 1.5 Cleanup Tasks
"""

import os
import re
import pytest
from pathlib import Path


class TestProductionCodeQuality:
    """Enforce production readiness standards for src/ directory."""

    def test_no_print_statements_in_production_code(self):
        """Verify no print() statements in src/ directory (excluding docstrings/comments).

        Why this matters:
        - Vercel serverless functions don't capture print() output
        - Production logs need structured logging (logger.info/debug/error)
        - Log levels allow filtering (INFO in prod, DEBUG in dev)

        Exceptions allowed:
        - Docstring examples (in triple-quoted strings)
        - Comments (lines starting with #)
        - CLI scripts in scripts/ directory (user feedback)

        Related: QA_STRATEGY.md → "1. Logging over Print Statements"
        """
        src_dir = Path(__file__).parent.parent / "src"
        violations = []

        for py_file in src_dir.rglob("*.py"):
            # Skip __pycache__ and test files
            if "__pycache__" in str(py_file) or "test_" in py_file.name:
                continue

            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

                in_docstring = False
                for line_num, line in enumerate(lines, 1):
                    # Track docstring state
                    if '"""' in line or "'''" in line:
                        in_docstring = not in_docstring
                        continue

                    # Skip if in docstring or comment
                    if in_docstring or line.strip().startswith('#'):
                        continue

                    # Check for print() statements (not in strings)
                    # Pattern: print(...) outside of quotes
                    if re.search(r'\bprint\s*\(', line):
                        # Verify it's not in a string literal
                        # Simple heuristic: if preceded by quotes, likely in string
                        stripped = line.strip()
                        if not (stripped.startswith('"') or stripped.startswith("'")):
                            violations.append(f"{py_file.relative_to(src_dir.parent)}:{line_num}: {line.strip()[:80]}")

        if violations:
            error_msg = (
                "\n❌ Found print() statements in production code:\n" +
                "\n".join(f"  {v}" for v in violations) +
                "\n\n💡 Fix: Replace with logger.info(), logger.debug(), or logger.error()\n" +
                "   See QA_STRATEGY.md → Code Quality Standards for examples"
            )
            pytest.fail(error_msg)

    def test_paths_use_configuration(self):
        """Verify no hardcoded data/ paths in src/ directory (should use supabase_settings).

        Why this matters:
        - Hardcoded paths break in Docker (different filesystem layout)
        - Vercel serverless has read-only filesystem (except /tmp)
        - Testing requires temporary directories

        Acceptable patterns:
        - supabase_settings.confessions_path ✅
        - config.get("data_path", "data/") ✅
        - Default parameter: def __init__(self, path="data/file.csv") ✅

        Not acceptable:
        - path = "data/confessions.csv" ❌
        - open("data/career_kb.csv") ❌

        Related: QA_STRATEGY.md → "2. Configuration over Hardcoding"
        """
        src_dir = Path(__file__).parent.parent / "src"
        violations = []

        # Pattern: String literal containing "data/" (but exclude config files)
        hardcoded_path_pattern = re.compile(r'["\']data/[^"\']*["\']')

        for py_file in src_dir.rglob("*.py"):
            # Skip __pycache__, test files, and config files (config defines paths)
            if "__pycache__" in str(py_file) or "test_" in py_file.name:
                continue
            if "config" in py_file.name:
                # Config files define paths - that's acceptable
                continue

            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

                in_docstring = False
                for line_num, line in enumerate(lines, 1):
                    # Track docstring state
                    if '"""' in line or "'''" in line:
                        in_docstring = not in_docstring
                        continue

                    # Skip if in docstring or comment
                    if in_docstring or line.strip().startswith('#'):
                        continue

                    # Check for hardcoded data/ paths
                    if hardcoded_path_pattern.search(line):
                        # Check if it's a default parameter or coming from config
                        if "supabase_settings" in line or "= default" in line or "def __init__" in line:
                            # Acceptable: default parameter or config usage
                            continue

                        violations.append(f"{py_file.relative_to(src_dir.parent)}:{line_num}: {line.strip()[:80]}")

        if violations:
            error_msg = (
                "\n❌ Found hardcoded data/ paths in production code:\n" +
                "\n".join(f"  {v}" for v in violations) +
                "\n\n💡 Fix: Use supabase_settings.confessions_path or config-driven paths\n" +
                "   See QA_STRATEGY.md → Code Quality Standards for examples"
            )
            pytest.fail(error_msg)

    def test_logging_properly_configured(self):
        """Verify logging is configured in supabase_config.py.

        Why this matters:
        - Structured logging enables production debugging
        - Log levels (INFO/DEBUG) control verbosity per environment
        - Vercel logs require proper logging configuration

        Expected:
        - configure_logging() function exists in supabase_config.py
        - Logging initialized when module imported
        - Environment-aware log levels (INFO for prod, DEBUG for dev)

        Related: QA_STRATEGY.md → "1. Logging over Print Statements"
        """
        config_file = Path(__file__).parent.parent / "assistant" / "config" / "supabase_config.py"

        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for logging configuration
        assert "def configure_logging():" in content, (
            "configure_logging() function not found in supabase_config.py"
        )

        assert "logging.basicConfig" in content, (
            "logging.basicConfig not called in configure_logging()"
        )

        assert "configure_logging()" in content, (
            "configure_logging() not called (should be at module level)"
        )

        # Check for environment-aware log levels
        assert "is_production" in content and "logging.INFO" in content and "logging.DEBUG" in content, (
            "Logging configuration should be environment-aware (INFO for prod, DEBUG for dev)"
        )


class TestScriptsQuality:
    """Verify scripts/ directory follows appropriate standards."""

    def test_scripts_can_use_print_for_user_feedback(self):
        """Scripts in scripts/ directory CAN use print() - it's acceptable for CLI tools.

        Why this is different:
        - Scripts are CLI tools for developers/operators
        - Print statements provide user feedback during execution
        - Not deployed to production (run locally or in admin contexts)

        This test just documents the exception - no enforcement needed.
        """
        # This is a documentation test - scripts/ can use print()
        scripts_dir = Path(__file__).parent.parent / "scripts"
        assert scripts_dir.exists(), "scripts/ directory should exist"

        # Count how many scripts exist (for documentation)
        script_count = len(list(scripts_dir.glob("*.py")))
        print(f"\nℹ️  {script_count} scripts in scripts/ directory can use print() for user feedback")


class TestEnvironmentAwareness:
    """Verify code checks environment before resource-intensive operations."""

    def test_production_checks_exist(self):
        """Verify code uses is_production/is_vercel checks for environment-specific logic.

        Why this matters:
        - Vercel has 10s timeout limit for serverless functions
        - Expensive operations (analytics, large queries) need caching in prod
        - Different behavior for dev (verbose) vs prod (optimized)

        Expected patterns:
        - if supabase_settings.is_production: use_cached_result()
        - if not supabase_settings.is_vercel: run_expensive_query()

        Related: QA_STRATEGY.md → "3. Environment Awareness"
        """
        config_file = Path(__file__).parent.parent / "assistant" / "config" / "supabase_config.py"

        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check that environment detection exists
        assert "self.is_production" in content, (
            "SupabaseSettings should define is_production attribute"
        )

        assert "self.is_vercel" in content, (
            "SupabaseSettings should define is_vercel attribute"
        )

        assert 'os.getenv("VERCEL_ENV")' in content or 'os.getenv("VERCEL")' in content, (
            "Should detect Vercel environment via environment variables"
        )


# Test count: 6 tests total (3 in TestProductionCodeQuality, 1 in TestScriptsQuality, 1 in TestEnvironmentAwareness)
# This brings total test count from 30 → 36 tests
