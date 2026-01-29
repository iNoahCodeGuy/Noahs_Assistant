"""Application settings and path configuration.

Simple module-level constants for paths and configuration.
All hardcoded paths should reference these constants instead.

Environment Variables:
- DEBUG_LOG_PATH: Path to debug log file (default: logs/debug.log)
- PROJECT_ROOT: Project root directory (auto-detected if not set)
"""

import os
from pathlib import Path


# Detect project root
def _detect_project_root() -> Path:
    """Detect project root directory.

    Tries in order:
    1. PROJECT_ROOT environment variable
    2. Git repository root (walks up from cwd)
    3. Current working directory
    """
    # Check environment variable first
    if os.getenv("PROJECT_ROOT"):
        return Path(os.getenv("PROJECT_ROOT")).resolve()

    # Try to find git root
    try:
        current = Path.cwd()
        while current != current.parent:
            if (current / ".git").exists():
                return current
            current = current.parent
    except Exception:
        pass

    # Fall back to current directory
    return Path.cwd()


# Project root
PROJECT_ROOT = _detect_project_root()

# Debug log path (configurable via environment)
DEBUG_LOG_PATH = os.getenv('DEBUG_LOG_PATH', 'logs/debug.log')


def get_debug_log_path() -> str:
    """Get the debug log file path as an absolute string.

    Returns:
        str: Absolute path to debug log file

    Example:
        from assistant.config.settings import get_debug_log_path

        with open(get_debug_log_path(), 'a') as f:
            f.write(json.dumps(data) + "\\n")
    """
    path = Path(DEBUG_LOG_PATH)

    # Make absolute if relative (relative to project root)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    # Ensure parent directory exists
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass  # If we can't create dir, let the file write fail naturally

    return str(path.resolve())
