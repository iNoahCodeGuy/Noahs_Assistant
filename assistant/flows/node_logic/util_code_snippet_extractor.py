"""Code snippet extraction utility for meta-teaching explanations.

This module extracts function code from Python files to show users how
Portfolia's programming handles edge cases. Used in meta-teaching explanations
to demonstrate actual implementation code.
"""

import ast
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def extract_function_code(file_path: str, function_name: str) -> Optional[str]:
    """Extract function definition and body from Python file.

    Reads a Python file, finds the specified function, and returns its
    complete code including docstring and body.

    Args:
        file_path: Path to Python file (relative or absolute)
        function_name: Name of function to extract

    Returns:
        Complete function code as string, or None if not found

    Example:
        >>> code = extract_function_code(
        ...     "assistant/flows/node_logic/util_edge_case_detection.py",
        ...     "_is_off_topic"
        ... )
        >>> print(code)
        def _is_off_topic(query: str, chat_history: list) -> bool:
            \"\"\"Detect queries completely unrelated...\"\"\"
            ...
    """
    try:
        # Resolve file path
        file_path_obj = Path(file_path)
        if not file_path_obj.is_absolute():
            # Try relative to project root
            project_root = Path(__file__).parent.parent.parent.parent
            file_path_obj = project_root / file_path

        if not file_path_obj.exists():
            logger.warning(f"File not found: {file_path}")
            return None

        # Read file content
        with open(file_path_obj, 'r', encoding='utf-8') as f:
            source_code = f.read()

        # Parse AST to find function
        tree = ast.parse(source_code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Found the function - extract lines
                start_line = node.lineno - 1  # AST is 1-indexed, list is 0-indexed
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 50

                lines = source_code.split('\n')
                function_lines = lines[start_line:end_line]

                return '\n'.join(function_lines)

        logger.warning(f"Function '{function_name}' not found in {file_path}")
        return None

    except Exception as e:
        logger.error(f"Error extracting function code: {e}", exc_info=True)
        return None


def extract_function_with_context(
    file_path: str,
    function_name: str,
    context_lines: int = 5
) -> Optional[Tuple[str, int]]:
    """Extract function code with surrounding context.

    Similar to extract_function_code but includes a few lines of context
    before the function (like imports or comments).

    Args:
        file_path: Path to Python file
        function_name: Name of function to extract
        context_lines: Number of lines of context before function

    Returns:
        Tuple of (code_string, start_line_number) or None if not found
    """
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.is_absolute():
            project_root = Path(__file__).parent.parent.parent.parent
            file_path_obj = project_root / file_path

        if not file_path_obj.exists():
            return None

        with open(file_path_obj, 'r', encoding='utf-8') as f:
            source_code = f.read()
            lines = source_code.split('\n')

        tree = ast.parse(source_code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                start_line = max(0, node.lineno - 1 - context_lines)
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else node.lineno + 50

                function_lines = lines[start_line:end_line]
                code = '\n'.join(function_lines)

                return (code, start_line + 1)  # Return 1-indexed line number

        return None

    except Exception as e:
        logger.error(f"Error extracting function with context: {e}", exc_info=True)
        return None


def get_function_signature(file_path: str, function_name: str) -> Optional[str]:
    """Get just the function signature (def line) without body.

    Useful for quick references in explanations.

    Args:
        file_path: Path to Python file
        function_name: Name of function

    Returns:
        Function signature string or None
    """
    code = extract_function_code(file_path, function_name)
    if not code:
        return None

    # Extract first line (function definition)
    first_line = code.split('\n')[0]
    return first_line.strip()


def format_code_for_display(code: str, language: str = "python") -> str:
    """Format code snippet for markdown display.

    Wraps code in markdown code block with syntax highlighting.

    Args:
        code: Code string to format
        language: Language identifier for syntax highlighting

    Returns:
        Formatted code block string
    """
    return f"```{language}\n{code}\n```"
