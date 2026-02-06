"""Code validation and sanitization utilities.

This module helps ensure that code snippets displayed to users are:
1. Actually valid Python code (not malformed metadata)
2. Free from SQL/retrieval artifacts that leak from the RAG pipeline

Junior dev note: These are "defensive" utilities that catch edge cases
where the retrieval system returns something that looks like code but isn't.
"""

import re


CODE_VALIDATION_KEYWORDS = [
    "def ", "class ", "import ", "from ", "return ",
    "async ", "await ", "try:", "except ", "lambda "
]


def is_valid_code_snippet(code: str) -> bool:
    """Check if retrieved text looks like actual Python source code.

    This uses simple heuristics to filter out malformed responses:
    - Must have minimum length (20 chars)
    - Must contain Python keywords (def, class, import, etc.)
    - Must have multiple lines (not just a single statement)
    - Must NOT be metadata (doc_id, query=, etc.)

    Args:
        code: The text to validate

    Returns:
        True if it looks like valid Python code, False otherwise
    """
    if not code:
        return False

    stripped = code.strip()

    # Too short to be meaningful code
    if len(stripped) < 20:
        return False

    # Looks like metadata leak (not actual code)
    if "doc_id" in stripped or "query=" in stripped or stripped.startswith("{'"):
        return False

    # Starts with closing braces (malformed)
    if stripped[0] in {"}", ")"} and stripped.count("\n") < 3:
        return False

    # Single-line code is probably not what we want to display
    if "\n" not in stripped:
        return False

    # Must contain at least one Python keyword
    if not any(keyword in stripped for keyword in CODE_VALIDATION_KEYWORDS):
        return False

    return True


SANITIZE_PREFIX_PATTERNS = [
    re.compile(r"^[\}\)\]\{\(\[]+$"),  # Lines with only braces/brackets
    re.compile(r"^SELECT\.?$", re.IGNORECASE),  # SQL SELECT tokens
    re.compile(r"^FROM\.?$", re.IGNORECASE),  # SQL FROM tokens
    re.compile(r"^END;?$", re.IGNORECASE),  # SQL block terminators like END or END;
    re.compile(r"^BEGIN\.?$", re.IGNORECASE),  # SQL BEGIN tokens
]


def sanitize_generated_answer(answer: str) -> str:
    """Strip leading SQL/artifact noise from LLM responses.

    Sometimes the RAG context includes SQL queries or metadata that the LLM
    accidentally includes at the start of its answer. This removes those.

    Example:
        Input:  "}\n\nSELECT\n\nHere is the answer you requested..."
        Output: "Here is the answer you requested..."

    Args:
        answer: The raw LLM-generated text

    Returns:
        Cleaned answer with leading artifacts removed
    """
    if not answer:
        return answer

    lines = answer.splitlines()
    sanitized_lines = []
    dropping_prefix = True

    for line in lines:
        stripped = line.strip()

        # Skip empty lines at the start
        if dropping_prefix and (not stripped):
            continue

        # Skip lines that match artifact patterns
        if dropping_prefix and any(pattern.match(stripped) for pattern in SANITIZE_PREFIX_PATTERNS):
            continue

        # Once we hit real content, stop dropping lines
        dropping_prefix = False
        sanitized_lines.append(line)

    if not sanitized_lines:
        return ""

    cleaned = "\n".join(sanitized_lines).lstrip()

    # SECONDARY CLEANUP: remove lines anywhere in the answer that are pure
    # SQL/PLPGSQL control tokens which sometimes leak from migration files
    # (examples: "END;", "BEGIN", etc.). We remove standalone lines only
    # to avoid accidentally changing legitimate prose.
    any_remove_patterns = [
        re.compile(r"^END;?$", re.IGNORECASE),
        re.compile(r"^BEGIN\.?$", re.IGNORECASE),
        re.compile(r"^\$\$;$"),  # PL/pgSQL dollar-quoted terminator lines
    ]

    final_lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if any(p.match(stripped) for p in any_remove_patterns):
            # Skip this standalone SQL artifact line
            continue
        final_lines.append(line)

    return "\n".join(final_lines).lstrip()
