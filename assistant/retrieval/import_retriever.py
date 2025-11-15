"""Import explanation retrieval for stack justification questions.

This module provides tier-appropriate explanations for every library and framework
used in the stack. It retrieves from imports_kb.csv based on user role and returns
explanations with enterprise context.
"""

import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Map roles to explanation tiers
ROLE_TO_TIER = {
    "Hiring Manager (technical)": "1",
    "Hiring Manager (nontechnical)": "1",
    "Software Developer": "2",
    "Just looking around": "1",
    "Looking to confess crush": "1",
}

# Path to imports knowledge base - try multiple strategies for Vercel compatibility
def _get_imports_kb_path() -> Path:
    """Resolve path to imports_kb.csv for both local and Vercel environments."""
    # Strategy 1: Relative from this file (local development)
    local_path = Path(__file__).parent.parent.parent / "data" / "imports_kb.csv"
    if local_path.exists():
        return local_path

    # Strategy 2: Absolute from current working directory (Vercel)
    cwd_path = Path.cwd() / "data" / "imports_kb.csv"
    if cwd_path.exists():
        return cwd_path

    # Strategy 3: From environment variable (explicit override)
    if env_path := os.getenv("IMPORTS_KB_PATH"):
        env_path_obj = Path(env_path)
        if env_path_obj.exists():
            return env_path_obj

    # Fallback: return local path and let FileNotFoundError be caught
    logger.warning(f"Could not find imports_kb.csv, tried: {local_path}, {cwd_path}")
    return local_path

IMPORTS_KB_PATH = _get_imports_kb_path()


def _load_imports_kb() -> List[Dict[str, str]]:
    """Load import justifications from CSV file.

    Returns:
        List of dictionaries containing import metadata and explanations
    """
    imports = []
    try:
        with open(IMPORTS_KB_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                imports.append(row)
        logger.info(f"Loaded {len(imports)} import explanations from {IMPORTS_KB_PATH}")
    except FileNotFoundError:
        logger.warning(f"Imports KB not found at {IMPORTS_KB_PATH}")
    except Exception as e:
        logger.error(f"Error loading imports KB: {e}")
    return imports


def get_import_explanation(
    import_name: str,
    role: str,
    tier: Optional[str] = None
) -> Optional[Dict[str, str]]:
    """Get tier-appropriate explanation for a specific import.

    Args:
        import_name: Name of the import/library (e.g., "openai", "supabase")
        role: User's role for tier determination
        tier: Optional explicit tier override (1, 2, or 3)

    Returns:
        Dictionary with explanation fields, or None if not found
    """
    imports = _load_imports_kb()

    # Determine tier from role if not explicitly provided
    if tier is None:
        tier = ROLE_TO_TIER.get(role, "1")

    # Find matching import at specified tier
    for imp in imports:
        if imp["import"].lower() == import_name.lower() and imp["tier"] == tier:
            return imp

    # Fallback to tier 1 if specific tier not found
    if tier != "1":
        for imp in imports:
            if imp["import"].lower() == import_name.lower() and imp["tier"] == "1":
                logger.info(f"Falling back to tier 1 for {import_name}")
                return imp

    logger.warning(f"No explanation found for import: {import_name}")
    return None


def get_all_imports_for_role(role: str) -> List[Dict[str, str]]:
    """Get all import explanations appropriate for a given role.

    Args:
        role: User's role

    Returns:
        List of import explanation dictionaries at appropriate tier
    """
    imports = _load_imports_kb()
    tier = ROLE_TO_TIER.get(role, "1")

    # Get unique import names
    import_names = list(set(imp["import"] for imp in imports))

    # Get tier-appropriate explanation for each
    results = []
    for name in import_names:
        explanation = get_import_explanation(name, role, tier)
        if explanation:
            results.append(explanation)

    return results


def search_import_explanations(query: str, role: str, top_k: int = 3) -> List[Dict[str, str]]:
    """Search import explanations based on query keywords.

    Args:
        query: User query (e.g., "why use supabase", "explain openai")
        role: User's role for tier determination
        top_k: Maximum number of results to return

    Returns:
        List of relevant import explanations
    """
    imports = _load_imports_kb()
    tier = ROLE_TO_TIER.get(role, "1")
    lowered_query = query.lower()

    # Score each import by relevance
    scored_imports = []
    for imp in imports:
        if imp["tier"] != tier:
            continue

        score = 0
        import_name = imp["import"].lower()

        # Exact match in query
        if import_name in lowered_query:
            score += 10

        # Category match (e.g., "database" matches supabase)
        if imp["category"].lower() in lowered_query:
            score += 5

        # Keyword matches in explanation
        explanation_text = imp["explanation"].lower()
        query_words = set(lowered_query.split())
        explanation_words = set(explanation_text.split())
        common_words = query_words & explanation_words
        score += len(common_words)

        # Alternative mentions (e.g., "pinecone" matches pgvector alternative)
        if "alternative" in imp and lowered_query in imp["enterprise_alternative"].lower():
            score += 3

        if score > 0:
            scored_imports.append((score, imp))

    # Sort by score and return top_k
    scored_imports.sort(key=lambda x: x[0], reverse=True)
    return [imp for _, imp in scored_imports[:top_k]]


def detect_import_in_query(query: str) -> Optional[str]:
    """Detect which import the user is asking about.

    Args:
        query: User query

    Returns:
        Import name if detected, else None
    """
    lowered = query.lower()

    # Known imports in the stack
    known_imports = [
        "openai", "supabase", "pgvector", "langchain", "langgraph",
        "vercel", "resend", "twilio", "langsmith", "streamlit",
        "pydantic", "storage"
    ]

    for import_name in known_imports:
        if import_name in lowered:
            return import_name

    return None
