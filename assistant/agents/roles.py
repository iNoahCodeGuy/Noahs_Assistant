# Centralized role definitions and capabilities
from enum import Enum
from typing import Dict

class Role(str, Enum):
    HIRING_MANAGER_NONTECH = "Hiring Manager (nontechnical)"
    HIRING_MANAGER_TECH = "Hiring Manager (technical)"
    SOFTWARE_DEVELOPER = "Software Developer"
    CASUAL = "Just looking around"
    CONFESSION = "Looking to confess crush"

ROLE_CONFIG: Dict[Role, Dict[str, bool]] = {
    Role.HIRING_MANAGER_NONTECH: {"include_code": False},
    Role.HIRING_MANAGER_TECH: {"include_code": True},
    Role.SOFTWARE_DEVELOPER: {"include_code": True},
    Role.CASUAL: {"include_code": False},
    Role.CONFESSION: {"include_code": False},
}

def role_include_code(role: str | None) -> bool:
    if not role:
        return False
    try:
        r = Role(role)
        return ROLE_CONFIG.get(r, {}).get("include_code", False)
    except ValueError:
        return False
