# Tests for role capability configuration
from src.agents.roles import Role, ROLE_CONFIG, role_include_code

def test_all_roles_have_config():
    for r in Role:
        assert r in ROLE_CONFIG, f"Missing ROLE_CONFIG entry for {r}"
        assert "include_code" in ROLE_CONFIG[r], f"Missing include_code flag for {r}"

def test_role_include_code_positive():
    assert role_include_code(Role.SOFTWARE_DEVELOPER.value) is True
    assert role_include_code(Role.HIRING_MANAGER_TECH.value) is True

def test_role_include_code_negative():
    assert role_include_code(Role.HIRING_MANAGER_NONTECH.value) is False
    assert role_include_code(Role.CASUAL.value) is False
    assert role_include_code(Role.CONFESSION.value) is False

def test_role_include_code_unknown():
    assert role_include_code("Unknown Role X") is False
    assert role_include_code(None) is False
