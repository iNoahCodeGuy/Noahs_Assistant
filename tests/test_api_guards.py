"""Abuse-guard tests for the /chat endpoint.

Hermetic: the pipeline is stubbed, so these exercise only the transport
layer — rate limiting and input caps must reject abusive traffic before
any embedding/LLM call could happen.
"""

import pytest
from fastapi.testclient import TestClient

import api.main as api_main


@pytest.fixture
def client(monkeypatch):
    # Stub the pipeline: guard tests must never reach generation
    monkeypatch.setattr(
        api_main,
        "run_conversation_flow",
        lambda state, engine, session_id: {**state, "answer": "stub answer"},
    )
    # Fresh limiter state per test
    api_main._request_log.clear()
    return TestClient(api_main.app)


def _post(client, message="hello", ip="1.2.3.4"):
    return client.post(
        "/chat",
        json={"message": message, "session_id": "guard-test"},
        headers={"x-forwarded-for": ip},
    )


def test_normal_request_passes(client):
    res = _post(client)
    assert res.status_code == 200
    assert res.json()["answer"] == "stub answer"


def test_oversized_message_rejected_with_400(client):
    res = _post(client, message="x" * (api_main.MAX_MESSAGE_CHARS + 1))
    assert res.status_code == 400
    assert "too long" in res.json()["detail"].lower()


def test_message_at_limit_passes(client):
    res = _post(client, message="x" * api_main.MAX_MESSAGE_CHARS)
    assert res.status_code == 200


def test_burst_over_limit_rejected_with_429(client):
    for _ in range(api_main.RATE_LIMIT_REQUESTS):
        assert _post(client).status_code == 200
    res = _post(client)
    assert res.status_code == 429


def test_rate_limit_is_per_client(client):
    for _ in range(api_main.RATE_LIMIT_REQUESTS):
        _post(client, ip="10.0.0.1")
    assert _post(client, ip="10.0.0.1").status_code == 429
    assert _post(client, ip="10.0.0.2").status_code == 200


class _FakeSupabase:
    """Minimal stand-in for the supabase client's table().select() chain."""

    def __init__(self, count):
        self._count = count

    def table(self, name):
        return self

    def select(self, *args, **kwargs):
        return self

    def limit(self, n):
        return self

    def execute(self):
        result = type("Result", (), {})()
        result.count = self._count
        return result


def test_health_ok_reports_chunk_count(client, monkeypatch):
    monkeypatch.setattr(api_main, "get_supabase_client", lambda: _FakeSupabase(234))
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "kb_chunks": 234}


def test_health_degraded_when_db_unreachable(client, monkeypatch):
    def _boom():
        raise RuntimeError("supabase unreachable")

    monkeypatch.setattr(api_main, "get_supabase_client", _boom)
    res = client.get("/health")
    assert res.status_code == 503
    assert res.json() == {"status": "degraded", "kb_chunks": None}


def test_window_expiry_unblocks(client, monkeypatch):
    for _ in range(api_main.RATE_LIMIT_REQUESTS):
        _post(client)
    assert _post(client).status_code == 429

    real_time = api_main.time.time()
    monkeypatch.setattr(
        api_main.time,
        "time",
        lambda: real_time + api_main.RATE_LIMIT_WINDOW_SECONDS + 1,
    )
    assert _post(client).status_code == 200
