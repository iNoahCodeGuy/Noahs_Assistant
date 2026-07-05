"""Throttle tests for capture-flow notification dispatch.

Hermetic: no Twilio/Resend client is ever constructed. The throttle guards
Noah's phone and inbox — capture submissions beyond the hourly cap must skip
the SMS/email ping (the Supabase write happens in the callers and is
unaffected).
"""

import pytest

from assistant.flows.capture import notifications


@pytest.fixture(autouse=True)
def fresh_throttle():
    notifications._dispatch_log.clear()
    yield
    notifications._dispatch_log.clear()


@pytest.fixture
def service_calls(monkeypatch):
    """Replace the lazy service getters with recording spies.

    The dispatch functions swallow service exceptions by design (graceful
    degradation), so a raising stub proves nothing — instead we record every
    attempt to obtain a client and assert on the call list. Returning None
    takes the "not configured" branch, so nothing is ever sent.
    """
    calls = []

    import assistant.services.twilio_service as twilio_service
    import assistant.services.resend_service as resend_service

    monkeypatch.setattr(
        twilio_service, "get_twilio_service", lambda: calls.append("twilio")
    )
    monkeypatch.setattr(
        resend_service, "get_resend_service", lambda: calls.append("resend")
    )
    return calls


def _fill_to_cap():
    for _ in range(notifications.NOTIFY_LIMIT):
        assert notifications._notify_throttled() is False


def test_under_limit_allows_dispatch():
    assert notifications._notify_throttled() is False


def test_at_limit_throttles():
    _fill_to_cap()
    assert notifications._notify_throttled() is True


def test_window_expiry_unblocks(monkeypatch):
    _fill_to_cap()
    assert notifications._notify_throttled() is True

    real_time = notifications.time.time()
    monkeypatch.setattr(
        notifications.time,
        "time",
        lambda: real_time + notifications.NOTIFY_WINDOW_SECONDS + 1,
    )
    assert notifications._notify_throttled() is False


def test_unthrottled_dispatch_reaches_services(service_calls):
    notifications._send_crush_notifications(anonymous=True, message="hey")
    assert service_calls == ["twilio", "resend"]


def test_throttled_crush_dispatch_skips_services(service_calls):
    _fill_to_cap()
    notifications._send_crush_notifications(
        anonymous=False, name="Sarah", contact="702-555-1234", message="hi"
    )
    assert service_calls == []


def test_throttled_lead_dispatch_skips_services(service_calls):
    _fill_to_cap()
    result = notifications._send_hm_lead_notifications(
        {"name": "Alex", "company": "Acme", "email": "alex@acme.com"}
    )
    assert result is False
    assert service_calls == []


def test_dispatch_functions_share_one_budget(service_calls):
    """Crush and lead notifications count toward the same cap — the scarce
    resource is Noah, not the flow type."""
    for _ in range(notifications.NOTIFY_LIMIT):
        notifications._send_crush_notifications(anonymous=True, message="hey")

    assert notifications._send_hm_lead_notifications({"name": "Alex"}) is False
    assert "twilio" in service_calls  # earlier dispatches did go through
