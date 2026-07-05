"""Twilio SMS / Resend email dispatch for the capture flows.

Side-effect helpers fired by the state machines (never by the LLM): a new
recruiter/hiring-manager lead or a crush confession triggers an SMS and a
transactional email to Noah. All service clients are imported lazily inside
try/except so a missing/misconfigured service degrades gracefully.
"""

import logging
import os
import time
from collections import deque
from typing import Deque

logger = logging.getLogger(__name__)


# --- Dispatch throttle ---
# The protected resource here is Noah's phone and inbox. The /chat rate
# limiter (api/main.py) caps per-IP traffic, but a patient client — or many
# IPs — could still turn every capture submission into an SMS + email.
# Sliding-window cap on total dispatches; in-memory is correct for the
# single-instance Railway deployment (upgrade path: Upstash/Redis if this
# ever scales horizontally). Throttled submissions are still written to
# Supabase by the callers — only the SMS/email ping is skipped, and the
# dashboard shows everything.
NOTIFY_LIMIT = 10
NOTIFY_WINDOW_SECONDS = 3600

_dispatch_log: Deque[float] = deque()


def _notify_throttled() -> bool:
    """Record this dispatch; True if the hourly cap is already exhausted."""
    now = time.time()
    while _dispatch_log and now - _dispatch_log[0] > NOTIFY_WINDOW_SECONDS:
        _dispatch_log.popleft()
    if len(_dispatch_log) >= NOTIFY_LIMIT:
        return True
    _dispatch_log.append(now)
    return False


def _send_hm_lead_notifications(info: dict) -> bool:
    """Send SMS and email notifications to Noah about a new hiring manager lead."""
    if _notify_throttled():
        logger.warning(
            "Notification throttle hit -- skipping HM lead SMS/email "
            "(the lead itself is saved)"
        )
        return False

    name = info.get('name') or 'Unknown'
    company = info.get('company') or 'Unknown company'
    email = info.get('email') or ''
    phone = info.get('phone') or ''
    referral = info.get('referral_source') or ''
    msg = info.get('message') or ''

    # ── SMS ──
    try:
        from assistant.services.twilio_service import get_twilio_service
        twilio = get_twilio_service()
        noah_phone = os.getenv("NOAH_PHONE_NUMBER")

        if noah_phone and twilio and twilio.enabled:
            sms_body = f"Portfolia Lead: {name} at {company}"
            if email:
                sms_body += f"\nEmail: {email}"
            if phone:
                sms_body += f"\nPhone: {phone}"
            if referral:
                sms_body += f"\nFound via: {referral[:100]}"
            if msg:
                sms_body += f"\nMessage: {msg[:200]}"

            if len(sms_body) > 1600:
                sms_body = sms_body[:1597] + "..."

            result = twilio.send_sms(to_phone=noah_phone, message=sms_body)
            logger.info(f"HM lead SMS sent to Noah: {result.get('status', 'unknown')}")
        else:
            logger.warning("Twilio not configured -- skipping HM lead SMS")
    except Exception as e:
        logger.error(f"HM lead SMS failed: {e}")

    # ── Email ──
    try:
        from assistant.services.resend_service import get_resend_service
        resend_svc = get_resend_service()

        if resend_svc and resend_svc.enabled:
            subject = f"Portfolia Lead: {name} at {company}"
            html = (
                "<h2>New Recruiter/Hiring Manager Lead</h2>"
                f"<p><strong>Name:</strong> {name}</p>"
                f"<p><strong>Company:</strong> {company}</p>"
                f"<p><strong>Email:</strong> {email or '(not provided)'}</p>"
                f"<p><strong>Phone:</strong> {phone or '(not provided)'}</p>"
                f"<p><strong>Referral Source:</strong> {referral or '(not provided)'}</p>"
                f"<p><strong>Message:</strong> {msg or '(none)'}</p>"
                "<p><em>Captured via Portfolia recruiter lead flow</em></p>"
            )
            resend_svc.send_email(
                to_email=resend_svc.admin_email,
                subject=subject,
                html=html,
            )
            logger.info("HM lead email sent to Noah")
        else:
            logger.warning("Resend not configured -- skipping HM lead email")
    except Exception as e:
        logger.error(f"HM lead email failed: {e}")

    return True


def _send_crush_notifications(
    anonymous: bool = False,
    alias: str | None = None,
    name: str | None = None,
    contact: str | None = None,
    message: str | None = None,
) -> None:
    """Send both SMS and email notifications to Noah about a crush confession."""
    if _notify_throttled():
        logger.warning(
            "Notification throttle hit -- skipping crush SMS/email "
            "(the confession itself is saved)"
        )
        return

    # ── SMS ──
    try:
        from assistant.services.twilio_service import get_twilio_service
        twilio = get_twilio_service()
        noah_phone = os.getenv("NOAH_PHONE_NUMBER")

        if noah_phone and twilio and twilio.enabled:
            if anonymous:
                sms_body = (
                    f"Portfolia: Secret admirer alert.\n"
                    f"Alias: {alias or 'Anonymous'}\n"
                    f"Message: {(message or '(none)')[:200]}"
                )
            else:
                sms_body = (
                    f"Portfolia: Someone chose the bold option.\n"
                    f"Name: {name or 'Unknown'}\n"
                    f"Contact: {contact or '(none)'}\n"
                    f"Message: {(message or '(none)')[:150]}"
                )
            if len(sms_body) > 1600:
                sms_body = sms_body[:1597] + "..."
            twilio.send_sms(to_phone=noah_phone, message=sms_body)
            logger.info("Crush SMS sent to Noah")
        else:
            logger.warning("Twilio not configured -- skipping crush SMS")
    except Exception as e:
        logger.error(f"Crush SMS failed: {e}")

    # ── Email ──
    try:
        from assistant.services.resend_service import get_resend_service
        resend_svc = get_resend_service()

        if resend_svc and resend_svc.enabled:
            if anonymous:
                subject = "Portfolia: Secret admirer"
                html = (
                    "<h2>Secret Admirer on Portfolia</h2>"
                    f"<p><strong>Alias:</strong> {alias or 'Anonymous'}</p>"
                    f"<p><strong>Message:</strong> {message or '(none)'}</p>"
                    "<p><em>Submitted via crush confession flow</em></p>"
                )
            else:
                subject = f"Portfolia: {name or 'Someone'} chose the bold option"
                html = (
                    "<h2>Crush Confession on Portfolia</h2>"
                    f"<p><strong>Name:</strong> {name or 'Unknown'}</p>"
                    f"<p><strong>Contact:</strong> {contact or '(none)'}</p>"
                    f"<p><strong>Message:</strong> {message or '(none)'}</p>"
                    "<p><em>Submitted via crush confession flow</em></p>"
                )
            resend_svc.send_email(
                to_email=resend_svc.admin_email,
                subject=subject,
                html=html,
            )
            logger.info("Crush email sent to Noah")
        else:
            logger.warning("Resend not configured -- skipping crush email")
    except Exception as e:
        logger.error(f"Crush email failed: {e}")
