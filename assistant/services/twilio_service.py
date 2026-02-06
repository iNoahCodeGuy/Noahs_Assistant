"""Twilio SMS service for notifications.

This service handles:
- SMS notifications for urgent contact requests
- Real-time alerts for high-priority messages
- Multi-channel communication (email + SMS)

Twilio provides reliable SMS delivery worldwide:
- 99.95% uptime SLA
- Delivery receipts
- International support
- Free trial includes $15 credit

Environment Variables Required:
- TWILIO_ACCOUNT_SID: Your Twilio account SID
- TWILIO_AUTH_TOKEN: Your Twilio auth token
- TWILIO_PHONE_NUMBER: Your Twilio phone number (E.164 format)
- ADMIN_PHONE_NUMBER: Phone to receive notifications

Setup:
1. Sign up at https://twilio.com/try-twilio
2. Get a phone number ($1/month)
3. Find credentials in Console Dashboard
4. Add to .env file

Usage:
    from services import TwilioService

    twilio = TwilioService()

    # Send urgent contact notification
    twilio.send_contact_alert(
        from_name="Jane Doe",
        from_email="jane@company.com",
        message_preview="Interested in senior role...",
        is_urgent=True
    )

    # Send custom SMS
    twilio.send_sms(
        to_phone="+1-555-0123",
        message="Your message here"
    )
"""

import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime

try:
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioRestException
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logging.warning("Twilio package not installed. SMS functionality will be degraded.")

logger = logging.getLogger(__name__)


class TwilioService:
    """Twilio SMS service wrapper."""

    def __init__(self):
        """Initialize Twilio service with credentials."""
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.from_phone = os.getenv('TWILIO_PHONE_NUMBER')
        self.admin_phone = os.getenv('ADMIN_PHONE_NUMBER')

        if not TWILIO_AVAILABLE:
            logger.warning("Twilio package not available. Install with: pip install twilio")
            self.enabled = False
            return

        if not all([self.account_sid, self.auth_token, self.from_phone]):
            logger.warning("Twilio credentials not set. SMS functionality disabled.")
            self.enabled = False
            return

        try:
            self.client = Client(self.account_sid, self.auth_token)
            self.enabled = True
            logger.info(f"TwilioService initialized. From: {self.from_phone}")

        except Exception as e:
            logger.error(f"Failed to initialize Twilio client: {e}")
            self.enabled = False

    def send_sms(
        self,
        to_phone: str,
        message: str,
        from_phone: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send an SMS via Twilio.

        Args:
            to_phone: Recipient phone number (E.164 format: +1-555-0123)
            message: SMS message text (max 1600 characters)
            from_phone: Sender phone (defaults to configured number)

        Returns:
            Dict with status and message SID

        Raises:
            Exception: If SMS sending fails

        Note:
            Phone numbers must be in E.164 format: +[country code][number]
            Example: +14155552671 for US number
        """
        if not self.enabled:
            logger.warning(f"SMS service disabled. Would send: {message[:50]}... to {to_phone}")
            return {'status': 'disabled', 'message': 'SMS service not configured'}

        # Validate phone number format
        if not to_phone.startswith('+'):
            logger.warning(f"Phone number should be in E.164 format: {to_phone}")
            to_phone = f"+1{to_phone.replace('-', '').replace(' ', '')}"  # Assume US if not specified

        try:
            sms = self.client.messages.create(
                body=message,
                from_=from_phone or self.from_phone,
                to=to_phone
            )

            logger.info(f"SMS sent to {to_phone} (SID: {sms.sid}, Status: {sms.status})")

            return {
                'status': 'sent',
                'message_sid': sms.sid,
                'to': to_phone,
                'delivery_status': sms.status,
                'message_preview': message[:50]
            }

        except TwilioRestException as e:
            logger.error(f"Twilio API error sending to {to_phone}: {e.msg}")
            raise Exception(f"SMS sending failed: {e.msg}")

        except Exception as e:
            logger.error(f"Failed to send SMS to {to_phone}: {e}")
            raise Exception(f"SMS sending failed: {e}")

    def send_contact_alert(
        self,
        from_name: str,
        from_email: str,
        message_preview: str,
        is_urgent: bool = False
    ) -> Dict[str, Any]:
        """Send contact form alert to admin phone.

        Args:
            from_name: Contact's name
            from_email: Contact's email
            message_preview: First ~100 chars of message
            is_urgent: If True, adds urgent prefix

        Returns:
            Dict with send status

        Example:
            twilio.send_contact_alert(
                from_name="Jane Doe",
                from_email="jane@company.com",
                message_preview="Interested in senior developer role. Looking to schedule...",
                is_urgent=True
            )
        """
        if not self.admin_phone:
            logger.warning("ADMIN_PHONE_NUMBER not configured. Cannot send SMS alert.")
            return {'status': 'skipped', 'reason': 'Admin phone not configured'}

        # Truncate message preview if too long
        if len(message_preview) > 100:
            message_preview = message_preview[:97] + "..."

        # Build SMS message (keep under 160 chars for single segment)
        urgent_prefix = "🚨 URGENT: " if is_urgent else ""

        message = (
            f"{urgent_prefix}New contact from {from_name} ({from_email})\n\n"
            f'"{message_preview}"\n\n'
            f"Check email for full details."
        )

        # Trim if still too long
        if len(message) > 160:
            message = message[:157] + "..."

        return self.send_sms(
            to_phone=self.admin_phone,
            message=message
        )

    def send_hiring_manager_alert(
        self,
        company_name: str,
        contact_name: str,
        interest_level: str = "high"
    ) -> Dict[str, Any]:
        """Send high-priority alert for hiring manager contacts.

        Args:
            company_name: Company name
            contact_name: Hiring manager's name
            interest_level: 'high', 'medium', 'low'

        Returns:
            Dict with send status

        Example:
            twilio.send_hiring_manager_alert(
                company_name="Google",
                contact_name="Jane Smith",
                interest_level="high"
            )
        """
        if not self.admin_phone:
            return {'status': 'skipped', 'reason': 'Admin phone not configured'}

        emoji_map = {
            'high': '🔥',
            'medium': '📬',
            'low': '📨'
        }

        emoji = emoji_map.get(interest_level, '📬')

        message = (
            f"{emoji} Hiring Manager Contact:\n"
            f"{contact_name} from {company_name}\n\n"
            f"Interest: {interest_level.upper()}\n"
            f"Check dashboard for details."
        )

        return self.send_sms(
            to_phone=self.admin_phone,
            message=message
        )

    def send_system_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "info"
    ) -> Dict[str, Any]:
        """Send system alert notification.

        Args:
            alert_type: Type of alert (e.g., 'error', 'warning', 'info')
            message: Alert message
            severity: 'critical', 'warning', 'info'

        Returns:
            Dict with send status

        Example:
            twilio.send_system_alert(
                alert_type="database_error",
                message="Vector search failing - check Supabase",
                severity="critical"
            )
        """
        if not self.admin_phone:
            return {'status': 'skipped', 'reason': 'Admin phone not configured'}

        # Only send SMS for critical/warning alerts to avoid spam
        if severity not in ['critical', 'warning']:
            return {'status': 'skipped', 'reason': f'Severity {severity} does not trigger SMS'}

        emoji_map = {
            'critical': '🚨',
            'warning': '⚠️',
            'info': 'ℹ️'
        }

        emoji = emoji_map.get(severity, 'ℹ️')

        sms_message = (
            f"{emoji} System Alert: {alert_type.upper()}\n\n"
            f"{message[:120]}\n\n"
            f"Time: {datetime.now().strftime('%I:%M %p')}"
        )

        return self.send_sms(
            to_phone=self.admin_phone,
            message=sms_message
        )

    def get_message_status(self, message_sid: str) -> Dict[str, Any]:
        """Get delivery status of a sent message.

        Args:
            message_sid: Message SID from send_sms response

        Returns:
            Dict with delivery status

        Possible statuses:
        - queued: Message queued for sending
        - sending: Message is being sent
        - sent: Message sent to carrier
        - delivered: Message delivered to recipient
        - failed: Message failed to send
        - undelivered: Message failed to deliver

        Example:
            result = twilio.send_sms("+1-555-0123", "Test message")
            status = twilio.get_message_status(result['message_sid'])
            print(status['delivery_status'])
        """
        if not self.enabled:
            return {'status': 'unavailable', 'reason': 'Twilio not configured'}

        try:
            message = self.client.messages(message_sid).fetch()

            return {
                'status': 'success',
                'message_sid': message_sid,
                'delivery_status': message.status,
                'to': message.to,
                'from': message.from_,
                'sent_at': message.date_sent,
                'error_code': message.error_code,
                'error_message': message.error_message
            }

        except Exception as e:
            logger.error(f"Failed to fetch message status for {message_sid}: {e}")
            return {
                'status': 'error',
                'reason': str(e)
            }

    def health_check(self) -> Dict[str, Any]:
        """Check if Twilio service is healthy.

        Returns:
            Dict with status and configuration info

        This checks:
        - Package availability
        - Credentials configured
        - Account status (via API)
        """
        if not self.enabled:
            return {
                'status': 'disabled',
                'reason': 'Credentials not configured or package not installed',
                'package_available': TWILIO_AVAILABLE
            }

        try:
            # Test API connection by fetching account info
            account = self.client.api.accounts(self.account_sid).fetch()

            return {
                'status': 'healthy',
                'account_status': account.status,
                'from_phone': self.from_phone,
                'admin_phone': self.admin_phone,
                'package_available': TWILIO_AVAILABLE
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'reason': str(e),
                'package_available': TWILIO_AVAILABLE
            }


# Global instance for convenience
_twilio_service = None

def get_twilio_service() -> TwilioService:
    """Get or create global Twilio service instance.

    Returns:
        TwilioService instance

    Example:
        from services import get_twilio_service

        twilio = get_twilio_service()
        twilio.send_contact_alert(...)
    """
    global _twilio_service
    if _twilio_service is None:
        _twilio_service = TwilioService()
    return _twilio_service
