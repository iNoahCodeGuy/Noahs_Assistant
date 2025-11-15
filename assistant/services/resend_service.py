"""Resend email service for transactional emails.

This service handles:
- Contact form submissions
- Resume delivery to hiring managers
- System notifications

Resend is a modern email API optimized for developers:
- Simple REST API
- Excellent deliverability
- React Email template support
- Generous free tier (100 emails/day, 3,000/month)

Environment Variables Required:
- RESEND_API_KEY: Your Resend API key (https://resend.com/api-keys)
- RESEND_FROM_EMAIL: Verified sender email (e.g., noah@yourdomain.com)

Setup:
1. Sign up at https://resend.com
2. Add and verify your domain
3. Generate API key
4. Add to .env file

Usage:
    from services import ResendService

    resend = ResendService()

    # Send contact form notification
    resend.send_contact_notification(
        from_name="Jane Doe",
        from_email="jane@company.com",
        message="I'd like to discuss the senior role",
        user_role="Hiring Manager (technical)"
    )

    # Send resume to hiring manager
    resend.send_resume_email(
        to_email="recruiter@company.com",
        to_name="Jane Doe",
        resume_url="https://supabase.co/storage/v1/object/sign/..."
    )
"""

import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import resend
    RESEND_AVAILABLE = True
except ImportError:
    RESEND_AVAILABLE = False
    logging.warning("Resend package not installed. Email functionality will be degraded.")

logger = logging.getLogger(__name__)


class ResendService:
    """Resend email service wrapper."""

    def __init__(self):
        """Initialize Resend service with API key."""
        self.api_key = os.getenv('RESEND_API_KEY')
        self.from_email = os.getenv('RESEND_FROM_EMAIL', 'noreply@yourdomain.com')
        self.admin_email = os.getenv('ADMIN_EMAIL', 'noah@yourdomain.com')

        if not RESEND_AVAILABLE:
            logger.warning("Resend package not available. Install with: pip install resend")
            self.enabled = False
            return

        if not self.api_key:
            logger.warning("RESEND_API_KEY not set. Email functionality disabled.")
            self.enabled = False
            return

        # Configure Resend
        resend.api_key = self.api_key
        self.enabled = True

        logger.info(f"ResendService initialized. From: {self.from_email}")

    def send_email(
        self,
        to_email: str,
        subject: str,
        html: str,
        from_email: Optional[str] = None,
        reply_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send an email via Resend.

        Args:
            to_email: Recipient email address
            subject: Email subject line
            html: HTML email body
            from_email: Sender email (defaults to configured from_email)
            reply_to: Reply-to email address

        Returns:
            Dict with status and message ID

        Raises:
            Exception: If email sending fails
        """
        if not self.enabled:
            logger.warning(f"Email service disabled. Would send: {subject} to {to_email}")
            return {'status': 'disabled', 'message': 'Email service not configured'}

        try:
            params = {
                "from": from_email or self.from_email,
                "to": [to_email],
                "subject": subject,
                "html": html
            }

            if reply_to:
                params["reply_to"] = [reply_to]

            response = resend.Emails.send(params)

            logger.info(f"Email sent to {to_email}: {subject} (ID: {response.get('id')})")

            return {
                'status': 'sent',
                'message_id': response.get('id'),
                'to': to_email,
                'subject': subject
            }

        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            raise Exception(f"Email sending failed: {e}")

    def send_contact_notification(
        self,
        from_name: str,
        from_email: str,
        message: str,
        user_role: str,
        phone: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send contact form notification to admin.

        Args:
            from_name: Contact's name
            from_email: Contact's email
            message: Contact's message
            user_role: User's selected role
            phone: Optional phone number

        Returns:
            Dict with send status

        Example:
            resend.send_contact_notification(
                from_name="Jane Doe",
                from_email="jane@company.com",
                message="Interested in senior developer role",
                user_role="Hiring Manager (technical)",
                phone="+1-555-0123"
            )
        """
        subject = f"🔔 Contact Form: {from_name} ({user_role})"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #4F46E5; color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f9fafb; padding: 30px; border: 1px solid #e5e7eb; }}
                .field {{ margin-bottom: 20px; }}
                .label {{ font-weight: 600; color: #6b7280; font-size: 12px; text-transform: uppercase; margin-bottom: 5px; }}
                .value {{ font-size: 16px; color: #111827; }}
                .message {{ background: white; padding: 15px; border-left: 4px solid #4F46E5; margin-top: 20px; }}
                .footer {{ text-align: center; margin-top: 30px; color: #6b7280; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2 style="margin: 0;">New Contact Form Submission</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Portfolia - Noah's AI Assistant</p>
                </div>
                <div class="content">
                    <div class="field">
                        <div class="label">From</div>
                        <div class="value">{from_name}</div>
                    </div>

                    <div class="field">
                        <div class="label">Email</div>
                        <div class="value"><a href="mailto:{from_email}">{from_email}</a></div>
                    </div>

                    {f'<div class="field"><div class="label">Phone</div><div class="value">{phone}</div></div>' if phone else ''}

                    <div class="field">
                        <div class="label">User Role</div>
                        <div class="value">{user_role}</div>
                    </div>

                    <div class="field">
                        <div class="label">Timestamp</div>
                        <div class="value">{datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
                    </div>

                    <div class="message">
                        <div class="label">Message</div>
                        <div class="value" style="white-space: pre-wrap;">{message}</div>
                    </div>
                </div>
                <div class="footer">
                    <p>This is an automated notification from Portfolia, Noah's AI Assistant contact form.</p>
                </div>
            </div>
        </body>
        </html>
        """

        return self.send_email(
            to_email=self.admin_email,
            subject=subject,
            html=html,
            reply_to=from_email
        )

    def send_resume_email(
        self,
        to_email: str,
        to_name: str,
        resume_url: str,
        message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send resume to hiring manager with signed URL.

        Args:
            to_email: Recipient's email
            to_name: Recipient's name
            resume_url: Signed URL to resume (from StorageService)
            message: Optional personalized message

        Returns:
            Dict with send status

        Example:
            storage = get_storage_service()
            resume_url = storage.get_signed_url('resumes/noah_resume.pdf', expires_in=86400)

            resend.send_resume_email(
                to_email="recruiter@company.com",
                to_name="Jane Doe",
                resume_url=resume_url,
                message="Thank you for your interest. Here's my resume."
            )
        """
        subject = "Noah De La Calzada - Resume"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ text-align: center; padding: 30px 0; }}
                .content {{ padding: 20px; }}
                .button {{ display: inline-block; background: #4F46E5; color: white; padding: 14px 30px; text-decoration: none; border-radius: 6px; margin: 20px 0; }}
                .button:hover {{ background: #4338CA; }}
                .note {{ background: #f3f4f6; padding: 15px; border-radius: 6px; margin-top: 20px; font-size: 14px; color: #6b7280; }}
                .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1 style="margin: 0; color: #111827;">Noah De La Calzada</h1>
                    <p style="margin: 10px 0 0 0; color: #6b7280;">Full-Stack Developer & AI Engineer</p>
                </div>

                <div class="content">
                    <p>Hi {to_name},</p>

                    {f'<p>{message}</p>' if message else '<p>Thank you for your interest in my profile. Please find my resume below.</p>'}

                    <div style="text-align: center;">
                        <a href="{resume_url}" class="button">📄 Download Resume (PDF)</a>
                    </div>

                    <div class="note">
                        <strong>📌 Note:</strong> This download link is valid for 24 hours for security purposes.
                        If you need another copy after that, please let me know.
                    </div>

                    <p style="margin-top: 30px;">Looking forward to connecting!</p>
                    <p style="margin: 5px 0;">Best regards,<br><strong>Noah De La Calzada</strong></p>
                </div>

                <div class="footer">
                    <p>
                        📧 <a href="mailto:noah@yourdomain.com">noah@yourdomain.com</a> |
                        🔗 <a href="https://linkedin.com/in/noah">LinkedIn</a> |
                        💻 <a href="https://github.com/noah">GitHub</a>
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

        return self.send_email(
            to_email=to_email,
            subject=subject,
            html=html
        )

    def send_welcome_email(self, to_email: str, to_name: str) -> Dict[str, Any]:
        """Send welcome email to new contacts.

        Args:
            to_email: Recipient's email
            to_name: Recipient's name

        Returns:
            Dict with send status
        """
        subject = "Thanks for reaching out! 🎉"

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .content {{ padding: 20px; }}
                .footer {{ text-align: center; margin-top: 30px; color: #6b7280; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="content">
                    <h2>Hi {to_name}! 👋</h2>

                    <p>Thank you for reaching out through Portfolia. I've received your message and will get back to you shortly.</p>

                    <p>In the meantime, feel free to:</p>
                    <ul>
                        <li>Explore my projects on <a href="https://github.com/noah">GitHub</a></li>
                        <li>Connect on <a href="https://linkedin.com/in/noah">LinkedIn</a></li>
                        <li>Check out my <a href="https://yourwebsite.com">portfolio</a></li>
                    </ul>

                    <p>Looking forward to connecting!</p>
                    <p>Best,<br><strong>Noah</strong></p>
                </div>

                <div class="footer">
                    <p>This is an automated confirmation. You'll receive a personal response soon.</p>
                </div>
            </div>
        </body>
        </html>
        """

        return self.send_email(
            to_email=to_email,
            subject=subject,
            html=html
        )

    def health_check(self) -> Dict[str, Any]:
        """Check if Resend service is healthy.

        Returns:
            Dict with status and configuration info
        """
        if not self.enabled:
            return {
                'status': 'disabled',
                'reason': 'API key not configured or package not installed'
            }

        return {
            'status': 'healthy',
            'from_email': self.from_email,
            'admin_email': self.admin_email,
            'package_available': RESEND_AVAILABLE
        }


# Global instance for convenience
_resend_service = None

def get_resend_service() -> ResendService:
    """Get or create global Resend service instance.

    Returns:
        ResendService instance

    Example:
        from services import get_resend_service

        resend = get_resend_service()
        resend.send_contact_notification(...)
    """
    global _resend_service
    if _resend_service is None:
        _resend_service = ResendService()
    return _resend_service
