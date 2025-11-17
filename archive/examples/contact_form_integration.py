"""Example integration of External Services services in Streamlit contact form.

This demonstrates:
1. Contact form with file upload
2. Email and SMS notifications
3. Resume delivery with signed URLs

Usage:
    streamlit run examples/contact_form_integration.py
"""

import streamlit as st
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from services import get_storage_service, get_resend_service, get_twilio_service


def contact_form_page():
    """Contact form with External Services integrations."""
    st.title("📬 Contact Noah")

    with st.form("contact_form"):
        # User inputs
        name = st.text_input("Your Name*", placeholder="Jane Doe")
        email = st.text_input("Your Email*", placeholder="jane@company.com")
        phone = st.text_input("Phone (optional)", placeholder="+1-555-0123")

        role = st.selectbox(
            "I am a...*",
            [
                "Hiring Manager (technical)",
                "Hiring Manager (nontechnical)",
                "Recruiter",
                "Software Developer",
                "Just curious"
            ]
        )

        message = st.text_area(
            "Message*",
            placeholder="Tell me about your role, company, or project...",
            height=150
        )

        # Optional: Request resume
        request_resume = st.checkbox("📄 Send me Noah's resume")

        # Submit button
        submitted = st.form_submit_button("Send Message", use_container_width=True)

        if submitted:
            # Validation
            if not all([name, email, message]):
                st.error("❌ Please fill out all required fields (*)")
                return

            # Process submission
            with st.spinner("Sending your message..."):
                try:
                    # 1. Send email notification to admin
                    resend = get_resend_service()
                    email_result = resend.send_contact_notification(
                        from_name=name,
                        from_email=email,
                        message=message,
                        user_role=role,
                        phone=phone if phone else None
                    )

                    # 2. Send SMS alert if urgent contact (hiring manager)
                    if "Hiring Manager" in role:
                        twilio = get_twilio_service()
                        sms_result = twilio.send_contact_alert(
                            from_name=name,
                            from_email=email,
                            message_preview=message[:100],
                            is_urgent=True
                        )

                    # 3. Send welcome email to user
                    resend.send_welcome_email(
                        to_email=email,
                        to_name=name
                    )

                    # 4. If resume requested, send it
                    if request_resume:
                        storage = get_storage_service()

                        # Get latest resume
                        resumes = storage.list_files('resumes')
                        if resumes:
                            latest_resume = max(resumes, key=lambda x: x['created_at'])
                            resume_path = latest_resume['name']

                            # Generate 24-hour signed URL
                            signed_url = storage.get_signed_url(
                                resume_path,
                                expires_in=86400
                            )

                            # Email resume
                            resend.send_resume_email(
                                to_email=email,
                                to_name=name,
                                resume_url=signed_url,
                                message="Thank you for your interest! Here's my resume."
                            )

                    # Success message
                    st.success("✅ Message sent successfully!")

                    st.info(f"""
                    📧 **Confirmation email sent to**: {email}

                    {f"📄 **Resume sent**: Check your inbox for download link (valid 24 hours)" if request_resume else ""}

                    💬 **I'll respond within 24 hours!**
                    """)

                except Exception as e:
                    st.error(f"❌ Error sending message: {str(e)}")
                    st.info("Please try emailing noah@yourdomain.com directly.")


def admin_dashboard():
    """Admin dashboard to view service health and manage files."""
    st.title("🔧 Admin Dashboard")

    # Service health checks
    st.header("Service Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📦 Storage")
        storage = get_storage_service()
        health = storage.health_check()

        if health['status'] == 'healthy':
            st.success("✅ Operational")
            st.caption(f"Public: {health['public_bucket']}")
            st.caption(f"Private: {health['private_bucket']}")
        else:
            st.error(f"❌ {health.get('error')}")

    with col2:
        st.subheader("📧 Email")
        resend = get_resend_service()
        health = resend.health_check()

        if health['status'] == 'healthy':
            st.success("✅ Operational")
            st.caption(f"From: {health['from_email']}")
        else:
            st.warning(f"⚠️ {health.get('reason')}")

    with col3:
        st.subheader("📱 SMS")
        twilio = get_twilio_service()
        health = twilio.health_check()

        if health['status'] == 'healthy':
            st.success("✅ Operational")
            st.caption(f"Phone: {health['from_phone']}")
        else:
            st.warning(f"⚠️ {health.get('reason')}")

    st.divider()

    # File management
    st.header("File Management")

    tab1, tab2 = st.tabs(["📄 Resumes", "📸 Images"])

    with tab1:
        storage = get_storage_service()
        resumes = storage.list_files('resumes')

        if resumes:
            st.write(f"**{len(resumes)} resume(s) uploaded**")

            for resume in sorted(resumes, key=lambda x: x['created_at'], reverse=True):
                col1, col2, col3 = st.columns([3, 2, 1])

                with col1:
                    st.text(resume['name'])

                with col2:
                    st.caption(resume['created_at'])

                with col3:
                    if st.button("🔗 URL", key=resume['name']):
                        url = storage.get_signed_url(resume['name'], expires_in=3600)
                        st.code(url, language=None)
        else:
            st.info("No resumes uploaded yet")

            # Upload form
            with st.form("upload_resume"):
                uploaded_file = st.file_uploader("Upload Resume", type=['pdf'])
                if st.form_submit_button("Upload") and uploaded_file:
                    # Save temporarily
                    temp_path = Path(f"/tmp/{uploaded_file.name}")
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                    # Upload to storage
                    path = storage.upload_resume(str(temp_path))
                    st.success(f"✅ Uploaded: {path}")
                    st.rerun()

    with tab2:
        headshots = storage.list_files('headshots')

        if headshots:
            st.write(f"**{len(headshots)} image(s) uploaded**")

            for img in headshots:
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.text(img['name'])
                    url = storage.get_public_url('public', f"headshots/{img['name']}")
                    st.image(url, width=200)

                with col2:
                    st.caption(img['created_at'])
        else:
            st.info("No images uploaded yet")


def main():
    """Main app with navigation."""
    st.set_page_config(
        page_title="Noah's AI Assistant - Contact",
        page_icon="📬",
        layout="wide"
    )

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Contact Form", "Admin Dashboard"]
    )

    if page == "Contact Form":
        contact_form_page()
    else:
        # Simple authentication
        password = st.sidebar.text_input("Admin Password", type="password")
        if password == "admin123":  # Change in production!
            admin_dashboard()
        else:
            st.warning("🔒 Enter admin password to access dashboard")


if __name__ == '__main__':
    main()
