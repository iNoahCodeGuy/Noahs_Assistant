"""External Services setup script: Storage, Email & SMS.

This script:
1. Creates Supabase storage buckets
2. Uploads resume and headshot
3. Tests email and SMS services
4. Verifies all integrations

Prerequisites:
- Supabase project created
- Environment variables configured
- Resume and headshot files available

Run:
    python scripts/setup_external_services.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services import StorageService, ResendService, TwilioService
from config.supabase_config import get_supabase_client


def create_storage_buckets():
    """Create public and private storage buckets."""
    print("\n" + "=" * 60)
    print("Step 1: Creating Storage Buckets")
    print("=" * 60)
    
    client = get_supabase_client()
    
    # Create public bucket
    try:
        client.storage.create_bucket(
            'public',
            options={'public': True}
        )
        print("✅ Created 'public' bucket")
    except Exception as e:
        if 'already exists' in str(e).lower():
            print("ℹ️  'public' bucket already exists")
        else:
            print(f"❌ Error creating public bucket: {e}")
    
    # Create private bucket
    try:
        client.storage.create_bucket(
            'private',
            options={'public': False}
        )
        print("✅ Created 'private' bucket")
    except Exception as e:
        if 'already exists' in str(e).lower():
            print("ℹ️  'private' bucket already exists")
        else:
            print(f"❌ Error creating private bucket: {e}")


def upload_files():
    """Upload resume and headshot to storage."""
    print("\n" + "=" * 60)
    print("Step 2: Uploading Files")
    print("=" * 60)
    
    storage = StorageService()
    
    # Define file paths (adjust these to match your actual files)
    resume_path = 'data/resume.pdf'
    headshot_path = 'data/headshot.jpg'
    
    # Upload resume (private)
    if Path(resume_path).exists():
        try:
            path = storage.upload_resume(resume_path)
            print(f"✅ Uploaded resume: {path}")
            
            # Generate signed URL
            signed_url = storage.get_signed_url(path, expires_in=3600)
            print(f"🔗 Signed URL (1 hour): {signed_url[:80]}...")
        except Exception as e:
            print(f"❌ Failed to upload resume: {e}")
    else:
        print(f"⚠️  Resume not found at {resume_path}")
        print(f"   Create a sample file: echo 'Sample Resume' > {resume_path}")
    
    # Upload headshot (public)
    if Path(headshot_path).exists():
        try:
            url = storage.upload_headshot(headshot_path)
            print(f"✅ Uploaded headshot: {url}")
        except Exception as e:
            print(f"❌ Failed to upload headshot: {e}")
    else:
        print(f"⚠️  Headshot not found at {headshot_path}")
        print(f"   Add your headshot as {headshot_path}")


def test_email_service():
    """Test Resend email service."""
    print("\n" + "=" * 60)
    print("Step 3: Testing Email Service (Resend)")
    print("=" * 60)
    
    resend = ResendService()
    
    health = resend.health_check()
    print(f"📧 Service status: {health['status']}")
    
    if health['status'] == 'healthy':
        print(f"   From email: {health['from_email']}")
        print(f"   Admin email: {health['admin_email']}")
        print("\n✅ Resend is configured and ready")
        
        # Optionally send test email
        send_test = input("\nSend test email? (y/n): ")
        if send_test.lower() == 'y':
            try:
                result = resend.send_contact_notification(
                    from_name="Test User",
                    from_email="test@example.com",
                    message="This is a test message from External Services setup",
                    user_role="Software Developer"
                )
                print(f"✅ Test email sent: {result.get('message_id')}")
            except Exception as e:
                print(f"❌ Test email failed: {e}")
    else:
        print(f"⚠️  {health.get('reason')}")
        print("\nTo enable email:")
        print("1. Sign up at https://resend.com")
        print("2. Add domain and verify")
        print("3. Generate API key")
        print("4. Add to .env:")
        print("   RESEND_API_KEY=re_...")
        print("   RESEND_FROM_EMAIL=noah@yourdomain.com")


def test_sms_service():
    """Test Twilio SMS service."""
    print("\n" + "=" * 60)
    print("Step 4: Testing SMS Service (Twilio)")
    print("=" * 60)
    
    twilio = TwilioService()
    
    health = twilio.health_check()
    print(f"📱 Service status: {health['status']}")
    
    if health['status'] == 'healthy':
        print(f"   Account status: {health.get('account_status')}")
        print(f"   From phone: {health['from_phone']}")
        print(f"   Admin phone: {health['admin_phone']}")
        print("\n✅ Twilio is configured and ready")
        
        # Optionally send test SMS
        send_test = input("\nSend test SMS? (y/n): ")
        if send_test.lower() == 'y':
            try:
                result = twilio.send_contact_alert(
                    from_name="Test User",
                    from_email="test@example.com",
                    message_preview="This is a test SMS from External Services setup",
                    is_urgent=False
                )
                print(f"✅ Test SMS sent: {result.get('message_sid')}")
            except Exception as e:
                print(f"❌ Test SMS failed: {e}")
    else:
        print(f"⚠️  {health.get('reason')}")
        print("\nTo enable SMS:")
        print("1. Sign up at https://twilio.com/try-twilio")
        print("2. Get a phone number ($1/month)")
        print("3. Find credentials in Console Dashboard")
        print("4. Add to .env:")
        print("   TWILIO_ACCOUNT_SID=AC...")
        print("   TWILIO_AUTH_TOKEN=...")
        print("   TWILIO_PHONE_NUMBER=+1-555-0123")
        print("   ADMIN_PHONE_NUMBER=+1-555-0456")


def verify_integration():
    """Verify all services are integrated."""
    print("\n" + "=" * 60)
    print("Step 5: Integration Verification")
    print("=" * 60)
    
    storage = StorageService()
    resend = ResendService()
    twilio = TwilioService()
    
    checks = {
        'Storage (Supabase)': storage.health_check()['status'] == 'healthy',
        'Email (Resend)': resend.health_check()['status'] == 'healthy',
        'SMS (Twilio)': twilio.health_check()['status'] == 'healthy'
    }
    
    print("\n📊 Service Status:")
    for service, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {service}")
    
    all_healthy = all(checks.values())
    
    if all_healthy:
        print("\n🎉 All services are operational!")
    else:
        print("\n⚠️  Some services need configuration. See above for setup instructions.")
    
    return all_healthy


def print_next_steps():
    """Print next steps for deployment."""
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    
    print("""
✅ External Services Setup Complete!

Your external services are now configured:
- 📦 Supabase Storage: Resume & headshot management
- 📧 Resend: Email notifications
- 📱 Twilio: SMS alerts

What's next?

1. **Test Contact Form**
   - Go to your Streamlit app
   - Submit a contact form
   - Check email and SMS notifications

2. **Test Resume Delivery**
   - Generate signed URL for resume
   - Email it to a test recipient
   - Verify 24-hour expiration works

3. **Monitor Service Health**
   - Add health check endpoint to your app
   - Monitor Resend/Twilio dashboards
   - Set up uptime alerts

4. **Deploy to Production**
   - Add environment variables to Vercel
   - Test in staging environment
   - Deploy to production

5. **Phase 4: API & Deployment**
   - Create Next.js API routes
   - Deploy to Vercel
   - Set up custom domain

Documentation:
- Supabase Storage: https://supabase.com/docs/guides/storage
- Resend API: https://resend.com/docs
- Twilio SMS: https://www.twilio.com/docs/sms

Need help? Check docs/EXTERNAL_SERVICES_COMPLETE.md
""")


def main():
    """Run External Services setup."""
    print("=" * 60)
    print("External Services: External Services & Storage Setup")
    print("=" * 60)
    print("\nThis script will:")
    print("1. Create Supabase storage buckets")
    print("2. Upload resume and headshot")
    print("3. Test email service (Resend)")
    print("4. Test SMS service (Twilio)")
    print("5. Verify all integrations")
    
    proceed = input("\nProceed with setup? (y/n): ")
    if proceed.lower() != 'y':
        print("Setup cancelled.")
        return
    
    try:
        # Run setup steps
        create_storage_buckets()
        upload_files()
        test_email_service()
        test_sms_service()
        all_healthy = verify_integration()
        
        # Print next steps
        print_next_steps()
        
        # Return exit code
        sys.exit(0 if all_healthy else 1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
