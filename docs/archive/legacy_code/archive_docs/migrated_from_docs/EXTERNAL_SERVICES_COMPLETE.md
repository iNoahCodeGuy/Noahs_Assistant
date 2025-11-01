# External Services Complete: External Services & Storage 🎉

## Overview

External Services adds external service integrations for file storage, email, and SMS notifications:

- ✅ **Supabase Storage**: Resume and headshot management with signed URLs
- ✅ **Resend**: Modern email API for transactional emails
- ✅ **Twilio**: SMS notifications for urgent contacts

---

## 📦 What Was Implemented

### 1. Supabase Storage Service

**File**: `src/services/storage_service.py`

Features:
- Public bucket for headshots and images
- Private bucket for resumes and sensitive documents
- Signed URL generation for secure temporary access
- Auto-detection of MIME types
- File upload, delete, and listing operations

**Key Methods**:
```python
storage = StorageService()

# Upload resume (private, requires signed URL)
path = storage.upload_resume('data/resume.pdf')

# Upload headshot (public, direct URL)
url = storage.upload_headshot('data/headshot.jpg')

# Generate 24-hour signed URL for resume delivery
signed_url = storage.get_signed_url(path, expires_in=86400)

# List uploaded files
files = storage.list_files('resumes')
```

**Use Cases**:
- Resume delivery to hiring managers with expiring links
- Public profile images accessible without authentication
- Secure document sharing
- Career portfolio file management

---

### 2. Resend Email Service

**File**: `src/services/resend_service.py`

Features:
- HTML email templates with modern styling
- Contact form notifications
- Resume delivery emails with signed URLs
- Welcome emails for new contacts
- Reply-to support for easy responses

**Key Methods**:
```python
resend = ResendService()

# Send contact form notification to admin
resend.send_contact_notification(
    from_name="Jane Doe",
    from_email="jane@company.com",
    message="Interested in senior developer role",
    user_role="Hiring Manager (technical)",
    phone="+1-555-0123"
)

# Send resume to hiring manager
resend.send_resume_email(
    to_email="recruiter@company.com",
    to_name="Jane Doe",
    resume_url="https://supabase.co/storage/v1/object/sign/...",
    message="Thank you for your interest!"
)

# Send welcome email to contact
resend.send_welcome_email(
    to_email="jane@company.com",
    to_name="Jane"
)
```

**Email Templates**:
- Contact notifications with formatted message preview
- Resume delivery with downloadable link and 24-hour expiration notice
- Welcome emails with links to portfolio and social profiles

**Pricing**:
- Free tier: 100 emails/day, 3,000/month
- Paid: $20/month for 50,000 emails
- Excellent deliverability (better than SendGrid/Mailgun)

---

### 3. Twilio SMS Service

**File**: `src/services/twilio_service.py`

Features:
- SMS notifications for urgent contact requests
- System alerts for critical errors
- Hiring manager priority notifications
- Delivery status tracking
- E.164 phone number validation

**Key Methods**:
```python
twilio = TwilioService()

# Send contact alert to admin
twilio.send_contact_alert(
    from_name="Jane Doe",
    from_email="jane@company.com",
    message_preview="Interested in senior role. Looking to schedule...",
    is_urgent=True  # Adds 🚨 to SMS
)

# Send hiring manager priority alert
twilio.send_hiring_manager_alert(
    company_name="Google",
    contact_name="Jane Smith",
    interest_level="high"  # Adds 🔥 emoji
)

# Send system alert
twilio.send_system_alert(
    alert_type="vector_search_error",
    message="pgvector similarity search failing",
    severity="critical"
)

# Check delivery status
status = twilio.get_message_status(message_sid)
```

**SMS Format** (optimized for 160 chars):
```
🚨 URGENT: New contact from Jane Doe (jane@company.com)

"Interested in senior role. Looking to schedule..."

Check email for full details.
```

**Pricing**:
- Phone number: $1/month
- SMS: $0.0079 per message (US)
- Free trial: $15 credit
- ~100 SMS per month budget

---

## 🔧 Setup Instructions

### Prerequisites

1. **Supabase Project**
   - Already configured from Supabase setup
   - Storage API enabled by default

2. **Resend Account**
   - Sign up: https://resend.com
   - Free tier sufficient for MVP

3. **Twilio Account** (optional)
   - Sign up: https://twilio.com/try-twilio
   - Get phone number for SMS

---

### Step 1: Configure Supabase Storage

**In Supabase Dashboard**:
1. Go to Storage → Buckets
2. Buckets will be created automatically by setup script
3. Or create manually:
   - Name: `public`, Public: ✅
   - Name: `private`, Public: ❌

**Set Bucket Policies** (optional for finer control):
```sql
-- Allow authenticated uploads to private bucket
CREATE POLICY "Allow authenticated uploads"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (bucket_id = 'private');

-- Allow public reads from public bucket
CREATE POLICY "Allow public reads"
ON storage.objects FOR SELECT
TO anon
USING (bucket_id = 'public');
```

---

### Step 2: Configure Resend

**1. Sign Up**:
- Go to https://resend.com
- Create account (GitHub OAuth recommended)

**2. Add Domain**:
- Settings → Domains → Add Domain
- Add DNS records (provided by Resend)
- Wait for verification (~5 minutes)

**3. Generate API Key**:
- Settings → API Keys → Create API Key
- Copy key (starts with `re_...`)

**4. Add to `.env`**:
```bash
RESEND_API_KEY=re_...
RESEND_FROM_EMAIL=noah@yourdomain.com
ADMIN_EMAIL=noah@personal-email.com
```

**Testing**:
```bash
# Quick test
python -c "
from services import get_resend_service
resend = get_resend_service()
print(resend.health_check())
"
```

---

### Step 3: Configure Twilio (Optional)

**1. Sign Up**:
- Go to https://twilio.com/try-twilio
- Verify your email and phone
- Get $15 free trial credit

**2. Get Phone Number**:
- Console → Phone Numbers → Buy a Number
- Choose local number ($1/month)
- Enable SMS capability

**3. Find Credentials**:
- Console Dashboard → Account Info
- Copy Account SID and Auth Token

**4. Add to `.env`**:
```bash
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1-555-0123
ADMIN_PHONE_NUMBER=+1-555-0456
```

**Testing**:
```bash
# Quick test
python -c "
from services import get_twilio_service
twilio = get_twilio_service()
print(twilio.health_check())
"
```

---

### Step 4: Run Setup Script

```bash
# Install new dependencies
pip install resend twilio

# Run External Services setup
python scripts/setup_external_services.py
```

**What the script does**:
1. ✅ Creates storage buckets (public & private)
2. ✅ Uploads resume and headshot (if present)
3. ✅ Tests email service configuration
4. ✅ Tests SMS service configuration
5. ✅ Verifies all integrations

**Expected Output**:
```
============================================================
External Services: External Services & Storage Setup
============================================================

Step 1: Creating Storage Buckets
✅ Created 'public' bucket
✅ Created 'private' bucket

Step 2: Uploading Files
✅ Uploaded resume: resumes/noah_resume_20251005.pdf
🔗 Signed URL (1 hour): https://supabase.co/storage/v1/object/sign/...
✅ Uploaded headshot: https://supabase.co/storage/v1/object/public/...

Step 3: Testing Email Service (Resend)
📧 Service status: healthy
   From email: noah@yourdomain.com
   Admin email: noah@personal-email.com
✅ Resend is configured and ready

Step 4: Testing SMS Service (Twilio)
📱 Service status: healthy
   Account status: active
   From phone: +1-555-0123
   Admin phone: +1-555-0456
✅ Twilio is configured and ready

Step 5: Integration Verification
📊 Service Status:
   ✅ Storage (Supabase)
   ✅ Email (Resend)
   ✅ SMS (Twilio)

🎉 All services are operational!
```

---

## 📊 Architecture Integration

### Data Flow: Contact Form → Notifications

```
User submits contact form
        ↓
Save to Supabase messages table
        ↓
    ┌───┴───┐
    ↓       ↓
  Email   SMS
(Resend) (Twilio)
    ↓       ↓
  Admin   Admin
 Inbox   Phone
```

### Resume Delivery Flow

```
Hiring manager requests resume
        ↓
Generate signed URL (24h expiration)
        ↓
Send email with download link
        ↓
Hiring manager downloads PDF
        ↓
Link expires after 24 hours
```

### Service Health Monitoring

```python
# Check all services at once
from services import StorageService, ResendService, TwilioService

def check_services():
    storage = StorageService()
    resend = ResendService()
    twilio = TwilioService()

    return {
        'storage': storage.health_check(),
        'email': resend.health_check(),
        'sms': twilio.health_check()
    }
```

---

## 🎯 Usage Examples

### Example 1: Complete Contact Flow

```python
from services import get_storage_service, get_resend_service, get_twilio_service

# User submits contact form
contact_data = {
    'name': 'Jane Doe',
    'email': 'jane@company.com',
    'phone': '+1-555-0123',
    'message': 'Interested in senior developer role. Looking to schedule interview.',
    'role': 'Hiring Manager (technical)'
}

# Send email notification
resend = get_resend_service()
email_result = resend.send_contact_notification(
    from_name=contact_data['name'],
    from_email=contact_data['email'],
    message=contact_data['message'],
    user_role=contact_data['role'],
    phone=contact_data['phone']
)

# Send SMS alert for urgent contacts
if contact_data['role'].startswith('Hiring Manager'):
    twilio = get_twilio_service()
    sms_result = twilio.send_contact_alert(
        from_name=contact_data['name'],
        from_email=contact_data['email'],
        message_preview=contact_data['message'][:100],
        is_urgent=True
    )

# Send welcome email to contact
resend.send_welcome_email(
    to_email=contact_data['email'],
    to_name=contact_data['name']
)
```

### Example 2: Resume Delivery

```python
from services import get_storage_service, get_resend_service

# Generate temporary access URL
storage = get_storage_service()
resume_path = 'resumes/noah_resume_20251005.pdf'
signed_url = storage.get_signed_url(resume_path, expires_in=86400)  # 24 hours

# Send to hiring manager
resend = get_resend_service()
resend.send_resume_email(
    to_email='recruiter@google.com',
    to_name='Jane Doe',
    resume_url=signed_url,
    message='Thank you for your interest! The download link is valid for 24 hours.'
)
```

### Example 3: File Management

```python
from services import get_storage_service

storage = get_storage_service()

# List all resumes
resumes = storage.list_files('resumes')
for file in resumes:
    print(f"{file['name']} - {file['created_at']}")

# Delete old resume
storage.delete_file('resumes/old_resume.pdf')

# Upload new headshot
url = storage.upload_headshot('data/new_headshot.jpg')
print(f"Public URL: {url}")
```

---

## 🧪 Testing

### Manual Testing

```bash
# Test storage
python -c "
from services import get_storage_service
storage = get_storage_service()
print(storage.health_check())
files = storage.list_files()
print(f'Files in private bucket: {len(files)}')
"

# Test email
python -c "
from services import get_resend_service
resend = get_resend_service()
result = resend.send_contact_notification(
    from_name='Test User',
    from_email='test@example.com',
    message='Test message',
    user_role='Software Developer'
)
print(result)
"

# Test SMS
python -c "
from services import get_twilio_service
twilio = get_twilio_service()
result = twilio.send_contact_alert(
    from_name='Test User',
    from_email='test@example.com',
    message_preview='Test SMS',
    is_urgent=False
)
print(result)
"
```

### Integration Tests

Create `tests/test_external_services.py`:
```python
import pytest
from services import StorageService, ResendService, TwilioService

def test_storage_health():
    storage = StorageService()
    health = storage.health_check()
    assert health['status'] == 'healthy'

def test_email_health():
    resend = ResendService()
    health = resend.health_check()
    assert health['status'] in ['healthy', 'disabled']

def test_sms_health():
    twilio = TwilioService()
    health = twilio.health_check()
    assert health['status'] in ['healthy', 'disabled']
```

---

## 💰 Cost Estimation

### Monthly Costs (Estimated)

| Service | Usage | Cost |
|---------|-------|------|
| **Supabase Storage** | 1 GB files, 10 GB bandwidth | $0 (within free tier) |
| **Resend** | 500 emails/month | $0 (within free tier) |
| **Twilio** | Phone + 100 SMS | $1 + $0.79 = $1.79 |
| **Total** | | **~$2/month** |

**Free Tier Limits**:
- Supabase: 1 GB storage, 2 GB bandwidth
- Resend: 3,000 emails/month, 100/day
- Twilio: $15 trial credit (~1,800 SMS)

**Scaling Costs**:
- Supabase: $25/month (100 GB storage, 200 GB bandwidth)
- Resend: $20/month (50,000 emails)
- Twilio: ~$8/month for 1,000 SMS

---

## 🚀 Next Steps

### Immediate (External Services Complete)
- [x] Supabase Storage configured
- [x] Resend email integration
- [x] Twilio SMS integration
- [x] Setup script and documentation

### Next phase: API & Deployment
- [ ] Create Next.js API routes for serverless functions
- [ ] Deploy to Vercel
- [ ] Set up custom domain
- [ ] Configure production environment variables
- [ ] Set up monitoring and alerts

### Future Enhancements
- [ ] React Email templates for better design
- [ ] SMS delivery receipts and tracking
- [ ] File upload from Streamlit UI
- [ ] Automated resume versioning
- [ ] Contact form analytics dashboard
- [ ] Email open/click tracking

---

## 📚 Resources

### Documentation
- **Supabase Storage**: https://supabase.com/docs/guides/storage
- **Resend API**: https://resend.com/docs
- **Twilio SMS**: https://www.twilio.com/docs/sms
- **React Email**: https://react.email (for advanced templates)

### Pricing Pages
- Supabase: https://supabase.com/pricing
- Resend: https://resend.com/pricing
- Twilio: https://www.twilio.com/pricing

### Tutorials
- Supabase Storage with Python: https://supabase.com/docs/guides/storage/uploads/standard-uploads
- Resend with Python: https://resend.com/docs/send-with-python
- Twilio SMS Quickstart: https://www.twilio.com/docs/sms/quickstart/python

---

**External Services Status**: ✅ **COMPLETE**

All external services are integrated and operational. Ready for Next phase: API & Deployment!

**Last Updated**: October 5, 2025
**Maintainer**: Noah De La Calzada
