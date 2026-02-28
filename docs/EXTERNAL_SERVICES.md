# 🚀 External Services Setup Guide

**Last Updated**: October 5, 2025  
**Estimated Time**: 30-45 minutes  
**Prerequisites**: Supabase setup completed, Supabase project created

---

## 📋 Setup Checklist

### ✅ Step 1: Supabase Configuration (5 min)

**If you haven't completed Supabase setup:**

1. **Create Supabase Project**
   - Go to: https://supabase.com/dashboard
   - Click "New Project"
   - Choose organization and region (US-West recommended)
   - Set a strong database password (save it!)
   - Wait 2-3 minutes for project provisioning

2. **Get API Credentials**
   - Navigate to: Settings → API
   - Copy these values to your `.env` file:
     ```bash
     SUPABASE_URL=https://your-project.supabase.co
     SUPABASE_SERVICE_ROLE_KEY=eyJ... (long JWT token)
     SUPABASE_ANON_KEY=eyJ... (long JWT token)
     ```

3. **Verify Database Setup**
   ```bash
   cd /Users/noahdelacalzada/NoahsAIAssistant/NoahsAIAssistant-
   python scripts/migrate_data_to_supabase.py
   ```

---

### ✅ Step 2: Resend Email Service (10 min)

**Purpose**: Send transactional emails (contact notifications, resume delivery)

1. **Create Resend Account**
   - Go to: https://resend.com
   - Sign up with GitHub or email
   - Free tier: 3,000 emails/month, 100 emails/day

2. **Get API Key**
   - Navigate to: https://resend.com/api-keys
   - Click "Create API Key"
   - Name it: "Noah AI Assistant Production"
   - Copy the key (starts with `re_`)
   - Add to `.env`:
     ```bash
     RESEND_API_KEY=re_abc123xyz...
     ```

3. **Add Domain** (Optional but recommended)
   - Navigate to: https://resend.com/domains
   - Add your domain (e.g., `yourdomain.com`)
   - Add DNS records (TXT, MX, CNAME) to your domain registrar
   - Wait 5-10 minutes for verification
   - Set your from email:
     ```bash
     RESEND_FROM_EMAIL=noah@yourdomain.com
     ```
   
   **OR use Resend's test domain** (for testing):
   ```bash
   RESEND_FROM_EMAIL=onboarding@resend.dev
   ```

4. **Set Admin Email**
   ```bash
   ADMIN_EMAIL=your.personal.email@gmail.com
   ```

---

### ✅ Step 3: Twilio SMS Service (15 min)

**Purpose**: Send urgent SMS notifications (high-priority contacts)

1. **Create Twilio Account**
   - Go to: https://www.twilio.com/try-twilio
   - Sign up (requires phone verification)
   - Free trial: $15 credit, ~1,900 SMS messages

2. **Get Phone Number**
   - Navigate to: Phone Numbers → Buy a Number
   - Search for US number with SMS capability
   - Cost: $1/month (deducted from trial credit)
   - Purchase the number

3. **Get API Credentials**
   - Navigate to: Console → Account Info
   - Copy these values:
     ```bash
     TWILIO_ACCOUNT_SID=AC... (34 characters)
     TWILIO_AUTH_TOKEN=... (32 characters)
     TWILIO_PHONE_NUMBER=+15551234567  # Your purchased number
     ```

4. **Set Admin Phone**
   ```bash
   ADMIN_PHONE_NUMBER=+15559876543  # Your personal phone
   ```

5. **Verify Phone Number** (Trial accounts only)
   - Navigate to: Console → Verified Caller IDs
   - Add your admin phone number
   - Enter verification code sent via SMS

---

### ✅ Step 4: Run Setup Script (5 min)

This script will:
- ✅ Verify all API credentials
- ✅ Create storage buckets
- ✅ Upload test files (resume, headshot)
- ✅ Test email delivery
- ✅ Test SMS delivery

**Run the setup:**
```bash
cd /Users/noahdelacalzada/NoahsAIAssistant/NoahsAIAssistant-
python scripts/setup_external_services.py
```

**Expected output:**
```
🚀 External Services Setup - External Services Integration
================================================

Step 1: Checking Supabase connection...
✅ Connected to Supabase successfully

Step 2: Creating storage buckets...
✅ Created 'resumes' bucket (private)
✅ Created 'headshots' bucket (public)

Step 3: Testing file uploads...
✅ Uploaded test resume
✅ Uploaded test headshot

Step 4: Testing email service...
✅ Sent test email to your.email@gmail.com

Step 5: Testing SMS service...
✅ Sent test SMS to +15559876543

🎉 External Services Setup Complete!
```

---

### ✅ Step 5: Verify Services (5 min)

1. **Check Email**
   - Open your admin email inbox
   - Look for email from `noah@yourdomain.com`
   - Subject: "Noah's AI Assistant - Test Email"
   - Verify formatting and links work

2. **Check SMS**
   - Check your admin phone
   - Look for SMS from your Twilio number
   - Message: "🧪 Test message from Noah's AI Assistant"

3. **Check Storage**
   - Go to: Supabase Dashboard → Storage
   - Verify buckets exist:
     - `resumes` (private) - contains test PDF
     - `headshots` (public) - contains test image
   - Try downloading test files

4. **Test Integration** (optional)
   ```bash
   cd examples
   python contact_form_integration.py
   ```

---

## 🔧 Configuration Summary

After completing all steps, your `.env` should look like:

```bash
# OpenAI
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002

# Supabase
SUPABASE_URL=https://yourproject.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...
SUPABASE_ANON_KEY=eyJ...

# Resend Email
RESEND_API_KEY=re_...
RESEND_FROM_EMAIL=noah@yourdomain.com
ADMIN_EMAIL=your.email@gmail.com

# Twilio SMS
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+15551234567
ADMIN_PHONE_NUMBER=+15559876543

# File Paths
VECTOR_STORE_PATH=vector_stores/
CAREER_KB_PATH=data/career_kb.csv
CODE_INDEX_PATH=vector_stores/code_index/
MMA_KB_PATH=data/mma_kb.csv
ANALYTICS_DB=sqlite:///analytics.db
```

---

## 💰 Cost Breakdown

| Service | Free Tier | Paid Tier | MVP Cost |
|---------|-----------|-----------|----------|
| **Supabase** | 500 MB DB, 1 GB storage | $25/mo (Pro) | $0 |
| **Resend** | 3K emails/mo, 100/day | $20/mo (10K emails) | $0 |
| **Twilio** | $15 trial credit | Pay-as-you-go | $1/mo + usage |
| **Total** | — | — | **~$1/mo** |

**Usage assumptions:**
- ~10 contact form submissions/day
- ~5 SMS alerts/day ($0.36/day = ~$11/month)
- Storage < 100 MB

---

## 🐛 Troubleshooting

### Issue: "Supabase connection failed"

**Solution:**
```bash
# Verify credentials
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
print('URL:', os.getenv('SUPABASE_URL'))
print('Key length:', len(os.getenv('SUPABASE_SERVICE_ROLE_KEY', '')))
"

# Should output:
# URL: https://yourproject.supabase.co
# Key length: 200+ characters
```

### Issue: "Resend API key invalid"

**Solutions:**
1. Verify key starts with `re_`
2. Check for extra spaces: `RESEND_API_KEY=re_abc` (no spaces)
3. Regenerate key if needed: https://resend.com/api-keys

### Issue: "Twilio SMS failed - 21608 error"

**Cause**: Unverified phone number (trial accounts)

**Solution:**
1. Go to: https://console.twilio.com/us1/develop/phone-numbers/manage/verified
2. Add admin phone number
3. Verify with code sent via SMS

### Issue: "StorageException: Bucket already exists"

**Solution**: This is expected if running setup twice. The script will skip existing buckets.

### Issue: "Email sent but not received"

**Solutions:**
1. Check spam folder
2. Verify `ADMIN_EMAIL` is correct in `.env`
3. Check Resend logs: https://resend.com/emails
4. For custom domains, verify DNS records are correct

---

## 📚 Next Steps

Once External Services setup is complete:

1. **Review Documentation**
   - Read: `docs/EXTERNAL_SERVICES_COMPLETE.md` (comprehensive guide)
   - Read: `docs/EXTERNAL_SERVICES_QUICK_REFERENCE.md` (quick reference)

2. **Test Integration Example**
   ```bash
   cd examples
   python contact_form_integration.py
   ```

3. **Move to Next phase**: API & Deployment
   - Create Next.js API routes
   - Deploy to Vercel
   - Configure production environment
   - Set up monitoring

---

## 🆘 Getting Help

- **Supabase Docs**: https://supabase.com/docs
- **Resend Docs**: https://resend.com/docs
- **Twilio Docs**: https://www.twilio.com/docs/sms
- **Project Docs**: `docs/EXTERNAL_SERVICES_COMPLETE.md`

---

**Status**: Ready for External Services setup ✅  
**Next**: Run `python scripts/setup_external_services.py`
