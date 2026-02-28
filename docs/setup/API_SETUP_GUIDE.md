# API Setup & Integration Guide

**Purpose**: Complete guide for setting up API keys, configuring services, and integrating with frontend.

**Status**: ✅ Production-ready  
**Last Updated**: October 16, 2025

---

## 📑 Table of Contents

### Quick Navigation
- [🚀 For New Developers](#for-new-developers-start-here) - Quick setup
- [🔑 API Keys Required](#api-keys-required) - What you need
- [⚡ Quick Start](#quick-start) - Get running in 5 minutes
- [🔧 Troubleshooting](#troubleshooting) - Common issues

### Core Sections
1. [API Keys & Services](#api-keys--services)
2. [Environment Configuration](#environment-configuration)
3. [API Endpoints](#api-endpoints)
4. [Frontend Integration](#frontend-integration)
5. [Testing](#testing)
6. [Deployment](#deployment)
7. [Security](#security)

---

## For New Developers (Start Here)

### What This Document Covers
Complete setup for all external services (OpenAI, Supabase, Resend, Twilio, LangSmith) and API integration with Next.js/React frontends.

### Quick Facts
- **4 required services**: OpenAI, Supabase (mandatory), Resend, Twilio (optional)
- **4 API endpoints**: `/api/chat`, `/api/email`, `/api/feedback`, `/api/confess`
- **Deployment**: Vercel serverless functions (auto-scales)
- **Local testing**: `vercel dev` + `python scripts/test_api_endpoints.py`

### 5-Minute Setup Checklist

```bash
# 1. Get API keys (see sections below)
# 2. Create .env file
cp .env.example .env

# 3. Add keys to .env
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://...
SUPABASE_SERVICE_ROLE_KEY=eyJ...

# 4. Test locally
vercel dev

# 5. Deploy
vercel --prod
```

---

## API Keys & Services

### 🔴 Required Services (Must Have)

#### 1. OpenAI API
**Purpose**: LLM generation (GPT-4o-mini), text embeddings (text-embedding-3-small)

**Get Key**:
1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy key (starts with `sk-`)

**Cost**: ~$0.10-0.50/day for typical usage

**Environment Variable**:
```bash
OPENAI_API_KEY=sk-proj-...
```

**Test**:
```bash
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

---

#### 2. Supabase (Database + Vector Store)
**Purpose**: PostgreSQL database, pgvector embeddings, analytics storage

**Get Keys**:
1. Go to https://supabase.com/dashboard
2. Create new project (or use existing)
3. Settings → API → Copy:
   - **Project URL** (`https://xxx.supabase.co`)
   - **Service Role Key** (`eyJhbGciOi...` - secret key, NOT anon key!)

**Cost**: Free tier covers typical usage

**Environment Variables**:
```bash
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOi...
```

**⚠️ CRITICAL**: Use **service_role** key (full access), NOT **anon** key (public access)

**Test**:
```bash
curl "$SUPABASE_URL/rest/v1/messages?select=count" \
  -H "apikey: $SUPABASE_SERVICE_ROLE_KEY" \
  -H "Authorization: Bearer $SUPABASE_SERVICE_ROLE_KEY"
```

---

### 🟡 Optional Services (Recommended)

#### 3. Resend (Email Service)
**Purpose**: Send resume/LinkedIn via email

**Get Key**:
1. Go to https://resend.com/api-keys
2. Create API key
3. Copy key (starts with `re_`)

**Cost**: 100 emails/day free, then $10/month

**Environment Variables**:
```bash
RESEND_API_KEY=re_...
RESEND_FROM_EMAIL=noah@yourdomain.com  # Must be verified domain
```

**Test**:
```bash
curl https://api.resend.com/emails \
  -H "Authorization: Bearer $RESEND_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"from":"noah@yourdomain.com","to":"test@example.com","subject":"Test","html":"<p>Works!</p>"}'
```

**Graceful Degradation**: If not configured, email actions return "Email service unavailable"

---

#### 4. Twilio (SMS Service)
**Purpose**: Send SMS notifications (contact requests, confessions)

**Get Keys**:
1. Go to https://console.twilio.com
2. Get Account SID + Auth Token
3. Buy phone number or use trial number

**Cost**: ~$0.0075/SMS (after $15 trial credit)

**Environment Variables**:
```bash
TWILIO_ACCOUNT_SID=ACxxxx...
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_FROM=+1234567890  # Your Twilio number
TWILIO_TO=+1987654321    # Noah's number (receives notifications)
```

**Test**:
```bash
curl -X POST "https://api.twilio.com/2010-04-01/Accounts/$TWILIO_ACCOUNT_SID/Messages.json" \
  -u "$TWILIO_ACCOUNT_SID:$TWILIO_AUTH_TOKEN" \
  -d "From=$TWILIO_FROM" \
  -d "To=$TWILIO_TO" \
  -d "Body=Test message"
```

**Graceful Degradation**: If not configured, SMS actions return "SMS service unavailable"

---

#### 5. LangSmith (Observability)
**Purpose**: Trace LLM calls, debug prompts, monitor production

**Get Key**:
1. Go to https://smith.langchain.com
2. Settings → API Keys → Create
3. Copy key (starts with `lsv2_pt_`)

**Cost**: Free tier: 5,000 traces/month

**Environment Variables**:
```bash
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=noahs-ai-assistant
```

**Test**: Run any chat query, then check https://smith.langchain.com for trace

**Graceful Degradation**: If not configured, tracing silently disabled (no errors)

---

## Environment Configuration

### Local Development (.env)

Create `.env` file in project root:

```bash
# Required
OPENAI_API_KEY=sk-proj-...
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOi...

# Optional (email)
RESEND_API_KEY=re_...
RESEND_FROM_EMAIL=noah@yourdomain.com

# Optional (SMS)
TWILIO_ACCOUNT_SID=ACxxxx...
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_FROM=+1234567890
TWILIO_TO=+1987654321

# Optional (observability)
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=noahs-ai-assistant
```

**⚠️ Security**: Never commit `.env` to git (add to `.gitignore`)

---

### Vercel Production

#### Method 1: Vercel CLI (Recommended)

```bash
# Add each variable
vercel env add OPENAI_API_KEY
vercel env add SUPABASE_URL
vercel env add SUPABASE_SERVICE_ROLE_KEY
vercel env add RESEND_API_KEY
vercel env add TWILIO_ACCOUNT_SID
vercel env add TWILIO_AUTH_TOKEN
vercel env add TWILIO_FROM
vercel env add TWILIO_TO
vercel env add LANGCHAIN_API_KEY

# Select environments: Production, Preview, Development
```

#### Method 2: Vercel Dashboard

1. Go to https://vercel.com/dashboard
2. Select project → Settings → Environment Variables
3. Add each variable with scope: Production, Preview, Development

**⚠️ CRITICAL**: Ensure no trailing newlines in API keys (causes `httpcore.LocalProtocolError`)

```bash
# Test for newlines
python -c "import os; key=os.getenv('OPENAI_API_KEY'); print(f'Length: {len(key)}, Has newline: {repr(key[-5:])}')"

# Should show: Length: 164, Has newline: '...abc'
# NOT: Length: 165, Has newline: '...abc\n'
```

---

## API Endpoints

All endpoints deployed as Vercel serverless functions at `https://noahsaiassistant.vercel.app/api/*`

### POST /api/chat
**Purpose**: Execute conversation flow with LangGraph backend

**Request**:
```json
{
  "query": "How does your RAG system work?",
  "role": "Software Developer",
  "session_id": "uuid-optional",
  "chat_history": [
    {"role": "user", "content": "Previous message"},
    {"role": "assistant", "content": "Previous response"}
  ],
  "user_email": "optional@example.com",
  "user_name": "Optional Name"
}
```

**Response**:
```json
{
  "success": true,
  "answer": "The RAG system uses pgvector for retrieval...",
  "actions_taken": ["retrieve_chunks", "generate_answer"],
  "session_id": "uuid",
  "metadata": {
    "latency_ms": 1234,
    "token_count": 450
  }
}
```

**Error Response**:
```json
{
  "success": false,
  "error": "OpenAI API error: Invalid API key"
}
```

---

### POST /api/email
**Purpose**: Send resume/LinkedIn via Resend

**Request**:
```json
{
  "type": "resume",  // or "linkedin"
  "to_email": "user@example.com",
  "to_name": "Jane Doe",
  "message": "Requested via chat"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Resume sent successfully!"
}
```

**Requires**: `RESEND_API_KEY` configured

---

### POST /api/feedback
**Purpose**: Log user feedback and contact requests

**Request**:
```json
{
  "message_id": "uuid",
  "rating": 5,
  "comment": "Very helpful, thank you!",
  "contact_requested": true,
  "user_email": "user@example.com",
  "user_name": "Jane Doe",
  "user_phone": "+1234567890"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Feedback received! Noah will reach out soon."
}
```

**Side Effects**: If `contact_requested=true` and Twilio configured, sends SMS to Noah

---

### POST /api/confess
**Purpose**: Handle confession submissions (anonymous or with contact)

**Request** (Anonymous):
```json
{
  "message": "I think you're amazing!",
  "is_anonymous": true
}
```

**Request** (With Contact):
```json
{
  "message": "I think you're amazing!",
  "is_anonymous": false,
  "name": "Secret Admirer",
  "contact": "admirer@example.com"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Your confession has been received! 💕"
}
```

**Side Effects**: If Twilio configured, sends SMS to Noah

---

### GET /api/analytics
**Purpose**: Fetch live analytics data (see [ANALYTICS_IMPLEMENTATION.md](../features/ANALYTICS_IMPLEMENTATION.md))

**Request**: None (GET request)

**Response**: See analytics docs for full schema

**Rate Limit**: 6 requests/minute per IP

**Requires**: `SUPABASE_SERVICE_ROLE_KEY` (server-side only)

---

## Frontend Integration

### React/Next.js Examples

#### Chat Component

```typescript
// components/Chat.tsx
import { useState } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!query.trim()) return;

    setLoading(true);
    
    // Add user message to UI
    const newMessages = [...messages, { role: 'user' as const, content: query }];
    setMessages(newMessages);
    setQuery('');

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          role: 'Software Developer',  // Get from role selector
          session_id: sessionStorage.getItem('session_id'),
          chat_history: messages
        })
      });

      const data = await response.json();

      if (data.success) {
        // Add assistant response
        setMessages([...newMessages, {
          role: 'assistant',
          content: data.answer
        }]);
        
        // Save session ID
        sessionStorage.setItem('session_id', data.session_id);
        
        // Handle actions
        if (data.actions_taken?.includes('send_resume')) {
          alert('Resume sent! Check your email.');
        }
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      alert(`Network error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            {msg.content}
          </div>
        ))}
      </div>
      
      <div className="input-area">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Ask me anything..."
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading}>
          {loading ? 'Thinking...' : 'Send'}
        </button>
      </div>
    </div>
  );
}
```

---

#### Resume Request Button

```typescript
// components/ResumeButton.tsx
export function ResumeButton({ userEmail, userName }: { userEmail: string; userName: string }) {
  const [loading, setLoading] = useState(false);

  const requestResume = async () => {
    setLoading(true);
    
    try {
      const response = await fetch('/api/email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'resume',
          to_email: userEmail,
          to_name: userName,
          message: 'Requested via chat interface'
        })
      });

      const data = await response.json();

      if (data.success) {
        alert('✅ Resume sent! Check your email.');
      } else {
        alert(`❌ ${data.error}`);
      }
    } catch (error) {
      alert(`Network error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <button onClick={requestResume} disabled={loading}>
      {loading ? 'Sending...' : '📄 Send Resume'}
    </button>
  );
}
```

---

#### Feedback Form

```typescript
// components/FeedbackForm.tsx
export function FeedbackForm({ messageId }: { messageId: string }) {
  const [rating, setRating] = useState(0);
  const [comment, setComment] = useState('');
  const [contactRequested, setContactRequested] = useState(false);
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');

  const submitFeedback = async () => {
    if (rating === 0) {
      alert('Please select a rating');
      return;
    }

    try {
      const response = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message_id: messageId,
          rating,
          comment,
          contact_requested: contactRequested,
          user_email: contactRequested ? email : undefined,
          user_name: contactRequested ? name : undefined
        })
      });

      const data = await response.json();
      alert(data.message);
    } catch (error) {
      alert(`Error: ${error.message}`);
    }
  };

  return (
    <div className="feedback-form">
      <h3>How was this response?</h3>
      
      <div className="star-rating">
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            onClick={() => setRating(star)}
            className={star <= rating ? 'active' : ''}
          >
            ⭐
          </button>
        ))}
      </div>
      
      <textarea
        placeholder="Additional comments (optional)"
        value={comment}
        onChange={(e) => setComment(e.target.value)}
      />
      
      <label>
        <input
          type="checkbox"
          checked={contactRequested}
          onChange={(e) => setContactRequested(e.target.checked)}
        />
        I'd like Noah to contact me
      </label>
      
      {contactRequested && (
        <>
          <input
            type="email"
            placeholder="Your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <input
            type="text"
            placeholder="Your name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </>
      )}
      
      <button onClick={submitFeedback}>Submit Feedback</button>
    </div>
  );
}
```

---

## Testing

### Local Testing

#### Step 1: Start Vercel Dev Server

```bash
# Terminal 1: Start server
vercel dev

# Should see:
# > Ready! Available at http://localhost:3000
```

#### Step 2: Run API Tests

```bash
# Terminal 2: Run tests
python scripts/test_api_endpoints.py

# Expected output:
# ✅ Testing /api/chat endpoint...
# ✅ Chat endpoint working!
# ✅ Testing /api/email endpoint...
# ✅ Email endpoint working!
# ✅ Testing /api/feedback endpoint...
# ✅ Feedback endpoint working!
# ✅ Testing /api/confess endpoint...
# ✅ Confess endpoint working!
```

#### Step 3: Manual cURL Tests

```bash
# Test chat endpoint
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does RAG work?",
    "role": "Software Developer"
  }'

# Test email endpoint
curl -X POST http://localhost:3000/api/email \
  -H "Content-Type: application/json" \
  -d '{
    "type": "resume",
    "to_email": "test@example.com",
    "to_name": "Test User",
    "message": "Test request"
  }'
```

---

### Production Testing

```bash
# Test chat endpoint
curl -X POST https://noahsaiassistant.vercel.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test query",
    "role": "Software Developer"
  }'

# Test analytics endpoint (GET)
curl https://noahsaiassistant.vercel.app/api/analytics
```

---

## Deployment

### Prerequisites Checklist

- [ ] All required API keys configured in Vercel
- [ ] Supabase migrations applied (`001_initial_schema.sql`, `002_add_confessions_and_sms.sql`, `003_analytics_helpers.sql`)
- [ ] Local tests passing
- [ ] `.env` file NOT committed to git
- [ ] Vercel project created and linked

### Deployment Steps

#### Step 1: Push to Main Branch

```bash
git add -A
git commit -m "feat: Add API integration and setup docs"
git push origin main
```

Vercel automatically deploys on push to `main`.

#### Step 2: Verify Deployment

```bash
# Check deployment status
vercel ls

# View latest deployment
vercel inspect <deployment-url>

# Check logs
vercel logs --follow
```

#### Step 3: Smoke Test Production

```bash
# Test each endpoint
./scripts/test_api_endpoints.sh production

# Or manually:
curl https://noahsaiassistant.vercel.app/api/chat \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"query":"test","role":"Software Developer"}'
```

#### Step 4: Monitor for Errors

```bash
# Watch logs for 5 minutes
vercel logs --follow

# Look for:
# - 5xx errors
# - API key errors
# - Timeout errors
# - Rate limit hits
```

---

## Troubleshooting

### Common Issues & Solutions

#### Issue 1: `httpcore.LocalProtocolError: Illegal header value`

**Symptoms**:
- Error in Vercel logs
- OpenAI API calls fail

**Root Cause**: API key has trailing newline

**Fix**:
```bash
# 1. Check for newlines
python -c "import os; print(repr(os.getenv('OPENAI_API_KEY')))"

# 2. Re-add key in Vercel (paste carefully, no extra spaces/newlines)
vercel env rm OPENAI_API_KEY production
vercel env add OPENAI_API_KEY production
```

---

#### Issue 2: `APIConnectionError: Connection error`

**Symptoms**:
- OpenAI API calls timeout
- Vercel function times out (10s limit)

**Root Cause**: Network issue or OpenAI outage

**Fix**:
```bash
# 1. Check OpenAI status
curl https://status.openai.com

# 2. Check Vercel logs for retries
vercel logs --follow | grep "OpenAI"

# 3. Increase timeout (if needed)
# Edit src/config/supabase_config.py:
self.openai_timeout = 30  # seconds
```

---

#### Issue 3: `FUNCTION_INVOCATION_FAILED` on `/api/email` or `/api/feedback`

**Symptoms**:
- Email/SMS endpoints return 500
- Error: "Resend service not configured"

**Root Cause**: Optional API keys not set

**Fix**:
```bash
# Option 1: Add missing keys
vercel env add RESEND_API_KEY production
vercel env add TWILIO_ACCOUNT_SID production

# Option 2: Accept graceful degradation
# Services handle missing keys and return helpful errors
```

---

#### Issue 4: Empty Embedding Warning

**Symptoms**:
- Logs show: "WARNING: Empty embedding, returning no results"
- Retrieval returns 0 results

**Root Cause**: OpenAI client fails before embedding generation

**Fix**:
```bash
# 1. Verify OpenAI API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# 2. Check for trailing newlines (see Issue 1)

# 3. Test embedding endpoint directly
curl https://api.openai.com/v1/embeddings \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":"test","model":"text-embedding-3-small"}'
```

---

#### Issue 5: Analytics Endpoint Returns 404

**Symptoms**:
- `/api/analytics` returns "Not Found"
- Frontend can't fetch live data

**Root Cause**: SQL helper functions not created

**Fix**:
```sql
-- Go to Supabase Dashboard → SQL Editor → New Query
-- Run: supabase/migrations/003_analytics_helpers.sql

-- Verify functions exist:
SELECT routine_name 
FROM information_schema.routines 
WHERE routine_type='FUNCTION' 
  AND specific_schema='public'
  AND routine_name LIKE '%summary%';
```

---

#### Issue 6: Rate Limit Exceeded (429)

**Symptoms**:
- User gets "Too many requests" error
- Analytics endpoint returns 429

**Root Cause**: Exceeded 6 requests/minute per IP

**Fix**:
```bash
# Option 1: Wait 60 seconds

# Option 2: Increase limit (for legitimate traffic)
# Edit api/analytics.py:
RATE_LIMIT = 12  # Increase from 6 to 12

# Option 3: Implement Redis (production)
# See ANALYTICS_IMPLEMENTATION.md → Future Enhancements
```

---

## Security Best Practices

### 1. API Key Protection

✅ **DO**:
- Store in environment variables
- Use `.env` for local development
- Use Vercel env vars for production
- Add `.env` to `.gitignore`
- Rotate keys quarterly

❌ **DON'T**:
- Hard-code in source files
- Commit to git (even private repos)
- Share in Slack/Discord
- Expose to client-side JavaScript

---

### 2. CORS Configuration

For production frontend:

```typescript
// vercel.json
{
  "headers": [
    {
      "source": "/api/(.*)",
      "headers": [
        { "key": "Access-Control-Allow-Origin", "value": "https://yourdomain.com" },
        { "key": "Access-Control-Allow-Methods", "value": "GET, POST, OPTIONS" },
        { "key": "Access-Control-Allow-Headers", "value": "Content-Type" }
      ]
    }
  ]
}
```

---

### 3. Input Validation

All endpoints validate required fields:

```python
# api/chat.py
def validate_chat_request(data):
    required = ["query", "role"]
    missing = [f for f in required if f not in data]
    
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"
    
    if not isinstance(data["query"], str) or len(data["query"]) > 5000:
        return False, "Invalid query: must be string under 5000 chars"
    
    return True, None
```

---

### 4. Rate Limiting

Currently: In-memory store (simple but not production-scale)

**Future**: Implement Redis-based rate limiting:

```python
# api/rate_limiter.py
import redis

redis_client = redis.from_url(os.getenv("REDIS_URL"))

def check_rate_limit(ip: str, limit: int = 6, window: int = 60) -> bool:
    key = f"rate_limit:{ip}"
    count = redis_client.incr(key)
    
    if count == 1:
        redis_client.expire(key, window)
    
    return count <= limit
```

---

## Next Steps

### Phase 1 ✅ Complete
- [x] API endpoints created
- [x] Environment configuration documented
- [x] Frontend integration examples
- [x] Testing procedures
- [x] Deployment guide

### Phase 2 (Optional Enhancements)
- [ ] Add JWT authentication
- [ ] Implement Redis rate limiting
- [ ] Add request/response logging
- [ ] Set up error tracking (Sentry)
- [ ] Add health check endpoint (`/api/health`)

### Phase 3 (Production Monitoring)
- [ ] LangSmith integration (see [LANGSMITH.md](../LANGSMITH.md))
- [ ] Vercel analytics dashboard
- [ ] Alert on 5xx error rate > 5%
- [ ] Cost tracking dashboard

---

## References

### Documentation
- **This Guide**: `docs/setup/API_SETUP_GUIDE.md`
- **Analytics Implementation**: `docs/features/ANALYTICS_IMPLEMENTATION.md`
- **Observability**: `docs/OBSERVABILITY.md`
- **Archived**: `docs/archive/setup/API_INTEGRATION_OCT_16_2025.md`

### Code Files
- **API Endpoints**: `api/*.py`
- **Configuration**: `src/config/supabase_config.py`
- **Services**: `src/services/*.py`
- **Tests**: `scripts/test_api_endpoints.py`

### External Resources
- **Vercel Docs**: https://vercel.com/docs/functions/serverless-functions
- **OpenAI API**: https://platform.openai.com/docs
- **Supabase Docs**: https://supabase.com/docs
- **Resend Docs**: https://resend.com/docs
- **Twilio Docs**: https://www.twilio.com/docs

---

**Last Updated**: October 16, 2025  
**Status**: ✅ Production-ready, fully documented  
**Maintainer**: @noah
