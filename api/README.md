# API Routes

Vercel serverless functions for Noah's AI Assistant.

## Endpoints

### POST /api/chat
Execute LangGraph conversation flow and return response.

**Request:**
```json
{
  "query": "What is your experience with Python?",
  "role": "Hiring Manager (technical)",
  "session_id": "user-123",
  "chat_history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ],
  "user_email": "recruiter@example.com",
  "user_name": "Jane Doe",
  "user_phone": "+15551234567"
}
```

**Response:**
```json
{
  "success": true,
  "answer": "I have extensive Python experience...",
  "role": "Hiring Manager (technical)",
  "session_id": "user-123",
  "analytics": {
    "latency_ms": 1250,
    "tokens_used": 450
  },
  "actions_taken": ["send_resume", "notify_resume_sent"],
  "retrieved_chunks": 4
}
```

### POST /api/email
Send resume or LinkedIn link via email.

**Request:**
```json
{
  "type": "resume",
  "to_email": "recruiter@example.com",
  "to_name": "Jane Doe",
  "message": "Here's my resume as requested"
}
```

**Response:**
```json
{
  "success": true,
  "type": "resume",
  "status": "sent",
  "resume_url": "https://storage.example.com/resume.pdf?token=..."
}
```

### POST /api/feedback
Submit user feedback and contact requests.

**Request:**
```json
{
  "message_id": "msg_123",
  "rating": 5,
  "comment": "Very helpful assistant!",
  "contact_requested": true,
  "user_email": "user@example.com",
  "user_name": "John Smith",
  "user_phone": "+15559876543"
}
```

**Response:**
```json
{
  "success": true,
  "feedback_id": "fb_456",
  "message": "Feedback received. Noah will reach out soon!"
}
```

### POST /api/confess
Submit anonymous or named confession.

**Request:**
```json
{
  "message": "I think you're amazing and wanted to tell you!",
  "is_anonymous": false,
  "name": "Sarah",
  "email": "sarah@example.com",
  "phone": "+15551112222"
}
```

**Response:**
```json
{
  "success": true,
  "confession_id": "conf_789",
  "message": "Thank you Sarah! Noah will reach out soon. 💌"
}
```

## Local Testing

```bash
# Install Vercel CLI
npm i -g vercel

# Run locally
vercel dev

# Test endpoints
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello", "role": "Just looking around"}'
```

## Deployment

```bash
# Deploy to Vercel
vercel --prod

# Set environment variables
vercel env add OPENAI_API_KEY
vercel env add SUPABASE_URL
vercel env add SUPABASE_SERVICE_KEY
# ... add all required env vars
```

## Error Handling

All endpoints return consistent error format:

```json
{
  "success": false,
  "error": "Error message description"
}
```

Common status codes:
- `400` - Bad request (missing/invalid fields)
- `500` - Internal server error
- `200` - Success

## Rate Limiting

Vercel free tier limits:
- 100GB bandwidth/month
- 100 hours serverless execution/month
- 1000 invocations/day per function

For production, upgrade to Pro plan or implement custom rate limiting.

## CORS

All endpoints support CORS for cross-origin requests from approved domains.
Adjust CORS headers in each handler as needed for production security.
