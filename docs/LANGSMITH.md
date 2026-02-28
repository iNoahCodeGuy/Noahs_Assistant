# 🔗 LangSmith Setup Guide

Complete guide to setting up LangSmith for observability in Noah's AI Assistant.

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Get LangSmith API Key](#get-langsmith-api-key)
3. [Configure Environment](#configure-environment)
4. [Verify Setup](#verify-setup)
5. [View Traces](#view-traces)
6. [Advanced Configuration](#advanced-configuration)

## Prerequisites

- OpenAI API key (already configured)
- Python 3.8+
- Active internet connection

## Get LangSmith API Key

### Step 1: Create LangSmith Account

1. Go to https://smith.langchain.com/
2. Click "Sign Up" (free tier available)
3. Sign up with Google/GitHub or email

### Step 2: Get API Key

1. After login, go to Settings → API Keys
2. Click "Create API Key"
3. Name it: `noahs-ai-assistant-prod`
4. Copy the key (starts with `lsv2_pt_...`)

**Important**: Save this key securely! You won't be able to see it again.

### Step 3: Create Project (Optional)

1. Go to Projects
2. Click "New Project"
3. Name: `noahs-ai-assistant`
4. Description: "RAG system for Noah's portfolio assistant"

## Configure Environment

### Option 1: Update .env File

```bash
# Open .env file
nano .env

# Add these lines
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_YOUR_ACTUAL_KEY_HERE
LANGCHAIN_PROJECT=noahs-ai-assistant
```

### Option 2: Use Environment Variables

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=lsv2_pt_YOUR_ACTUAL_KEY_HERE
export LANGCHAIN_PROJECT=noahs-ai-assistant
```

### Full .env Example

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-...

# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key

# LangSmith Configuration (OBSERVABILITY)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=noahs-ai-assistant
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

# Database
DATABASE_URL=postgresql://...

# Streamlit
STREAMLIT_PORT=8501
```

## Verify Setup

### Method 1: Python Script

Create `test_langsmith.py`:
```python
#!/usr/bin/env python3
from observability import initialize_langsmith

if initialize_langsmith():
    print("✅ LangSmith is configured correctly!")
    print("View traces at: https://smith.langchain.com/")
else:
    print("❌ LangSmith setup failed. Check your configuration.")
```

Run:
```bash
python test_langsmith.py
```

### Method 2: Check Logs

```bash
# Run the app
streamlit run src/main.py

# Look for this log message
# ✅ LangSmith tracing enabled
```

### Method 3: Test Query

```python
from core.rag_engine import RagEngine

engine = RagEngine()
response = engine.generate_response("What are Noah's skills?")
print(response)

# Then check LangSmith dashboard
```

## View Traces

### Access Dashboard

1. Go to https://smith.langchain.com/
2. Select project: `noahs-ai-assistant`
3. View traces in real-time

### Trace Structure

```
📊 RAG Pipeline (1.2s)
├─ 🔍 Retrieval (150ms)
│  ├─ Query: "What are Noah's skills?"
│  ├─ Chunks: 3
│  └─ Avg Similarity: 0.82
├─ 🤖 Generation (980ms)
│  ├─ Model: gpt-4
│  ├─ Tokens: 150 prompt + 200 completion
│  ├─ Cost: $0.012
│  └─ Response: "Noah is proficient in..."
└─ ✅ Evaluation (50ms)
   ├─ Faithfulness: 0.92
   ├─ Relevance: 0.88
   └─ Quality: 0.90
```

When `LANGGRAPH_FLOW_ENABLED=true` (default) you’ll also see sub-runs for `classify_query`, `retrieve_chunks`, `generate_answer`, `plan_actions`, `apply_role_context`, and `execute_actions`, mirroring the conversation node order.

### Filter Traces

- **By Date**: Last hour, today, this week
- **By Status**: Success, Error, Pending
- **By Latency**: > 1s, > 2s, > 5s
- **By Model**: gpt-4, gpt-3.5-turbo

### Export Data

```python
from observability import get_langsmith_client

client = get_langsmith_client()
runs = client.list_runs(project_name="noahs-ai-assistant")

for run in runs:
    print(f"Run: {run.id}, Latency: {run.total_time}ms")
```

## Advanced Configuration

### Custom Project Name

```bash
# Use environment-specific projects
LANGCHAIN_PROJECT=noahs-ai-assistant-dev    # Development
LANGCHAIN_PROJECT=noahs-ai-assistant-prod   # Production
```

### Sampling (Reduce Costs)

```python
# In src/core/rag_engine.py
import os

# Only trace 10% of requests in production
TRACE_SAMPLING_RATE = 0.1 if os.getenv("ENVIRONMENT") == "production" else 1.0

if random.random() < TRACE_SAMPLING_RATE:
    # Trace this request
    pass
```

### Custom Metadata

```python
from observability import log_trace_metadata

# Add custom metadata to trace
log_trace_metadata(
    run_id="abc-123",
    metadata={
        "user_role": "Hiring Manager",
        "query_type": "technical",
        "custom_field": "value"
    }
)
```

### Disable for Tests

```python
# tests/conftest.py
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
```

### Multiple Environments

```bash
# .env.development
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=noahs-ai-assistant-dev

# .env.production
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=noahs-ai-assistant-prod

# .env.test
LANGCHAIN_TRACING_V2=false
```

Load with:
```python
from dotenv import load_dotenv
import os

env = os.getenv("ENVIRONMENT", "development")
load_dotenv(f".env.{env}")
```

## Troubleshooting

### Issue: "LangSmith not available"

**Cause**: `langsmith` package not installed

**Solution**:
```bash
pip install langsmith
# or
pip install -r requirements.txt
```

### Issue: "LANGCHAIN_API_KEY not set"

**Cause**: API key not in environment

**Solution**:
```bash
# Check if .env exists
cat .env | grep LANGCHAIN_API_KEY

# If missing, add it
echo "LANGCHAIN_API_KEY=lsv2_pt_..." >> .env

# Restart app
```

### Issue: Traces Not Appearing

**Diagnosis**:
```python
import os
print("Tracing enabled:", os.getenv("LANGCHAIN_TRACING_V2"))
print("API key set:", bool(os.getenv("LANGCHAIN_API_KEY")))
print("Project:", os.getenv("LANGCHAIN_PROJECT"))
```

**Solutions**:
1. Verify API key is correct
2. Check internet connectivity
3. Try manual trace:
   ```python
   from langsmith import Client
   client = Client()
   print("Connected to LangSmith!")
   ```

### Issue: High Latency

**Cause**: LangSmith adds ~10-50ms overhead

**Solutions**:
1. Use async tracing (coming soon)
2. Disable in latency-critical code
3. Sample traces (10-50% instead of 100%)

### Issue: Rate Limits

**Cause**: Free tier limit exceeded (5k traces/month)

**Solutions**:
1. Upgrade to paid plan ($39/month)
2. Reduce sampling rate
3. Use multiple projects to distribute load

## Pricing

| Tier | Traces/Month | Cost/Month | Notes |
|------|--------------|------------|-------|
| **Free** | 5,000 | $0 | Good for development |
| **Team** | 100,000 | $39 | Recommended for production |
| **Enterprise** | Unlimited | Custom | For large deployments |

### Cost Estimation

```python
# Estimate monthly cost
queries_per_day = 100
traces_per_month = queries_per_day * 30

if traces_per_month <= 5000:
    print("Free tier sufficient")
elif traces_per_month <= 100000:
    print("Team tier required: $39/month")
else:
    print("Enterprise tier required: Contact sales")
```

## Security Best Practices

### 1. Protect API Keys

```bash
# Never commit .env to git
echo ".env" >> .gitignore

# Use environment variables in production
export LANGCHAIN_API_KEY=lsv2_pt_...
```

### 2. Rotate Keys Regularly

1. Create new API key in LangSmith
2. Update `.env` file
3. Delete old key from LangSmith

### 3. Use Different Keys per Environment

```bash
# Development
LANGCHAIN_API_KEY=lsv2_pt_dev_...

# Production
LANGCHAIN_API_KEY=lsv2_pt_prod_...
```

### 4. Limit Key Permissions

In LangSmith settings:
- ✅ Read traces
- ✅ Write traces
- ❌ Delete projects (admin only)

## Integration with CI/CD

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        env:
          LANGCHAIN_TRACING_V2: false  # Disable in CI
        run: pytest
```

### Vercel Deployment

```bash
# vercel.json or dashboard
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=noahs-ai-assistant-prod
```

## Next Steps

1. ✅ Complete this setup
2. 📖 Read [Observability Guide](OBSERVABILITY_GUIDE.md)
3. 🧪 Run test queries and view traces
4. 📊 Set up dashboards in LangSmith
5. 🔔 Configure alerts for errors

## Resources

- **LangSmith Docs**: https://docs.smith.langchain.com/
- **Python SDK**: https://github.com/langchain-ai/langsmith-sdk
- **Pricing**: https://www.langchain.com/pricing
- **Support**: support@langchain.com

---

**Last Updated**: December 2024  
**Status**: ✅ Production Ready
