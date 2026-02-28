# Portfolia - Noah's AI Assistant

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![CI/CD Tests](https://github.com/iNoahCodeGuy/ai_assistant/workflows/Quality%20Assurance%20Tests/badge.svg)](https://github.com/iNoahCodeGuy/ai_assistant/actions)

**Portfolia** is Noah's **educational generative AI assistant** that teaches users how GenAI systems work and their enterprise value by **using herself as a hands-on case study**.

This is a **complete full-stack AI system** demonstrating every component enterprises need: **frontend** (UI/UX), **backend** (APIs, business logic), **data pipelines** (ETL, embeddings), **architecture** (RAG, vector search), **QA** (testing strategies), and **DevOps** (deployment, monitoring). Through interactive conversation, it explains its own implementation and shows how each pattern applies to enterprise use cases like customer support, internal documentation, and sales enablement.

**🎓 The Learning Approach**: Instead of abstract explanations, the assistant shows you the actual code, architecture diagrams, and data flows that power the conversation you're having.

**📘 [Start Here: Complete System Learning Guide](docs/LEARNING_GUIDE_COMPLETE_SYSTEM.md)** - Comprehensive walkthrough of every component with enterprise mapping.

**Quick explorations:**
- "Show me the backend stack" → See real Python code with annotations
- "How does RAG work?" → Walk through the retrieval pipeline with examples
- "Explain the data pipeline" → Document processing and embedding generation
- "Show me the frontend code" → UI patterns and session management
- "How do you test everything?" → QA strategies and mocking patterns
- "Display data analytics" → View live metrics and understand observability
- "What makes this valuable to enterprises?" → Connect technical patterns to business ROI

The assistant adapts its teaching style based on user roles (technical vs. non-technical), making complex AI concepts accessible while showcasing production-ready implementation patterns across the entire stack.

## 🎉 Current Status (February 2026)

**✅ System Fully Operational!**

Recent improvements:
- **Knowledge Base Cleanup**: Removed 661 broken chunks (89.2% of KB), now 100% high-quality content
- **Performance Optimization**: Native pgvector RPC for fast semantic retrieval
- **Interactive Chat**: New terminal interface (`chat_with_portfolia.py`) with debug mode
- **Import Path Resolution**: Fixed all module imports to use `assistant/` directory
- **Template Ready**: 26-section KB template available for expansion

**Quick Test**:
```bash
# Interactive chat with Portfolia
python3 chat_with_portfolia.py

# Test retrieval performance
python3 test_rpc_performance.py
```

**See Also**:
- [KB_CLEANUP_COMPLETED.md](KB_CLEANUP_COMPLETED.md) - Detailed cleanup report
- [SESSION_SUMMARY.md](SESSION_SUMMARY.md) - Previous session accomplishments
- [KB_REGENERATION_GUIDE.md](KB_REGENERATION_GUIDE.md) - Guide for expanding KB content

## Table of Contents
- [🚀 Quickstart](#-quickstart)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Role-Specific Behaviors](#role-specific-behaviors)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Quickstart

**Get Portfolia running locally in 5 minutes**

### Prerequisites
- ✅ Python 3.11+ installed
- ✅ OpenAI API key ([get one here](https://platform.openai.com/api-keys)) - $5 free credit for new accounts
- ✅ Supabase account ([sign up free](https://supabase.com/dashboard)) - Free tier is sufficient

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/noahcal/noahs-ai-assistant.git
cd noahs-ai-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
# Create a .env file with your API keys:
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGc...
```

**Get Supabase credentials**:
1. Create a new project in [Supabase Dashboard](https://supabase.com/dashboard)
2. Go to Settings → API
3. Copy `URL` and `service_role` key (⚠️ keep this secret!)

```bash
# 4. Set up the database schema
# Follow the step-by-step guide in docs/PHASE_1_SETUP.md
# This creates tables for knowledge base, messages, and analytics

# 5. Run the data migration (one-time)
python scripts/migrate_data_to_supabase.py

# 6. Start the Streamlit app
streamlit run src/main.py
```

**🎉 Success!** Open http://localhost:8501 in your browser.

**First steps**:
1. Select a role (try "Software Developer" to explore technical architecture)
2. Ask questions like:
   - "How does RAG work in this application?"
   - "Show me the backend architecture"
   - "Explain the data pipeline"
   - "What makes this valuable to enterprises?"
3. See the assistant explain its own implementation with code examples!

**Need help?** See [PHASE_1_SETUP.md](docs/PHASE_1_SETUP.md) for detailed setup instructions.

**Next**: Learn about the [Complete System](docs/LEARNING_GUIDE_COMPLETE_SYSTEM.md) covering frontend, backend, data pipelines, architecture, QA, and DevOps.

## 📚 Complete System Components

This assistant demonstrates **every layer of a production AI application**:

### 🎨 Frontend Development
- **UI/UX Patterns**: Chat interface, role selection, session management
- **Professional Rendering**: Markdown tables, data visualization, action buttons
- **State Management**: UUID-based sessions, conversation history, error boundaries
- **Enterprise Application**: Maps to customer portals, internal tools, CRM widgets
- **Ask:** "Show me your frontend code" | "How does the chat interface work?"

### ⚙️ Backend Architecture
- **API Design**: Serverless routes (`/api/chat`, `/api/analytics`, `/api/email`)
- **Business Logic**: LangGraph orchestration, service layer, state management
- **Error Handling**: Graceful degradation, retry logic, timeout management
- **Enterprise Application**: Same patterns for support bots, doc assistants, sales tools
- **Ask:** "Show me the backend stack" | "Explain the conversation pipeline"

### 📊 Data Pipeline Management
- **ETL Process**: Document ingestion, parsing, chunking (500-token strategy)
- **Embedding Generation**: OpenAI text-embedding-3-small (1536 dimensions, $0.00002/1K tokens)
- **Storage**: Supabase Postgres + pgvector, idempotent migrations
- **Enterprise Application**: Scales to Confluence, Zendesk, SharePoint ingestion
- **Ask:** "Show me the data pipeline" | "How do you process documents?"

### 🏗️ System Architecture
- **RAG Pipeline**: Query embedding → Vector search → Context assembly → LLM generation
- **Vector Search**: pgvector with IVFFLAT index, cosine similarity, top-k retrieval
- **LLM Orchestration**: GPT-4o-mini with temperature control, token optimization
- **Enterprise Application**: Proven patterns for accuracy, cost, and maintainability
- **Ask:** "How does RAG work?" | "Show me vector search code"

### 🧪 QA & Testing
- **Test Strategy**: Pytest framework, unit + integration tests, mocking patterns
- **Edge Cases**: Empty queries, malformed input, XSS attempts, concurrent sessions
- **Quality Gates**: Linting, type checking, coverage thresholds
- **Enterprise Application**: Same patterns + domain-specific tests (PII, compliance)
- **Ask:** "Show me your testing strategy" | "How do you mock Supabase?"

### 🚀 DevOps & Deployment
- **Infrastructure**: Vercel serverless (auto-scaling), Supabase (managed DB), LangSmith (monitoring)
- **CI/CD**: Git push → automated tests → zero-downtime deployment
- **Cost Tracking**: $25/month for production system, scales to $3200/month for 100k users
- **Enterprise Application**: Kubernetes, Redis caching, enterprise SLA, security hardening
- **Ask:** "What's your deployment process?" | "Show me cost at 100k users"

**→ [Complete System Learning Guide](docs/LEARNING_GUIDE_COMPLETE_SYSTEM.md)** - Deep dive into each component with enterprise use case mapping.

---

## What You'll Learn

Through interactive conversation, this assistant teaches you:

### For Technical Audiences (Full-Stack Developers, DevOps Engineers)
- **Frontend Patterns**: UI components, session management, error boundaries, professional data rendering
- **Backend Design**: API routes, service layer, state management, LangGraph orchestration
- **Data Pipelines**: ETL processes, chunking strategies, embedding generation, incremental updates
- **RAG Architecture**: How Retrieval-Augmented Generation works (vector embeddings, semantic search, context injection)
- **Vector Search**: Supabase pgvector implementation with IVFFLAT indexing and similarity scoring
- **LLM Orchestration**: Node-based conversation flows with temperature control and token optimization
- **QA Strategies**: Unit testing, integration testing, mocking external services, edge case validation
- **DevOps Practices**: Serverless deployment, environment management, CI/CD, cost optimization
- **System Design**: Hybrid deployment (Streamlit + Vercel), monitoring, observability, scaling patterns
- **Production Patterns**: Analytics logging, PII redaction, rate limiting, security (RLS policies)
- **Cost Optimization**: Model selection, caching strategies, and infrastructure choices
- **Testing & Reliability**: Pytest patterns, Supabase mocking, error degradation modes

### For Non-Technical Audiences
- **GenAI Business Value**: Why enterprises invest in conversational AI like Portfolia (cost savings, scalability, 24/7 availability)
- **Accuracy & Trust**: How RAG prevents hallucinations by grounding responses in real data
- **Data Governance**: PII protection, privacy controls, compliance considerations
- **ROI & Metrics**: Success measurement (response time, accuracy, user satisfaction, conversion rates)
- **Competitive Advantage**: Using AI to improve customer experience and operational efficiency
- **Implementation Roadmap**: High-level steps to build similar systems for your organization

### Live Demonstrations
The assistant demonstrates concepts by showing its own implementation:
- **Ask "show me the backend stack"** → See real Python code with explanations
- **Ask "display data analytics"** → Live dashboard with real-time metrics
- **Ask "how do you prevent hallucinations?"** → RAG pipeline walkthrough with examples
- **Ask "what's your data pipeline?"** → Document processing flow with architecture diagrams

## Features

### 🎓 Educational Features (Learn GenAI Through Demonstration)
- **Self-Referential Teaching**: Explains RAG, vector search, and LLM orchestration by showing its own implementation
- **Live Code Walkthroughs**: Displays actual Python files with explanations of GenAI patterns and enterprise applications
- **Architecture Tours**: Interactive exploration of stack, data pipelines, backend design, and deployment strategies
- **Enterprise Value Mapping**: Connects every technical concept to business ROI (cost reduction, scalability, reliability)
- **Role-Adaptive Teaching**: Adjusts depth for technical (with code) vs. non-technical (with analogies) audiences
- **Real Metrics Dashboard**: Shows live analytics to explain observability, logging, and data governance in production

### 🏗️ Technical Features (Production-Ready Patterns)
- **Retrieval-Augmented Generation (RAG)**: Supabase pgvector with IVFFLAT indexing for semantic search
- **LangGraph Orchestration**: Node-based conversation flows with immutable state management
- **Dual-Audience Formatting**: Technical details + plain-English summaries for mixed audiences
- **Code & Career Grounding**: File:line citations proving all responses are grounded in real data
- **PII Redaction**: Automated masking of emails/phones in user feedback (demonstrates data governance)
- **Rate Limiting**: 6 req/min per IP with graceful degradation (demonstrates production reliability)
- **Live Analytics Dashboard**: Real-time data visualization with PII redaction and role-based access
- **MMA Query Routing**: Direct fight link for MMA-related queries (demonstrates multi-modal routing)
- **Confession Mode**: Privacy-protected personal messaging (demonstrates data governance)
- **Analytics Tracking**: Interaction logging, retrieval tracking, user feedback with Supabase
- **Contact Requests**: Email delivery via Resend, SMS notifications via Twilio
- **Extensible Orchestration**: LangGraph-style node-based conversation flow
- **Observability**: LangSmith integration for tracing and performance monitoring

## Tech Stack

### Current Architecture (Supabase + Vercel)
- **Frontend/UI**: Streamlit (chat UI, role selector, session management)
- **Core Framework**: LangChain (loaders, embeddings, retrieval pipeline)
- **Database**: Supabase Postgres with pgvector extension
- **Vector Search**: pgvector with IVFFLAT indexing for semantic similarity
- **Models**: OpenAI GPT-3.5/4 (generation), OpenAI ada-002 (embeddings)
- **Analytics**: Direct Supabase writes (messages, retrieval_logs, feedback)
- **External Services**:
  - Resend (email delivery)
  - Twilio (SMS notifications)
- **API Layer**: Next.js API routes for external integrations
- **Security**: Row Level Security (RLS) policies, environment variables
- **Deployment**: Hybrid (Streamlit + Vercel serverless)
- **Observability**: LangSmith integration for tracing
- **Testing**: Pytest with Supabase mocking

### Cost Estimates
- **Supabase Pro**: ~$25/month (includes PostgreSQL + pgvector + Storage)
- **OpenAI API**: Variable based on usage (~$10-30/month for moderate traffic)
- **Vercel**: Free tier (serverless functions for API routes)
- **Resend**: Free tier up to 3,000 emails/month
- **Twilio**: Pay-as-you-go (~$0.0075/SMS)
- **Total**: ~$35-60/month for production deployment

**Previous GCP Architecture** (archived): Cost ~$100-200/month
- Used Cloud SQL, Pub/Sub, Secret Manager, Vertex AI, Cloud Run
- Migrated to Supabase for 50-75% cost reduction and simplified maintenance

## Role-Specific Behaviors

### 1. Hiring Manager (nontechnical)
- Goal: High-level résumé narrative
- Sources: Career KB CSV, résumé text
- Format: Career Overview → Notable Outcomes → Source Citations (lightweight)
- UI: Contact CTA (email / LinkedIn)

### 2. Hiring Manager (technical)
- Goal: Blend career signal + engineering depth
- Sources: Career KB + Code Index
- Format:
  - Section 1: Engineer Detail (with file:line citations)
  - Section 2: Plain-English Summary
- UI: Expandable code reference (future enhancement)

### 3. Software Developer
- Goal: Deep technical explanation, architecture, design tradeoffs
- Sources: Code Index + Career KB fallback
- Format: Engineer Detail → Summary (same dual format)

### 4. Just Looking Around
- Goal: Fun facts + MMA info
- MMA queries: Direct fight link
- Other queries: Light career / personal facts

### 5. Looking to Confess Crush
- Goal: Safe, minimal interaction channel
- Storage: Only explicit submission, no hidden PII retention
- Response: Acknowledgment only (no model overreach)

## Installation

### Prerequisites
- Python 3.10+
- Supabase account (free tier available)
- OpenAI API key
- (Optional) Resend API key for email
- (Optional) Twilio credentials for SMS

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/iNoahCodeGuy/NoahsAIAssistant-.git
cd NoahsAIAssistant-
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Supabase**
   - Create a new project at [supabase.com](https://supabase.com)
   - Run the SQL migration in `supabase/migrations/001_initial_schema.sql` in your Supabase SQL Editor
   - Copy your project URL and service role key

4. **Set environment variables**
```bash
cp .env.example .env
# Edit .env and add:
# SUPABASE_URL=your_project_url
# SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
# OPENAI_API_KEY=your_openai_key
# (Optional) RESEND_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM
```

5. **Populate knowledge base** (optional)
```bash
python scripts/migrate_data_to_supabase.py
```

## Developer Setup

### Quality Assurance & Pre-Commit Hooks

This project uses **automated quality checks** to maintain code quality and prevent regressions. All tests and hooks are configured to run automatically.

#### Install Pre-Commit Hooks

**One-time setup** (required for contributors):

```bash
# 1. Install pre-commit framework
pip install pre-commit

# 2. Install git hooks
pre-commit install

# 3. (Optional) Run hooks manually on all files
pre-commit run --all-files
```

#### What the Hooks Do

Pre-commit hooks run automatically **before every commit** and check:

✅ **Conversation Quality Tests** (18 tests)
- No markdown headers in LLM responses
- No duplicate follow-up prompts
- No information overload (response length limits)
- Professional formatting consistency

✅ **Documentation Alignment Tests** (12 tests)
- Function names in docs match actual code
- File paths are valid (no broken references)
- Configuration values are correct

✅ **Documentation Drift Prevention**
- New `.md` files must be registered in master docs
- Feature docs follow naming conventions
- Suggests CHANGELOG updates

✅ **Code Hygiene**
- Trailing whitespace removal (auto-fix)
- End-of-file fixes (auto-fix)
- YAML syntax validation

**Execution time:** ~2-3 seconds total

#### What Happens If a Check Fails?

```bash
# If a test fails, the commit is blocked:
❌ Conversation Quality Tests (18 tests).....Failed
- hook id: conversation-quality-tests
- exit code: 1

5 tests failed. Fix the issues and try again.

# Fix the issues, then:
git add -A
git commit -m "your message"  # Hooks run again
```

#### Running Tests Manually

```bash
# Run all tests (30 total)
pytest tests/ -v

# Run only conversation quality tests
pytest tests/test_conversation_quality.py -v

# Run only documentation alignment tests
pytest tests/test_documentation_alignment.py -v
```

#### Bypassing Hooks (Emergency Only)

**⚠️ Warning:** Only use this for emergency hotfixes. CI/CD will still run tests on push.

```bash
# Skip pre-commit hooks (NOT RECOMMENDED)
git commit --no-verify -m "emergency hotfix"

# Better approach: Fix the failing test, then commit normally
```

#### CI/CD Integration

Even if you bypass pre-commit hooks locally, **all tests run automatically** on every push via GitHub Actions. Failed tests will block PR merges.

See [QA_STRATEGY.md](docs/QA_STRATEGY.md) for complete quality assurance documentation.

---

## Usage

### Development
Run the Streamlit application locally:
```bash
streamlit run src/main.py
```

### Production Deployment

**Option 1: Streamlit Community Cloud**
- Deploy directly from GitHub
- Set environment variables in Streamlit Cloud dashboard
- Best for chat interface only

**Option 2: Hybrid (Streamlit + Vercel)**
- Deploy Streamlit for chat UI
- Deploy Next.js API routes to Vercel for email/SMS integrations
- Configure `vercel.json` for serverless functions

### Database Schema
The Supabase database includes:
- `kb_chunks`: Knowledge base with pgvector embeddings
- `messages`: Chat interaction logs
- `retrieval_logs`: RAG pipeline tracking
- `links`: External resource URLs
- `feedback`: User ratings and contact requests

### Architecture Overview
```
User → Streamlit UI → RagEngine (pgvector search) → OpenAI GPT → Response
                ↓
        Supabase Analytics (messages, retrieval_logs)
                ↓
        Feedback → Email (Resend) / SMS (Twilio)
```

## File Structure
```
noahs-ai-assistant
├── src
│   ├── main.py
│   ├── config/
│   ├── core/
│   ├── retrieval/
│   ├── agents/
│   ├── ui/
│   ├── analytics/
│   └── utils/
├── data/
├── vector_stores/
├── tests/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Contributing
1. Create a branch: feature/<name>
2. Keep commits focused
3. Open PR → request review

## License
MIT (see LICENSE if present). Add LICENSE file if not yet created.
