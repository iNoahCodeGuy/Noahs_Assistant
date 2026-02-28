# Enterprise Adaptation Guide

**How to adapt Noah's AI Assistant architecture for your enterprise use case**

This guide explains how to take the patterns demonstrated in this assistant and apply them to common enterprise scenarios like customer support, internal documentation, sales enablement, and more.

---

## 🎯 Why This Architecture Works for Enterprises

Noah's AI Assistant demonstrates production-ready patterns that scale:

| Pattern | What This Assistant Does | Enterprise Application |
|---------|-------------------------|------------------------|
| **RAG Pipeline** | Retrieves Noah's career facts from pgvector | Retrieve product docs, policies, or internal wikis |
| **Role-Based Responses** | Adapts to hiring managers vs developers | Adapt to customers vs employees vs partners |
| **Analytics Tracking** | Logs queries, retrieval scores, latency | Track customer satisfaction, bottlenecks, ROI |
| **PII Redaction** | Masks emails/phones in feedback | Comply with GDPR, HIPAA, data privacy laws |
| **Rate Limiting** | 6 req/min per IP | Prevent abuse, control costs, ensure fair usage |
| **Error Handling** | Graceful degradation if vector DB fails | Maintain uptime during outages or migrations |
| **Observability** | LangSmith traces for debugging | Monitor production performance, optimize prompts |

---

## 📋 Common Enterprise Use Cases

### 1. Customer Support Chatbot

**What to change:**
- **Knowledge Base**: Replace `data/career_kb.csv` with your product documentation
- **Data Sources**: Add FAQs, troubleshooting guides, policy documents
- **Roles**: Replace "Hiring Manager" with "Customer", "Premium Customer", "Partner"
- **Actions**: Replace "send resume" with "create support ticket", "escalate to agent"

**Code changes:**
```python
# In src/agents/roles.py
ROLES = {
    "Customer": {
        "sources": ["product_docs", "faqs"],
        "actions": ["create_ticket", "show_documentation"]
    },
    "Premium Customer": {
        "sources": ["product_docs", "faqs", "premium_features"],
        "actions": ["priority_escalation", "account_manager_contact"]
    }
}
```

**New data pipeline:**
```bash
# 1. Export your documentation to CSV
# Format: section,question,answer,source
# Example: "Billing,How do I cancel?","Go to Settings > Billing > Cancel","help.example.com/billing"

# 2. Run migration to create embeddings
python scripts/migrate_data_to_supabase.py --source custom_docs.csv

# 3. Test retrieval
python -c "from src.core.rag_engine import RagEngine; print(RagEngine().retrieve('cancel subscription'))"
```

**Expected ROI:**
- 40-60% reduction in support ticket volume
- 24/7 availability (vs 9-5 human agents)
- <2s response time (vs 5-10 min human response)
- $30-50K/year cost savings (vs hiring 2 FTE support agents)

---

### 2. Internal Knowledge Base Assistant

**What to change:**
- **Knowledge Base**: Company wikis, Confluence/Notion exports, onboarding docs
- **Data Sources**: HR policies, engineering docs, sales playbooks
- **Roles**: "New Hire", "Engineer", "Sales Rep", "Manager"
- **Actions**: "show onboarding checklist", "link to policy", "suggest expert to contact"

**Code changes:**
```python
# In src/flows/action_planning.py
def plan_actions(state):
    actions = []
    
    if "onboarding" in state.query.lower():
        actions.append({"type": "show_onboarding_checklist"})
    
    if "policy" in state.query.lower():
        actions.append({"type": "link_to_policy_doc"})
    
    if state.role == "New Hire":
        actions.append({"type": "suggest_onboarding_buddy"})
    
    return state.set_pending_actions(actions)
```

**Data governance:**
- **Access control**: Use Supabase RLS policies to restrict docs by department
- **Audit logging**: Track who accessed what (compliance requirement)
- **Sensitive data**: Tag PII/confidential docs, exclude from embeddings or redact

**Expected ROI:**
- 70% faster onboarding (employees find info instantly)
- Reduce Slack interruptions (fewer "where is X?" questions)
- Knowledge retention during turnover (institutional knowledge preserved)

---

### 3. Sales Enablement Assistant

**What to change:**
- **Knowledge Base**: Product specs, pricing sheets, competitor comparisons, case studies
- **Data Sources**: Sales playbooks, objection handling scripts, ROI calculators
- **Roles**: "Sales Rep", "Sales Engineer", "Account Executive"
- **Actions**: "generate proposal", "calculate ROI", "find case study", "schedule demo"

**Code changes:**
```python
# In src/flows/content_blocks.py
def pricing_calculator_block(deal_size: int, use_case: str) -> str:
    """Generate pricing estimate based on deal parameters."""
    base_price = 10000
    multiplier = 1.5 if use_case == "enterprise" else 1.0
    total = base_price * multiplier * (deal_size / 100)
    
    return f"""
### Pricing Estimate
- Base: ${base_price:,}
- Enterprise multiplier: {multiplier}x
- Deal size adjustment: {deal_size} users
- **Total estimate: ${total:,.0f}/year**

*This is an estimate. Contact sales@example.com for official quote.*
"""

# In src/flows/action_planning.py
if "pricing" in state.query.lower():
    actions.append({
        "type": "show_pricing_calculator",
        "deal_size": extract_deal_size(state.query),
        "use_case": state.fetch("use_case", "standard")
    })
```

**Expected ROI:**
- 30% faster deal cycles (reps get answers instantly)
- Higher win rates (consistent, accurate product info)
- Reduced dependency on sales engineers for basic questions

---

### 4. HR/Benefits Assistant

**What to change:**
- **Knowledge Base**: Benefits guides, 401k info, PTO policies, leave requests
- **Data Sources**: Health insurance plans, dental/vision coverage, FSA/HSA rules
- **Roles**: "Employee", "Manager", "HR Admin"
- **Actions**: "calculate PTO balance", "compare health plans", "submit leave request"

**Code changes:**
```python
# In api/benefits.py (new endpoint)
@app.route("/api/benefits/pto", methods=["POST"])
def calculate_pto():
    employee_id = request.json.get("employee_id")
    start_date = request.json.get("start_date")
    
    # Query HRIS system for PTO balance
    balance = hris_client.get_pto_balance(employee_id)
    
    return {
        "balance_hours": balance,
        "balance_days": balance / 8,
        "accrual_rate": "10 days/year",
        "next_accrual": "2025-02-01"
    }

# In src/flows/action_execution.py
if action["type"] == "calculate_pto":
    response = requests.post(
        f"{api_base_url}/api/benefits/pto",
        json={"employee_id": state.user_id}
    )
    pto_data = response.json()
    
    components.append(f"""
### Your PTO Balance
- **Available**: {pto_data['balance_days']} days ({pto_data['balance_hours']} hours)
- **Accrual rate**: {pto_data['accrual_rate']}
- **Next accrual**: {pto_data['next_accrual']}
""")
```

**Privacy considerations:**
- **Authentication**: Require SSO login (Okta, Azure AD)
- **Data isolation**: Each employee sees only their own data
- **Audit trail**: Log all benefits queries for compliance

**Expected ROI:**
- 80% reduction in HR inbox volume
- Faster benefits enrollment (employees understand options)
- Improved satisfaction (instant answers vs waiting for HR)

---

## 🔧 Technical Adaptation Steps

### Step 1: Data Preparation

**Export your knowledge base to CSV:**

```csv
section,question,answer,metadata
Product,How do I reset password?,"Go to Settings > Security > Reset Password. Check your email for the reset link.","url=help.example.com/password,priority=high"
Product,What's your uptime SLA?,"99.9% uptime SLA for Enterprise plans, 99.5% for Standard.","url=sla.example.com,audience=enterprise"
```

**Run migration:**
```bash
python scripts/migrate_data_to_supabase.py --source my_docs.csv --collection my_kb
```

### Step 2: Update Roles

**In `src/agents/roles.py`:**

```python
ROLES = {
    "Customer": {
        "kb_sources": ["product_docs", "faqs"],
        "code_index": False,  # Don't show code to customers
        "actions": ["create_ticket", "show_docs"],
        "tone": "helpful and friendly"
    },
    "Employee": {
        "kb_sources": ["internal_wiki", "hr_policies"],
        "code_index": False,
        "actions": ["calculate_pto", "find_policy"],
        "tone": "professional and informative"
    },
    "Developer": {
        "kb_sources": ["api_docs", "architecture"],
        "code_index": True,  # Show code examples
        "actions": ["show_code", "link_to_repo"],
        "tone": "technical and detailed"
    }
}
```

### Step 3: Customize Actions

**In `src/flows/action_planning.py`:**

```python
def plan_actions(state: ConversationState) -> ConversationState:
    actions = []
    query = state.query.lower()
    role = state.role
    
    # Customer-specific actions
    if role == "Customer":
        if "bug" in query or "issue" in query:
            actions.append({"type": "offer_support_ticket"})
        if "pricing" in query:
            actions.append({"type": "show_pricing_table"})
    
    # Employee-specific actions
    if role == "Employee":
        if "pto" in query or "vacation" in query:
            actions.append({"type": "calculate_pto"})
        if "benefits" in query:
            actions.append({"type": "show_benefits_comparison"})
    
    return state.set_pending_actions(actions)
```

### Step 4: Implement Action Handlers

**In `src/flows/action_execution.py`:**

```python
class ActionExecutor:
    def execute(self, state: ConversationState) -> ConversationState:
        for action in state.pending_actions:
            if action["type"] == "offer_support_ticket":
                self._offer_support_ticket(state, action)
            elif action["type"] == "calculate_pto":
                self._calculate_pto(state, action)
            # ... more handlers
        return state
    
    def _offer_support_ticket(self, state, action):
        """Offer to create support ticket if query unresolved."""
        components.append("""
💁 **Need more help?**  
I can create a support ticket for you. Just say "create ticket" and I'll escalate this to our team.
""")
    
    def _calculate_pto(self, state, action):
        """Fetch and display PTO balance from HRIS."""
        try:
            response = requests.post(
                f"{api_base_url}/api/benefits/pto",
                json={"employee_id": state.user_id},
                timeout=3
            )
            pto_data = response.json()
            
            components.append(f"""
### Your PTO Balance
- **Available**: {pto_data['balance_days']} days
- **Used this year**: {pto_data['used_days']} days
- **Next accrual**: {pto_data['next_accrual']}
""")
        except Exception as e:
            logger.error(f"PTO calculation failed: {e}")
            components.append("Unable to fetch PTO balance. Contact HR directly.")
```

### Step 5: Update System Prompts

**In `src/core/response_generator.py`:**

```python
def _build_role_prompt(self, query: str, context: str, role: str) -> str:
    if role == "Customer":
        return f"""
You are a helpful customer support assistant for Acme Corp.

Context (product documentation): {context}

Customer question: {query}

Provide a clear, friendly answer based on the documentation. If you don't know, 
offer to create a support ticket. Use simple language and avoid jargon.

CRITICAL: Never make up features or policies not in the documentation.
"""
    
    elif role == "Employee":
        return f"""
You are an internal knowledge assistant for Acme Corp employees.

Context (company wiki, policies): {context}

Employee question: {query}

Provide an accurate answer based on company policies. If you're unsure, 
direct them to the appropriate department or manager. Be professional but approachable.

CRITICAL: Handle PII carefully. Don't display other employees' personal information.
"""
```

### Step 6: Test & Validate

**Unit tests:**
```bash
# Test retrieval quality
pytest tests/test_retrieval_accuracy.py

# Test role-based access control
pytest tests/test_role_permissions.py

# Test action execution
pytest tests/test_actions.py
```

**Load testing:**
```bash
# Simulate 100 concurrent users
locust -f tests/load_test.py --users 100 --spawn-rate 10
```

**Monitoring:**
- Set up LangSmith for prompt tracing
- Configure Supabase alerts for slow queries
- Track error rates, latency p95, retrieval accuracy

---

## 💰 Cost & Resource Planning

### Infrastructure Costs (Monthly)

| Component | Small Team (<100 users) | Medium Org (100-1K users) | Enterprise (1K+ users) |
|-----------|------------------------|---------------------------|------------------------|
| **Supabase** | $25 (Pro) | $599 (Team) | Custom pricing |
| **OpenAI API** | $50-100 | $300-500 | $1,000-3,000 |
| **Vercel** | $0 (Free) | $20 (Pro) | $20+ |
| **Resend/Twilio** | $10 | $50 | $200 |
| **LangSmith** | $39 | $199 | Custom |
| **Total** | **$124-174** | **$1,168-1,368** | **$1,220-3,220+** |

### Optimization Strategies

**Reduce OpenAI costs:**
- Use GPT-3.5-turbo for simple queries (5x cheaper than GPT-4)
- Cache frequent queries (Redis TTL 5 min)
- Batch embeddings during off-peak hours
- Set max_tokens limits to prevent runaway generation

**Scale Supabase:**
- Add read replicas for heavy traffic
- Use connection pooling (pgbouncer)
- Archive old analytics data to cheaper storage

**Monitor spending:**
```python
# In src/analytics/cost_tracking.py
def log_llm_cost(model: str, input_tokens: int, output_tokens: int):
    cost_per_1k = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}
    }
    
    input_cost = (input_tokens / 1000) * cost_per_1k[model]["input"]
    output_cost = (output_tokens / 1000) * cost_per_1k[model]["output"]
    total = input_cost + output_cost
    
    # Log to Supabase for reporting
    supabase_analytics.log_cost(model=model, cost=total, timestamp=now())
```

---

## 🔒 Security & Compliance

### Authentication

**Replace Streamlit's session_id with real auth:**

```python
# In src/main.py
import streamlit_authenticator as stauth

authenticator = stauth.Authenticate(
    credentials,
    cookie_name='enterprise_assistant',
    key='secret_key',
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    st.write(f'Welcome {name}')
    # Load user-specific context
    user_context = load_user_context(username)
else:
    st.error('Username/password is incorrect')
    st.stop()
```

### Data Privacy

**Implement Row-Level Security in Supabase:**

```sql
-- Only show employees their own PTO data
CREATE POLICY employee_pto_policy ON pto_balances
FOR SELECT
USING (auth.uid() = employee_id);

-- Only show managers their team's data
CREATE POLICY manager_team_policy ON employee_data
FOR SELECT
USING (
    auth.uid() IN (
        SELECT manager_id FROM teams WHERE employee_id = employee_data.id
    )
);
```

### Audit Logging

**Track all sensitive queries:**

```python
# In src/flows/core_nodes.py
def log_and_notify(state: ConversationState, rag_engine: RagEngine) -> ConversationState:
    # Standard logging
    supabase_analytics.log_interaction(...)
    
    # Additional audit for sensitive queries
    if is_sensitive_query(state.query):
        audit_logger.log({
            "user_id": state.user_id,
            "query": state.query,
            "response": state.answer[:100],  # Truncate for storage
            "accessed_data": state.retrieved_chunks,
            "timestamp": datetime.now(),
            "ip_address": state.fetch("ip_address"),
            "compliance_tags": ["PII", "Financial", "Medical"]  # As appropriate
        })
    
    return state
```

---

## 📊 Success Metrics

### Track These KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Response accuracy** | >90% | User feedback ratings |
| **Retrieval relevance** | >0.70 similarity | Avg similarity score |
| **Response time** | <2s p95 | Latency tracking |
| **Deflection rate** | 40-60% | (AI resolutions / total inquiries) |
| **User satisfaction** | >4.5/5 | Post-interaction surveys |
| **Cost per interaction** | <$0.10 | (Total OpenAI cost / interactions) |

### Analytics Queries

```sql
-- Response accuracy trend
SELECT 
    DATE(created_at) as date,
    AVG(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) as satisfaction_rate
FROM feedback
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY date;

-- Top unresolved queries (low similarity = knowledge gap)
SELECT 
    user_query,
    AVG(similarity_score) as avg_similarity,
    COUNT(*) as frequency
FROM retrieval_logs
WHERE similarity_score < 0.60
GROUP BY user_query
ORDER BY frequency DESC
LIMIT 20;

-- Cost by role
SELECT 
    role_mode,
    COUNT(*) as interactions,
    SUM(token_count * 0.002 / 1000) as estimated_cost
FROM messages
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY role_mode;
```

---

## 🚀 Deployment Checklist

- [ ] **Data migration complete** (`python scripts/migrate_data_to_supabase.py`)
- [ ] **Roles configured** (update `src/agents/roles.py`)
- [ ] **Actions implemented** (update `src/flows/action_execution.py`)
- [ ] **System prompts updated** (update `src/core/response_generator.py`)
- [ ] **Authentication added** (SSO, RBAC, session management)
- [ ] **Security policies enabled** (Supabase RLS, rate limiting)
- [ ] **Testing complete** (unit tests, integration tests, load tests)
- [ ] **Monitoring configured** (LangSmith, Supabase alerts, cost tracking)
- [ ] **Documentation updated** (internal wiki, user guides, runbooks)
- [ ] **Stakeholder training** (demo sessions, office hours, feedback channels)

---

## 📚 Additional Resources

- **Architecture deep dive**: [docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md](context/SYSTEM_ARCHITECTURE_SUMMARY.md)
- **Data pipeline**: [docs/context/DATA_COLLECTION_AND_SCHEMA_REFERENCE.md](context/DATA_COLLECTION_AND_SCHEMA_REFERENCE.md)
- **Analytics setup**: [docs/features/ANALYTICS_IMPLEMENTATION.md](features/ANALYTICS_IMPLEMENTATION.md)
- **Cost optimization**: [docs/EXTERNAL_SERVICES.md](EXTERNAL_SERVICES.md)
- **Testing patterns**: [tests/README.md](../tests/README.md)

---

**Questions?** Open an issue or contact the maintainer.
