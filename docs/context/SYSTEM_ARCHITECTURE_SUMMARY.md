# SYSTEM_ARCHITECTURE_SUMMARY.md
> *Let me walk you through my architecture like a senior engineer teaching a new team member. I'll explain the design decisions, show you the tradeoffs, and help you understand how production GenAI systems are built. This is your hands-on case study for learning RAG, vector search, and LLM orchestration.*

## 0) Educational Mission
This system exists to **teach how generative AI applications work** by using itself as a transparent example:
- **Every component is explorable:** Ask about any part and I'll show you the code
- **Design decisions are explained:** Learn why I chose pgvector over FAISS, serverless over dedicated servers, temperature 0.7 (balanced factual + conversational)
- **Patterns map to enterprise use cases:** See how this architecture adapts for customer support, internal docs, sales enablement
- **Live observability:** View real metrics showing retrieval performance, costs, user satisfaction
- **Production-ready patterns:** Authentication, rate limiting, PII handling, error management, cost optimization

## 1) Control flow (LangGraph nodes) - **The GenAI Conversation Pipeline**

### Conceptual Flow (What's Happening)
```
Classify user intent → Retrieve relevant knowledge → Generate grounded answer →
Enhance with role context → Execute side effects → Log for observability
```

### Actual Implementation (The Code)
```python
# Pipeline defined in assistant/flows/conversation_flow.py
handle_greeting
  → Detects first-turn "hello" and returns greeting without RAG
  → Short-circuits pipeline if user just said hi (efficiency!)

classify_query
  → Analyzes user intent: teaching moment? code request? data request?
  → Sets flags: needs_longer_response, code_would_help, data_would_help
  → Source: assistant/flows/query_classification.py

retrieve_chunks (THIS IS RAG!)
  → Converts query to embedding via text-embedding-3-small (768 dims)
  → Searches Supabase kb_chunks using pgvector cosine similarity
  → Returns top-k relevant context chunks with similarity scores
  → Source: assistant/flows/core_nodes.py → assistant/retrieval/pgvector_retriever.py

generate_answer
  → Calls OpenAI GPT-4o-mini with retrieved context
  → Injects dynamic instructions based on query classification
  → Handles narrative (explain concepts) and code display (show implementation)
  → Uses role-specific prompts (Technical HM, Developer, General)
  → Source: assistant/flows/core_nodes.py → assistant/core/response_generator.py

plan_actions
  → Determines side effects needed: send analytics? offer contact? log feedback?
  → Creates action plan without executing yet (separation of concerns)
  → Source: assistant/flows/conversation_nodes.py

apply_role_context
  → Adds role-specific enhancements (follow-ups, contact offers, personality)
  → Software Developer → technical follow-ups + code examples
  → Hiring Manager → business value + Noah's contact offer
  → Just exploring → fun facts + casual tone
  → Source: assistant/flows/conversation_nodes.py

execute_actions
  → Runs planned side effects: email via Resend, SMS via Twilio, analytics logging
  → Handles failures gracefully (degraded mode - logs errors but doesn't crash)
  → Source: assistant/flows/action_execution.py

log_and_notify
  → Logs interaction to Supabase messages + retrieval_logs tables
  → Tracks latency, tokens, success/failure for observability
  → Source: assistant/flows/conversation_nodes.py
```

**Teaching insights:**
- **Determinism first:** Factual answers only after retrieval (no hallucinations)
- **Proactive intelligence:** Show code/data when it would clarify, not just when explicitly requested
- **Role adaptation:** Technical users get code, business users get value propositions
- **Graceful degradation:** External services fail → log errors, continue with core functionality
- **Observability:** Every step logged for continuous improvement

**Try it:** Ask "show me the retrieval code" or "how does classification work?" to see these nodes in action!

## 2) RAG pipeline - **How I Avoid Hallucinations (The Core GenAI Pattern)**
1. **Embedding:** `text-embedding-3-small` converts your query into a vector (768 dimensions capturing semantic meaning)
2. **Vector search:** `SELECT ... FROM kb_chunks ORDER BY embedding <=> $query LIMIT k` using IVFFLAT index (approximate nearest neighbor)
3. **Context assembly:** Merge top-k chunks + role instructions + dynamic affordances (code snippets, contact links)
4. **Generation:** `gpt-4o-mini` with grounded context; temperature 0.7 (balanced - factual but not robotic)
5. **Attribution:** Cite KB sections so you can verify sources (transparency!)

**Why this matters for enterprises:**
- **Accuracy:** Grounded in your data, not model hallucinations (see our 94% grounding rate in analytics)
- **Cost:** $0.0001 embeddings + $0.002 per LLM call vs $50k+ fine-tuning
- **Maintainability:** Update KB without retraining models
- **Auditability:** Every answer traces to specific KB chunks (compliance-ready)

**Try it:** Ask "display analytics" to see retrieval performance metrics, or "show me the vector search code"

## 3) Data layer (Supabase)
- **Tables:**
  - `messages(id, session_id, role_mode, user_query, assistant_answer, latency_ms, token_count, created_at, success)`
  - `retrieval_logs(id, message_id, chunk_id, similarity_score, grounded, created_at)`
  - `feedback(id, message_id, rating, comment(redacted), contact_requested, created_at)`
  - `kb_chunks(id, section, embedding, created_at)`
  - Optional: `confessions(id, is_anonymous, created_at)`, `sms_logs(...)`
- **RLS:** Enabled per table; server routes use service key only on the backend.
- **Storage:** Private bucket for résumé (signed URLs); public for headshot.

## 4) Frontend (Vercel)
- Single‑page chat with role selector.
- Professional tables for analytics (fixed column sets, ISO timestamps, units).
- Buttons for **Send Résumé**, **Open LinkedIn**, **Request Contact** (logs event + optional notifications).
- Error states and loading spinners; retries on transient fetch errors.

## 5) Backend/API
- **/api/chat:** Role → retrieve → generate → log → respond.
- **/api/analytics:** Returns inventory & last‑50 rows per table with PII redaction.
- **/api/email:** Generates signed résumé URL and sends via Resend.
- **/api/sms:** Twilio wrapper for alerts (resume sent, contact requested).
- **/api/feedback:** Persists rating/comment; flags contact intent.

## 6) Reasoning about presentation - **Teaching Through Demonstration**
- **Show code** when: It clarifies GenAI patterns (RAG retrieval, prompt engineering, error handling); ≤40 lines with inline teaching annotations; cite file path so users can explore
- **Show tables/metrics** when: User asks about performance, cost, accuracy, or "how is this measured?"; demonstrates observability practices
- **Long explanations** when: User asks "how/why" about architecture, GenAI patterns, or enterprise scaling; otherwise summarize with offer to drill down
- **Follow‑ups**: Invite deeper exploration: "Want to see the prompt engineering?", "Curious about vector search optimization?", "Should I explain enterprise adaptation?"
- **Adapt depth**: Technical users get implementation details; business users get value framing; explorers get accessible analogies

**The meta-lesson:** Every presentation choice demonstrates how to build user-centric AI interfaces

## 7) Enterprise adaptation path - **How to Build This for Your Organization**
This architecture maps directly to common enterprise use cases:

**Customer Support Bot:**
- Replace career KB with product docs + troubleshooting KB
- Add ticket creation actions (Zendesk/Intercom API)
- Same RAG pipeline, different knowledge source

**Internal Documentation Assistant:**
- Ingest Confluence/Notion/SharePoint
- Add SSO (SAML/OIDC) for access control
- Per-department KB chunks with role-based retrieval

**Sales Enablement Tool:**
- KB: Product specs, competitor analysis, objection handling
- Actions: Log to CRM, send proposals, schedule demos
- Same observability patterns for coaching insights

**Technical implementation paths:**
- **Security:** Add SSO, Secrets Manager, per-tenant RLS
- **Scale:** Async work → queue (Vercel Queue, Cloud Tasks, or Neon logical replication + worker)
- **Vector search:** pgvector scales to millions; dedicated vector DB (Pinecone, Weaviate) only if needed
- **Observability:** LangSmith traces + cost/latency dashboards + alert rules
- **APIs:** API Gateway + WAF + rate-limits per tenant

**Try it:** Ask "how would this work for customer support?" or "show me the cost breakdown"

## 8) Tradeoffs & rationale - **Learning from Design Decisions**
**pgvector vs FAISS/Pinecone:**
- ✅ pgvector: SQL-auditable, single database, simpler ops, RLS for security
- ❌ FAISS: Faster for 100M+ vectors but adds deployment complexity
- ❌ Pinecone: Managed vector DB but $70/month minimum, vendor lock-in
- **Decision:** pgvector keeps retrieval **governed and observable** — perfect for learning and most enterprise scales

**Serverless (Vercel) vs long-running containers:**
- ✅ Serverless: Zero-to-deploy in minutes, auto-scales, pay-per-use ($5/month for this project)
- ❌ Containers: Better for high-throughput (>10k req/day) but needs K8s/ECS management
- **Decision:** Serverless is perfect for teaching pattern and handles 1k+ users; graduate to containers if needed

**Single DB (Postgres) vs polyglot persistence:**
- ✅ Postgres: Relational + vectors + full-text + JSON in one system, atomic transactions
- ❌ Polyglot: Separate vector DB + warehouse + cache adds operational complexity
- **Decision:** Start simple; add BigQuery/Snowflake only when analytics query volume demands it

**Temperature 0.7 (balanced):**
- **Why 0.7:** Sweet spot between deterministic (0.0) and creative (1.0)
- **Effect:** Grounded in retrieved facts but with natural conversational phrasing
- **Alternative approaches:** Some systems use 0.0 for maximum determinism, 0.7+ for creative writing
- **Decision:** Balance accuracy with readability - users get factual answers that don't sound robotic

**Try it:** Ask "why pgvector over Pinecone?" or "show me cost comparison" to dive deeper into any tradeoff

---

**Want to explore?**
- "Show me the pipeline architecture"
- "Display the /api/analytics contract"
- "How does vector search work?"
- "What's the cost per query?"

I'll show you exactly how the parts fit and how I keep facts grounded. This is your GenAI learning laboratory!
