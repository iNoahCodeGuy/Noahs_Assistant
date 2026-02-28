# CONTRIBUTING.md

Welcome! This repo powers **Portfolia (Noah's AI Assistant)** — a role‑aware, RAG‑backed portfolio system.
Please read these short conventions before making changes.

---

## 1) Quick Links (Copilot context first)
- 📘 Project Overview → `docs/context/PROJECT_REFERENCE_OVERVIEW.md`
- 🧩 System Architecture → `docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md`
- 🧮 Data & Schema Reference → `docs/context/DATA_COLLECTION_AND_SCHEMA_REFERENCE.md`
- 💬 Conversation Personality → `docs/context/CONVERSATION_PERSONALITY.md`

> **Tip:** Open these files side‑by‑side when prompting Copilot so it grounds suggestions in the intended architecture and tone.

---

## 2) Branching & PRs
- Use feature branches: `feature/<short-topic>` (e.g., `feature/rag-retriever-pgvector`).
- Keep PRs small, focused, and scoped to a single concern.
- Include **Before/After** notes and **Testing steps** in the PR body.

---

## 3) Code Organization (expected layout)
```
src/
  agents/             # role router, response formatter, follow-up generator
  api/                # serverless endpoints (/api/chat, /api/analytics, /api/email, /api/sms, /api/feedback)
  core/               # rag_engine, retriever (pgvector), embedding utils
  analytics/          # supabase_analytics.py (messages, retrieval_logs, feedback writers)
  services/           # wrappers: supabase_client, resend_email, twilio_sms
  ui/                 # chat UI helpers (if not in Next.js)
docs/
  context/            # architecture & schema docs for Copilot
tests/
  unit/               # fast tests for pure functions
  integration/        # api route + supabase test doubles
```

---

## 4) Data Contracts (do not break without migrations)
- Postgres tables: `messages`, `retrieval_logs`, `feedback`, `kb_chunks`, optional `confessions`, `sms_logs`.
- If you change columns or types, add a SQL migration under `supabase/migrations/` and update
  `docs/context/DATA_COLLECTION_AND_SCHEMA_REFERENCE.md` accordingly.

---

## 5) Copilot Prompt Patterns
When asking Copilot to implement/modify code, **paste a short task prompt** and include links to the relevant docs.

**Examples**

**A. Implement pgvector retriever (k=4)**
```
You are updating src/core/rag_engine.py to use Supabase pgvector.
Follow docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md (RAG pipeline section).
Implement a function retrieve(query_embedding, k=4) using:
SELECT id, section, similarity(embedding, $1) AS score
FROM kb_chunks
ORDER BY embedding <=> $1
LIMIT $2;
Return list[Chunk] with id, section, score. Add inline comments.
```

**B. Add analytics writer**
```
Add log_message(user_query, assistant_answer, role_mode, latency_ms, token_count) to src/analytics/supabase_analytics.py.
Use docs/context/DATA_COLLECTION_AND_SCHEMA_REFERENCE.md as schema source (messages table).
Insert row and return inserted id. Include error handling + docstring.
```

**C. Role-aware follow‑up generator**
```
In src/agents/response_formatter.py, add generate_followup(role_mode, topic, last_query).
Follow docs/context/PROJECT_REFERENCE_OVERVIEW.md (Conversation style) for tone and brevity.
Return <= 50 tokens, role-aware, with 2-3 helpful next-step options.
```

---

## 6) Quality Bar
- **Accuracy first:** All facts must be grounded in KB or Supabase.
- **Small, commented snippets:** Prefer ≤40 lines when showing code to users.
- **Tests:** Add/adjust unit tests for pure logic; add minimal integration tests for new API routes.
- **Docs:** If behavior changes, update the relevant file in `docs/context/`.

---

## 7) Local Dev Notes
- Python ≥ 3.10
- Env vars: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `OPENAI_API_KEY`, optional `TWILIO_*`, `RESEND_API_KEY`.
- Use `.env` for local; never commit secrets.

---

## 8) Release & Deploy
- CI should run tests + lint on PRs.
- Vercel auto-deploys main; ensure environment variables are set in Vercel dashboard.
- Supabase migrations must be applied before deploying code depending on them.

Thanks for contributing — this assistant’s *life’s purpose is to help Noah get a software engineering role*, so every improvement that increases clarity, stability, or reliability supports that mission.
