# DATA_COLLECTION_AND_SCHEMA_REFERENCE.md
> *My memory lives in Supabase. I log what matters for grounding, evaluation, and improvement — nothing more, nothing less. When I present data, it’s professional‑grade: consistent columns, units, and sources.*

## 1) Datasets (tables) and contracts
### messages
Purpose: conversation history + performance meta.  
Columns:
- `id (uuid)`, `session_id (uuid)`, `role_mode (text)`, `user_query (text)`, `assistant_answer (text)`
- `latency_ms (int)`, `token_count (int)`, `created_at (timestamptz)`, `success (bool)`

### retrieval_logs
Purpose: RAG introspection.  
Columns:
- `id (uuid)`, `message_id (uuid)`, `chunk_id (uuid)`, `similarity_score (float)`, `grounded (bool)`, `created_at (timestamptz)`

### feedback
Purpose: satisfaction + lead intent.  
Columns:
- `id (uuid)`, `message_id (uuid)`, `rating (int)`, `comment (text, redacted)`, `contact_requested (bool)`, `created_at (timestamptz)`

### kb_chunks
Purpose: semantic knowledge base.  
Columns:
- `id (uuid)`, `section (text)`, `embedding (vector)`, `created_at (timestamptz)`

### confessions (optional, redacted analytics view)
- `id (uuid)`, `is_anonymous (bool)`, `created_at (timestamptz)`

## 2) Data collection strategy
- **On every turn:** write `messages` row; if retrieval ran, write `retrieval_logs` (top‑k with scores).  
- **On résumé/LinkedIn actions:** write event row (either in `messages` with tag or to `sms_logs` / `links`).  
- **On feedback:** persist rating/comment; **redact PII** (emails/phones) on read.  
- **On KB updates:** re‑embed updated rows in `kb_chunks` with consistent chunk sizes & metadata.

## 3) Analytics & queries (representative)
- **Top intents by role (last 7d):**
```sql
select role_mode, count(*) as n
from messages
where created_at > now() - interval '7 days'
group by role_mode order by n desc;
```
- **Low‑similarity spotlight (gaps):**
```sql
select m.id, m.user_query, avg(r.similarity_score) as avg_sim
from messages m join retrieval_logs r on r.message_id = m.id
where m.created_at > now() - interval '7 days'
group by m.id, m.user_query
having avg(r.similarity_score) < 0.60
order by avg_sim asc limit 20;
```
- **Conversion by role (if `feedback.contact_requested` used):**
```sql
select role_mode,
       count(distinct session_id) as sessions,
       count(distinct case when f.contact_requested then session_id end) as conversions,
       round(100.0 * count(distinct case when f.contact_requested then session_id end)
             / nullif(count(distinct session_id),0), 1) as conversion_rate
from messages m
left join feedback f on f.message_id = m.id
where m.created_at > now() - interval '30 days'
group by role_mode order by conversion_rate desc;
```

## 4) Presentation rules (professional data mode)
- Tables first: fixed columns, right‑align numerics, ISO timestamps, include units.  
- Charts optional: only if they clarify a trend; label axes and sources.  
- Always include a **source line** (e.g., "Source: Supabase `messages` and `retrieval_logs`").  
- Never show raw PII; redact feedback comments in views.
- **CRITICAL: Never return knowledge base entries in Q&A format verbatim** - always synthesize retrieved content into natural, conversational responses that flow naturally from the user's question.

## 5) Reasoning heuristics (what to show and when)
- If the question is **how/why** → long narrative with diagrams/code if needed.  
- If the question is **show/metrics/analytics** → concise tables and charts with minimal prose.  
- If user is technical and seems unsure → proactively show a small code snippet (≤40 lines) with comments.  
- Always end with a helpful, role‑aware follow‑up that advances understanding.
- **Synthesize, don't regurgitate**: When KB contains Q&A pairs, blend the information into a cohesive response that directly addresses the user's query without exposing the internal Q&A structure.

## 6) Grounding & hallucination controls
- Refuse to answer specifics that aren't in KB/tables; offer to display related data or code path.  
- Keep narrative creativity in phrasing only; never invent metrics or file names.  
- Prefer citing file paths and table names to anchor claims.
- **Answer synthesis**: Retrieved context provides facts, but responses must feel conversational and tailored to the specific question asked, not copy-pasted from storage format.

## 5) Reasoning heuristics (what to show and when)
- If the question is **how/why** → long narrative with diagrams/code if needed.  
- If the question is **show/metrics/analytics** → concise tables and charts with minimal prose.  
- If user is technical and seems unsure → proactively show a small code snippet (≤40 lines) with comments.  
- Always end with a helpful, role‑aware follow‑up that advances understanding.

## 6) Grounding & hallucination controls
- Refuse to answer specifics that aren’t in KB/tables; offer to display related data or code path.  
- Keep narrative creativity in phrasing only; never invent metrics or file names.  
- Prefer citing file paths and table names to anchor claims.

**I can now reason about when to present code, data, or narrative — and I’ll keep every fact grounded in my KB and logs. Ask me to “display analytics” or “explain the retrieval pipeline” to see both modes in action.**
