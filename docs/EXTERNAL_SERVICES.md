# External Services

What the backend talks to besides the LLM APIs, what each service does here,
and how to configure it. Every variable named below is read by the code
(central config: `assistant/config/supabase_config.py`); `.env.example` mirrors
this list. Local dev loads `.env`; production reads Railway environment
variables with the same names.

| Service | Role in this system | Required? |
| --- | --- | --- |
| Supabase | pgvector knowledge base, conversation analytics, leads, confessions | Yes |
| OpenAI | Embeddings only (`text-embedding-3-small`, 1536 dims) | Yes |
| Anthropic | Generation (Claude Sonnet 4.5) + intent classification (Claude Haiku) | Yes |
| Resend | Transactional email to Noah (lead + confession alerts) | Optional |
| Twilio | SMS to Noah (lead + confession alerts) | Optional |
| LangSmith | Tracing every LLM call (see [LANGSMITH.md](LANGSMITH.md)) | Optional |
| Sentry | Backend error reporting | Optional |

Optional services degrade gracefully: unset means the feature is skipped with
a log line, never a crash.

## Supabase (required)

1. Create a project at https://supabase.com/dashboard and apply the schema in
   [supabase/migrations](../supabase/migrations/) (applied-state log:
   `APPLIED.md` there).
2. Settings → API → copy into `.env`:

```bash
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...   # server-side only, never exposed
```

3. Load the knowledge base (embeds `data/*.csv` and inserts into `kb_chunks`):

```bash
python3 scripts/migrate_data_to_supabase.py
```

Vector search runs through the `match_kb_chunks` RPC (see
`assistant/retrieval/pgvector_retriever.py`).

## Resend email (optional)

Free tier: 3,000 emails/month. Create an API key at
https://resend.com/api-keys.

```bash
RESEND_API_KEY=re_...
RESEND_FROM_EMAIL=onboarding@resend.dev   # or a verified domain sender
ADMIN_EMAIL=you@example.com               # where alerts land
```

## Twilio SMS (optional)

Trial accounts must verify the destination number (Console → Verified Caller
IDs), or sends fail with error 21608.

```bash
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM=+15551234567        # your Twilio number (legacy alias: TWILIO_PHONE_NUMBER)
NOAH_PHONE_NUMBER=+15559876543  # crush-confession alerts go TO this number
ADMIN_PHONE_NUMBER=+15559876543 # general lead alerts
```

Notification dispatch is throttled (10/hour across all capture flows,
`assistant/flows/capture/notifications.py`) — throttled submissions are still
saved to Supabase; only the SMS/email ping is skipped.

## Sentry error monitoring (optional)

Free tier is sufficient. Create a Python/FastAPI project at
https://sentry.io, copy its DSN:

```bash
SENTRY_DSN=https://...@....ingest.sentry.io/...
SENTRY_ENVIRONMENT=production   # optional label, defaults to "production"
```

When set, `api/main.py` initializes the SDK at startup and reports both
unhandled route errors and pipeline errors that the `/chat` endpoint masks
behind its graceful-fallback response. Errors only — performance tracing is
deliberately off because LangSmith owns tracing.

## Verifying a setup

```bash
# 1. Config sanity: fails loudly if a required var is missing
python3 -c "from assistant.config.supabase_config import supabase_settings; print('config ok')"

# 2. End to end: ask a KB question through the real pipeline
python3 chat_with_portfolia.py

# 3. Notifications: complete the crush flow in the terminal client and
#    check your phone/inbox (or the crush_confessions table if Twilio/Resend
#    are unset)
```
