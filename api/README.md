# api/

FastAPI app serving the Portfolia backend. Production runs this on Railway via the
repo's [Dockerfile](../Dockerfile); the deployed frontend
([portfolia-frontend](https://github.com/iNoahCodeGuy/portfolia-frontend)) calls it at
`${NEXT_PUBLIC_API_URL}/chat`.

## Run locally

```bash
uvicorn api.main:app --reload --port 8000
```

## Endpoints

### `GET /health`

Liveness plus a cheap Supabase probe. Returns `{"status": "ok", "kb_chunks": <count>}`
(200) when the KB table is reachable, `{"status": "degraded", "kb_chunks": null}` (503)
when it isn't. The scheduled keep-alive workflow
([.github/workflows/keepalive.yml](../.github/workflows/keepalive.yml)) hits this to keep
the Supabase free tier active and to catch silent retrieval outages.

### `POST /chat`

Request body (all fields optional — sensible defaults apply):

```json
{
  "message": "What is Noah's professional background?",
  "session_id": "any-stable-string",
  "role": "Learn more about Noah",
  "chat_history": [{"role": "user", "content": "..."}]
}
```

- `message` — the user's text. Welcome-button labels (e.g. `"See what Noah has built"`)
  are recognized and mapped to the pipeline's menu options. `query` is accepted as an
  alias for `message`.
- `session_id` — omit to have one generated; reuse it to continue a conversation.
- `role` — welcome-button context; falls back to the session's stored role.
- `chat_history` — optional stateless mode: pass the full history each turn and no
  server-side session state is relied upon.

Response:

```json
{
  "success": true,
  "response": "…the assistant's answer…",
  "answer": "…mirror of response (the field the frontend reads)…",
  "session_id": "any-stable-string",
  "form": null
}
```

`form` is the pipeline's structured form signal — `"crush"` or `"contact"` while the
corresponding capture flow is awaiting input, `null` otherwise. The frontend renders
forms from this field rather than pattern-matching the answer text.

On pipeline errors the endpoint still returns 200 with `"success": false` and a generic
message — the frontend renders it as a normal bubble.

## Notes

- Sessions are held in an in-memory dict, so they reset on redeploy and don't share
  across instances. The frontend's stateless `chat_history` mode is the durable path.
- CORS: `FRONTEND_URL` (env) is added to the allow-list alongside `localhost:3000`.
- The heavy lifting happens in `assistant/flows/conversation_flow.py`
  (`run_conversation_flow`) — this module is just transport, session glue, and
  menu-button mapping.
