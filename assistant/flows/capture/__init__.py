"""Capture flows package — the two chat-history-marker state machines.

Portfolia's multi-step flows (crush confession, contact/lead capture) are
stateless: each turn, the current step is recovered by scanning chat_history
for marker strings defined in `constants`. This package was mechanically
split out of `assistant/flows/node_logic/stage1_intent_router.py`, which
still re-exports every moved name for compatibility.

Modules:
- constants:      shared marker strings, form-text templates, engagement counters
- crush_flow:     crush confession FSM (form parsing, Supabase insert)
- lead_capture:   contact/lead capture FSM (parsing, recruiter_leads insert)
- notifications:  Twilio SMS / Resend email dispatch helpers
"""

from assistant.flows.capture.crush_flow import (
    handle_crush_confession,
    handle_crush_flow_continuation,
)
from assistant.flows.capture.lead_capture import handle_hm_capture_continuation

__all__ = [
    "handle_crush_confession",
    "handle_crush_flow_continuation",
    "handle_hm_capture_continuation",
]
