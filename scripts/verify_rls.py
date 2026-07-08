#!/usr/bin/env python3
"""Verify RLS actually blocks the public anon key across all tables and views.

Probes PostgREST as `anon` — SELECT on every table and view, plus INSERT on
the four tables migration 010 fixed — and fails loudly on any exposure. Run
this after applying any migration that touches RLS; APPLIED.md cites the
output as evidence.

Requires SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY
(reads .env locally). Exit 0 = locked down, exit 1 = exposure found.
"""

import json
import re
import sys
import urllib.error
import urllib.request

SENTINEL = "rls-verify-probe"

# Tables that must be invisible to anon (RLS on, no anon policy).
HIDDEN_TABLES = [
    "crush_confessions",
    "recruiter_leads",
    "conversation_sessions",
    "conversation_messages",
    "confessions",
    "messages",
    "retrieval_logs",
    "feedback",
    "sms_logs",
]

# Tables anon may read by design (001 grants SELECT for a future public API).
READABLE_TABLES = ["kb_chunks", "links"]

# Views must not leak their (RLS-protected) base tables to anon.
VIEWS = [
    "messages_with_retrieval",
    "analytics_by_role",
    "recent_confessions",
    "pending_contacts",
]

# Minimal valid payloads for anon INSERT probes on the tables 010 fixed.
INSERT_PAYLOADS = {
    "crush_confessions": {"session_id": SENTINEL},
    "recruiter_leads": {"session_id": SENTINEL},
    "conversation_sessions": {"session_id": SENTINEL},
    "conversation_messages": {
        "session_id": SENTINEL,
        "turn_number": 1,
        "role": "user",
        "content": SENTINEL,
    },
}


def load_env(path=".env"):
    env = {}
    for line in open(path):
        m = re.match(r"^([A-Z_0-9]+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return env


def request(url, key, method="GET", payload=None, prefer=None):
    """Returns (status, body). HTTP errors come back as (status, body), not raises."""
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    if prefer:
        headers["Prefer"] = prefer
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:120]


def main():
    env = load_env()
    base = env["SUPABASE_URL"].rstrip("/") + "/rest/v1"
    anon = env["SUPABASE_ANON_KEY"]
    service = env["SUPABASE_SERVICE_ROLE_KEY"]

    failures = []

    def report(kind, target, ok, detail):
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {kind:12s} {target:26s} {detail}")
        if not ok:
            failures.append(f"{kind} {target}: {detail}")

    # Pre-clean sentinel rows so INSERT probes are deterministic
    # (conversation_sessions.session_id is UNIQUE).
    for table in INSERT_PAYLOADS:
        request(
            f"{base}/{table}?session_id=eq.{SENTINEL}", service, method="DELETE"
        )

    print("anon SELECT — RLS-protected tables (expect 0 rows visible):")
    for table in HIDDEN_TABLES:
        status, body = request(f"{base}/{table}?select=id&limit=3", anon)
        if status != 200:
            # 401/403/404 also means anon can't read it — fine, but say so.
            report("select", table, True, f"HTTP {status} (not readable)")
            continue
        n = len(json.loads(body))
        report("select", table, n == 0, f"HTTP 200, rows_visible={n}")

    print("anon SELECT — public-by-design tables (informational):")
    for table in READABLE_TABLES:
        status, body = request(f"{base}/{table}?select=*&limit=1", anon)
        n = len(json.loads(body)) if status == 200 else 0
        print(f"  [info] select       {table:26s} HTTP {status}, rows_visible={n}")

    print("anon SELECT — views (expect 0 rows visible or error):")
    for view in VIEWS:
        status, body = request(f"{base}/{view}?select=*&limit=3", anon)
        if status != 200:
            report("select", view, True, f"HTTP {status} (not readable)")
            continue
        n = len(json.loads(body))
        report("select", view, n == 0, f"HTTP 200, rows_visible={n}")

    print("anon INSERT — tables fixed by migration 010 (expect denial):")
    for table, payload in INSERT_PAYLOADS.items():
        status, body = request(
            f"{base}/{table}", anon, method="POST", payload=payload
        )
        if status < 300:
            # The write landed — exposure. Clean it up via service role.
            request(
                f"{base}/{table}?session_id=eq.{SENTINEL}",
                service,
                method="DELETE",
            )
            report("insert", table, False, f"HTTP {status} — WRITE LANDED (cleaned up)")
        else:
            report("insert", table, True, f"HTTP {status} (denied)")

    print()
    if failures:
        print(f"EXPOSED — {len(failures)} probe(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL LOCKED DOWN — every probe denied or returned zero rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
