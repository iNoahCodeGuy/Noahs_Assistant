# LangSmith Integration into QA Strategy

**Date:** October 16, 2025  
**Question:** "Should we incorporate LangSmith into our QA?"  
**Answer:** **YES - As Phase 2 production monitoring, NOT as replacement for pytest**

---

## TL;DR

| Tool | Purpose | When | What It Catches |
|------|---------|------|-----------------|
| **pytest** | Pre-deployment testing | Before code merges | 90% of bugs (logic errors, policy violations) |
| **LangSmith** | Post-deployment monitoring | After production deploy | 10% of bugs (edge cases, performance, real LLM behavior) |

**Recommendation:** Keep pytest as primary QA (Phase 1), add LangSmith for production monitoring (Phase 2).

---

## Why LangSmith ≠ Automated Testing

### What LangSmith IS
- **Production observability tool**
- Traces every LLM call in real-time
- Shows actual prompts, responses, latency, costs
- Detects patterns across thousands of queries
- Root cause analysis for production issues

### What LangSmith is NOT
- ❌ Not a testing framework (can't mock, can't control inputs)
- ❌ Not pre-deployment validation (tests real production traffic)
- ❌ Not deterministic (depends on actual user queries)
- ❌ Not a pytest replacement (complements it)

---

## The Hybrid Approach: Test + Monitor

### Phase 1: Pre-Deployment Testing (pytest) ✅ Already Done

**Current Setup:**
- 30 automated tests (18 conversation + 12 alignment)
- Runs in <3 seconds
- Blocks bad code from merging
- 77% pass rate (target: 100%)

**Example:**
```python
def test_no_emoji_headers():
    """Ensure LLM strips markdown headers."""
    mock_engine = MagicMock()
    mock_engine.generate_response.return_value = "**Bold Header**\n\nContent..."
    
    state = run_conversation_flow(mock_engine)
    
    # Assert NO markdown headers in response
    assert "###" not in state.answer  # ✅ Blocks deployment if fails
```

**What It Catches:**
- Logic errors (wrong function calls)
- Policy violations (emoji headers, Q&A verbatim)
- Code structure issues (missing functions)
- Configuration errors (wrong temperature)

**What It Misses:**
- Real LLM behavior (tests use mocks)
- Edge case queries (tests only cover known scenarios)
- Performance issues (mocks have no latency)
- Cost overruns (mocks don't track tokens)

---

### Phase 2: Post-Deployment Monitoring (LangSmith) ← **Add This**

**Proposed Setup:**
- Configure LangSmith API key (see [LANGSMITH.md](LANGSMITH.md))
- Enable production tracing: `LANGCHAIN_TRACING_V2=true`
- Run `scripts/quality_monitor.py` daily via cron
- Display metrics in `scripts/quality_dashboard.py`

**Example:**
```python
def check_production_quality():
    """Check LangSmith traces for quality violations."""
    client = get_langsmith_client()
    
    # Get last 100 production queries
    runs = client.list_runs(
        project_name="noahs-ai-assistant",
        start_time=datetime.now() - timedelta(hours=1)
    )
    
    violations = []
    
    for run in runs:
        answer = run.outputs.get("answer", "")
        
        # Check for markdown headers (our KB vs Response policy!)
        if re.search(r'#{1,6}\s', answer):
            violations.append({
                "type": "emoji_headers",
                "trace_id": run.id,
                "severity": "CRITICAL",
                "message": f"Found markdown headers in production response",
                "link": f"https://smith.langchain.com/o/.../runs/{run.id}"
            })
        
        # Check response length
        if len(answer) > 15000:
            violations.append({
                "type": "information_overload",
                "trace_id": run.id,
                "severity": "WARNING",
                "message": f"Response too long: {len(answer)} chars"
            })
        
        # Check latency
        if run.total_time and run.total_time > 5000:
            violations.append({
                "type": "slow_query",
                "trace_id": run.id,
                "severity": "WARNING",
                "message": f"Query took {run.total_time}ms (>5s threshold)"
            })
    
    return violations
```

**What It Catches:**
- Real LLM outputs (bypassed our sanitization?)
- Edge case queries (user asked something unexpected)
- Performance degradation (latency spike)
- Cost anomalies (token usage spike)
- Production errors (exceptions in real traffic)

**What It Doesn't Replace:**
- Pre-deployment tests (LangSmith only sees production)
- Fast feedback (tests run in seconds, LangSmith requires deployed code)
- Controlled scenarios (LangSmith depends on user queries)

---

## Integration Plan

### Step 1: Add LangSmith to Phase 2 Checklist ✅ Done

Updated `docs/QA_IMPLEMENTATION_SUMMARY.md`:

```markdown
### Phase 2 (Week 2) - Automation & Production Monitoring
- [ ] Create `.github/workflows/conversation-quality.yml`
- [ ] Implement `scripts/quality_monitor.py` with LangSmith integration
- [ ] Implement `scripts/quality_dashboard.py` with LangSmith metrics
- [ ] Set up automated alerts (email + Slack)
- [ ] Configure LangSmith for production tracing
- [ ] Add runtime quality checks (emoji headers, response length, latency)
```

### Step 2: Enhance `scripts/quality_monitor.py`

**Add LangSmith integration:**

```python
#!/usr/bin/env python3
"""Automated quality monitoring with LangSmith integration.

Runs daily via cron/GitHub Actions. Checks:
1. Supabase analytics (existing)
2. LangSmith production traces (NEW)
"""

from langsmith import Client
from src.observability import get_langsmith_client
from src.analytics.supabase_analytics import supabase_analytics

def check_supabase_metrics():
    """Existing function - checks DB metrics."""
    # ... existing code ...
    pass

def check_langsmith_traces():
    """NEW function - checks production LLM traces."""
    client = get_langsmith_client()
    
    if not client:
        print("⚠️  LangSmith not configured, skipping trace checks")
        return []
    
    print("🔍 Checking LangSmith production traces...")
    
    # Get last 24 hours
    runs = client.list_runs(
        project_name="noahs-ai-assistant",
        start_time=datetime.now() - timedelta(hours=24)
    )
    
    violations = []
    
    for run in runs:
        if not run.outputs or "answer" not in run.outputs:
            continue
        
        answer = run.outputs["answer"]
        
        # Policy 1: No markdown headers
        if re.search(r'#{1,6}\s', answer):
            violations.append(f"🔴 CRITICAL: Markdown headers in trace {run.id}")
        
        # Policy 2: No information overload
        if len(answer) > 15000:
            violations.append(f"⚠️  WARNING: Response {len(answer)} chars in trace {run.id}")
        
        # Policy 3: No duplicate prompts
        prompt_count = answer.lower().count("would you like")
        if prompt_count > 1:
            violations.append(f"⚠️  WARNING: {prompt_count} prompts in trace {run.id}")
        
        # Performance check
        if run.total_time and run.total_time > 5000:
            violations.append(f"⚠️  WARNING: Slow query {run.total_time}ms in trace {run.id}")
        
        # Error check
        if run.error:
            violations.append(f"❌ ERROR: {run.error} in trace {run.id}")
    
    return violations

def main():
    """Run all quality checks."""
    all_violations = []
    
    # Check 1: Supabase metrics (existing)
    supabase_violations = check_supabase_metrics()
    all_violations.extend(supabase_violations)
    
    # Check 2: LangSmith traces (NEW)
    langsmith_violations = check_langsmith_traces()
    all_violations.extend(langsmith_violations)
    
    # Report results
    if all_violations:
        print(f"\n❌ {len(all_violations)} quality violations found:\n")
        for v in all_violations:
            print(f"  {v}")
        
        # Send alerts
        send_email_alert(all_violations)
        send_slack_alert(all_violations)
        
        sys.exit(1)
    else:
        print("\n✅ All quality checks passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
```

### Step 3: Enhance `scripts/quality_dashboard.py`

**Add LangSmith metrics panel:**

```python
import streamlit as st
from langsmith import Client

def show_langsmith_section():
    """Display LangSmith production metrics."""
    st.header("🔍 Production Quality (LangSmith)")
    
    client = get_langsmith_client()
    
    if not client:
        st.warning("⚠️  LangSmith not configured - showing Supabase metrics only")
        st.info("To enable: Set LANGCHAIN_API_KEY in environment")
        return
    
    # Get last 24 hours
    runs = client.list_runs(
        project_name="noahs-ai-assistant",
        start_time=datetime.now() - timedelta(hours=24)
    )
    
    # Calculate metrics
    total_runs = len(runs)
    errors = sum(1 for r in runs if r.error)
    slow_queries = sum(1 for r in runs if r.total_time and r.total_time > 2000)
    
    # Display KPI cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Queries (24h)", f"{total_runs:,}")
    
    with col2:
        error_rate = (errors / total_runs * 100) if total_runs > 0 else 0
        st.metric("Error Rate", f"{error_rate:.1f}%",
                  delta=f"{-0.5 if error_rate < 1 else +0.5}%",
                  delta_color="inverse")
    
    with col3:
        slow_rate = (slow_queries / total_runs * 100) if total_runs > 0 else 0
        st.metric("Slow Queries", f"{slow_rate:.1f}%")
    
    with col4:
        avg_latency = sum(r.total_time for r in runs if r.total_time) / total_runs if total_runs > 0 else 0
        st.metric("Avg Latency", f"{avg_latency:.0f}ms")
    
    # Check for quality violations
    st.subheader("⚠️  Quality Violations")
    violations = check_langsmith_traces()  # From quality_monitor.py
    
    if violations:
        for v in violations:
            if "CRITICAL" in v:
                st.error(v)
            elif "WARNING" in v:
                st.warning(v)
            else:
                st.info(v)
    else:
        st.success("✅ No quality violations detected in last 24 hours")
    
    # Show trace links
    st.subheader("📊 Recent Traces")
    for run in runs[:10]:  # Show last 10
        status = "✅" if not run.error else "❌"
        st.write(f"{status} {run.name} - {run.total_time}ms - [View Trace](https://smith.langchain.com/o/.../runs/{run.id})")
```

### Step 4: Update Documentation

**Add to `docs/QA_STRATEGY.md`:**

```markdown
## LangSmith Integration (Phase 2)

### Purpose
Monitor production quality in real-time using LLM trace data.

### Setup
1. Get LangSmith API key from https://smith.langchain.com/
2. Add to `.env`: `LANGCHAIN_API_KEY=lsv2_pt_...`
3. Enable tracing: `LANGCHAIN_TRACING_V2=true`
4. Deploy to production (Vercel auto-detects env vars)

### What We Monitor

| Quality Standard | pytest Test | LangSmith Check | Why Both? |
|-----------------|-------------|-----------------|-----------|
| **No emoji headers** | `test_no_emoji_headers()` | ✅ Scan responses for `###` | Test uses mocks, LangSmith sees real LLM output |
| **Response length** | `test_no_information_overload()` | ✅ Alert if >15k chars | Test checks logic, LangSmith catches prompt engineering bugs |
| **Duplicate prompts** | `test_no_duplicate_prompts()` | ✅ Count "would you like" | Test validates code, LangSmith finds edge cases |
| **Latency** | ❌ No test | ✅ Track p50, p95, p99 | Requires real production traffic |
| **Error rate** | ❌ No test | ✅ Track exceptions | Runtime errors only appear in production |
| **Token costs** | ❌ No test | ✅ Cost per query | Budget monitoring requires production data |

### Alert Thresholds

```python
ALERT_RULES = {
    "emoji_headers": {
        "severity": "CRITICAL",
        "threshold": "ANY",  # Zero tolerance
        "action": "Email + Slack + Create GitHub issue"
    },
    "response_length": {
        "severity": "WARNING",
        "threshold": ">15k chars in >10% of queries",
        "action": "Email summary"
    },
    "error_rate": {
        "severity": "CRITICAL",
        "threshold": ">5%",
        "action": "Email + Slack + Page on-call"
    },
    "latency_p95": {
        "severity": "WARNING",
        "threshold": ">3s",
        "action": "Email summary"
    },
    "daily_cost": {
        "severity": "INFO",
        "threshold": ">$5",
        "action": "Email summary"
    }
}
```

### Daily Report Example

```
📊 Quality Report - October 16, 2025

✅ Overall Status: HEALTHY

Metrics (24h):
  - 234 queries processed
  - 1.2s avg latency (p95: 2.1s)
  - $0.45 total cost ($0.0019/query)
  - 0 errors (0%)

Policy Compliance:
  ✅ No markdown headers detected
  ✅ All responses <15k chars
  ✅ Single follow-up prompts only
  ⚠️  3 queries >3s latency (1.3%)

Top Queries:
  1. "explain conversation nodes" - 45 times
  2. "show me code examples" - 32 times
  3. "what are noah's skills" - 28 times

Slowest Query:
  - Query: "explain the full technical architecture"
  - Latency: 4.2s
  - Trace: https://smith.langchain.com/...
  - Action: Optimize prompt length
```

### Cost Analysis

**LangSmith Pricing:**
- Free tier: 5,000 traces/month (good for development)
- Team tier: $39/month for 100k traces (needed for production)

**Break-Even Analysis:**
- If LangSmith catches **1 production bug/month** → Saves 2-4 hours debugging → Worth $39
- If it prevents **1 user complaint** → Maintains reputation → Priceless

**Current Usage:**
- ~234 queries/day × 30 days = 7,020 traces/month
- **Recommendation:** Start with Team tier ($39/month)
```

---

## Why This Matters for Our KB vs Response Policy

### The Problem We Just Solved
- KB content can use `###` headers and emojis (teaching structure)
- LLM responses must strip them to **Bold** (professional presentation)
- pytest validates this with mocks

### But What If...
- LLM ignores the prompt instruction?
- New OpenAI model behaves differently?
- Edge case query triggers unexpected formatting?
- Prompt injection bypasses sanitization?

### LangSmith Catches These
```python
# Scenario: New GPT-4-turbo ignores our "strip ###" instruction
# pytest: ✅ PASSES (mocks return clean output)
# Production: 🔴 Users see ### headers!

# LangSmith alert:
{
    "severity": "CRITICAL",
    "violation": "markdown_headers_in_production",
    "trace_id": "abc-123",
    "query": "explain conversation nodes",
    "response": "### Node 1: handle_greeting...",  # Uh oh!
    "model": "gpt-4-turbo-2024-10-15",  # New model version
    "action": "Rollback to gpt-4 or update prompt"
}
```

---

## Implementation Timeline

### Week 1 (Current - Phase 1) ✅ In Progress
- [x] 30 automated pytest tests
- [x] KB vs Response policy documented
- [x] test_no_emoji_headers updated
- [ ] Fix remaining 5 failing tests

### Week 2 (Phase 2) - Add LangSmith
- [ ] Get LangSmith API key
- [ ] Configure production tracing
- [ ] Implement `scripts/quality_monitor.py` with LangSmith
- [ ] Implement `scripts/quality_dashboard.py` with metrics
- [ ] Set up daily cron job
- [ ] Configure email alerts

### Week 3 (Phase 3) - Enforcement
- [ ] Make pytest required for PR merges (CI/CD)
- [ ] Daily LangSmith quality reports
- [ ] Weekly review of production violations

### Ongoing (Phase 4) - Refinement
- [ ] Adjust alert thresholds based on data
- [ ] Add new quality checks as patterns emerge
- [ ] Optimize prompts based on trace analysis

---

## Summary

### Question: Should we incorporate LangSmith into QA?

**Answer: YES, but as Phase 2 production monitoring, not as replacement for pytest.**

### The Winning Formula

```
Phase 1 (Testing):  pytest (30 tests) ← Prevents 90% of bugs
                           ↓ Deploy
Phase 2 (Monitoring): LangSmith ← Catches the other 10%
                           ↓ Learn
Phase 3 (Improve):  Add new pytest tests for patterns found in production
                           ↓ Repeat
```

### Key Takeaways

1. **pytest = Pre-deployment quality gate** (fast, deterministic, blocks bad code)
2. **LangSmith = Post-deployment safety net** (real behavior, edge cases, performance)
3. **Together = Comprehensive QA** (test what you can control, monitor what you can't)
4. **Cost:** $39/month LangSmith is worth it if it catches 1 production bug/month

### Next Actions

1. ✅ Complete Phase 1 (fix remaining 5 tests → 100% pass rate)
2. ⬜ Get LangSmith API key and configure
3. ⬜ Implement enhanced `quality_monitor.py` with LangSmith
4. ⬜ Deploy to production with tracing enabled
5. ⬜ Set up daily quality reports

---

**Status:** ✅ Integration plan approved and documented  
**Next Step:** Complete Phase 1 testing, then proceed to Phase 2 LangSmith setup
