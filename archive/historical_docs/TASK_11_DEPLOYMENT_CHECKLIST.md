# Task 11: Deployment & Testing Checklist

**Date**: October 16, 2025
**Feature**: Intelligent Resume Distribution System
**Environments**: Local (Streamlit) + Production (Vercel)

---

## Phase 1: Local Streamlit Testing ⏳ IN PROGRESS

### Pre-Testing Setup ✅

- [ ] Environment variables configured in `.env`
  - [ ] `OPENAI_API_KEY`
  - [ ] `SUPABASE_URL`
  - [ ] `SUPABASE_SERVICE_ROLE_KEY`
  - [ ] `RESEND_API_KEY`
  - [ ] `TWILIO_ACCOUNT_SID`
  - [ ] `TWILIO_AUTH_TOKEN`
  - [ ] `TWILIO_PHONE_NUMBER`
  - [ ] `NOAH_PHONE_NUMBER`

- [ ] All automated tests passing
  ```bash
  pytest tests/test_conversation_quality.py tests/test_documentation_alignment.py tests/test_resume_distribution.py -v
  # Expected: 70 passed, 1 skipped
  ```

- [ ] Streamlit server started
  ```bash
  streamlit run src/main.py
  # Server: http://localhost:8501
  ```

---

### Scenario 1: Mode 1 - Pure Education ⏳ PENDING

**Test**: Educational query with ZERO resume mentions

- [ ] Selected role: "Hiring Manager (technical)"
- [ ] Asked: "How do RAG systems work?"
- [ ] Response is 100% educational
- [ ] NO resume mentions anywhere
- [ ] Follow-up prompt is professional (not pushy)
- [ ] Code examples displayed (if applicable)
- [ ] NO aggressive CTAs

**Status**: ⏳ PENDING MANUAL TEST

---

### Scenario 2: Mode 2 - Hiring Signals ⏳ PENDING

**Test**: Query with hiring context triggers subtle mention

- [ ] Cleared chat (new conversation)
- [ ] Selected role: "Hiring Manager (technical)"
- [ ] Asked: "We're hiring a Senior GenAI Engineer. How do RAG systems work?"
- [ ] Response primarily educational (≥50%)
- [ ] Exactly ONE subtle availability mention
- [ ] Mention at end of response
- [ ] No pressure language
- [ ] Follow-up question: "What's the difference between FAISS and pgvector?"
- [ ] NO additional availability mention (already mentioned once)

**Status**: ⏳ PENDING MANUAL TEST

---

### Scenario 3: Mode 3 - Explicit Request ⏳ PENDING

**Test**: Full resume distribution flow with email/SMS

**Step 1 - Request Resume**:
- [ ] Cleared chat (new conversation)
- [ ] Selected role: "Hiring Manager (technical)"
- [ ] Asked: "Can I get your resume?"
- [ ] Immediate email collection prompt (no delays)
- [ ] NO qualification questions

**Step 2 - Provide Email**:
- [ ] Provided test email: `your-test-email@example.com`
- [ ] Name collection prompt shown
- [ ] Natural ask, not interrogative

**Step 3 - Provide Name**:
- [ ] Provided name: "I'm Jane Smith"
- [ ] Confirmation message shown
- [ ] Professional thank you

**Email Validation**:
- [ ] Email received at test address
- [ ] Subject: "Noah de la Calzada - Resume"
- [ ] Resume PDF attached
- [ ] Professional email body
- [ ] From address correct (via Resend)

**SMS Validation**:
- [ ] SMS received on Noah's phone
- [ ] Format: "Resume sent to [email] from [name]"
- [ ] Via Twilio
- [ ] Timestamp current

**Status**: ⏳ PENDING MANUAL TEST

---

### Scenario 4: Job Details Gathering ⏳ PENDING

**Test**: Post-resume conversational extraction

**Prerequisites**: Complete Scenario 3 first (same conversation)

- [ ] Continued from Scenario 3 (resume already sent)
- [ ] Asked: "How do you handle retrieval failures in RAG?"
- [ ] Educational response about retrieval failures
- [ ] Natural job details question at end
- [ ] Conversational tone (not interrogative)
- [ ] Provided: "I'm with Acme Corp, hiring for a Senior AI Engineer starting next month."
- [ ] System acknowledged: "Got it — Acme Corp, Senior AI Engineer."
- [ ] Correctly extracted: company, position, timeline
- [ ] System doesn't ask again (only once)

**Status**: ⏳ PENDING MANUAL TEST

---

### Scenario 5: Duplicate Prevention ⏳ PENDING

**Test**: Once-per-session enforcement

**Prerequisites**: Complete Scenario 3 (same conversation)

- [ ] Continued from Scenario 3
- [ ] Asked: "Can you send me your resume again?"
- [ ] Polite duplicate prevention message
- [ ] Reminds user of email where resume was sent
- [ ] Offers continued assistance
- [ ] NO second email received (checked inbox)
- [ ] NO second SMS received (checked phone)

**Status**: ⏳ PENDING MANUAL TEST

---

### Scenario 6: Cross-Role Consistency ⏳ PENDING

**Test**: Resume distribution only for hiring managers

- [ ] Cleared chat (new conversation)
- [ ] Selected role: "Software Developer"
- [ ] Asked: "We're hiring. How do RAG systems work?"
- [ ] Educational content only
- [ ] NO availability mention (even with "hiring" keyword)
- [ ] Developer-specific enhancements (code focus)

**Status**: ⏳ PENDING MANUAL TEST

---

### Scenario 7: Invalid Email Detection ⏳ PENDING

**Test**: Graceful error handling

- [ ] Cleared chat
- [ ] Selected role: "Hiring Manager (technical)"
- [ ] Asked: "Can I get your resume?"
- [ ] Provided invalid email: `not-an-email`
- [ ] Error message shown: "That doesn't look like a valid email..."
- [ ] Retry allowed
- [ ] NO resume sent with invalid email

**Status**: ⏳ PENDING MANUAL TEST

---

### Local Testing Summary

**Scenarios Completed**: 0/7
**Scenarios Passing**: 0/7
**Scenarios Failing**: 0/7
**Ready for Vercel**: ❌ NO (complete local testing first)

---

## Phase 2: Vercel Deployment ⏳ AWAITING LOCAL VALIDATION

### Pre-Deployment Checklist

- [ ] All local Streamlit tests passing (7/7 scenarios)
- [ ] All automated tests passing (70/71)
- [ ] Documentation updated (CHANGELOG.md current)
- [ ] Environment variables ready for Vercel
- [ ] Vercel CLI installed: `npm install -g vercel`
- [ ] Logged in to Vercel: `vercel login`

---

### Deployment Steps

**Step 1: Configure Environment Variables**

```bash
# Set each variable (paste value when prompted)
vercel env add OPENAI_API_KEY
vercel env add SUPABASE_URL
vercel env add SUPABASE_SERVICE_ROLE_KEY
vercel env add RESEND_API_KEY
vercel env add TWILIO_ACCOUNT_SID
vercel env add TWILIO_AUTH_TOKEN
vercel env add TWILIO_PHONE_NUMBER
vercel env add NOAH_PHONE_NUMBER

# Optional: LangSmith monitoring
vercel env add LANGSMITH_API_KEY
vercel env add LANGCHAIN_TRACING_V2
```

- [ ] All environment variables set
- [ ] Verified NO trailing newlines (test locally first)
- [ ] Listed variables: `vercel env ls`

---

**Step 2: Deploy to Production**

```bash
# Deploy
vercel --prod

# Note deployment URL
```

- [ ] Deployment succeeded
- [ ] Deployment URL: `https://_____________________________.vercel.app`
- [ ] No build errors
- [ ] All functions deployed

---

**Step 3: Health Check**

```bash
# Test basic connectivity
curl https://YOUR-DEPLOYMENT-URL.vercel.app/api/chat \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "role": "Software Developer", "session_id": "health-check"}'
```

- [ ] API responds (200 status)
- [ ] JSON response valid
- [ ] No errors in Vercel logs: `vercel logs --follow`

---

### Vercel Testing Scenarios

#### Scenario V1: Production API - Pure Education ⏳ PENDING

**Test**: Educational query via API

```bash
curl -X POST https://YOUR-DEPLOYMENT-URL.vercel.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain vector databases",
    "role": "Hiring Manager (technical)",
    "session_id": "vercel-test-1"
  }'
```

**Validation**:
- [ ] Response contains educational content
- [ ] NO resume mentions
- [ ] Response time <5s (check logs)
- [ ] No errors in Vercel function logs

**Status**: ⏳ PENDING DEPLOYMENT

---

#### Scenario V2: Production API - Hiring Signals ⏳ PENDING

**Test**: Subtle mention on production

```bash
curl -X POST https://YOUR-DEPLOYMENT-URL.vercel.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "We are hiring a Senior AI Engineer. How do RAG systems work?",
    "role": "Hiring Manager (technical)",
    "session_id": "vercel-test-2"
  }'
```

**Validation**:
- [ ] Educational content primary (≥50%)
- [ ] ONE subtle availability mention
- [ ] No aggressive CTAs
- [ ] hiring_signals logged to Supabase

**Status**: ⏳ PENDING DEPLOYMENT

---

#### Scenario V3: Production API - Resume Request ⏳ PENDING

**Test**: Full flow via API

**Step 1 - Request**:
```bash
SESSION_ID="vercel-test-3-$(date +%s)"

curl -X POST https://YOUR-DEPLOYMENT-URL.vercel.app/api/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"Can I get your resume?\",
    \"role\": \"Hiring Manager (technical)\",
    \"session_id\": \"$SESSION_ID\"
  }"
```

**Step 2 - Email**:
```bash
curl -X POST https://YOUR-DEPLOYMENT-URL.vercel.app/api/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"test-vercel@example.com\",
    \"role\": \"Hiring Manager (technical)\",
    \"session_id\": \"$SESSION_ID\"
  }"
```

**Step 3 - Name**:
```bash
curl -X POST https://YOUR-DEPLOYMENT-URL.vercel.app/api/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"I'm Jane Vercel\",
    \"role\": \"Hiring Manager (technical)\",
    \"session_id\": \"$SESSION_ID\"
  }"
```

**Validation**:
- [ ] Email received at test-vercel@example.com
- [ ] SMS received on Noah's phone
- [ ] Supabase logs show resume_sent
- [ ] Confirmation in API response

**Status**: ⏳ PENDING DEPLOYMENT

---

### Vercel Production Validation

**Functionality**:
- [ ] Mode 1 (Education): ZERO mentions ✅
- [ ] Mode 2 (Hiring Signals): ONE subtle mention ✅
- [ ] Mode 3 (Explicit Request): Resume sent ✅
- [ ] Job Details: Extraction working ✅
- [ ] Duplicate Prevention: Once per session ✅

**Production Services**:
- [ ] OpenAI API working
- [ ] Supabase logging working
- [ ] Resend email working
- [ ] Twilio SMS working
- [ ] pgvector retrieval working

**Performance**:
- [ ] Cold start <10s (Vercel limit)
- [ ] Warm response <3s average
- [ ] No timeout errors
- [ ] Memory usage <512MB

**Error Handling**:
- [ ] Invalid email returns error (not crash)
- [ ] Missing env vars handled gracefully
- [ ] Service failures return friendly messages
- [ ] All errors logged

**Observability**:
- [ ] Vercel logs showing requests
- [ ] Supabase analytics updated
- [ ] LangSmith tracing (if enabled)
- [ ] No sensitive data in logs

---

### Vercel Deployment Summary

**Scenarios Completed**: 0/3
**Scenarios Passing**: 0/3
**Scenarios Failing**: 0/3
**Production Ready**: ❌ NO (complete testing first)

---

## Final Approval Checklist

**Code Quality**:
- [x] All 71 automated tests passing (99% pass rate)
- [ ] All 7 local Streamlit scenarios passing
- [ ] All 3 Vercel API scenarios passing
- [ ] No console errors
- [ ] No exceptions in logs

**Documentation**:
- [x] Feature doc complete (INTELLIGENT_RESUME_DISTRIBUTION.md)
- [x] QA standards updated (QA_STRATEGY.md)
- [x] CHANGELOG.md current
- [x] Testing guide created (STREAMLIT_TESTING_GUIDE.md)
- [ ] Deployment checklist complete (this file)

**External Services**:
- [ ] Email via Resend working (local + production)
- [ ] SMS via Twilio working (local + production)
- [ ] Resume PDF sending correctly
- [ ] Job details in SMS (if gathered)
- [ ] Supabase logging all actions

**Quality Standards**:
- [ ] No pushy language anywhere
- [ ] Education remains primary (≥50%)
- [ ] Natural conversational tone
- [ ] Professional formatting
- [ ] No aggressive CTAs

**Deployment**:
- [ ] Vercel environment variables set
- [ ] Production deployment successful
- [ ] API endpoints responding
- [ ] Performance within limits
- [ ] Monitoring enabled

---

## Sign-Off

**Local Testing**:
- [ ] Tested by: ________________
- [ ] Date: ________________
- [ ] Result: ☐ PASS  ☐ FAIL  ☐ PENDING
- [ ] Notes: ________________________________________________

**Vercel Deployment**:
- [ ] Deployed by: ________________
- [ ] Date: ________________
- [ ] Result: ☐ PASS  ☐ FAIL  ☐ PENDING
- [ ] Notes: ________________________________________________

**Production Validation**:
- [ ] Validated by: ________________
- [ ] Date: ________________
- [ ] Result: ☐ PASS  ☐ FAIL  ☐ PENDING
- [ ] Notes: ________________________________________________

---

## Rollback Plan (If Needed)

**If critical issues found**:

1. **Rollback Vercel deployment**:
   ```bash
   vercel ls  # Get previous deployment URL
   vercel rollback https://previous-deployment-url.vercel.app
   ```

2. **Document issue**:
   - Issue description: ________________________________
   - Steps to reproduce: ________________________________
   - Expected behavior: ________________________________
   - Actual behavior: ________________________________

3. **Fix locally**:
   - Make changes
   - Run tests: `pytest tests/ -v`
   - Test in Streamlit: `streamlit run src/main.py`

4. **Redeploy when fixed**:
   ```bash
   vercel --prod
   ```

5. **Re-test failed scenario**

---

## Next Steps After Approval

**When all tests pass**:

1. [ ] Mark Task 11 as complete ✅
2. [ ] Update CHANGELOG.md: "Manual testing complete - production ready"
3. [ ] Create git tag: `git tag v1.0.0-resume-distribution`
4. [ ] Push to GitHub: `git push origin --tags`
5. [ ] Monitor production for 24 hours
6. [ ] Set up LangSmith alerts (Phase 2)
7. [ ] Document lessons learned
8. [ ] Celebrate! 🎉

---

**Status**: Task 11 in progress - Local testing phase
**Next Action**: Run Scenario 1 (Pure Education) in Streamlit
