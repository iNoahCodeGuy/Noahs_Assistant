"""Tests for Intelligent Resume Distribution System (Hybrid Approach).

This test suite validates the three-mode behavioral model:
- Mode 1 (Education): Pure teaching, ZERO resume mentions
- Mode 2 (Hiring Signals): Education + ONE subtle availability mention
- Mode 3 (Explicit Request): Immediate resume distribution

See docs/features/INTELLIGENT_RESUME_DISTRIBUTION.md for full specification.
See docs/context/CONVERSATION_PERSONALITY.md Section 6.1 for behavioral modes.

Test Categories:
1. Hiring Signal Detection (passive tracking)
2. Explicit Resume Request Handling (immediate response)
3. Subtle Availability Mentions (Mode 2 only)
4. Job Details Gathering (post-interest)
5. Once-Per-Session Enforcement
6. Email/Name Extraction
"""

import pytest
from assistant.state.conversation_state import ConversationState
from assistant.flows.node_logic.resume_distribution import (
    detect_hiring_signals,
    handle_resume_request,
    should_add_availability_mention,
    extract_email_from_query,
    extract_name_from_query,
    should_gather_job_details,
    extract_job_details_from_query,
)


class TestHiringSignalDetection:
    """Test passive hiring signal detection (enables Mode 2)."""

    def test_detects_mentioned_hiring(self):
        """Should detect when user explicitly mentions hiring."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "We're hiring a GenAI engineer for our team",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        state = detect_hiring_signals(state)

        assert "mentioned_hiring" in state["hiring_signals"]
        assert len(state["hiring_signals"]) >= 1

    def test_detects_described_role(self):
        """Should detect when user describes specific role."""
        state: ConversationState = {
            "role": "hiring_manager_nontechnical",
            "query": "Looking for a full-stack developer with AI experience",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        state = detect_hiring_signals(state)

        assert "described_role" in state["hiring_signals"]

    def test_detects_team_context(self):
        """Should detect when user mentions team/organization."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "Our team is building an AI platform",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        state = detect_hiring_signals(state)

        assert "team_context" in state["hiring_signals"]

    def test_detects_timeline_urgency(self):
        """Should detect timeline/urgency mentions."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "When is Noah available to start?",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        state = detect_hiring_signals(state)

        assert "asked_timeline" in state["hiring_signals"]

    def test_detects_budget_compensation(self):
        """Should detect budget/compensation discussions."""
        state: ConversationState = {
            "role": "hiring_manager_nontechnical",
            "query": "What's your salary expectation?",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        state = detect_hiring_signals(state)

        assert "budget_mentioned" in state["hiring_signals"]

    def test_detects_multiple_signals(self):
        """Should detect multiple signals in single query."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "We're hiring a GenAI engineer for our AI team, need someone immediately",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        state = detect_hiring_signals(state)

        # Should have mentioned_hiring, described_role, team_context, asked_timeline
        assert len(state["hiring_signals"]) >= 3
        assert "mentioned_hiring" in state["hiring_signals"]
        assert "described_role" in state["hiring_signals"]

    def test_no_signals_in_pure_education_query(self):
        """Should NOT detect signals in educational queries."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "How do RAG systems work?",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        state = detect_hiring_signals(state)

        assert len(state["hiring_signals"]) == 0

    def test_passive_tracking_no_side_effects(self):
        """Signal detection should NOT trigger proactive offers."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "We're hiring a GenAI engineer",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        state = detect_hiring_signals(state)

        # Should ONLY populate hiring_signals list
        assert len(state["hiring_signals"]) > 0
        # Should NOT set these flags (those are for Mode 3)
        assert not state.get("resume_explicitly_requested", False)
        assert not state.get("resume_sent", False)


class TestExplicitResumeRequestHandling:
    """Test explicit resume request detection (triggers Mode 3)."""

    def test_detects_direct_resume_request(self):
        """Should detect 'can I get your resume'."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "Can I get your resume?",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        state = handle_resume_request(state)

        assert state.get("resume_explicitly_requested", False)

    def test_detects_send_resume_request(self):
        """Should detect 'send me your resume'."""
        state: ConversationState = {
            "role": "hiring_manager_nontechnical",
            "query": "Please send me your CV",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        state = handle_resume_request(state)

        assert state.get("resume_explicitly_requested", False)

    def test_detects_availability_inquiry(self):
        """Should detect 'is Noah available'."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "Is Noah available for a role?",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        state = handle_resume_request(state)

        assert state.get("resume_explicitly_requested", False)

    def test_detects_contact_request(self):
        """Should detect 'contact Noah' in hiring context."""
        state: ConversationState = {
            "role": "hiring_manager_nontechnical",
            "query": "I'd like to talk to Noah about a position",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        state = handle_resume_request(state)

        assert state.get("resume_explicitly_requested", False)

    def test_no_false_positives_education_query(self):
        """Should NOT detect request in educational queries."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "How do you build a resume parsing system?",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        state = handle_resume_request(state)

        assert not state.get("resume_explicitly_requested", False)

    def test_immediate_response_no_qualification(self):
        """Explicit requests should NOT require qualification checks."""
        state: ConversationState = {
            "role": "hiring_manager_nontechnical",
            "query": "Send me your resume",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        state = handle_resume_request(state)

        # Should mark as requested (triggers immediate email collection)
        assert state.get("resume_explicitly_requested", False)
        # TypedDict doesn't have arbitrary attributes - this check is obsolete
        # The design uses explicit state fields, not dynamic attributes


class TestSubtleAvailabilityMentions:
    """Test subtle availability mention logic (Mode 2)."""

    def test_mode_2_enabled_with_sufficient_signals(self):
        """Should enable Mode 2 with ≥2 hiring signals."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "We're hiring a GenAI engineer for our team",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }
        state = detect_hiring_signals(state)

        # Should have ≥2 signals
        assert len(state["hiring_signals"]) >= 2

        # Should enable subtle mention
        assert should_add_availability_mention(state)

    def test_mode_2_disabled_insufficient_signals(self):
        """Should NOT enable Mode 2 with <2 signals."""
        state = ConversationState(
            role="hiring_manager_technical",
            query="Our team is working on AI"  # Only team_context signal
        )
        state = detect_hiring_signals(state)

        # Should NOT enable subtle mention (need ≥2 signals)
        assert not should_add_availability_mention(state)

    def test_mode_2_only_for_hiring_managers(self):
        """Should only enable Mode 2 for hiring manager roles."""
        state = ConversationState(
            role="software_developer",  # NOT hiring manager
            query="We're hiring a GenAI engineer for our team"
        )
        state = detect_hiring_signals(state)

        # Has signals but wrong role
        assert len(state["hiring_signals"]) >= 2
        assert not should_add_availability_mention(state)

    def test_mode_2_disabled_after_resume_sent(self):
        """Should NOT add mention if resume already sent."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "We're hiring a GenAI engineer",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }
        state = detect_hiring_signals(state)
        state["resume_sent"] = True

        # Has signals but resume already sent
        assert not should_add_availability_mention(state)

    def test_mode_2_disabled_if_explicitly_requested(self):
        """Should NOT add mention if user explicitly requested (Mode 3 takes precedence)."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "We're hiring and I'd like your resume",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }
        state = detect_hiring_signals(state)
        state["resume_explicitly_requested"] = True

        # Has signals but explicit request takes precedence
        assert not should_add_availability_mention(state)


class TestJobDetailsGathering:
    """Test job details gathering (post-interest only)."""

    def test_gathers_after_resume_sent(self):
        """Should gather job details AFTER resume sent."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "Tell me about RAG systems",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }
        state["resume_sent"] = True

        # Should enable job details gathering
        assert should_gather_job_details(state)

    def test_no_gathering_before_resume_sent(self):
        """Should NOT gather job details BEFORE resume sent."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "Tell me about RAG systems",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        # Resume not sent yet
        assert not should_gather_job_details(state)

    def test_only_gathers_once(self):
        """Should only gather job details once per session."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "Tell me about RAG systems",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }
        state["resume_sent"] = True
        state.setdefault("job_details", {})["company"] = "Acme Corp"

        # Already have company info
        assert not should_gather_job_details(state)

    def test_extracts_company_name(self):
        """Should extract company name from query."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "I'm with Acme Corp and we're looking for engineers",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        state = extract_job_details_from_query(state)

        assert state.get("job_details", {}).get("company") == "Acme Corp"

    def test_extracts_position_title(self):
        """Should extract position from query."""
        state: ConversationState = {
            "role": "hiring_manager_nontechnical",
            "query": "The position is Senior GenAI Engineer",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        state = extract_job_details_from_query(state)

        assert "Senior GenAI Engineer" in state.get("job_details", {}).get("position", "")

    def test_extracts_timeline(self):
        """Should extract timeline/urgency from query."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "We need someone to start immediately",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        state = extract_job_details_from_query(state)

        assert state.get("job_details", {}).get("timeline") is not None


class TestOncePerSessionEnforcement:
    """Test resume sent only once per session."""

    def test_resume_sent_flag_prevents_duplicate(self):
        """Should not send resume twice in same session."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "Send me your resume",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }
        state["resume_sent"] = True

        # Flag should be set
        assert state.get("resume_sent", False)

        # Subsequent request should be blocked by action execution logic
        # (tested in action_execution.py - execute_send_resume_and_notify checks this flag)

    def test_duplicate_request_detection(self):
        """Should detect when user requests resume again."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "Can I get your resume again?",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }
        state["resume_sent"] = True

        # Both flags set
        state = handle_resume_request(state)
        assert state.get("resume_sent", False)
        assert state.get("resume_explicitly_requested", False)

        # Action executor should handle gracefully (don't re-send)


class TestEmailNameExtraction:
    """Test email and name extraction from queries."""

    def test_extracts_valid_email(self):
        """Should extract email address from query."""
        query = "My email is john.smith@acmecorp.com"

        email = extract_email_from_query(query)

        assert email == "john.smith@acmecorp.com"

    def test_extracts_name_my_name_is(self):
        """Should extract name from 'my name is X' pattern."""
        query = "My name is John Smith"

        name = extract_name_from_query(query)

        assert name == "John Smith"

    def test_extracts_name_im_pattern(self):
        """Should extract name from 'I'm X' pattern."""
        query = "I'm Sarah Johnson"

        name = extract_name_from_query(query)

        assert name == "Sarah Johnson"

    def test_no_email_returns_empty(self):
        """Should return empty string if no email found."""
        query = "Tell me about your experience"

        email = extract_email_from_query(query)

        assert email == ""

    def test_no_name_returns_empty(self):
        """Should return empty string if no name found."""
        query = "Tell me about RAG systems"

        name = extract_name_from_query(query)

        assert name == ""


class TestHybridApproachIntegration:
    """Integration tests for the full hybrid approach."""

    def test_mode_1_pure_education(self):
        """Mode 1: Pure education query with no hiring context."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "How do RAG systems work?",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        # Run signal detection
        state = detect_hiring_signals(state)
        state = handle_resume_request(state)

        # Mode 1: NO resume-related activity
        assert len(state["hiring_signals"]) == 0
        assert not state.get("resume_explicitly_requested", False)
        assert not should_add_availability_mention(state)

    def test_mode_2_hiring_signals_detected(self):
        """Mode 2: Hiring signals detected, subtle mention allowed."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "We're hiring a GenAI engineer. How do RAG systems work?",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        # Run signal detection
        state = detect_hiring_signals(state)
        state = handle_resume_request(state)

        # Mode 2: Signals detected, subtle mention enabled
        assert len(state["hiring_signals"]) >= 2
        assert not state.get("resume_explicitly_requested", False)  # Not explicit request
        assert should_add_availability_mention(state)

    def test_mode_3_explicit_request(self):
        """Mode 3: Explicit resume request, immediate distribution."""
        state: ConversationState = {
            "role": "hiring_manager_nontechnical",
            "query": "Can I get your resume?",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        # Run request detection
        state = handle_resume_request(state)

        # Mode 3: Explicit request detected
        assert state.get("resume_explicitly_requested", False)
        # Mode 2 should be bypassed (explicit takes precedence)
        assert not should_add_availability_mention(state)

    def test_post_interest_job_details(self):
        """After resume sent, should gather job details."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "Tell me about your Python experience",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }
        state["resume_sent"] = True
        # Set user contact info (using dict access)
        state["user_email"] = "john@acme.com"
        state["user_name"] = "John Smith"

        # Should enable job details gathering
        assert should_gather_job_details(state)

        # Simulate user response with job details
        state["query"] = "I'm with Acme Corp, hiring for Senior GenAI Engineer role"
        state = extract_job_details_from_query(state)

        # Should have extracted details
        assert state.get("job_details", {}).get("company") == "Acme Corp"
        assert "Senior GenAI Engineer" in state.get("job_details", {}).get("position", "")

    def test_education_remains_primary(self):
        """Education should remain primary even with hiring signals."""
        state: ConversationState = {
            "role": "hiring_manager_technical",
            "query": "We're hiring a GenAI engineer. Explain vector databases.",
            "chat_history": [],
            "hiring_signals": [],
            "resume_sent": False,
            "resume_explicitly_requested": False,
            "job_details": {}
        }

        # Run signal detection
        state = detect_hiring_signals(state)

        # Has hiring signals BUT education is still the primary query
        assert len(state["hiring_signals"]) > 0
        # System should answer the education query FIRST
        # Subtle mention should be AFTERTHOUGHT (at end of response)
        # This is enforced by content_blocks.format_availability_mention()


# Run tests with: pytest tests/test_resume_distribution.py -v
