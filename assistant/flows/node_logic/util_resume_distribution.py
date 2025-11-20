"""Resume distribution nodes for intelligent hiring manager interactions.

This module implements the Intelligent Resume Distribution System (Hybrid Approach):
- Mode 1 (Education): Pure teaching, zero resume mentions
- Mode 2 (Hiring Signals): Education + subtle availability mention when signals detected
- Mode 3 (Explicit Request): Immediate resume distribution without qualification

See docs/features/INTELLIGENT_RESUME_DISTRIBUTION.md for full specification.
See docs/context/CONVERSATION_PERSONALITY.md Section 6.1 for behavioral modes.

Philosophy: "Portfolia needs to mainly offer education on generative AI applications;
she only goes deeper if the user asks for the resume to be sent."
"""

from __future__ import annotations
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from assistant.state.conversation_state import ConversationState


def detect_hiring_signals(state: ConversationState) -> ConversationState:
    """Passively detect hiring signals in user query (Mode 2 enabler).

    This function scans the query for indicators that the user is actively hiring
    or exploring talent, but does NOT trigger proactive offers. It only enables
    subtle availability mentions (ONE per conversation) in educational responses.

    Hiring Signals (from feature doc):
    - mentioned_hiring: "we're hiring", "looking for", "need someone"
    - described_role: "GenAI engineer", "ML specialist", specific title
    - team_context: "our team", "my team", organizational mention
    - asked_timeline: "when available", "start date", urgency mention
    - budget_mentioned: "salary range", "compensation", financial discussion

    Args:
        state: ConversationState with query to scan

    Returns:
        Updated state with hiring_signals list populated (passive tracking)

    Example:
        Query: "We're hiring a GenAI engineer for our AI team"
        Result: hiring_signals = ["mentioned_hiring", "described_role", "team_context"]
        Effect: Enables ONE subtle mention in educational response (Mode 2)
    """
    query_lower = state["query"].lower()
    hiring_signals = state.get("hiring_signals", [])

    # Pattern 1: Mentioned hiring explicitly
    hiring_patterns = [
        r'\b(hiring|looking for|need someone|recruiting|seeking)\b',
        r'\b(open position|job opening|role available)\b',
        r'\b(candidates|applicants)\b'
    ]
    if any(re.search(pattern, query_lower) for pattern in hiring_patterns):
        if "mentioned_hiring" not in hiring_signals:
            hiring_signals.append("mentioned_hiring")

    # Pattern 2: Described specific role
    role_patterns = [
        r'\b(engineer|developer|architect|specialist|lead)\b',
        r'\b(genai|gen ai|generative ai|ml|machine learning|ai)\b.*\b(engineer|developer|role)\b',
        r'\b(full.?stack|backend|frontend|data|software)\b.*\b(engineer|developer)\b'
    ]
    if any(re.search(pattern, query_lower) for pattern in role_patterns):
        if "described_role" not in hiring_signals:
            hiring_signals.append("described_role")

    # Pattern 3: Team context mentioned
    team_patterns = [
        r'\b(our team|my team|the team)\b',
        r'\b(organization|company|startup|enterprise)\b',
        r'\b(we are|we\'re)\b.*\b(building|creating|developing)\b'
    ]
    if any(re.search(pattern, query_lower) for pattern in team_patterns):
        if "team_context" not in hiring_signals:
            hiring_signals.append("team_context")

    # Pattern 4: Timeline/urgency mentioned
    timeline_patterns = [
        r'\b(when available|start date|immediately|asap)\b',
        r'\b(timeline|schedule|availability|available)\b',
        r'\b(notice period|can start|when.*start)\b'
    ]
    if any(re.search(pattern, query_lower) for pattern in timeline_patterns):
        if "asked_timeline" not in hiring_signals:
            hiring_signals.append("asked_timeline")

    # Pattern 5: Budget/compensation mentioned
    budget_patterns = [
        r'\b(salary|compensation|budget|rate)\b',
        r'\b(pay|payment|\$\d+k?|k salary)\b',
        r'\b(benefits|equity|stock)\b'
    ]
    if any(re.search(pattern, query_lower) for pattern in budget_patterns):
        if "budget_mentioned" not in hiring_signals:
            hiring_signals.append("budget_mentioned")

    # Return state with updated hiring_signals and strength metadata
    state["hiring_signals"] = hiring_signals
    strength = len(hiring_signals)
    state["hiring_signals_strength"] = strength
    state["hiring_signals_strong"] = strength >= 2
    return state


def handle_resume_request(state: ConversationState) -> ConversationState:
    """Detect explicit resume requests and trigger Mode 3 (immediate distribution).

    This function scans for explicit requests like:
    - "Can I get your resume?"
    - "Send me your CV"
    - "Is Noah available?"
    - "Share Noah's resume"

    If detected, sets state.resume_explicitly_requested = True, which:
    1. Triggers email collection flow (no qualification needed)
    2. Sends resume immediately after email provided
    3. Bypasses subtle mention logic (user asked directly)

    Args:
        state: ConversationState with query to scan

    Returns:
        Updated state with resume_explicitly_requested flag if pattern matched

    Example:
        Query: "Can I get your resume?"
        Result: resume_explicitly_requested = True
        Effect: Immediate email collection → resume send (Mode 3)
    """
    query_lower = state["query"].lower()

    # Pattern 1: Direct resume request
    resume_patterns = [
        r'\b(can i get|send me|share|forward|email me)\b.*\b(resume|cv|curriculum vitae)\b',
        r'\b(resume|cv)\b.*\b(available|access|view|see)\b',
        r'\byour resume\b',
        r'\bnoah\'s resume\b'
    ]
    if any(re.search(pattern, query_lower) for pattern in resume_patterns):
        state["resume_explicitly_requested"] = True
        return state

    # Pattern 2: Availability inquiry
    availability_patterns = [
        r'\bis noah available\b',
        r'\bcan noah\b.*\b(interview|meet|talk|discuss)\b',
        r'\bavailable for\b.*\b(hire|hiring|role|position|work)\b'
    ]
    if any(re.search(pattern, query_lower) for pattern in availability_patterns):
        state["resume_explicitly_requested"] = True
        return state

    # Pattern 3: Contact request
    contact_patterns = [
        r'\bcontact noah\b',
        r'\bconnect with noah\b',
        r'\btalk to noah\b.*\b(about|regarding)\b.*\b(role|position|opportunity)\b'
    ]
    if any(re.search(pattern, query_lower) for pattern in contact_patterns):
        state["resume_explicitly_requested"] = True
        return state

    return state


def should_add_availability_mention(state: ConversationState) -> bool:
    """Determine if subtle availability mention should be added (Mode 2 check).

    Conditions for adding mention:
    1. Role is hiring manager (technical or non-technical)
    2. Hiring signals detected (≥2 signals)
    3. Resume not already sent (once-per-session)
    4. Resume not explicitly requested (Mode 3 takes precedence)

    Returns:
        True if subtle availability mention should be added to educational response
    """
    # Check if hiring manager role
    role = state.get("role", "")
    is_hiring_manager = role in [
        "hiring_manager_technical",
        "hiring_manager_nontechnical"
    ]
    if not is_hiring_manager:
        return False

    # Check if enough signals detected
    hiring_signals = state.get("hiring_signals", [])
    if len(hiring_signals) < 2:
        return False

    # Don't add if already sent or explicitly requested
    resume_sent = state.get("resume_sent", False)
    resume_explicitly_requested = state.get("resume_explicitly_requested", False)
    if resume_sent or resume_explicitly_requested:
        return False

    return True


def extract_email_from_query(query: str) -> str:
    """Extract email address from user query if provided.

    Args:
        query: User query that may contain email

    Returns:
        Email address if found, empty string otherwise
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, query)
    return match.group(0) if match else ""


def extract_name_from_query(query: str) -> str:
    """Extract name from user query if provided (simple heuristic).

    Args:
        query: User query that may contain name

    Returns:
        Name if detected, empty string otherwise
    """
    # Simple pattern: "My name is X" or "I'm X" (case-insensitive)
    name_patterns = [
        r'(?:my name is|i\'m|this is)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)',
    ]

    for pattern in name_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)

    return ""


def should_gather_job_details(state: ConversationState) -> bool:
    """Determine if we should ask for job details (post-interest gathering).

    Job details are gathered AFTER the user has expressed interest in Noah
    (resume sent) to help Noah prepare for the conversation. This is done
    naturally and conversationally, not as an interrogation.

    Conditions for gathering:
    1. Resume has been sent (user expressed explicit interest)
    2. Job details not yet gathered (first time only)
    3. Not confessing crush (wrong context)

    Returns:
        True if we should naturally ask about company/position
    """
    # Only gather after resume sent (user expressed interest)
    if not state.get("resume_sent", False):
        return False

    # Only gather once (check if we already have company info)
    job_details = state.get("job_details", {})
    if job_details.get("company"):
        return False

    # Don't gather in confession context
    if state.get("role") == "confession":
        return False

    return True


def get_job_details_prompt() -> str:
    """Generate natural job details gathering prompt.

    This prompt is added to the LLM's system instructions AFTER resume sent
    to naturally ask about the hiring context. The tone is conversational,
    not interrogative.

    Returns:
        Additional system instruction for gathering job details
    """
    return """
IMPORTANT: The user has requested Noah's resume (sent successfully).
Now, to help Noah prepare for your conversation, naturally ask about the hiring context.

Add this to the END of your response (after answering their question):

"Just curious — what company are you with? And what's the position you're hiring for?"

Keep it conversational and friendly. If they don't want to share, that's completely fine.
The goal is to help Noah understand the context, not to interrogate.
"""


def extract_job_details_from_query(state: ConversationState) -> ConversationState:
    """Extract job details from user query after resume sent.

    This function uses simple heuristics to extract company name, position,
    and other details when the user responds to the job details prompt.

    Patterns detected:
    - Company: "I'm with [Company]", "at [Company]", "[Company] is"
    - Position: "[Title] role", "[Title] position", "hiring a/an [Title]"
    - Timeline: "immediately", "ASAP", "in X weeks/months"

    Args:
        state: ConversationState with query containing potential job details

    Returns:
        Updated state with job_details populated if extracted
    """
    query = state.get("query", "")
    if not query:
        return state  # No query to extract from
    job_details = state.get("job_details", {})

    # Extract company name (case-insensitive)
    company_patterns = [
        r'(?:i\'m with|with)\s+([A-Z][A-Za-z0-9\s&.]+?)(?:\s*,|\s+(?:and|hiring|looking))',
        r'(?:i work at|at)\s+([A-Z][A-Za-z0-9\s&.]+?)(?:\s*,|\s+(?:and|hiring))',
        r'([A-Z][A-Za-z0-9\s&.,]+?)\s+(?:is hiring|is looking)',
        r'(?:company is|our company is)\s+([A-Z][A-Za-z0-9\s&.,]+)',
        r'(?:for)\s+([A-Z][A-Za-z0-9\s&.]+?)(?:\s+(?:and|,|hiring))',
    ]
    for pattern in company_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            company = match.group(1).strip()
            # Clean up common false positives
            if len(company) > 2 and company not in ['The', 'A', 'An', 'We', 'They', 'I']:
                job_details["company"] = company
                break

    # Extract position/title
    position_patterns = [
        r'(?:position is|role is)\s+([A-Z][A-Za-z\s]+(?:Engineer|Developer|Manager|Specialist|Architect)?)',
        r'(?:hiring (?:a|an|for))\s+([A-Z][A-Za-z\s]+(?:Engineer|Developer|Manager|Specialist|Architect)?)',
        r'([A-Z][A-Za-z\s]+?)\s+(?:role|position)(?:\s|,|\.|!|\?)',
        r'(?:for (?:the|a|an))\s+([A-Z][A-Za-z\s]+?)(?:\s+(?:role|position))',
    ]
    for pattern in position_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            position = match.group(1).strip()
            if len(position) > 2:
                job_details["position"] = position
                break

    # Extract timeline
    timeline_patterns = [
        r'\b(immediately|ASAP|as soon as possible)\b',
        r'(?:start|begin|available)\s+(?:in\s+)?(\d+\s+(?:weeks?|months?))',
        r'(?:within|in)\s+(\d+\s+(?:weeks?|months?))',
    ]
    for pattern in timeline_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            timeline = match.group(0).strip()
            job_details["timeline"] = timeline
            break

    # Return state with updated job_details
    state["job_details"] = job_details
    return state
