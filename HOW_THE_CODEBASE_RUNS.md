# How The Codebase Actually Runs - Complete Execution Flow

## Table of Contents
1. [Startup Sequence](#startup-sequence)
2. [Request Flow (Local Dev)](#request-flow-local-dev)
3. [Request Flow (Production)](#request-flow-production)
4. [LangGraph Execution Pipeline](#langgraph-execution-pipeline)
5. [State Management](#state-management)
6. [Data Layer](#data-layer)
7. [External Services](#external-services)

---

## Startup Sequence

### Local Development (Streamlit)

**Command**: `streamlit run src/main.py`

```
1. Python interpreter loads src/main.py
   └─> Imports trigger module initialization cascade

2. Configuration Loading (src/config/supabase_config.py)
   ├─> Read environment variables (.env file)
   │   ├─ OPENAI_API_KEY
   │   ├─ SUPABASE_URL
   │   ├─ SUPABASE_SERVICE_ROLE_KEY
   │   ├─ LANGSMITH_API_KEY (optional)
   │   └─ TWILIO_*/RESEND_* (optional)
   │
   └─> Create supabase_settings singleton
       └─> Detects runtime: is_vercel=False, is_production=False

3. Service Initialization (Factory Pattern)
   ├─> get_supabase_client() → PostgreSQL + pgvector connection
   ├─> get_resend_service() → Email service (or None if keys missing)
   ├─> get_twilio_service() → SMS service (or None if keys missing)
   └─> get_storage_service() → Supabase Storage for resume uploads

4. LangSmith Tracing Setup (src/observability/langsmith_tracer.py)
   ├─> If LANGSMITH_API_KEY exists:
   │   └─> Initialize LangSmith client
   │       └─> Project: "noahs-ai-assistant"
   └─> Else: Tracing disabled (degraded mode)

5. RAG Engine Initialization (src/core/rag_engine.py)
   ├─> PgVectorRetriever connects to Supabase
   │   ├─ Embedding function: text-embedding-3-small (1536 dims)
   │   ├─ Similarity threshold: 0.3
   │   └─ RPC function: match_documents (pgvector search)
   │
   └─> ResponseGenerator initialized
       ├─ Default model: gpt-4o-mini
       └─ Temperature: 0.7 (configurable)

6. Streamlit UI Rendering (src/main.py)
   ├─> Page config: wide layout, custom theme
   ├─> Session state initialization
   │   ├─ chat_history = []
   │   ├─ selected_role = None
   │   ├─ session_id = uuid4()
   │   └─ rag_engine = RagEngine()
   │
   └─> Render UI components
       ├─ Role selection dropdown (5 roles)
       ├─ Chat message container
       └─ Input box + Send button

7. Server Ready
   └─> Listening on http://localhost:8501
       └─> WebSocket connection for real-time updates
```

### Production (Vercel Serverless)

**Trigger**: HTTP request to `/api/chat`

```
1. Vercel Runtime Starts Python Container
   ├─> Cold start: ~2-4 seconds
   └─> Warm start: ~50-200ms

2. Import api/chat.py
   └─> Triggers same initialization cascade as above
       └─> But: supabase_settings.is_vercel=True

3. HTTP Handler Ready
   └─> Awaiting POST requests at /api/chat
       └─> Timeout: 10 seconds max (Vercel limit)
```

---

## Request Flow (Local Dev)

### User Interaction Example

```
USER: Selects "Hiring Manager (technical)" → Types "Show me the full tech stack" → Clicks Send
```

### Execution Trace

```python
# 1. STREAMLIT EVENT HANDLER (src/main.py ~line 250)
if st.button("Send") or user_input:
    # Store user message in session state
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    # Call conversation flow
    with st.spinner("Thinking..."):
        result = run_conversation_flow(
            query=user_input,
            role=st.session_state.selected_role,
            session_id=st.session_state.session_id,
            chat_history=st.session_state.chat_history,
            rag_engine=st.session_state.rag_engine
        )

    # Display assistant response
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result.answer
    })
    st.rerun()  # Refresh UI with new messages
```

```python
# 2. CONVERSATION FLOW ORCHESTRATOR (src/flows/conversation_flow.py ~line 45)
def run_conversation_flow(
    query: str,
    role: str,
    session_id: str,
    chat_history: List[Dict],
    rag_engine: RagEngine
) -> ConversationState:

    # Initialize conversation state
    state = ConversationState(
        session_id=session_id,
        role=role,
        query=query,
        chat_history=chat_history,
        turn_number=len([m for m in chat_history if m["role"] == "user"])
    )

    # Execute pipeline (18 nodes in sequence)
    # Each node receives state, mutates it, returns updated state
    pipeline = [
        initialize_conversation_state,  # Stage 0: Session setup
        handle_greeting,                # Stage 1: Greeting detection
        classify_role_mode,             # Stage 2: Role normalization
        classify_intent,                # Stage 2: Query type detection
        extract_entities,               # Stage 2: Entity extraction
        check_clarification_needed,     # Stage 3: Ambiguity detection
        compose_retrieval_query,        # Stage 3: Query enhancement
        set_depth_level,                # Stage 3: Response depth
        set_display_preferences,        # Stage 3: Display toggles
        retrieve,                       # Stage 4: pgvector search
        generate_draft,                 # Stage 5: LLM generation
        plan_actions,                   # Stage 6: Action planning
        format_answer,                  # Stage 6: Layout formatting
        execute_actions,                # Stage 6: Side effects
        log_and_notify,                 # Stage 6: Analytics
    ]

    # Sequential execution with LangSmith tracing
    for node in pipeline:
        with trace_node(node.__name__):
            state = node(state, rag_engine)

    return state
```

### Stage-by-Stage Breakdown

#### **Stage 0: Session Management**

```python
# src/flows/node_logic/stage0_session_management.py
def initialize_conversation_state(state: ConversationState, rag_engine: RagEngine):
    """Set up session tracking and context."""

    # Ensure session ID exists
    if not state.session_id:
        state.session_id = str(uuid4())

    # Calculate turn number
    user_turns = len([m for m in state.chat_history if m["role"] == "user"])
    state.turn_number = user_turns

    # Initialize empty collections
    state.entities = {}
    state.pending_actions = []
    state.retrieved_chunks = []

    logger.info(f"🚀 Session {state.session_id[:8]}... | Turn {state.turn_number}")
    return state
```

**Output**: `state` with session tracking initialized

---

#### **Stage 1: Greeting Detection**

```python
# src/flows/node_logic/stage1_greetings.py
def handle_greeting(state: ConversationState, rag_engine: RagEngine):
    """Detect and respond to greetings immediately."""

    query_lower = state.query.lower()

    # Check for greeting patterns
    greeting_patterns = ["hi", "hello", "hey", "good morning", "good afternoon"]
    is_greeting = any(pattern in query_lower for pattern in greeting_patterns)

    if is_greeting and len(query_lower.split()) <= 3:
        # Short greeting → immediate response
        role = state.role or "Just looking around"
        state.answer = get_role_welcome_message(role)
        state.query_type = "greeting"
        state.skip_pipeline = True  # Short-circuit remaining nodes
        logger.info("✅ Greeting detected, returning welcome message")

    return state
```

**Output**: If greeting detected, `state.answer` is set and `skip_pipeline=True`

---

#### **Stage 2a: Role Classification**

```python
# src/flows/node_logic/stage2_role_routing.py
def classify_role_mode(state: ConversationState, rag_engine: RagEngine):
    """Normalize role to internal role_mode."""

    # Map UI role names to internal role_mode
    role_map = {
        "Hiring Manager (nontechnical)": "hiring_manager_nontechnical",
        "Hiring Manager (technical)": "hiring_manager_technical",
        "Software Developer": "software_developer",
        "Just looking around": "casual_visitor",
        "Looking to confess crush": "confession"
    }

    state.role_mode = role_map.get(state.role, "casual_visitor")
    logger.info(f"🎯 Role mode: {state.role_mode}")

    return state
```

**Output**: `state.role_mode` set to internal identifier

---

#### **Stage 2b: Intent Classification**

```python
# src/flows/node_logic/stage2_query_classification.py
def classify_intent(state: ConversationState, rag_engine: RagEngine):
    """Classify query type: technical, career, menu_selection, analytics, etc."""

    query_lower = state.query.lower()

    # Menu selection detection (for technical hiring managers)
    menu_match = re.match(r'^(\d)\.?\s*$', state.query.strip())
    if menu_match and state.role_mode == "hiring_manager_technical":
        menu_choice = menu_match.group(1)
        state.query_type = "menu_selection"
        state.menu_choice = menu_choice
        logger.info(f"✅ Menu choice SET: menu_choice={menu_choice}")
        return state

    # Technical keywords
    tech_keywords = ["code", "architecture", "technical", "implementation", "system"]
    if any(kw in query_lower for kw in tech_keywords):
        state.query_type = "technical"
        return state

    # Career keywords
    career_keywords = ["experience", "background", "worked", "role", "job"]
    if any(kw in query_lower for kw in career_keywords):
        state.query_type = "career"
        return state

    # Default
    state.query_type = "general"
    return state
```

**Output**: `state.query_type` and optionally `state.menu_choice`

**Real Example**:
```
Input: "1"
Output: state.query_type="menu_selection", state.menu_choice="1"
```

---

#### **Stage 2c: Entity Extraction**

```python
# src/flows/node_logic/stage2_entity_extraction.py
def extract_entities(state: ConversationState, rag_engine: RagEngine):
    """Extract companies, roles, contact info, menu context."""

    entities = {}

    # Menu context mapping (for menu selections)
    if state.query_type == "menu_selection" and state.menu_choice:
        menu_context_map = {
            "1": "full_tech_stack",
            "2": "orchestration_layer",
            "3": "data_layer",
            "4": "production_deployment"
        }
        entities["menu_selection"] = state.menu_choice
        entities["menu_context"] = menu_context_map.get(state.menu_choice, "unknown")
        logger.info(f"✅ Extracted menu entity: {entities}")

    # Company extraction (hiring signals)
    company_patterns = [r'\b([A-Z][a-z]+ (?:Inc|Corp|LLC))\b', r'\bfrom ([A-Z][a-z]+)\b']
    for pattern in company_patterns:
        match = re.search(pattern, state.query)
        if match:
            entities["company"] = match.group(1)

    # Email extraction
    email_match = re.search(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b', state.query)
    if email_match:
        entities["email"] = email_match.group(0)

    state.entities = entities
    return state
```

**Output**: `state.entities` dictionary

**Real Example**:
```
Input: menu_choice="1"
Output: state.entities = {"menu_selection": "1", "menu_context": "full_tech_stack"}
```

---

#### **Stage 3: Query Enhancement**

```python
# src/flows/node_logic/stage3_query_composition.py
def compose_retrieval_query(state: ConversationState, rag_engine: RagEngine):
    """Enhance query for better retrieval."""

    # For menu selections, expand to full query
    if state.query_type == "menu_selection" and state.entities.get("menu_context"):
        context = state.entities["menu_context"]

        query_map = {
            "full_tech_stack": "[hiring_manager_technical] full technology stack architecture with 5 layers: frontend, backend, data, observability, deployment",
            "orchestration_layer": "[hiring_manager_technical] LangGraph orchestration and conversation pipeline",
            # ... etc
        }

        state.composed_query = query_map.get(context, state.query)
        logger.info(f"🔍 Composed query: {state.composed_query[:80]}...")
    else:
        # Use original query with role prefix
        state.composed_query = f"[{state.role_mode}] {state.query}"

    return state
```

**Output**: `state.composed_query` (enhanced version of original query)

**Real Example**:
```
Input: query="1", menu_context="full_tech_stack"
Output: state.composed_query = "[hiring_manager_technical] full technology stack architecture with 5 layers..."
```

---

#### **Stage 4: Retrieval (pgvector Search)**

```python
# src/flows/node_logic/stage4_retrieval_nodes.py
def retrieve(state: ConversationState, rag_engine: RagEngine):
    """Semantic search in Supabase pgvector."""

    query = state.composed_query or state.query

    # Call pgvector retriever
    results = rag_engine.retriever.retrieve(
        query=query,
        top_k=4,  # Get top 4 most similar chunks
        threshold=0.3  # Minimum cosine similarity
    )

    state.retrieved_chunks = results.get("chunks", [])
    state.retrieval_scores = results.get("scores", [])

    logger.info(f"Retrieved {len(state.retrieved_chunks)} chunks, avg_similarity={results.get('avg_similarity', 0):.3f}")

    return state
```

**Under the Hood (Supabase RPC Call)**:

```sql
-- Executed in PostgreSQL via RPC
SELECT
    id,
    content,
    metadata,
    1 - (embedding <=> query_embedding) as similarity
FROM documents
WHERE 1 - (embedding <=> query_embedding) > 0.3
ORDER BY embedding <=> query_embedding
LIMIT 4;
```

**Process**:
1. `query` → OpenAI embeddings API → `query_embedding` (1536-dim vector)
2. Supabase RPC `match_documents(query_embedding, threshold, limit)`
3. pgvector's `<=>` operator computes cosine distance
4. Returns top 4 chunks with similarity > 0.3

**Output**: `state.retrieved_chunks` contains 4 text chunks with metadata

**Real Example**:
```json
state.retrieved_chunks = [
    {
        "content": "Frontend: Streamlit UI with role selection and session management. Next.js migration planned for production...",
        "metadata": {"source": "technical_kb.csv", "category": "architecture"},
        "similarity": 0.581
    },
    {
        "content": "Backend: LangGraph StateGraph with 18 nodes orchestrating the conversation flow...",
        "metadata": {"source": "technical_kb.csv", "category": "architecture"},
        "similarity": 0.580
    },
    // ... 2 more chunks
]
```

---

#### **Stage 5: Generation (LLM Call)**

```python
# src/flows/node_logic/stage5_generation_nodes.py
def generate_draft(state: ConversationState, rag_engine: RagEngine):
    """Call OpenAI to generate response with retrieved context."""

    # Build context from retrieved chunks
    context_str = "\n\n".join([
        f"[Chunk {i+1} - similarity: {chunk.get('similarity', 0):.3f}]\n{chunk['content']}"
        for i, chunk in enumerate(state.retrieved_chunks)
    ])

    # Special handling for menu option 1 (full tech stack)
    extra_instructions = []
    if (state.query_type == "menu_selection" and
        state.menu_choice == "1" and
        state.role_mode == "hiring_manager_technical"):

        # Extract layer-specific facts to prevent verbatim copying
        layer_outline = _extract_layer_outline(state.retrieved_chunks)

        extra_instructions.append(
            "CRITICAL INSTRUCTION - FULL TECH STACK WALKTHROUGH:\n"
            "⚠️ DO NOT copy text verbatim. Synthesize into cohesive narrative.\n"
            "📋 REQUIRED OUTPUT FORMAT:\n"
            "[Opening paragraph]\n"
            "**Frontend Layer:** [2-3 sentences: " + layer_outline.get("frontend", "") + "]\n"
            "**Backend/Orchestration Layer:** [2-3 sentences: " + layer_outline.get("backend", "") + "]\n"
            "**Data Layer:** [2-3 sentences: " + layer_outline.get("data", "") + "]\n"
            "**Observability Layer:** [2-3 sentences: " + layer_outline.get("observability", "") + "]\n"
            "**Deployment Layer:** [2-3 sentences: " + layer_outline.get("deployment", "") + "]\n"
            "[Closing paragraph]\n"
            "🎯 TARGET: 300-350 words"
        )

    # Model selection based on query complexity
    model = _select_model_for_task(state)  # Usually gpt-4o-mini

    # Call OpenAI Chat Completions API
    answer = rag_engine.response_generator.generate_contextual_response(
        query=state.query,
        context=context_str,
        role=state.role,
        chat_history=state.chat_history,
        extra_instructions="\n".join(extra_instructions),
        model_name=model
    )

    # Validate output quality for menu option 1
    if state.query_type == "menu_selection" and state.menu_choice == "1":
        required_layers = [
            "**Frontend Layer:**",
            "**Backend/Orchestration Layer:**",
            "**Data Layer:**",
            "**Observability Layer:**",
            "**Deployment Layer:**"
        ]
        missing_layers = [layer for layer in required_layers if layer not in answer]
        word_count = len(answer.split())

        if missing_layers or word_count < 250:
            state.generation_quality_warning = f"Missing layers: {missing_layers}"
            logger.warning(f"⚠️ Generated answer missing {len(missing_layers)} layers")

    state.draft_answer = answer
    state.answer = answer

    return state
```

**OpenAI API Call**:
```python
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.7,
    max_tokens=1500
)
answer = response.choices[0].message.content
```

**Output**: `state.draft_answer` contains LLM-generated response (300-350 words for menu 1)

**Real Example**:
```
Input: retrieved_chunks (4 chunks about tech stack)
Output: state.draft_answer = "This AI assistant is built on a modern, scalable five-layer architecture:\n\n**Frontend Layer:** Streamlit provides the conversational UI..."
```

---

#### **Stage 6a: Action Planning**

```python
# src/flows/node_logic/stage6_action_planning.py
def plan_actions(state: ConversationState, rag_engine: RagEngine):
    """Decide what enrichments to add to response."""

    actions = []

    # For hiring manager + menu option 1 → include architecture code
    if (state.query_type == "menu_selection" and
        state.menu_choice == "1" and
        state.role_mode == "hiring_manager_technical"):

        actions.append({
            "type": "include_code_reference",
            "context": "architecture"
        })
        actions.append({
            "type": "include_adaptation_diagram"
        })
        logger.info("✅ Adding architecture code reference for menu option 1")

    # After 2+ turns, offer resume to hiring managers
    if (state.role_mode in ["hiring_manager_nontechnical", "hiring_manager_technical"] and
        state.turn_number >= 2):
        actions.append({
            "type": "offer_resume_prompt"
        })

    state.pending_actions = actions
    return state
```

**Output**: `state.pending_actions` list of enrichment actions

**Real Example**:
```python
state.pending_actions = [
    {"type": "include_code_reference", "context": "architecture"},
    {"type": "include_adaptation_diagram"}
]
```

---

#### **Stage 6b: Formatting**

```python
# src/flows/node_logic/stage6_formatting_nodes.py
def format_answer(state: ConversationState, rag_engine: RagEngine):
    """Structure answer with headings, toggles, and action blocks."""

    # Check if retry needed (for menu option 1 quality issues)
    base_answer = _retry_generation_if_insufficient(state, rag_engine)
    if not base_answer:
        base_answer = state.draft_answer

    # Split answer and sources
    body_text, sources_text = _split_answer_and_sources(base_answer)

    # Build formatted sections
    sections = []

    # ❌ CURRENT ISSUE: Always adds Teaching Takeaways
    sections.append("**Teaching Takeaways**")
    sections.extend(_summarize_answer(body_text, state.depth_level))

    # ❌ CURRENT ISSUE: Always wraps in <details> toggle
    details_block = content_blocks.render_block(
        "Full Walkthrough",
        body_text,
        summary="Expand for the detailed explanation",
        open_by_default=state.depth_level >= 2
    )
    sections.append("")
    sections.append(details_block)

    # Execute planned actions
    for action in state.pending_actions:
        if action["type"] == "include_code_reference":
            # Add code snippet
            code_block = _get_code_for_context(action["context"])
            sections.append("")
            sections.append(code_block)

        elif action["type"] == "offer_resume_prompt":
            # Add resume CTA
            sections.append("")
            sections.append("---\n\nWould you like me to email you Noah's resume and LinkedIn profile?")

    # Add sources
    if sources_text:
        sections.append("")
        sections.append(content_blocks.render_block(
            "Sources",
            sources_text,
            summary="See where this info came from",
            open_by_default=False
        ))

    # Assemble final answer
    state.answer = "\n".join(sections)

    return state
```

**Output**: `state.answer` with full formatting

**Real Example (Current - with issues)**:
```markdown
**Teaching Takeaways**
- Five-layer architecture: frontend, backend, data, observability, deployment
- Modern stack with Streamlit, LangGraph, Supabase
- Production-ready with Vercel serverless

<details open>
<summary>Expand for the detailed explanation</summary>

**Full Walkthrough**
This AI assistant is built on a modern, scalable five-layer architecture:

**Frontend Layer:** Streamlit provides...
**Backend/Orchestration Layer:** LangGraph StateGraph...
**Data Layer:** Supabase PostgreSQL...
**Observability Layer:** LangSmith...
**Deployment Layer:** Vercel...

</details>

```python
# src/flows/conversation_flow.py
def create_conversation_graph():
    """LangGraph StateGraph with 18 nodes"""
    # ...
```

---

Would you like me to email you Noah's resume and LinkedIn profile?

<details>
<summary>See where this info came from</summary>

**Sources**
- docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md
- data/technical_kb.csv

</details>
```

---

#### **Stage 6c: Action Execution**

```python
# src/flows/node_logic/stage6_action_execution.py
def execute_actions(state: ConversationState, rag_engine: RagEngine):
    """Perform side effects: email, SMS, storage."""

    for action in state.pending_actions:
        if action["type"] == "send_resume_email":
            email = action.get("email")
            if email:
                resend = get_resend_service()
                if resend:
                    result = resend.send_resume_email(
                        to_email=email,
                        candidate_name="Noah",
                        resume_url="https://..."
                    )
                    logger.info(f"📧 Resume email sent to {email}: {result}")

        elif action["type"] == "send_hiring_manager_alert":
            company = action.get("company")
            contact = action.get("contact")
            twilio = get_twilio_service()
            if twilio:
                result = twilio.send_hiring_manager_alert(
                    company_name=company,
                    contact_name=contact,
                    interest_level="high"
                )
                logger.info(f"📱 SMS alert sent: {result}")

    return state
```

**Output**: Side effects executed (emails sent, SMS sent, storage uploads)

---

#### **Stage 6d: Logging & Analytics**

```python
# src/flows/node_logic/stage7_logging_nodes.py
def log_and_notify(state: ConversationState, rag_engine: RagEngine):
    """Persist interaction to Supabase analytics."""

    analytics_data = {
        "session_id": state.session_id,
        "role": state.role,
        "query": state.query,
        "query_type": state.query_type,
        "answer": state.answer,
        "retrieval_count": len(state.retrieved_chunks),
        "model_used": state.analytics_metadata.get("selected_model"),
        "turn_number": state.turn_number,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Insert into Supabase
    supabase = get_supabase_client()
    result = supabase.table("user_interactions").insert(analytics_data).execute()

    logger.info(f"Logged interaction for session {state.session_id}, message_id: {result.data[0]['id']}")

    return state
```

**Database Operation**:
```sql
INSERT INTO user_interactions (
    session_id, role, query, query_type, answer,
    retrieval_count, model_used, turn_number, timestamp
) VALUES (
    '550e8400-...',
    'Hiring Manager (technical)',
    '1',
    'menu_selection',
    'This AI assistant is built on...',
    4,
    'gpt-4o-mini',
    2,
    '2025-11-10T22:14:19Z'
);
```

**Output**: Analytics persisted to database

---

### Complete Timeline (Menu Option 1 Example)

```
T+0ms    User clicks "Send" in Streamlit UI
T+1ms    run_conversation_flow() called
T+2ms    initialize_conversation_state → session tracking
T+3ms    handle_greeting → Not a greeting, continue
T+4ms    classify_role_mode → "hiring_manager_technical"
T+5ms    classify_intent → query_type="menu_selection", menu_choice="1"
T+6ms    extract_entities → {"menu_selection": "1", "menu_context": "full_tech_stack"}
T+7ms    compose_retrieval_query → Enhanced query with 5 layer keywords
T+8ms    retrieve → Calls Supabase pgvector
  T+8ms    → OpenAI embeddings API: query → 1536-dim vector (50ms)
  T+58ms   → Supabase RPC match_documents (42ms)
  T+100ms  → Returns 4 chunks
T+100ms  generate_draft → Calls OpenAI Chat Completions
  T+100ms  → Builds prompt with context + special menu 1 instructions
  T+110ms  → OpenAI API call (gpt-4o-mini, 350 tokens) (1800ms)
  T+1910ms → Validates output (checks for 5 layers, word count)
  T+1912ms → Sets generation_quality_warning if issues detected
T+1912ms plan_actions → Adds "include_code_reference", "include_adaptation_diagram"
T+1913ms format_answer → Structures answer with toggles
  T+1913ms → Check if retry needed (generation_quality_warning)
  T+1914ms → If warning: Call OpenAI again with reinforced prompt (1500ms)
  T+3414ms → Assemble sections with Teaching Takeaways, HTML toggles, code blocks
T+3415ms execute_actions → No side effects for menu selection
T+3416ms log_and_notify → Insert into Supabase analytics (15ms)
T+3431ms Return final state to Streamlit
T+3432ms Streamlit rerenders UI with new message
T+3450ms User sees response in chat

Total: ~3.5 seconds (2 OpenAI calls if retry triggered)
```

---

## Request Flow (Production)

### Vercel Serverless Endpoint

**File**: `api/chat.py`

```python
from http.server import BaseHTTPRequestHandler
import json
from src.flows.conversation_flow import run_conversation_flow
from src.core.rag_engine import RagEngine

# Module-level singleton (persists across warm invocations)
_rag_engine = None

def get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RagEngine()
    return _rag_engine

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Parse request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        data = json.loads(body)

        # Extract parameters
        query = data.get('query')
        role = data.get('role')
        session_id = data.get('session_id')
        chat_history = data.get('chat_history', [])

        # Run conversation flow (same as local)
        rag_engine = get_rag_engine()
        result = run_conversation_flow(
            query=query,
            role=role,
            session_id=session_id,
            chat_history=chat_history,
            rag_engine=rag_engine
        )

        # Return JSON response
        self._send_json(200, {
            'answer': result.answer,
            'query_type': result.query_type,
            'session_id': result.session_id
        })

    def _send_json(self, status, data):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
```

### Client Request (Next.js/React)

```typescript
// app/components/hooks/useChat.ts
const sendMessage = async (content: string) => {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: content,
      role: selectedRole,
      session_id: sessionId,
      chat_history: messages
    })
  });

  const data = await response.json();
  return data.answer;
};
```

---

## LangGraph Execution Pipeline

### StateGraph Definition

```python
# src/flows/conversation_flow.py
from langgraph.graph import StateGraph

def create_conversation_graph() -> StateGraph:
    """Creates the 18-node conversation pipeline."""

    graph = StateGraph(ConversationState)

    # Add all nodes
    graph.add_node("initialize", initialize_conversation_state)
    graph.add_node("greetings", handle_greeting)
    graph.add_node("classify_role", classify_role_mode)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("extract_entities", extract_entities)
    graph.add_node("clarification", check_clarification_needed)
    graph.add_node("compose_query", compose_retrieval_query)
    graph.add_node("set_depth", set_depth_level)
    graph.add_node("set_display", set_display_preferences)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate_draft)
    graph.add_node("plan_actions", plan_actions)
    graph.add_node("format", format_answer)
    graph.add_node("execute", execute_actions)
    graph.add_node("log", log_and_notify)

    # Define edges (execution order)
    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "greetings")
    graph.add_edge("greetings", "classify_role")
    graph.add_edge("classify_role", "classify_intent")
    graph.add_edge("classify_intent", "extract_entities")
    graph.add_edge("extract_entities", "clarification")
    graph.add_edge("clarification", "compose_query")
    graph.add_edge("compose_query", "set_depth")
    graph.add_edge("set_depth", "set_display")
    graph.add_edge("set_display", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "plan_actions")
    graph.add_edge("plan_actions", "format")
    graph.add_edge("format", "execute")
    graph.add_edge("execute", "log")
    graph.set_finish_point("log")

    return graph.compile()
```

### LangGraph Dev Server

**Command**: `langgraph dev`

```
1. Reads langgraph.json configuration
   ├─> graph_path: "src/flows/conversation_flow.py:create_conversation_graph"
   └─> Imports and compiles graph

2. Starts local API server on port 2024
   └─> HTTP endpoints:
       ├─ POST /threads/{thread_id}/runs/stream
       ├─ GET /threads/{thread_id}/state
       └─ GET /graphs/{graph_id}

3. LangGraph Studio connects via WebSocket
   └─> Real-time visualization of node execution

4. Watches for file changes
   └─> Auto-reloads graph on save
```

---

## State Management

### ConversationState TypedDict

```python
# src/state/conversation_state.py
from typing import TypedDict, List, Dict, Any, Optional

class ConversationState(TypedDict, total=False):
    # Session tracking
    session_id: str
    turn_number: int

    # User context
    role: str                    # UI role name
    role_mode: str              # Internal role identifier
    query: str                  # Original user query
    chat_history: List[Dict[str, str]]

    # Query classification
    query_type: str             # technical, career, menu_selection, etc.
    menu_choice: str            # "1", "2", "3", "4" for menus
    entities: Dict[str, Any]    # Extracted companies, emails, menu context

    # Query enhancement
    composed_query: str         # Enhanced query for retrieval
    depth_level: int           # 1-5 (affects toggles, code display)
    layout_variant: str        # "mixed", "technical", "narrative"

    # Retrieval
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_scores: List[float]

    # Generation
    draft_answer: str          # Raw LLM output
    answer: str               # Final formatted output
    generation_quality_warning: Optional[str]

    # Actions
    pending_actions: List[Dict[str, Any]]

    # Analytics
    analytics_metadata: Dict[str, Any]
```

### State Update Pattern

**❌ WRONG (direct mutation)**:
```python
def my_node(state):
    state["query_type"] = "technical"  # Mutates original
    return state  # Returns mutated object
```

**✅ CORRECT (partial update)**:
```python
def my_node(state):
    update = {}
    update["query_type"] = "technical"
    update["depth_level"] = 2
    return update  # LangGraph merges into state
```

### Why Partial Updates?

LangGraph uses **reducer pattern**:
1. Node returns `update` dict
2. LangGraph merges `update` into current `state`
3. Next node receives merged `state`

**Critical**: Fields MUST be declared in `ConversationState` TypedDict for LangGraph to propagate them. Undeclared fields are silently dropped.

---

## Data Layer

### Supabase Tables

```sql
-- user_interactions: Analytics for every query/response
CREATE TABLE user_interactions (
    id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    role TEXT,
    query TEXT,
    query_type TEXT,
    answer TEXT,
    retrieval_count INT,
    model_used TEXT,
    turn_number INT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- documents: Vector store for semantic search
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1536),  -- OpenAI text-embedding-3-small
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create pgvector index for fast similarity search
CREATE INDEX documents_embedding_idx ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- confessions: Crush confession messages
CREATE TABLE confessions (
    id SERIAL PRIMARY KEY,
    message TEXT,
    submitted_at TIMESTAMPTZ DEFAULT NOW()
);

-- sms_logs: Twilio SMS delivery tracking
CREATE TABLE sms_logs (
    id SERIAL PRIMARY KEY,
    to_phone TEXT,
    message TEXT,
    status TEXT,
    sent_at TIMESTAMPTZ DEFAULT NOW()
);
```

### RPC Functions

```sql
-- match_documents: Vector similarity search
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(1536),
    match_threshold FLOAT,
    match_count INT
)
RETURNS TABLE (
    id INT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) AS similarity
    FROM documents
    WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
    ORDER BY documents.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
```

### Data Migration

**Script**: `scripts/migrate_data_to_supabase.py`

```python
# Run once to populate documents table from CSV files

import pandas as pd
from openai import OpenAI
from supabase import create_client

# Load data
career_df = pd.read_csv("data/career_kb.csv")
technical_df = pd.read_csv("data/technical_kb.csv")

# Generate embeddings
openai_client = OpenAI()
for _, row in career_df.iterrows():
    text = row["question"] + " " + row["answer"]

    # Call OpenAI embeddings API
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = response.data[0].embedding  # 1536 floats

    # Insert into Supabase
    supabase.table("documents").insert({
        "content": text,
        "embedding": embedding,
        "metadata": {
            "source": "career_kb.csv",
            "category": row["category"]
        }
    }).execute()
```

---

## External Services

### OpenAI (Required)

**Models Used**:
- `text-embedding-3-small`: Embeddings (1536 dims, $0.02/1M tokens)
- `gpt-4o-mini`: Default chat (simple queries, $0.15/1M input tokens)
- `gpt-4o`: Complex reasoning (menu option 1 retries, $2.50/1M input tokens)
- `o1-preview`: Advanced reasoning (future use, $15/1M input tokens)

**API Calls Per Query**:
1. Embedding generation (1 call): ~50ms, 100 tokens
2. Chat completion (1-2 calls): ~1800ms, 350 tokens output

**Cost Per Query**: ~$0.0001 (embedding) + ~$0.0005 (chat) = **$0.0006 total**

### LangSmith (Optional)

**Purpose**: Distributed tracing and debugging

**Configuration**:
```bash
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=noahs-ai-assistant
```

**Traces Captured**:
- Node execution times
- LLM prompts and responses
- Retrieval queries and results
- Error stack traces

**Access**: https://smith.langchain.com/

### Resend (Optional)

**Purpose**: Transactional emails (resume delivery)

**Usage**:
```python
resend.send_resume_email(
    to_email="hiring@company.com",
    candidate_name="Noah",
    resume_url="https://storage.supabase.co/..."
)
```

### Twilio (Optional)

**Purpose**: SMS alerts for high-value hiring signals

**Usage**:
```python
twilio.send_hiring_manager_alert(
    company_name="Google",
    contact_name="Jane Smith",
    interest_level="high"
)
```

### Supabase Storage (Optional)

**Purpose**: Resume PDF hosting

**Bucket**: `resumes`

**Usage**:
```python
storage.upload_resume(
    file_path="data/noah_resume.pdf",
    file_name="noah_resume_2025.pdf"
)
# Returns: https://storage.supabase.co/object/public/resumes/noah_resume_2025.pdf
```

---

## Performance Characteristics

### Latency Breakdown (Menu Option 1)

| Stage | Operation | Time | Percentage |
|-------|-----------|------|------------|
| Session init | State setup | 1ms | 0.03% |
| Classification | Regex + rules | 4ms | 0.11% |
| Entity extraction | Regex | 2ms | 0.06% |
| Query composition | String building | 1ms | 0.03% |
| **Retrieval** | **Embedding + pgvector** | **100ms** | **2.9%** |
| **Generation (1st)** | **OpenAI gpt-4o-mini** | **1800ms** | **52%** |
| Validation | String checks | 2ms | 0.06% |
| **Generation (retry)** | **OpenAI gpt-4o** | **1500ms** | **43%** |
| Action planning | Rule evaluation | 1ms | 0.03% |
| Formatting | String assembly | 2ms | 0.06% |
| Action execution | None (menu 1) | 0ms | 0% |
| Analytics logging | Supabase insert | 15ms | 0.43% |
| **TOTAL** | | **3431ms** | **100%** |

**Optimization Opportunities**:
1. Cache embeddings for common queries (saves 50ms)
2. Use streaming for generation (perceived latency improvement)
3. Parallelize embedding + initial classification (saves 20ms)
4. Implement prompt caching (saves 500ms on retries)

### Throughput

**Concurrent Requests**: Limited by OpenAI rate limits
- Free tier: 3 RPM (requests per minute)
- Tier 1: 500 RPM
- Tier 4: 10,000 RPM

**Bottleneck**: OpenAI API calls (1800ms each)

**Scalability**: Stateless design allows horizontal scaling
- Vercel: Auto-scales to 100+ concurrent functions
- Supabase: 500 connections per pool

---

## Error Handling & Degraded Mode

### Graceful Degradation

```python
# src/config/supabase_config.py
class SupabaseSettings:
    def validate_supabase(self):
        """Check if Supabase is configured."""
        if not self.supabase_config.url or not self.supabase_config.service_role_key:
            logger.warning("⚠️ Supabase not configured - analytics disabled")
            return False
        return True
```

```python
# src/flows/node_logic/stage7_logging_nodes.py
def log_and_notify(state, rag_engine):
    """Log interaction - fails gracefully if Supabase unavailable."""
    try:
        supabase = get_supabase_client()
        if supabase:
            supabase.table("user_interactions").insert({...}).execute()
    except Exception as e:
        logger.error(f"Analytics logging failed: {e}")
        # Continue - don't block user response

    return state
```

### Error Recovery

**OpenAI API Failures**:
```python
try:
    answer = openai.chat.completions.create(...)
except openai.APIError as e:
    logger.error(f"OpenAI API error: {e}")
    answer = "I'm having trouble generating a response right now. Please try again."
```

**Supabase Connection Failures**:
```python
try:
    results = supabase.rpc("match_documents", {...}).execute()
except Exception as e:
    logger.error(f"Retrieval failed: {e}")
    results = {"chunks": []}  # Proceed with empty retrieval
```

---

## Debugging & Observability

### Logging

**Format**: Emoji-prefixed structured logs
```
2025-11-10T22:14:15.848Z [INFO] ✅ Menu choice SET: menu_choice=1, query_type=menu_selection
2025-11-10T22:14:15.855Z [INFO] 🔍 Entity extraction START: menu_choice=1, query_type=menu_selection
2025-11-10T22:14:19.465Z [INFO] 📝 Layer outline extracted for synthesis: ['frontend', 'backend', 'data', 'observability', 'deployment']
2025-11-10T22:14:19.466Z [WARN] ⚠️ Generated answer missing 5 layers: ['**Frontend Layer:**', ...]
```

### LangSmith Traces

**Trace Hierarchy**:
```
ConversationFlow (3431ms)
├─ initialize_conversation_state (1ms)
├─ handle_greeting (1ms)
├─ classify_role_mode (1ms)
├─ classify_intent (1ms)
├─ extract_entities (1ms)
├─ retrieve (100ms)
│  ├─ OpenAI Embeddings (50ms)
│  └─ Supabase RPC (42ms)
├─ generate_draft (1812ms)
│  └─ OpenAI Chat Completion (1800ms)
├─ format_answer (1515ms)
│  └─ _retry_generation_if_insufficient (1500ms)
│     └─ OpenAI Chat Completion (1500ms)
└─ log_and_notify (15ms)
```

### LangGraph Studio

**Visual Debugging**:
1. Shows node execution order
2. Displays state changes at each node
3. Highlights failed nodes in red
4. Allows manual state editing and replay

---

## Summary

**Key Takeaways**:

1. **Dual runtime**: Streamlit (local dev) + Vercel serverless (production)
2. **18-node pipeline**: Sequential execution via LangGraph StateGraph
3. **State propagation**: Partial update pattern, fields must be in TypedDict
4. **Bottleneck**: OpenAI API calls (1800ms per generation)
5. **Data layer**: Supabase pgvector for semantic search, analytics persistence
6. **Graceful degradation**: Optional services fail silently, core flow continues
7. **Observability**: LangSmith tracing + emoji-prefixed logs for debugging

**Total latency for menu option 1**: 3.4 seconds (with retry)
**Cost per query**: $0.0006
**Scalability**: Stateless, horizontally scalable to 100+ concurrent users
