# Affinity Tracking System: Practical Examples

## Overview

The affinity tracking system uses **progressive decay scores** to dynamically adjust presentation style based on conversation context. This creates natural, adaptive responses that match user interest without explicit mode switching.

## Three-Layer Architecture

### Layer 1: Top-Level Affinities (Presentation Flags)

Two independent boolean flags control broad presentation strategy:

1. **`relate_to_enterprise`** (Boolean)
   - Enabled when `enterprise_relevance_score >= 2`
   - Controls: Enterprise tie-ins, business value framing, ROI mentions
   - Use case: Technical hiring managers who need both depth AND business context

2. **`show_technical_depth`** (Boolean)
   - Enabled when `technical_relevance_score >= 2`
   - Controls: Code snippets, architecture diagrams, implementation details
   - Use case: Developers and technical users drilling into specifics

### Layer 2: Relevance Scores (Session Memory)

Stored in `session_memory.persona_hints`:

```python
{
    "enterprise_relevance_score": 3,  # 0-4, threshold >= 2
    "technical_relevance_score": 4,   # 0-4, threshold >= 2
}
```

**Score mechanics:**
- **+2** when relevant keywords detected (capped at 4)
- **-1** when opposite keywords detected (floor at 0)
- Scores persist across turns, creating momentum

### Layer 3: Technical Subcategories (Retrieval Hints)

Fine-grained scores guide retrieval strategy and content selection:

```python
{
    "stack_depth_score": 2,           # Tech stack, dependencies
    "architecture_depth_score": 4,    # System design, patterns
    "data_pipeline_depth_score": 1,   # RAG, vectors, retrieval
    "state_management_depth_score": 3 # LangGraph nodes, state flow
}
```

**Usage:**
- **Retrieval source selection**: High `stack_depth_score` → prioritize `technical_kb.csv`
- **Code example selection**: High `state_management_depth_score` → show `ConversationState` manipulation
- **Diagram selection**: High `architecture_depth_score` → include system architecture diagram
- **Explanation depth**: High `data_pipeline_depth_score` → explain pgvector retrieval flow in detail

---

## Example Conversation 1: Technical Hiring Manager Journey

**User Context:** Technical hiring manager evaluating Noah for full-stack role. Starts with enterprise concerns, drills into architecture, then returns to business value.

### Turn 1: Enterprise Opening
**Query:** "How does this portfolio demonstrate enterprise readiness?"

**Score Updates:**
```python
enterprise_relevance_score: 0 → 2 (+2 for "enterprise")
technical_relevance_score: 0 → 0
stack_depth_score: 0 → 0
architecture_depth_score: 0 → 0
data_pipeline_depth_score: 0 → 0
state_management_depth_score: 0 → 0
```

**Flags:**
- `relate_to_enterprise`: **True** ✅ (score >= 2)
- `show_technical_depth`: **False** (score < 2)

**Response Style:**
- High-level overview with enterprise tie-ins
- Mentions: scalability, production-readiness, observability (LangSmith)
- NO code snippets (technical depth flag off)
- Emphasizes business outcomes

---

### Turn 2: Architecture Deep-Dive
**Query:** "Show me the system architecture and how the nodes are orchestrated"

**Score Updates:**
```python
enterprise_relevance_score: 2 → 1 (-1 for "how", "show me" = technical trigger)
technical_relevance_score: 0 → 2 (+2 for "architecture", "orchestrated")
stack_depth_score: 0 → 0 (no decay yet, no keywords)
architecture_depth_score: 0 → 2 (+2 for "architecture", "orchestrated")
data_pipeline_depth_score: 0 → 0
state_management_depth_score: 0 → 2 (+2 for "nodes", "orchestrated")
```

**Flags:**
- `relate_to_enterprise`: **False** (score < 2)
- `show_technical_depth`: **True** ✅ (score >= 2)

**Active Subcategories:** `["architecture_depth", "state_management_depth"]`

**Response Style:**
- Shows system architecture diagram (architecture_depth >= 2)
- Explains LangGraph node pipeline (state_management_depth >= 2)
- Includes code snippet of node definitions
- NO enterprise tie-ins (flag off)
- Deep technical explanation of orchestration

**Retrieval Strategy:**
- Pulls from `architecture_kb.csv` (high architecture_depth)
- Includes `conversation_flow.py` code examples (high state_management_depth)

---

### Turn 3: Data Pipeline Specifics
**Query:** "How does the pgvector retrieval work?"

**Score Updates:**
```python
enterprise_relevance_score: 1 → 0 (-1 for "how does" = technical)
technical_relevance_score: 2 → 4 (+2 for "how does", "retrieval", capped at 4)
stack_depth_score: 0 → 1 (-1 decay, not mentioned)
architecture_depth_score: 2 → 1 (-1 decay, not mentioned)
data_pipeline_depth_score: 0 → 2 (+2 for "pgvector", "retrieval")
state_management_depth_score: 2 → 1 (-1 decay, not mentioned)
```

**Flags:**
- `relate_to_enterprise`: **False** (score = 0)
- `show_technical_depth`: **True** ✅ (score = 4, maxed out)

**Active Subcategories:** `["data_pipeline_depth"]`

**Response Style:**
- Pure technical explanation (no business framing)
- Shows pgvector RPC call code
- Explains embedding generation, similarity search, re-ranking
- Includes latency metrics (technical context, not business ROI)

**Retrieval Strategy:**
- Prioritizes `data/` folder references, `pgvector_retriever.py` code
- Shows SQL RPC function for vector search

---

### Turn 4: Return to Business Value
**Query:** "What's the ROI of using pgvector over FAISS for enterprise deployment?"

**Score Updates:**
```python
enterprise_relevance_score: 0 → 2 (+2 for "ROI", "enterprise", "deployment")
technical_relevance_score: 4 → 3 (-1 for "ROI" = business keyword)
stack_depth_score: 1 → 0 (-1 decay)
architecture_depth_score: 1 → 0 (-1 decay)
data_pipeline_depth_score: 2 → 1 (-1 decay, mentioned but offset by business context)
state_management_depth_score: 1 → 0 (-1 decay)
```

**Flags:**
- `relate_to_enterprise`: **True** ✅ (score = 2)
- `show_technical_depth`: **True** ✅ (score = 3)

**Active Subcategories:** `[]` (all below threshold now)

**Response Style:**
- **Hybrid mode**: Both flags active!
- Technical explanation of pgvector vs FAISS (technical depth flag)
- WITH enterprise framing (enterprise flag):
  - "For multi-tenant deployments..."
  - "Centralized vector storage reduces operational complexity..."
  - "No need to sync FAISS indexes across instances..."
  - Cost comparison ($X/month Supabase vs self-hosted FAISS infra)

**Retrieval Strategy:**
- No specific subcategory prioritization (all decayed)
- Retrieves general comparison content

---

### Turn 5: Pure Business Question
**Query:** "How does this demonstrate leadership in technical decision-making?"

**Score Updates:**
```python
enterprise_relevance_score: 2 → 4 (+2 for "leadership", "decision-making", capped)
technical_relevance_score: 3 → 2 (-1 for "leadership", "demonstrate" = business context)
stack_depth_score: 0 → 0 (floor, no keywords)
architecture_depth_score: 0 → 0
data_pipeline_depth_score: 1 → 0 (-1 decay)
state_management_depth_score: 0 → 0
```

**Flags:**
- `relate_to_enterprise`: **True** ✅ (score = 4, maxed out)
- `show_technical_depth`: **True** ✅ (score = 2, barely holding)

**Active Subcategories:** `[]`

**Response Style:**
- Primarily enterprise-focused (high enterprise score)
- Light technical depth (score = 2, threshold):
  - Mentions technical choices (pgvector, LangSmith, Vercel)
  - But frames them as strategic decisions, not implementation details
- Emphasizes: architecture decisions, observability, production-readiness
- NO code snippets (technical depth at minimum threshold)

---

### Turn 6: Back to Code
**Query:** "Show me the ConversationState dataclass implementation"

**Score Updates:**
```python
enterprise_relevance_score: 4 → 3 (-1 for "show me", "implementation" = technical)
technical_relevance_score: 2 → 4 (+2 for "implementation", "dataclass")
stack_depth_score: 0 → 0
architecture_depth_score: 0 → 0
data_pipeline_depth_score: 0 → 0
state_management_depth_score: 0 → 2 (+2 for "ConversationState", "state")
```

**Flags:**
- `relate_to_enterprise`: **True** ✅ (score = 3)
- `show_technical_depth`: **True** ✅ (score = 4)

**Active Subcategories:** `["state_management_depth"]`

**Response Style:**
- Shows full `ConversationState` TypedDict code
- Explains state management pattern (state_management_depth active)
- Still includes enterprise tie-in (flag still on):
  - "This immutable state design enables audit trails..."
  - "State persistence supports multi-turn enterprise workflows..."

**Retrieval Strategy:**
- Pulls from `src/state/conversation_state.py`
- Includes related node examples (`conversation_nodes.py`)

---

## Example Conversation 2: Developer Exploring Stack

**User Context:** Software developer curious about implementation, no business concerns.

### Turn 1: General Question
**Query:** "What technologies did you use?"

**Score Updates:**
```python
enterprise_relevance_score: 0 → 0
technical_relevance_score: 0 → 0 (no strong technical keywords)
stack_depth_score: 0 → 2 (+2 for "technologies", "use")
```

**Flags:**
- `relate_to_enterprise`: **False**
- `show_technical_depth`: **False**

**Active Subcategories:** `["stack_depth"]`

**Response Style:**
- High-level list of technologies (Python, LangChain, Supabase, etc.)
- NO code snippets (technical depth flag off)
- NO business framing (enterprise flag off)
- Clean, concise answer

**Retrieval Strategy:**
- Prioritizes `technical_kb.csv` (stack_depth active)

---

### Turn 2: Dive into Stack
**Query:** "Why did you choose LangChain and Supabase over other options?"

**Score Updates:**
```python
enterprise_relevance_score: 0 → 0
technical_relevance_score: 0 → 2 (+2 for "why", "choose" = decision reasoning)
stack_depth_score: 2 → 4 (+2 for "LangChain", "Supabase", capped)
```

**Flags:**
- `relate_to_enterprise`: **False**
- `show_technical_depth`: **True** ✅

**Active Subcategories:** `["stack_depth"]`

**Response Style:**
- Technical comparison (LangChain vs LlamaIndex, Supabase vs Pinecone)
- Shows code examples of LangChain usage
- Explains trade-offs (developer experience, observability, cost)
- NO business ROI framing (flag off)

**Retrieval Strategy:**
- Heavy `technical_kb.csv` retrieval

---

### Turn 3: Shift to Architecture
**Query:** "How do the nodes communicate in the pipeline?"

**Score Updates:**
```python
enterprise_relevance_score: 0 → 0
technical_relevance_score: 2 → 4 (+2 for "how do", "pipeline")
stack_depth_score: 4 → 3 (-1 decay, not mentioned)
architecture_depth_score: 0 → 2 (+2 for "nodes", "pipeline", "communicate")
state_management_depth_score: 0 → 2 (+2 for "nodes", "pipeline")
```

**Flags:**
- `relate_to_enterprise`: **False**
- `show_technical_depth`: **True** ✅

**Active Subcategories:** `["stack_depth", "architecture_depth", "state_management_depth"]`

**Response Style:**
- Deep technical explanation of node orchestration
- Shows `conversation_flow.py` pipeline code
- Explains immutable state pattern (state_management_depth)
- Includes architecture diagram (architecture_depth)
- Mentions LangChain's LCEL pattern (stack_depth)

**Retrieval Strategy:**
- Pulls from `architecture_kb.csv` + `conversation_flow.py` + state management docs

---

## Key Benefits of This Approach

### 1. **Natural Conversations**
No explicit "switch to technical mode" commands. The system adapts based on what users ask about.

### 2. **Momentum & Memory**
Scores persist across turns, so users can build depth gradually without re-explaining their interest level each turn.

### 3. **Multi-Dimensional Context**
Can simultaneously be:
- Technical AND enterprise-focused (HM technical use case)
- Technical with stack focus AND architecture focus
- Pure business OR pure technical

### 4. **Graceful Transitions**
Scores decay naturally, allowing conversations to flow from technical → business → technical without jarring shifts.

### 5. **Retrieval Intelligence**
Subcategories inform what KB sources to prioritize, making retrieval more precise without complex routing logic.

### 6. **Analytics Gold**
Every turn logs affinity scores, enabling:
- Persona clustering (which users go deep on architecture?)
- Content gap analysis (are stack questions getting poor responses?)
- A/B testing presentation strategies

---

## Implementation Notes

### Where Flags Are Used

**`relate_to_enterprise` (Boolean):**
- `format_answer()`: Adds enterprise content blocks (governance, scale, ROI)
- `suggest_followups()`: Includes business-oriented follow-up questions
- `plan_actions()`: May trigger enterprise case study attachments

**`show_technical_depth` (Boolean):**
- `format_answer()`: Includes code snippets, architecture diagrams
- `display_controller()`: Enables "show code" toggles
- `depth_controller()`: Pushes depth_level to 3 (deep dive)

**Subcategory Scores (Dict in session_memory):**
- `retrieve_chunks()`: Prioritizes KB sources based on active subcategories
- `format_answer()`: Selects relevant code examples (e.g., state management code when `state_management_depth_score >= 2`)
- `plan_actions()`: May attach specific diagrams (architecture vs data flow)

### Why Not Separate Nodes?

Having 4 separate nodes (`update_stack_affinity`, `update_architecture_affinity`, etc.) would:
- **Bloat the pipeline**: 6+ affinity nodes instead of 2
- **Increase latency**: Each node adds 5-10ms overhead
- **Complicate state**: 8+ boolean flags instead of 2 + metadata
- **Harder to reason about**: "Why did we show this code?" → Check 6 flags vs 2 flags + metadata

**Nested approach keeps pipeline lean while enabling rich context.**

---

## Future Enhancements

### 1. **Temporal Decay Rates**
Currently all scores decay at -1/turn. Could implement:
- Fast decay for transient interests (stack mentions decay -2)
- Slow decay for deep interests (architecture decay -0.5)

### 2. **Cross-Category Boosting**
"How does LangChain orchestrate the RAG pipeline?" should boost:
- `stack_depth_score` (LangChain mention)
- `architecture_depth_score` (orchestrate = architecture)
- `data_pipeline_depth_score` (RAG pipeline)

Currently only applies to one category per turn.

### 3. **Confidence Scores**
Track how confident we are in affinity scores:
- High confidence: Multiple keyword matches over 3+ turns
- Low confidence: Single keyword match, could be noise

Use confidence to gate aggressive retrieval strategies.

### 4. **Persona Clustering**
After 100+ conversations, cluster users by affinity patterns:
- "Deep divers": High technical + high architecture scores
- "Stack shoppers": High stack + low architecture
- "Business evaluators": High enterprise + low technical

Use clusters to pre-warm retrieval strategies.

---

## Testing Strategy

### Unit Tests
Test score calculations in isolation:
```python
def test_stack_affinity_boost():
    state = {"query": "What's in your tech stack?", "session_memory": {}}
    result = update_technical_affinity(state)
    assert result["session_memory"]["persona_hints"]["stack_depth_score"] == 2
```

### Integration Tests
Test multi-turn conversations:
```python
def test_technical_to_business_transition():
    # Turn 1: Technical
    state = conversation_flow({"query": "Show me the code"})
    assert state["show_technical_depth"] == True
    
    # Turn 2: Business
    state = conversation_flow({"query": "What's the ROI?"}, state)
    assert state["relate_to_enterprise"] == True
    assert state["show_technical_depth"] == False  # Decayed below threshold
```

### LangSmith Evaluation
Create evaluation dataset with expected affinity progressions:
- Technical deep-dive → All subcategories should activate sequentially
- Business evaluation → Enterprise flag on, technical flag off
- Hybrid conversation → Both flags toggle dynamically

---

## Conclusion

The nested affinity tracking system provides **maximum presentation flexibility with minimal state complexity**. By keeping two top-level flags backed by rich metadata scores, we enable:

1. **Simple boolean checks** in formatting logic
2. **Rich contextual understanding** for retrieval
3. **Natural conversation flow** without mode switches
4. **Analytics-ready data** for optimization

This architecture scales better than separate nodes while maintaining clarity and debuggability.
