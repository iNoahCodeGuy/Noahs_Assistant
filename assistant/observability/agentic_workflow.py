"""LangGraph-based agentic workflow for Noah's AI Assistant.

This module implements a multi-step agentic flow using LangGraph:
    classify_intent → retrieve → answer → tool_call → log_eval

Why LangGraph:
- Orchestrates complex multi-step workflows
- Conditional routing based on intent
- Retry logic for failed operations
- State management across steps
- Visual workflow debugging

Architecture:
    User Query
        ↓
    [classify_intent] → Technical/Career/MMA/Fun/General
        ↓
    [retrieve] → Get relevant chunks from pgvector
        ↓
    [answer] → Generate response with OpenAI
        ↓
    [tool_call] → Optional: Search web, run code, etc.
        ↓
    [log_eval] → Evaluate and log metrics
        ↓
    Final Response

State Flow:
    {
        "query": "What are Noah's skills?",
        "intent": "technical",
        "retrieved_chunks": [...],
        "answer": "...",
        "tool_results": {...},
        "metrics": {...}
    }
"""

import logging
from typing import TypedDict, List, Dict, Any, Literal
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import LangGraph (graceful degradation)
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolExecutor
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not available. Install with: pip install langgraph")


class AgentState(TypedDict, total=False):
    """State passed between workflow nodes.

    Attributes:
        query: User query text
        role_mode: User role (Hiring Manager, Developer, etc.)
        intent: Classified intent (technical, career, mma, fun, general)
        retrieved_chunks: Chunks from pgvector retrieval
        retrieval_scores: Similarity scores
        answer: Generated response
        tool_calls: List of tool calls made
        tool_results: Results from tool execution
        metrics: Evaluation metrics
        error: Error message if any step fails
        retry_count: Number of retries for current step
    """
    query: str
    role_mode: str
    intent: str
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_scores: List[float]
    answer: str
    tool_calls: List[Dict[str, Any]]
    tool_results: Dict[str, Any]
    metrics: Dict[str, Any]
    error: str
    retry_count: int


def classify_intent(state: AgentState) -> AgentState:
    """Classify user query intent.

    Intent types:
    - technical: Programming, architecture, code questions
    - career: Work experience, projects, achievements
    - mma: MMA training, skills, competitions
    - fun: Personal interests, hobbies
    - general: Other questions

    Args:
        state: Current agent state with 'query' key

    Returns:
        Updated state with 'intent' key
    """
    from openai import OpenAI
    import os

    query = state['query']
    role_mode = state.get('role_mode', 'General')

    # Use simple heuristics for fast classification
    query_lower = query.lower()

    # Technical keywords
    if any(kw in query_lower for kw in [
        'code', 'programming', 'python', 'javascript', 'api', 'database',
        'architecture', 'tech stack', 'framework', 'library', 'algorithm'
    ]):
        intent = 'technical'

    # Career keywords
    elif any(kw in query_lower for kw in [
        'experience', 'work', 'job', 'project', 'achievement', 'education',
        'degree', 'company', 'team', 'leadership', 'management'
    ]):
        intent = 'career'

    # MMA keywords
    elif any(kw in query_lower for kw in [
        'mma', 'martial arts', 'fighting', 'boxing', 'wrestling', 'jiu-jitsu',
        'training', 'competition', 'fight', 'combat'
    ]):
        intent = 'mma'

    # Fun/Personal keywords
    elif any(kw in query_lower for kw in [
        'hobby', 'interest', 'favorite', 'fun', 'enjoy', 'like', 'love',
        'music', 'movie', 'book', 'game', 'sport'
    ]):
        intent = 'fun'

    # Default to general
    else:
        intent = 'general'

    logger.info(f"Classified intent: {intent} for query: {query[:50]}...")

    state['intent'] = intent
    return state


def retrieve(state: AgentState) -> AgentState:
    """Retrieve relevant chunks from pgvector.

    Uses role-aware retrieval with intent-based filtering.

    Args:
        state: Current agent state with 'query' and 'intent' keys

    Returns:
        Updated state with 'retrieved_chunks' and 'retrieval_scores' keys
    """
    from retrieval.pgvector_retriever import get_retriever

    query = state['query']
    intent = state.get('intent', 'general')

    try:
        # Get retriever with appropriate threshold
        threshold = 0.7 if intent in ['technical', 'career'] else 0.6
        retriever = get_retriever(similarity_threshold=threshold)

        # Retrieve chunks
        chunks = retriever.retrieve(query, top_k=5)

        # Extract scores
        scores = [c.get('score', 0.0) for c in chunks]

        logger.info(f"Retrieved {len(chunks)} chunks with avg score {sum(scores)/len(scores) if scores else 0:.3f}")

        state['retrieved_chunks'] = chunks
        state['retrieval_scores'] = scores

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        state['error'] = f"Retrieval error: {str(e)}"
        state['retrieved_chunks'] = []
        state['retrieval_scores'] = []

    return state


def answer(state: AgentState) -> AgentState:
    """Generate answer using OpenAI.

    Uses retrieved chunks as context.

    Args:
        state: Current agent state with 'query' and 'retrieved_chunks' keys

    Returns:
        Updated state with 'answer' key
    """
    from openai import OpenAI
    import os

    query = state['query']
    chunks = state.get('retrieved_chunks', [])
    intent = state.get('intent', 'general')
    role_mode = state.get('role_mode', 'General')

    # Build context from chunks
    context = "\n\n".join([
        f"[Chunk {i+1}] {chunk.get('content', '')}"
        for i, chunk in enumerate(chunks)
    ])

    # Build role-aware prompt
    if role_mode.startswith("Hiring Manager"):
        system_prompt = "You are Noah's AI assistant talking to a hiring manager. Be professional and highlight relevant experience."
    elif role_mode == "Software Developer":
        system_prompt = "You are Noah's AI assistant talking to a fellow developer. Be technical and include code examples when relevant."
    else:
        system_prompt = "You are Noah's AI assistant. Be helpful and conversational."

    prompt = f"""Context from Noah's knowledge base:
{context}

User question: {query}

Provide a helpful answer based on the context. If the context doesn't contain enough information, say so honestly."""

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        answer_text = response.choices[0].message.content

        logger.info(f"Generated answer: {answer_text[:100]}...")

        state['answer'] = answer_text

    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        state['error'] = f"Generation error: {str(e)}"
        state['answer'] = "I apologize, but I encountered an error generating a response."

    return state


def tool_call(state: AgentState) -> AgentState:
    """Execute optional tool calls.

    Tools might include:
    - Web search for recent information
    - Code execution for technical demos
    - Calendar lookup for availability

    Currently a placeholder for future extensions.

    Args:
        state: Current agent state

    Returns:
        Updated state with 'tool_results' key
    """
    # Placeholder - tools can be added here
    state['tool_calls'] = []
    state['tool_results'] = {}

    logger.debug("Tool call node (no tools executed)")

    return state


def log_eval(state: AgentState) -> AgentState:
    """Evaluate and log metrics.

    Uses LLM-as-judge for quality assessment.

    Args:
        state: Current agent state with all previous results

    Returns:
        Updated state with 'metrics' key
    """
    from observability.evaluators import evaluate_response
    from observability.metrics import calculate_retrieval_metrics

    query = state['query']
    chunks = state.get('retrieved_chunks', [])
    answer = state.get('answer', '')

    try:
        # Calculate retrieval metrics
        retrieval_metrics = calculate_retrieval_metrics(
            query=query,
            chunks=chunks,
            latency_ms=0  # Would track in production
        )

        # Evaluate response (sample 10% to save costs)
        import random
        if random.random() < 0.1:
            context = [c.get('content', '') for c in chunks]
            eval_metrics = evaluate_response(query, context, answer)

            state['metrics'] = {
                'retrieval': {
                    'num_chunks': retrieval_metrics.num_chunks,
                    'avg_similarity': retrieval_metrics.avg_similarity,
                },
                'evaluation': {
                    'faithfulness': eval_metrics.faithfulness_score,
                    'relevance': eval_metrics.relevance_score,
                    'quality': eval_metrics.answer_quality_score,
                    'overall': eval_metrics.overall_score(),
                }
            }

            logger.info(f"Evaluation metrics: {state['metrics']}")
        else:
            state['metrics'] = {
                'retrieval': {
                    'num_chunks': retrieval_metrics.num_chunks,
                    'avg_similarity': retrieval_metrics.avg_similarity,
                },
                'evaluation': None  # Not sampled
            }

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        state['metrics'] = {'error': str(e)}

    return state


def should_retry(state: AgentState) -> Literal["retrieve", "end"]:
    """Decide if we should retry retrieval.

    Retry if:
    - No chunks retrieved
    - Error occurred
    - Retry count < 2

    Args:
        state: Current agent state

    Returns:
        "retrieve" to retry, "end" to finish
    """
    chunks = state.get('retrieved_chunks', [])
    error = state.get('error')
    retry_count = state.get('retry_count', 0)

    if error and retry_count < 2:
        state['retry_count'] = retry_count + 1
        logger.warning(f"Retrying due to error (attempt {state['retry_count']})")
        return "retrieve"

    if not chunks and retry_count < 2:
        state['retry_count'] = retry_count + 1
        logger.warning(f"Retrying due to no chunks (attempt {state['retry_count']})")
        return "retrieve"

    return "end"


def create_agentic_workflow():
    """Create LangGraph workflow for agentic RAG.

    Returns:
        Compiled StateGraph workflow
    """
    if not LANGGRAPH_AVAILABLE:
        logger.error("LangGraph not installed. Cannot create workflow.")
        return None

    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("answer", answer)
    workflow.add_node("tool_call", tool_call)
    workflow.add_node("log_eval", log_eval)

    # Add edges
    workflow.set_entry_point("classify_intent")
    workflow.add_edge("classify_intent", "retrieve")
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("answer", "tool_call")
    workflow.add_edge("tool_call", "log_eval")

    # Conditional edge for retry logic
    workflow.add_conditional_edges(
        "log_eval",
        should_retry,
        {
            "retrieve": "retrieve",
            "end": END
        }
    )

    # Compile
    app = workflow.compile()

    logger.info("✅ Agentic workflow created")

    return app


def run_agentic_rag(query: str, role_mode: str = "General") -> Dict[str, Any]:
    """Run agentic RAG workflow.

    Args:
        query: User query
        role_mode: User role

    Returns:
        Final state with answer and metrics
    """
    workflow = create_agentic_workflow()

    if not workflow:
        # Fallback to simple RAG
        logger.warning("Using fallback RAG (LangGraph not available)")
        from core.rag_engine import RagEngine
        engine = RagEngine()
        answer = engine.generate_response(query)
        return {'query': query, 'answer': answer}

    # Initialize state
    initial_state: AgentState = {
        'query': query,
        'role_mode': role_mode,
        'retry_count': 0
    }

    # Run workflow
    try:
        result = workflow.invoke(initial_state)
        logger.info(f"Workflow completed: {result.get('intent')} intent")
        return result

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        return {
            'query': query,
            'answer': f"I apologize, but I encountered an error: {str(e)}",
            'error': str(e)
        }


# Initialize workflow on module load (if LangGraph available)
if LANGGRAPH_AVAILABLE:
    logger.info("✅ LangGraph agentic workflow available")
else:
    logger.info("❌ LangGraph not available (install with: pip install langgraph)")
