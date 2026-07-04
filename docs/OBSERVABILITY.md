# 🔍 Observability & Monitoring Guide

## Overview

Noah's AI Assistant includes comprehensive observability features for monitoring RAG (Retrieval Augmented Generation) performance, tracking metrics, and evaluating response quality.

The default runtime executes the LangGraph-style `run_conversation_flow`, so traces often show nodes for classification, retrieval, generation, action planning, and notifications. Legacy RoleRouter traces appear only when `LANGGRAPH_FLOW_ENABLED=false` for troubleshooting.

## 📊 Features

### 1. **LangSmith Integration**
- 🔎 **Trace LLM Calls**: Visual traces of every Claude and embedding interaction
- 📈 **Token Usage Tracking**: Monitor costs and optimize prompts
- ⏱️ **Latency Monitoring**: Identify performance bottlenecks
- 🐛 **Error Debugging**: Detailed error traces with context

### 2. **Retrieval Metrics**
- **Similarity Scores**: Track pgvector retrieval quality
- **Chunk Count**: Monitor how many chunks are retrieved
- **Relevance**: Evaluate if retrieved context matches query
- **Latency**: Measure retrieval performance

### 3. **Generation Metrics**
- **Token Usage**: Track prompt and completion tokens
- **Cost Estimation**: Calculate Anthropic + OpenAI API costs
- **Response Latency**: Monitor generation time
- **Model Performance**: Compare generation (Sonnet) vs classification (Haiku) calls

### 4. **LLM-as-Judge Evaluation**
- **Faithfulness**: Does response cite retrieved context?
- **Relevance**: Are retrieved chunks relevant to query?
- **Answer Quality**: Is response helpful and accurate?
- **Groundedness**: Are claims supported by evidence?

### 5. **LangGraph Agentic Workflow** (Optional)
- Multi-step agentic flow with conditional routing
- Intent classification (technical/career/mma/fun)
- Retry logic for failed operations
- Tool calling capabilities

## 🚀 Quick Start

### Prerequisites
```bash
pip install langsmith langgraph
```

### 1. Configure LangSmith

Add to your `.env` file:
```bash
# LangSmith Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_YOUR_API_KEY_HERE
LANGCHAIN_PROJECT=noahs-ai-assistant
```

Get your API key from: https://smith.langchain.com/

### 2. Verify Installation

```python
from observability import initialize_langsmith

if initialize_langsmith():
    print("✅ LangSmith enabled")
else:
    print("❌ LangSmith disabled")
```

### 3. Use Traced RAG

```python
from core.rag_engine import RagEngine

# Automatic tracing - no code changes needed!
engine = RagEngine()
response = engine.generate_response("What are Noah's skills?")
```

Check LangSmith dashboard to see the trace 🎉

## 📖 Usage Examples

### Basic Retrieval with Metrics

```python
from observability import calculate_retrieval_metrics

# Retrieve chunks
chunks = retriever.retrieve("What programming languages does Noah know?", top_k=3)

# Calculate metrics
metrics = calculate_retrieval_metrics(
    query="What programming languages does Noah know?",
    chunks=chunks,
    latency_ms=150
)

print(f"Retrieved {metrics.num_chunks} chunks")
print(f"Avg similarity: {metrics.avg_similarity:.3f}")
```

### Evaluate Response Quality

```python
from observability import evaluate_response

# After generating a response
evaluation = evaluate_response(
    query="Explain Noah's RAG architecture",
    context=["Noah built a RAG system using pgvector...", "..."],
    answer="Noah's RAG system uses Supabase pgvector for..."
)

print(f"Faithfulness: {evaluation.faithfulness_score:.2f}")
print(f"Relevance: {evaluation.relevance_score:.2f}")
print(f"Quality: {evaluation.answer_quality_score:.2f}")
print(f"Overall: {evaluation.overall_score():.2f}")
```

### Custom Tracing

```python
from observability import trace_retrieval, trace_generation

@trace_retrieval
def my_custom_retriever(query: str):
    # Your retrieval logic
    return chunks

@trace_generation
def my_custom_generator(prompt: str):
    # Your generation logic
    return response
```

### Agentic Workflow (Advanced)

```python
from observability.agentic_workflow import run_agentic_rag

# Run full agentic pipeline
result = run_agentic_rag(
    query="What are Noah's skills?",
    role_mode="Hiring Manager (technical)"
)

print(f"Intent: {result['intent']}")
print(f"Answer: {result['answer']}")
print(f"Metrics: {result['metrics']}")
```

## 🏗️ Architecture

### RAG Pipeline with Observability

```
User Query
    ↓
[LangSmith Trace Start]
    ↓
┌─────────────────┐
│  Classify       │ → Intent: technical/career/mma/fun
│  Intent         │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Retrieve       │ → pgvector similarity search
│  Context        │ → Log: similarity scores, chunk count
└────────┬────────┘
         ↓
┌─────────────────┐
│  Generate       │ → Claude Sonnet 4.5
│  Response       │ → Log: tokens, latency, cost
└────────┬────────┘
         ↓
┌─────────────────┐
│  Evaluate       │ → LLM-as-judge
│  Quality        │ → Log: faithfulness, relevance, quality
└────────┬────────┘
         ↓
[LangSmith Trace End]
    ↓
Final Response + Metrics
```

### Data Flow

1. **Trace Creation**: LangSmith creates parent trace
2. **Retrieval Span**: Logs pgvector query, scores, latency
3. **Generation Span**: Logs the Claude call, tokens, response
4. **Evaluation Span**: Logs quality metrics (sampled)
5. **Trace Completion**: Full pipeline visible in dashboard

## 📊 Metrics Reference

### RetrievalMetrics
```python
@dataclass
class RetrievalMetrics:
    query: str                    # User query text
    num_chunks: int               # Number of chunks retrieved
    similarity_scores: List[float] # Cosine similarity (0-1)
    avg_similarity: float         # Average score
    latency_ms: int               # Retrieval time
    chunk_sources: List[str]      # Source identifiers
    timestamp: datetime           # When retrieval occurred
```

### GenerationMetrics
```python
@dataclass
class GenerationMetrics:
    prompt: str                   # Input prompt
    response: str                 # Generated response
    tokens_prompt: int            # Prompt tokens
    tokens_completion: int        # Completion tokens
    total_tokens: int             # Total tokens
    latency_ms: int               # Generation time
    model: str                    # Model name (e.g. claude-sonnet-4-5)
    cost_usd: float               # Estimated cost
    timestamp: datetime           # When generation occurred
```

### EvaluationMetrics
```python
@dataclass
class EvaluationMetrics:
    faithfulness_score: float     # Response cites context (0-1)
    relevance_score: float        # Context matches query (0-1)
    answer_quality_score: float   # Overall quality (0-1)
    groundedness: float           # Claims supported (0-1)
    citation_accuracy: float      # Citations correct (0-1)
    explanation: str              # Human-readable reasoning
    timestamp: datetime           # When evaluated
```

## 🎯 Best Practices

### 1. Sampling Strategy
Don't evaluate every response (too expensive):
```python
from observability.evaluators import should_evaluate_sample

if should_evaluate_sample(sample_rate=0.1):  # 10% of responses
    metrics = evaluate_response(query, context, answer)
```

### 2. Cost Management
- **Development**: Use sampling for evaluation
- **Production**: Evaluate only flagged responses
- **Claude Haiku**: cheapest option for evaluation calls
- **Budget**: a few dollars/month for 10% sampling at 1K queries/day

### 3. Dashboard Monitoring
View traces in LangSmith dashboard:
- **Traces**: https://smith.langchain.com/
- **Filters**: Filter by project, date, status
- **Compare**: A/B test different prompts
- **Debug**: Click traces to see full context

### 4. Error Handling
Observability fails gracefully:
```python
# If LangSmith unavailable, tracing is no-op
# App continues working without observability
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LANGCHAIN_TRACING_V2` | Yes | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | Yes | - | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | `noahs-ai-assistant` | Project name in LangSmith |
| `LANGCHAIN_ENDPOINT` | No | `https://api.smith.langchain.com` | LangSmith API endpoint |

### Disable Observability
To disable (e.g., for cost savings):
```bash
# In .env
LANGCHAIN_TRACING_V2=false
```

Or set `LANGCHAIN_API_KEY` to empty:
```bash
LANGCHAIN_API_KEY=
```

## 🧪 Testing

### Unit Tests
```bash
pytest tests/test_observability.py
```

### Manual Testing
```python
from observability import trace_rag_call

@trace_rag_call
def test_function():
    return "Hello from traced function!"

result = test_function()
# Check LangSmith dashboard for trace
```

## 📚 Additional Resources

- **LangSmith Docs**: https://docs.smith.langchain.com/
- **LangGraph Guide**: https://langchain-ai.github.io/langgraph/
- **Evaluation Guide**: https://docs.smith.langchain.com/evaluation
- **Pricing**: https://www.langchain.com/pricing

## 🆘 Troubleshooting

### Issue: Traces Not Appearing

**Solution**:
1. Check `LANGCHAIN_TRACING_V2=true` in `.env`
2. Verify `LANGCHAIN_API_KEY` is set
3. Check internet connectivity
4. View logs: `grep "LangSmith" logs/app.log`

### Issue: High Evaluation Costs

**Solution**:
1. Reduce sampling rate: `sample_rate=0.05` (5%)
2. Use Claude Haiku instead of Sonnet for evaluation
3. Disable evaluation in development:
   ```python
   if os.getenv("ENVIRONMENT") == "production":
       evaluate_response(...)
   ```

### Issue: Slow Performance

**Solution**:
1. LangSmith adds ~10-50ms latency
2. Disable if latency critical: `LANGCHAIN_TRACING_V2=false`
3. Use async tracing (coming soon)

## 🎓 Learn More

### Example Projects
- **Supabase Analytics**: `assistant/analytics/supabase_analytics.py`
- **pgvector Retrieval**: `assistant/retrieval/pgvector_retriever.py`
- **RAG Engine**: `assistant/core/rag_engine.py`

### Extending Observability
Add custom metrics:
```python
from observability.metrics import log_metrics_to_supabase

# Log custom metrics
log_metrics_to_supabase(
    retrieval_metrics=my_metrics,
    message_id=123
)
```

---

**Last Updated**: December 2024
**Status**: ✅ Production Ready
**Maintainer**: Noah's AI Team
