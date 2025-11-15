# 🔍 Observability Module

Comprehensive monitoring and evaluation for Noah's AI Assistant RAG system.

## Overview

This module provides:
- **LangSmith Integration**: Trace OpenAI calls and RAG pipeline
- **Metrics Collection**: Track retrieval quality, token usage, costs
- **LLM-as-Judge**: Automated quality evaluation
- **Agentic Workflow**: Optional LangGraph multi-step flow

## Quick Start

### 1. Install Dependencies

```bash
pip install langsmith langgraph
```

### 2. Configure LangSmith

Add to `.env`:
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_YOUR_KEY_HERE
LANGCHAIN_PROJECT=noahs-ai-assistant
```

Get API key: https://smith.langchain.com/

### 3. Use It!

```python
from core.rag_engine import RagEngine

# Automatically traced!
engine = RagEngine()
response = engine.generate_response("What are Noah's skills?")

# View trace: https://smith.langchain.com/
```

## Module Structure

```
src/observability/
├── __init__.py              # Public API exports
├── langsmith_tracer.py      # LangSmith integration
├── metrics.py               # Metric data structures
├── evaluators.py            # LLM-based evaluation
└── agentic_workflow.py      # LangGraph workflow
```

## Features

### 1. Automatic Tracing

```python
from observability import trace_rag_call

@trace_rag_call
def my_rag_function(query: str):
    # Automatically traced to LangSmith
    return engine.generate_response(query)
```

### 2. Metrics Collection

```python
from observability import calculate_retrieval_metrics

metrics = calculate_retrieval_metrics(
    query="What are Noah's skills?",
    chunks=retrieved_chunks,
    latency_ms=150
)

print(f"Avg similarity: {metrics.avg_similarity:.3f}")
print(f"Latency: {metrics.latency_ms}ms")
```

### 3. Quality Evaluation

```python
from observability import evaluate_response

metrics = evaluate_response(
    query="What are Noah's skills?",
    context=["Noah is proficient in Python...", "..."],
    answer="Noah has expertise in Python and JavaScript..."
)

print(f"Faithfulness: {metrics.faithfulness_score:.2f}")
print(f"Overall: {metrics.overall_score():.2f}")
```

### 4. Agentic Workflow

```python
from observability.agentic_workflow import run_agentic_rag

result = run_agentic_rag(
    query="What are Noah's skills?",
    role_mode="Hiring Manager"
)

print(f"Intent: {result['intent']}")
print(f"Answer: {result['answer']}")
print(f"Metrics: {result['metrics']}")
```

## API Reference

### Tracing Decorators

- `@trace_rag_call` - Trace full RAG pipeline
- `@trace_retrieval` - Trace retrieval operations
- `@trace_generation` - Trace LLM generation

### Metrics Classes

- `RetrievalMetrics` - Retrieval performance metrics
- `GenerationMetrics` - LLM generation metrics
- `EvaluationMetrics` - Quality evaluation scores

### Evaluation Functions

- `evaluate_faithfulness(query, context, answer)` → (score, explanation)
- `evaluate_relevance(query, context)` → (score, explanation)
- `evaluate_answer_quality(query, answer)` → (score, explanation)
- `evaluate_response(query, context, answer)` → EvaluationMetrics

### Workflow Functions

- `run_agentic_rag(query, role_mode)` → Dict with answer and metrics
- `create_agentic_workflow()` → Compiled LangGraph workflow

## Examples

See `examples/` directory:
- `observability_basic.py` - Basic tracing and metrics
- `observability_evaluation.py` - LLM-as-judge evaluation

Run examples:
```bash
python examples/observability_basic.py
python examples/observability_evaluation.py
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LANGCHAIN_TRACING_V2` | No | `false` | Enable tracing |
| `LANGCHAIN_API_KEY` | Yes* | - | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | `noahs-ai-assistant` | Project name |

*Required only if tracing enabled

### Disable Observability

```bash
# In .env
LANGCHAIN_TRACING_V2=false
```

Observability is fully optional - app works without it!

## Cost Estimation

### LangSmith Pricing

- **Free**: 5,000 traces/month
- **Team**: $39/month for 100k traces
- **Enterprise**: Custom pricing

### Evaluation Costs

- GPT-3.5-turbo: ~$0.0004 per evaluation
- 10% sampling: ~$1-5/month for 1,000 queries/day

### Total: ~$40-50/month for production

## Performance Impact

- Tracing overhead: +10-30ms per request
- Metric calculation: +1-5ms
- Evaluation (sampled): +500-1000ms

**Tip**: Use sampling to minimize cost and latency

## Testing

Run tests:
```bash
pytest tests/test_observability.py -v
```

Test coverage: ~95%

## Documentation

Full guides:
- [OBSERVABILITY_GUIDE.md](../../docs/OBSERVABILITY_GUIDE.md) - Complete user guide
- [LANGSMITH_SETUP.md](../../docs/LANGSMITH_SETUP.md) - Setup instructions
- [OBSERVABILITY_IMPLEMENTATION_SUMMARY.md](../../docs/OBSERVABILITY_IMPLEMENTATION_SUMMARY.md) - Implementation details

## Troubleshooting

### Issue: Traces not appearing

**Solution**:
1. Check `LANGCHAIN_TRACING_V2=true` in `.env`
2. Verify `LANGCHAIN_API_KEY` is set
3. Check internet connectivity

### Issue: Import errors

**Solution**:
```bash
pip install langsmith langgraph
```

### Issue: High costs

**Solution**:
1. Reduce sampling rate (10% → 5%)
2. Use GPT-3.5-turbo for evaluation
3. Disable in development

## Support

- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory
- **Tests**: See `tests/test_observability.py`
- **LangSmith Docs**: https://docs.smith.langchain.com/

## Contributing

When adding features:
1. Add tests to `tests/test_observability.py`
2. Update this README
3. Add examples to `examples/`
4. Update documentation in `docs/`

## License

Part of Noah's AI Assistant project.

---

**Status**: ✅ Production Ready
**Last Updated**: October 2025
