# LangSmith Complete Guide

> **Quick Start**: For immediate setup, see [LANGSMITH_STUDIO_QUICKSTART.md](LANGSMITH_STUDIO_QUICKSTART.md) in root.

## Table of Contents

1. [Studio Setup & Development](#studio-setup)
2. [Tracing & Observability](#tracing-observability)
3. [Advanced Features](#advanced-features)

---

## Studio Setup

See: `LANGSMITH_STUDIO_QUICKSTART.md` (root directory)

**One-line start**:
```bash
./start_studio.sh
```

**Manual setup** and troubleshooting in the quickstart guide.

---

## Tracing & Observability

See: `docs/LANGSMITH_TRACING_SETUP.md`

### What Gets Traced

- All OpenAI API calls (embeddings, chat)
- Supabase pgvector queries
- Node transitions in conversation flow
- Token usage, latency, costs

### View Traces

Dashboard: https://smith.langchain.com/o/project/noahs-ai-assistant

### Environment Variables

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=noahs-ai-assistant
```

---

## Advanced Features

See: `docs/LANGSMITH_ADVANCED_FEATURES.md`

### Evaluation Pipeline

```bash
python scripts/run_evaluation.py
```

### Prompt Hub

```python
from assistant.prompts import get_prompt
prompt = get_prompt("basic_qa")
```

### LangGraph Studio

```bash
langgraph dev
open http://127.0.0.1:2024
```

---

## Documentation References

- **Setup**: `LANGSMITH_STUDIO_QUICKSTART.md` (root)
- **Tracing**: `docs/LANGSMITH_TRACING_SETUP.md`
- **Advanced**: `docs/LANGSMITH_ADVANCED_FEATURES.md`
- **Comparison**: `docs/LANGSMITH_COMPARISON_WITH_ELI5.md`

All individual guides remain available for detailed reference.
