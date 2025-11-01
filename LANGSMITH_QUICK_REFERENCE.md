# LangSmith Features - Quick Reference

## 1. Run Type Classification ✅
**Already working** - View at https://smith.langchain.com/

## 2. LangGraph Studio 🎨
```bash
langgraph dev                    # Start Studio
open http://127.0.0.1:2024      # Access UI
```

## 3. Prompt Hub 📝
```python
# Initialize (one-time)
from src.prompts.prompt_hub import initialize_prompt_hub
initialize_prompt_hub()

# Use in code
from src.prompts import get_prompt
prompt = get_prompt("basic_qa")
```

## 4. Evaluation Pipeline 📊
```bash
# Run full evaluation
python3 scripts/run_evaluation.py

# A/B testing
python3 scripts/run_evaluation.py --experiment "baseline"
python3 scripts/run_evaluation.py --experiment "variant"

# Specific evaluators
python3 scripts/run_evaluation.py --evaluators accuracy tone

# Upload dataset only
python3 scripts/run_evaluation.py --upload-only
```

## Verify Installation
```bash
python3 scripts/test_langsmith_features.py
```

## Available Prompts
- `basic_qa` - Main RAG prompt
- `role_hiring_manager_technical` - Technical persona
- `role_hiring_manager_nontechnical` - Business persona
- `role_developer` - Developer persona
- `faithfulness_evaluator` - Hallucination detection
- `relevance_evaluator` - Context relevance

## Available Evaluators
- `accuracy_evaluator` - Answer correctness
- `tone_evaluator` - Role appropriateness
- `grounding_evaluator` - Hallucination check
- `relevance_evaluator` - Context quality
- `response_time_evaluator` - Latency check
- `conciseness_evaluator` - Detail level

## Documentation
- Full guide: `docs/LANGSMITH_ADVANCED_FEATURES.md`
- Summary: `LANGSMITH_FEATURES_IMPLEMENTATION_SUMMARY.md`

## Environment Variables
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=noahs-ai-assistant
```
