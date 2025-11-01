# LangSmith Advanced Features Guide

This guide covers the four advanced LangSmith/LangGraph features now integrated into Noah's AI Assistant:

1. ✅ **Run Type Classification** - Proper categorization of operations
2. ✅ **LangGraph Studio** - Visual debugging interface
3. ✅ **Prompt Hub** - Version-controlled prompts
4. ✅ **Evaluation Pipeline** - A/B testing framework

## 1. Run Type Classification

### What It Does
Categorizes different operations in LangSmith traces (retrieval, generation, chains, tools) for better observability.

### Current Implementation
Already implemented in `src/observability/langsmith_tracer.py`:

```python
@traceable(name="retrieval", run_type="retriever")  # Vector search operations
@traceable(name="generation", run_type="llm")        # LLM generation calls
@traceable(name="rag_pipeline", run_type="chain")   # Full RAG workflows
```

### How to View
1. Go to https://smith.langchain.com/
2. Select your project (e.g., "noahs-ai-assistant")
3. Click on any trace
4. See operations categorized by type in the trace tree

### Benefits
- **Easier debugging**: Quickly identify which component is slow or failing
- **Performance analysis**: Compare latency across operation types
- **Cost tracking**: See token usage broken down by LLM vs retriever calls

---

## 2. LangGraph Studio (Visual Debugging)

### What It Does
Provides a visual interface to debug conversation flows, inspect state at each node, and replay with different inputs.

### Setup
Created `langgraph.json` configuration:

```json
{
  "dependencies": ["."],
  "graphs": {
    "conversation_flow": "./src/flows/conversation_flow.py:graph"
  },
  "env": ".env"
}
```

### How to Use

**Start Studio:**
```bash
langgraph dev
```

**Access UI:**
Open browser to http://127.0.0.1:2024

**Test Queries:**
1. Select "conversation_flow" graph
2. Enter test input:
   ```json
   {
     "role": "Software Developer",
     "query": "What's Noah's Python experience?",
     "session_id": "studio-test"
   }
   ```
3. Click "Run"
4. Inspect each node's output in the visualization

### Features
- **Visual graph**: See all 24 nodes and their connections
- **State inspection**: View conversation state at each step
- **Conditional branches**: See which path was taken (e.g., greeting short-circuit)
- **Replay**: Re-run with different inputs without restarting
- **Time travel**: Inspect intermediate states

### Troubleshooting
If Studio shows errors:
- Ensure `langgraph` package installed: `pip install langgraph`
- Check that all imports in `conversation_flow.py` resolve
- Verify `.env` has required keys (OPENAI_API_KEY, SUPABASE_URL, etc.)

---

## 3. Prompt Hub (Version Control)

### What It Does
Stores prompts in LangSmith Hub for version control, A/B testing, and collaborative editing.

### Files
- `src/prompts/prompt_hub.py` - Main implementation
- `src/prompts/__init__.py` - Public API

### Local Templates
Pre-defined prompts in `LOCAL_PROMPTS`:
- `basic_qa` - Main QA prompt
- `role_hiring_manager_technical` - Technical hiring manager persona
- `role_hiring_manager_nontechnical` - Business hiring manager persona
- `role_developer` - Software developer persona
- `faithfulness_evaluator` - Hallucination detection prompt
- `relevance_evaluator` - Context relevance scoring prompt

### Usage

**Initialize Hub (one-time setup):**
```python
from src.prompts.prompt_hub import initialize_prompt_hub

initialize_prompt_hub()  # Pushes all local templates to hub
```

**Pull Prompt from Hub:**
```python
from src.prompts import get_prompt

# Try hub first, fall back to local
prompt = get_prompt("basic_qa")
filled = prompt.format(context="...", question="...")
```

**Push New Prompt Version:**
```python
from src.prompts import push_prompt

push_prompt(
    "basic_qa",
    "Context: {context}\n\nQuestion: {question}\n\nAnswer:",
    description="v2 - more concise"
)
```

**List Available Prompts:**
```python
from src.prompts import list_prompts

prompts = list_prompts()
print(f"Available: {', '.join(prompts.keys())}")
```

### Benefits
- **Version control**: Track prompt changes over time
- **Rollback**: Revert to previous version if quality degrades
- **A/B testing**: Compare different prompt variants
- **Collaboration**: Team can iterate on prompts in hub
- **Centralized**: Same prompts across local/Vercel deployments

### Viewing in LangSmith
1. Go to https://smith.langchain.com/
2. Click "Prompts" in sidebar
3. Browse/edit uploaded prompts
4. See version history

---

## 4. Evaluation Pipeline (A/B Testing)

### What It Does
Systematically evaluates RAG quality using golden datasets and multiple metrics.

### Files
- `data/evaluation/golden_dataset.csv` - 10 test queries with expected outputs
- `src/evaluation/evaluators.py` - 6 custom evaluator functions
- `scripts/run_evaluation.py` - Evaluation runner script

### Metrics

1. **Accuracy** (LLM-as-judge): Does answer match expected output?
2. **Tone** (LLM-as-judge): Is response appropriate for role?
3. **Grounding** (LLM-as-judge): Are claims supported by context?
4. **Relevance** (LLM-as-judge): Is retrieved context useful?
5. **Response Time** (rule-based): Is latency acceptable (<1s ideal)?
6. **Conciseness** (rule-based): Is answer appropriately detailed?

### Golden Dataset Format
CSV with columns:
- `role` - User role (e.g., "Software Developer")
- `query` - User question
- `expected_output` - Ideal answer
- `evaluation_criteria` - Criteria for tone evaluation

### Running Evaluations

**Full Evaluation Suite:**
```bash
python scripts/run_evaluation.py
```

**Specific Evaluators Only:**
```bash
python scripts/run_evaluation.py --evaluators accuracy tone
```

**Custom Dataset:**
```bash
python scripts/run_evaluation.py --dataset my_dataset.csv
```

**A/B Test with Experiment Name:**
```bash
# Baseline
python scripts/run_evaluation.py --experiment "baseline"

# Test variant (e.g., after changing prompt)
python scripts/run_evaluation.py --experiment "prompt-v2"
```

**Upload Dataset Only (no eval):**
```bash
python scripts/run_evaluation.py --upload-only
```

### Output Example
```
==============================================================
EVALUATION SUMMARY
==============================================================

Metric Averages:
  accuracy            : 0.875 (n=10)
  tone                : 0.920 (n=10)
  grounding           : 0.950 (n=10)
  conciseness         : 0.810 (n=10)

  OVERALL             : 0.889

==============================================================
View detailed results at: https://smith.langchain.com/
==============================================================
```

### Viewing Results in LangSmith
1. Go to https://smith.langchain.com/
2. Click "Datasets & Testing" → "Experiments"
3. Compare experiments side-by-side
4. Click individual runs to see traces

### Use Cases

**Prompt Engineering:**
```bash
# Baseline
python scripts/run_evaluation.py --experiment "prompt-v1"

# Edit prompt in src/prompts/prompt_hub.py
# Push to hub: push_prompt(...)

# Test new version
python scripts/run_evaluation.py --experiment "prompt-v2"

# Compare in LangSmith UI
```

**Model Comparison:**
```bash
# GPT-4
export OPENAI_MODEL=gpt-4-turbo-preview
python scripts/run_evaluation.py --experiment "gpt4"

# GPT-3.5
export OPENAI_MODEL=gpt-3.5-turbo
python scripts/run_evaluation.py --experiment "gpt35"
```

**Retrieval Tuning:**
```bash
# Current top_k=4
python scripts/run_evaluation.py --experiment "topk-4"

# Change in rag_engine.py: top_k=6
python scripts/run_evaluation.py --experiment "topk-6"
```

---

## Quick Start Checklist

### First Time Setup

1. **Verify LangSmith credentials:**
   ```bash
   grep LANGSMITH .env
   # Should show:
   # LANGCHAIN_TRACING_V2=true
   # LANGCHAIN_API_KEY=lsv2_pt_...
   # LANGCHAIN_PROJECT=noahs-ai-assistant
   ```

2. **Install LangGraph (if not already):**
   ```bash
   pip install langgraph
   ```

3. **Initialize Prompt Hub:**
   ```bash
   python -c "from src.prompts.prompt_hub import initialize_prompt_hub; initialize_prompt_hub()"
   ```

4. **Upload evaluation dataset:**
   ```bash
   python scripts/run_evaluation.py --upload-only
   ```

### Daily Workflow

**Visual debugging session:**
```bash
langgraph dev
# Open http://127.0.0.1:2024
```

**Check trace classification:**
- Run any query in Streamlit
- View at https://smith.langchain.com/
- Verify run types are correct (retriever, llm, chain)

**A/B test prompt change:**
```bash
# Baseline
python scripts/run_evaluation.py --experiment "baseline"

# Make change, test
python scripts/run_evaluation.py --experiment "variant"

# Compare in LangSmith UI
```

---

## Integration with Existing Code

### Using Prompt Hub in Production

**Before (hardcoded):**
```python
template = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
prompt = template.format(context=..., question=...)
```

**After (versioned):**
```python
from src.prompts import get_prompt

template = get_prompt("basic_qa")
prompt = template.format(context=..., question=...)
```

### Adding Custom Evaluators

**Create new evaluator:**
```python
# src/evaluation/evaluators.py

def my_custom_evaluator(run: Any, example: Any) -> Dict[str, Any]:
    """Evaluate custom metric."""
    score = # ... your logic
    return {
        "score": score,
        "key": "my_metric",
        "reasoning": "..."
    }
```

**Use in evaluation:**
```python
# scripts/run_evaluation.py

EVALUATOR_MAP = {
    # ...existing...
    "my_metric": my_custom_evaluator,
}
```

**Run:**
```bash
python scripts/run_evaluation.py --evaluators accuracy my_metric
```

---

## Troubleshooting

### LangGraph Studio won't start
```bash
# Check graph exports correctly
python -c "from src.flows.conversation_flow import graph; print(graph)"

# Should print: <langgraph.graph...CompiledGraph...>
# If None: install langgraph, check imports
```

### Prompt Hub push fails
```bash
# Check credentials
python -c "from src.observability.langsmith_tracer import get_langsmith_client; print(get_langsmith_client())"

# Should print: <Client object>
# If None: check LANGCHAIN_API_KEY in .env
```

### Evaluation errors
```bash
# Test dataset loads
python -c "import pandas as pd; print(pd.read_csv('data/evaluation/golden_dataset.csv').head())"

# Test predict function
python -c "from scripts.run_evaluation import predict_answer; print(predict_answer({'role': 'Software Developer', 'query': 'test'}))"
```

---

## Next Steps

### Expand Golden Dataset
Add more test cases to `data/evaluation/golden_dataset.csv`:
- Edge cases (vague queries, confessions, MMA)
- Multi-turn conversations
- Error scenarios

### Custom Evaluators
Implement domain-specific metrics:
- Citation accuracy (check if URLs valid)
- Code quality (if showing code snippets)
- Personality consistency (warmth, enthusiasm)

### Continuous Evaluation
Set up automated evaluation:
```bash
# Add to CI/CD pipeline
python scripts/run_evaluation.py --experiment "commit-${GIT_SHA}"
```

### Prompt Optimization
Iterate on prompts using evaluation feedback:
1. Run baseline eval
2. Tweak prompt in hub
3. Run comparison eval
4. Deploy best performer

---

## Resources

- **LangSmith Docs**: https://docs.smith.langchain.com/
- **LangGraph Studio**: https://langchain-ai.github.io/langgraph/tutorials/langgraph-studio/
- **Prompt Hub**: https://smith.langchain.com/hub
- **Evaluation Guide**: https://docs.smith.langchain.com/evaluation

## Support

For issues or questions:
1. Check `docs/LANGSMITH_TRACING_SETUP.md` for detailed setup
2. View example traces in LangSmith dashboard
3. Test with ELI5 reference: https://github.com/xuro-langchain/eli5
