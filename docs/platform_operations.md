# 🚨 Platform Operations Guide

Unified reference for observability, tracing, and monitoring across Noah's AI Assistant.

## Core Concepts
- **Default runtime**: LangGraph conversation flow (`run_conversation_flow`) with nodes for classification → retrieval → answer generation → action planning → enrichment → side effects.
- **Observability stack**: LangSmith for traces and evaluations, Supabase analytics tables for persisted metrics, optional Twilio/Resend notifications during action execution.
- **Environments**: Local development uses `.env`; production deployments rely on platform-specific secrets (Vercel, Streamlit Cloud, etc.).

## Quick Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure LangSmith credentials in `.env`:
   ```bash
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=lsv2_pt_your_key
   LANGCHAIN_PROJECT=noahs-ai-assistant
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   ```
3. Export the same values in CI/hosting environments if tracing is required. Disable with `LANGCHAIN_TRACING_V2=false` for tests or latency-sensitive runs.

## Monitoring Workflow
- Traces appear automatically whenever `LANGCHAIN_TRACING_V2=true`. Each flow run produces spans for `classify_query`, `retrieve_chunks`, `generate_answer`, `plan_actions`, `apply_role_context`, and `execute_actions`.
- Retrieval metrics (chunk ids, scores, latency) are written to Supabase via `supabase_analytics.log_interaction`.
- Action execution updates analytics keys such as `resume_email_status`, `linkedin_offer`, and `message_id` for downstream reporting.

### Dashboards & Tools
| Tool | Purpose | Location |
| --- | --- | --- |
| LangSmith | Live traces, token usage, latency graphs | https://smith.langchain.com/ |
| Supabase | Interaction logs, feedback, retrieval KPIs | Supabase Dashboard → SQL Editor |
| Logs | Local debugging | `logs/` directory or deployment logs |

## Sampling & Cost Control
- **Tracing**: Set `LANGCHAIN_TRACING_V2=false` in `.env.test` or override in CI to avoid test noise.
- **Evaluations**: Use sampling helpers (e.g., `should_evaluate_sample(rate=0.1)`) to run LLM-as-judge on a subset of responses.
- **Model Choices**: Prefer GPT-3.5 for evaluation jobs; reserve GPT-4 for high-fidelity production checks.

## Key Metrics
- Retrieval: chunk count, average similarity, latency.
- Generation: prompt tokens, completion tokens, model, total cost.
- Evaluation: faithfulness, relevance, answer quality, groundedness (optional sampling).
- Actions: resume emails sent, contact notifications dispatched, signed URL generation.

## Troubleshooting
- **Traces missing**: Verify `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` is populated; run `python -c "from observability import initialize_langsmith; print(initialize_langsmith())"`.
- **High latency**: Disable tracing temporarily or reduce sampling. LangSmith adds 10–50 ms per span.
- **Credential errors**: Ensure secrets are set in hosting platform dashboards (Vercel/Streamlit). Never commit `.env`.
- **Cost spikes**: Lower evaluation sampling and prefer lighter models for diagnostics.

## Additional References
- `docs/runtime_dependencies.md`: Comprehensive module inventory invoked at runtime.
- `src/flows/conversation_nodes.py`: Action planning and execution hooks described above.
- `docs/context/SYSTEM_ARCHITECTURE_SUMMARY.md`: Full system architecture and control flow.
- `docs/context/PROJECT_REFERENCE_OVERVIEW.md`: Project purpose, roles, and stack overview.
