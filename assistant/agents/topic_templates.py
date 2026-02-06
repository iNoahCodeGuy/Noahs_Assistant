"""Topic-aware cinematic response templates.

This module centralizes fallback narratives for core topics. Each template
keeps the cinematic, curious tone described in docs/context/CONVERSATION_PERSONALITY.md
and CONVERSATION_PERSONALITY.md while grounding every statement in the current stack
(LangGraph orchestration, Supabase pgvector, Vercel, GPT-4o).

Design principles applied:
- SRP: Only stores and returns narrative templates.
- Maintainability: Shared helper ensures consistent style across callers.
- Defensibility: Fallback never returns empty strings, preventing formatting bugs.
"""

from __future__ import annotations

from typing import Callable, Dict

# Micro-pause phrasing reused across templates for consistency.
_MICRO_PAUSE = "Now here's the fun part..."
_CLOSING = "Pretty elegant, right?"


def _backend_template() -> str:
    return (
    "Let's dive into the part where all the logic lives - the backend.\n\n"
        "🧠 Backend\n"
    "Here's how I handle that in production: LangGraph orchestrates each node -"
        " classify, retrieve, generate, and log. Every hop carries context so the"
        " answer stays grounded.\n\n"
        f"{_MICRO_PAUSE}\n\n"
        "🛠️ Orchestration Rhythm\n"
    "Each request streams through the LangGraph-style pipeline: the query"
        " gets embedded, I search Supabase pgvector for the sharpest matches,"
        " then GPT-4o assembles a grounded answer. Observability hooks fire along"
        " the way so we can replay the trace in LangSmith.\n\n"
        f"{_CLOSING}"
    )


def _architecture_template() -> str:
    return (
        "That's a great question. Let me frame the whole system so you can see"
        " how the layers cooperate.\n\n"
        "🧭 System Flow\n"
    "User input hits Vercel, the Python runtime hands it to LangGraph,"
    " and the stateful nodes take turns - classification, retrieval,"
        " generation, action planning, and logging.\n\n"
        f"{_MICRO_PAUSE}\n\n"
        "🧠 Intelligence Layer\n"
        "RagEngine stitches together Supabase pgvector results with the current"
        " conversation history, then GPT-4o-mini produces the narrative with"
        " temperature pinned low for enterprise accuracy.\n\n"
        f"{_CLOSING}"
    )


def _frontend_template() -> str:
    return (
        "Let's start on the surface where people feel the experience first.\n\n"
        "🎨 Frontend Flow\n"
        "Next.js handles the production interface on Vercel, streaming responses"
        " as I reason. Streamlit stays ready for local explorations so new"
        " features ship safely before they reach hiring managers.\n\n"
        f"{_MICRO_PAUSE}\n\n"
        "🧩 Experience Design\n"
    "Every response keeps cinematic pacing, but it's powered by practical"
    " engineering - role-aware prompts, latency dashboards in the UI,"
        " and session memory so conversations feel alive.\n\n"
        f"{_CLOSING}"
    )


def _data_template() -> str:
    return (
        "That's a great question. Let me show you how the data layer behaves in"
        " production.\n\n"
        "📊 Supabase Memory\n"
        "Every knowledge slice lives in Supabase Postgres with pgvector."
        " Cosine similarity search pulls the right chunks in under 400ms, and"
        " every match carries metadata for auditing.\n\n"
        f"{_MICRO_PAUSE}\n\n"
        "🧮 Retrieval Signals\n"
        "I log similarity scores, latency, and precision so we can see how"
        " the memory performs over time. That data powers the analytics tables"
        " in Supabase and the LangSmith dashboards.\n\n"
        f"{_CLOSING}"
    )


def _retrieval_template() -> str:
    return (
    "Love that you're curious about retrieval. Here's how I keep answers"
    " anchored to source material.\n\n"
        "🔍 Vector Search\n"
        "Queries get embedded with the compat OpenAIEmbeddings wrapper, then"
        " pgvector returns the closest chunks with their similarity scores."
        " Anything below the safety threshold triggers clarifying prompts"
        " instead of risky guesses.\n\n"
        f"{_MICRO_PAUSE}\n\n"
        "🧵 Grounded Generation\n"
    "I weave those chunks into the GPT-4o prompt with citations so every"
    " sentence maps back to real context. Guardrails log the trace so we"
        " can replay it if something looks off.\n\n"
        f"{_CLOSING}"
    )


def _testing_template() -> str:
    return (
    "Testing questions are my favorite - they show you're thinking like an"
        " operator. Here's the rundown.\n\n"
        "🧪 Quality Gates\n"
        "Pytest suites cover retrieval, formatting, and resume flows."
        " Pre-commit hooks stop shaky responses before they ever ship.\n\n"
        f"{_MICRO_PAUSE}\n\n"
        "📈 Production Feedback\n"
        "LangSmith traces and Supabase analytics flag latency spikes,"
        " retrieval misses, and formatting drift. When something slips, we"
        " replay the trace, patch the node, and deploy through Vercel.\n\n"
        f"{_CLOSING}"
    )


def _career_template() -> str:
    return (
    "Happy to talk about Noah's background. Think of it as the story that"
    " makes this assistant possible.\n\n"
        "👣 Career Track\n"
    "Noah shipped production-grade AI systems, from Tesla workflows to"
    " enterprise GenAI assistants. Each role sharpened the patterns you're"
    " seeing here - modular orchestration, rigorous testing, and grounded"
        " data strategy.\n\n"
        f"{_MICRO_PAUSE}\n\n"
        "🚀 Recent Focus\n"
    "He built this LangGraph-driven resume assistant to prove those"
    " patterns work end to end. Retrieval, reasoning, analytics - all"
        " wrapped in one cinematic experience.\n\n"
        f"{_CLOSING}"
    )


def _general_template() -> str:
    return (
        "That's a great question. Let me walk you through how I keep the whole"
        " experience tight.\n\n"
        "🧠 Reasoning Loop\n"
    "LangGraph orchestrates the flow, GPT-4o handles narrative polish,"
    " and Supabase tracks every fact so we can audit anything on demand.\n\n"
        f"{_MICRO_PAUSE}\n\n"
        "🛰️ Deployment Posture\n"
        "Everything ships through Vercel with observability wired in."
        " Streamlit stays in play for local experiments so we can test new"
        " roles before pushing live.\n\n"
        f"{_CLOSING}"
    )


def _fun_template() -> str:
    return (
    "Let's keep it light for a second - Noah's story has plenty of fun"
        " along the way.\n\n"
        "🎉 Off the Clock\n"
        "He mixes MMA training, chess puzzles, and mentoring youth wrestling"
        " with building AI systems. It's balance through curiosity.\n\n"
        f"{_MICRO_PAUSE}\n\n"
        "🎬 Hidden Easter Eggs\n"
        "I even carry a few MMA callbacks in the knowledge base. Ask about"
        " the fight breakdowns if you want a cinematic detour.\n\n"
        f"{_CLOSING}"
    )


_TEMPLATE_MAP: Dict[str, Callable[[], str]] = {
    "backend": _backend_template,
    "architecture": _architecture_template,
    "frontend": _frontend_template,
    "data": _data_template,
    "retrieval": _retrieval_template,
    "testing": _testing_template,
    "qa": _testing_template,
    "career": _career_template,
    "general": _general_template,
    "fun": _fun_template,
}

# Alias common synonyms to keep detection flexible.
_TEMPLATE_ALIASES = {
    "data layer": "data",
    "data_layer": "data",
    "rag": "retrieval",
    "quality": "testing",
    "qa": "testing",
    "ui": "frontend",
    "interface": "frontend",
}


def response_template(topic: str | None) -> str:
    """Return a cinematic fallback template for the given topic.

    Args:
        topic: Detected topic focus (may be None or unknown).

    Returns:
        Non-empty cinematic narrative covering the requested topic.
    """
    if not topic:
        resolved = "general"
    else:
        lowered = topic.lower()
        resolved = _TEMPLATE_ALIASES.get(lowered, lowered)

    builder = _TEMPLATE_MAP.get(resolved, _TEMPLATE_MAP["general"])
    return builder()
