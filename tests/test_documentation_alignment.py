"""
Documentation Alignment Tests

These tests verify that documentation matches actual code implementation.
Prevents drift where docs claim files, models, or flows that don't exist.

Run: pytest tests/test_documentation_alignment.py -v
"""

import os
import re
import inspect

import pytest


README = "README.md"
DOCS_DIR = "docs"
KEPT_DOCS = [
    "docs/README.md",
    "docs/GLOSSARY.md",
    "docs/EXTERNAL_SERVICES.md",
    "docs/LANGSMITH.md",
    "docs/OBSERVABILITY.md",
]


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class TestReadmeAlignment:
    """The README is the front door — its claims must match the code."""

    def test_readme_exists_and_links_live_site(self):
        content = _read(README)
        assert "noahdelacalzada.com" in content, "README must link the live demo"
        assert "portfolia-frontend" in content, "README must link the frontend repo"

    def test_readme_embedding_model_matches_code(self):
        """The embedding model named in the README is the one the retriever uses."""
        readme = _read(README)
        retriever_src = _read("assistant/retrieval/pgvector_retriever.py")

        code_models = set(re.findall(r"text-embedding-[\w.-]+", retriever_src))
        assert code_models, "Expected an embedding model in pgvector_retriever.py"
        for model in code_models:
            assert model in readme, (
                f"Retriever uses '{model}' but the README doesn't mention it. "
                "Update the README stack section."
            )

    def test_readme_rpc_name_matches_code(self):
        """The Supabase RPC named in the README exists in the retrieval code."""
        readme = _read(README)
        rpc_names = re.findall(r"`(match_[a-z_]+)`", readme)
        assert rpc_names, "README should name the vector-search RPC"
        retriever_src = _read("assistant/retrieval/pgvector_retriever.py")
        for rpc in set(rpc_names):
            assert rpc in retriever_src, (
                f"README names RPC '{rpc}' but pgvector_retriever.py doesn't use it."
            )

    def test_readme_file_references_exist(self):
        """Every repo-relative path the README cites must exist."""
        readme = _read(README)
        # Paths in backticks or markdown links, e.g. `api/main.py`, [x](data/)
        candidates = set(
            re.findall(r"[`(]((?:assistant|api|data|docs|scripts|supabase|tests)/[\w./-]*)", readme)
        ) | set(re.findall(r"`((?:chat_with_portfolia\.py|Dockerfile|LICENSE|\.env\.example))`", readme))
        missing = [p for p in candidates if not os.path.exists(p.rstrip(")"))]
        assert not missing, f"README references nonexistent paths: {sorted(missing)}"

    def test_readme_quickstart_commands_reference_real_entrypoints(self):
        readme = _read(README)
        assert "uvicorn api.main:app" in readme
        assert os.path.exists("api/main.py")
        assert "chat_with_portfolia.py" in readme
        assert os.path.exists("chat_with_portfolia.py")


class TestConversationFlowAlignment:
    """Pipeline concepts the README sells must exist in the orchestrator."""

    def test_pipeline_concepts_documented(self):
        readme = _read(README).lower()
        for concept in ["intent", "retrieval", "generation", "pgvector"]:
            assert concept in readme, f"Core concept '{concept}' missing from README"

    def test_orchestrator_is_a_pipeline(self):
        from assistant.flows.conversation_flow import run_conversation_flow

        source = inspect.getsource(run_conversation_flow)
        assert "pipeline" in source.lower(), (
            "Expected a pipeline pattern in conversation_flow.py"
        )


class TestDocsIntegrity:
    """The kept reference docs exist, are non-trivial, and are indexed."""

    def test_kept_docs_exist_and_not_empty(self):
        for doc in KEPT_DOCS:
            assert os.path.exists(doc), f"Missing doc: {doc}"
            assert len(_read(doc)) > 500, f"{doc} is too short to be a real reference"

    def test_docs_index_covers_all_docs(self):
        """Every markdown file in docs/ is linked from docs/README.md."""
        index = _read("docs/README.md")
        for name in sorted(os.listdir(DOCS_DIR)):
            if not name.endswith(".md") or name == "README.md":
                continue
            assert name in index, (
                f"docs/{name} is not listed in docs/README.md — add it to the index "
                "or delete the file."
            )


class TestNoStaleTechReferences:
    """Docs must not describe the retired stack (Streamlit UI, GPT generation, src/)."""

    FORBIDDEN = [
        r"streamlit run",
        r"\bsrc/",
        r"gpt-3\.5",
        r"gpt-4o",
        r"FAISS_CAREER_PATH",
        r"SUPABASE_SERVICE_KEY=",
    ]

    def test_readme_and_docs_have_no_stale_references(self):
        offenders = []
        for path in [README, *KEPT_DOCS, "CONTRIBUTING.md", "CLAUDE.md", ".env.example"]:
            content = _read(path)
            for pattern in self.FORBIDDEN:
                if re.search(pattern, content, re.IGNORECASE):
                    offenders.append(f"{path}: /{pattern}/")
        assert not offenders, "Stale tech references found:\n" + "\n".join(offenders)


class TestEnvExampleAlignment:
    """.env.example documents the variables the config layer actually requires."""

    def test_required_vars_present(self):
        example = _read(".env.example")
        for var in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "SUPABASE_URL",
            "SUPABASE_SERVICE_ROLE_KEY",
        ]:
            assert re.search(rf"^{var}=", example, re.MULTILINE), (
                f"{var} missing from .env.example — the code requires it."
            )


class TestChangelogIntegrity:
    """CHANGELOG.md exists and is structured correctly."""

    def test_changelog_exists(self):
        assert os.path.exists("CHANGELOG.md")

    def test_changelog_has_dated_entries(self):
        content = _read("CHANGELOG.md")
        assert len(content) > 1000, "CHANGELOG.md seems incomplete"
        assert re.search(r"\[20\d{2}-\d{2}(-\d{2})?\]", content), (
            "CHANGELOG.md should have dated entries like [2026-07-04]"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
