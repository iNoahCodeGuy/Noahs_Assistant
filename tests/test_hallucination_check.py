"""Unit tests for the hallucination gate (stage5).

The gate deterministically verifies checkable claims — percentages, dollar
amounts, URLs — against retrieved chunks and the generation prompt's own
facts. Rollout mode comes from HALLUCINATION_GATE (log by default).
"""

from typing import Any, Dict

import pytest

from assistant.flows.node_logic.stage5_generation_nodes import (
    _find_unsupported_claims,
    hallucination_check,
)


CHUNKS = [
    {"content": "The logistic regression model reached 94.75% accuracy.", "section": "ml"},
    {"content": "Repo: https://github.com/iNoahCodeGuy/portfolia-backend", "section": "links"},
]


def _state(draft: str, chunks=None) -> Dict[str, Any]:
    return {
        "draft_answer": draft,
        "retrieved_chunks": CHUNKS if chunks is None else chunks,
        "analytics_metadata": {},
    }


class TestClaimFinder:
    def test_supported_percentage_passes(self):
        assert _find_unsupported_claims("Accuracy hit 94.75% on that data.", CHUNKS) == []

    def test_unsupported_percentage_is_found(self):
        assert _find_unsupported_claims("Roughly 62% of users agreed.", CHUNKS) == ["62%"]

    def test_percentage_supported_by_bare_number(self):
        chunks = [{"content": "accuracy of 91.2 on the holdout set"}]
        assert _find_unsupported_claims("It scored 91.2%.", chunks) == []

    def test_prompt_facts_count_as_support(self):
        # 94.75% and the GitHub URL live in the generation prompt's fact list
        claims = _find_unsupported_claims(
            "See https://github.com/iNoahCodeGuy — 94.75% accuracy.", []
        )
        assert claims == []

    def test_unsupported_url_is_found(self):
        assert _find_unsupported_claims(
            "Docs at https://example.com/made-up.", CHUNKS
        ) == ["https://example.com/made-up"]

    def test_url_trailing_punctuation_is_normalized(self):
        assert _find_unsupported_claims(
            "Repo: https://github.com/iNoahCodeGuy/portfolia-backend.", CHUNKS
        ) == []

    def test_unsupported_dollar_amount_is_found(self):
        assert _find_unsupported_claims("It saved $12,000 a year.", CHUNKS) == ["$12,000"]


class TestGateModes:
    def test_log_mode_records_findings_but_ships(self, monkeypatch):
        monkeypatch.setenv("HALLUCINATION_GATE", "log")
        state = _state("A made-up 62% improvement.")
        result = hallucination_check(state)
        state.update(result or {})

        assert state["hallucination_safe"] is True
        assert state["analytics_metadata"]["hallucination_findings"] == ["62%"]
        assert "62%" in state["draft_answer"], "log mode must not modify the answer"

    def test_enforce_mode_replaces_the_answer(self, monkeypatch):
        monkeypatch.setenv("HALLUCINATION_GATE", "enforce")
        state = _state("A made-up 62% improvement.")
        result = hallucination_check(state)
        state.update(result or {})

        assert state["hallucination_safe"] is False
        assert "62%" not in state["draft_answer"]
        assert state["analytics_metadata"]["hallucination_findings"] == ["62%"]

    def test_clean_answer_is_safe_in_any_mode(self, monkeypatch):
        for mode in ("log", "enforce"):
            monkeypatch.setenv("HALLUCINATION_GATE", mode)
            state = _state("Accuracy hit 94.75% — details in the repo.")
            state.update(hallucination_check(state) or {})
            assert state["hallucination_safe"] is True
            assert "hallucination_findings" not in state["analytics_metadata"]

    def test_off_mode_skips_entirely(self, monkeypatch):
        monkeypatch.setenv("HALLUCINATION_GATE", "off")
        state = _state("A made-up 62% improvement.")
        state.update(hallucination_check(state) or {})
        assert state["hallucination_safe"] is True
        assert "hallucination_findings" not in state["analytics_metadata"]

    def test_no_chunks_means_nothing_to_verify(self, monkeypatch):
        monkeypatch.setenv("HALLUCINATION_GATE", "enforce")
        state = _state("A made-up 62% improvement.", chunks=[])
        state.update(hallucination_check(state) or {})
        assert state["hallucination_safe"] is True

    def test_self_knowledge_chunks_are_excluded(self, monkeypatch):
        monkeypatch.setenv("HALLUCINATION_GATE", "enforce")
        state = _state(
            "A made-up 62% improvement.",
            chunks=[{"content": "synthetic", "source": "self_knowledge"}],
        )
        state.update(hallucination_check(state) or {})
        assert state["hallucination_safe"] is True
