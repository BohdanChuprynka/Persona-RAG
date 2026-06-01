"""Decoding-side voice levers: best-of-N style selection + paren logit bias."""

from __future__ import annotations

from persona_rag.config import get_settings
from persona_rag.generate.llm_client import paren_logit_bias
from persona_rag.generate.select import pick_best


class TestPickBest:
    def test_returns_max_score_candidate(self) -> None:
        assert pick_best(["a", "b", "c"], [0.1, 0.9, 0.5]) == "b"

    def test_tie_returns_first(self) -> None:
        assert pick_best(["a", "b"], [0.5, 0.5]) == "a"

    def test_single_candidate(self) -> None:
        assert pick_best(["only"], [0.3]) == "only"

    def test_empty_returns_empty_string(self) -> None:
        assert pick_best([], []) == ""

    def test_mismatched_lengths_returns_first(self) -> None:
        # defensive: never crash the generate node on a scorer hiccup
        assert pick_best(["a", "b"], [0.5]) == "a"


class TestParenLogitBias:
    def test_off_returns_none(self, monkeypatch) -> None:
        monkeypatch.setenv("PAREN_LOGIT_BIAS", "0")
        get_settings.cache_clear()
        try:
            assert paren_logit_bias() is None
        finally:
            get_settings.cache_clear()

    def test_on_maps_paren_tokens_to_bias(self, monkeypatch) -> None:
        monkeypatch.setenv("PAREN_LOGIT_BIAS", "3")
        get_settings.cache_clear()
        try:
            bias = paren_logit_bias()
            assert bias  # non-empty dict
            assert all(isinstance(k, int) for k in bias)
            assert set(bias.values()) == {3}
        finally:
            get_settings.cache_clear()
