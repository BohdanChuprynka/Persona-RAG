"""Decoding-side voice levers: best-of-N style selection + paren logit bias."""

from __future__ import annotations

from persona_rag.config import get_settings
from persona_rag.generate.llm_client import exclaim_logit_bias, paren_logit_bias, voice_logit_bias
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


class TestExclaimLogitBias:
    def test_off_returns_none(self, monkeypatch) -> None:
        monkeypatch.setenv("EXCLAIM_LOGIT_BIAS", "0")
        get_settings.cache_clear()
        try:
            assert exclaim_logit_bias() is None
        finally:
            get_settings.cache_clear()

    def test_on_maps_exclaim_tokens_to_negative_bias(self, monkeypatch) -> None:
        monkeypatch.setenv("EXCLAIM_LOGIT_BIAS", "-5")
        get_settings.cache_clear()
        try:
            bias = exclaim_logit_bias()
            assert bias
            assert all(isinstance(k, int) for k in bias)
            assert set(bias.values()) == {-5}
        finally:
            get_settings.cache_clear()


class TestVoiceLogitBias:
    def test_merges_paren_positive_and_exclaim_negative(self, monkeypatch) -> None:
        monkeypatch.setenv("PAREN_LOGIT_BIAS", "2")
        monkeypatch.setenv("EXCLAIM_LOGIT_BIAS", "-5")
        get_settings.cache_clear()
        try:
            merged = voice_logit_bias()
            assert merged
            assert 2 in merged.values()  # paren nudged up
            assert -5 in merged.values()  # exclaim pushed down
        finally:
            get_settings.cache_clear()

    def test_both_off_returns_none(self, monkeypatch) -> None:
        monkeypatch.setenv("PAREN_LOGIT_BIAS", "0")
        monkeypatch.setenv("EXCLAIM_LOGIT_BIAS", "0")
        get_settings.cache_clear()
        try:
            assert voice_logit_bias() is None
        finally:
            get_settings.cache_clear()


class TestVoiceLogitBiasBackendGate:
    """The token ids come from the OpenAI tokenizer (tiktoken). On the ollama
    backend the model is Qwen — those ids map to unrelated tokens, and Ollama's
    OpenAI-compatible API drops logit_bias anyway. So the lever must be OpenAI
    -only; on the LoRA the ")" tic and absent "!" are learned from the data."""

    def test_ollama_backend_disables_logit_bias(self, monkeypatch) -> None:
        monkeypatch.setenv("PAREN_LOGIT_BIAS", "2")
        monkeypatch.setenv("EXCLAIM_LOGIT_BIAS", "-5")
        monkeypatch.setenv("GENERATION_BACKEND", "ollama")
        get_settings.cache_clear()
        try:
            assert voice_logit_bias() is None
        finally:
            get_settings.cache_clear()

    def test_openai_backend_keeps_logit_bias(self, monkeypatch) -> None:
        monkeypatch.setenv("PAREN_LOGIT_BIAS", "2")
        monkeypatch.setenv("EXCLAIM_LOGIT_BIAS", "-5")
        monkeypatch.setenv("GENERATION_BACKEND", "openai")
        get_settings.cache_clear()
        try:
            assert voice_logit_bias()  # non-empty on the OpenAI path
        finally:
            get_settings.cache_clear()
