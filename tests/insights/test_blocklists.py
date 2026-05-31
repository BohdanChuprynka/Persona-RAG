# ruff: noqa: RUF001
# Reason: tests use intentional Cyrillic strings to exercise the slavic
# stopword blocklist.
from __future__ import annotations

from persona_rag.insights.blocklists import (
    SLAVIC_FUNCTION_WORDS,
    passes_entity_filter,
)

_OK = {"count": 10, "n_sessions": 3, "all_zero_positions": False}


def test_blocks_uk_pronouns():
    for tok in ["мене", "тебе", "себе", "мені", "тобі", "мій", "моя", "її"]:
        assert passes_entity_filter(tok, **_OK) is False, f"should block {tok!r}"


def test_blocks_ru_pronouns():
    for tok in ["меня", "тебя", "его", "мне", "тебе", "что-то"]:
        assert passes_entity_filter(tok, **_OK) is False, f"should block {tok!r}"


def test_blocks_clear_particles():
    for tok in ["нічого", "щось", "хтось", "якось", "якийсь", "така"]:
        assert passes_entity_filter(tok, **_OK) is False, f"should block {tok!r}"


def test_keeps_real_topics():
    # Names, family terms, hobbies, places — must survive. Tokens shorter
    # than min_len=4 are filtered upstream by TOKEN_RE in algo.py so we don't
    # exercise them here.
    for tok in [
        "python",
        "школа",
        "мама",
        "тато",
        "брат",
        "cyberpunk",
        "running",
        "Оlexiy",
        "Chicago",
    ]:
        assert passes_entity_filter(tok, **_OK) is True, f"should keep {tok!r}"


def test_keeps_verbs_like_хочу():
    # Deliberate omission from blocklist (spec §5.1.a) — verbs carry signal.
    for tok in ["хочу", "можу", "буде", "було"]:
        assert passes_entity_filter(tok, **_OK) is True, f"should keep {tok!r}"


def test_keeps_adverbs_like_дуже():
    # Deliberate omission — opinion-shading adverbs colour insights, useful to keep.
    for tok in [
        "дуже",
        "тільки",
        "взагалі",
        "просто",
        "майже",
        "лише",
        "зараз",
        "зара",
    ]:
        assert passes_entity_filter(tok, **_OK) is True, f"should keep {tok!r}"


def test_whitelist_overrides_blocklist():
    # User-blessed entries in synonyms.yaml always pass, even if they happen
    # to match a blocklist term (forward compat).
    wl = {"мене"}  # nonsense entry; just exercising the hook
    assert passes_entity_filter("мене", **_OK, whitelist=wl) is True


def test_whitelist_keeps_short_token_if_long_enough_for_tokenizer():
    # `шк` is len=2 — below default min_len=4 — but whitelisted, so it passes.
    # In practice the TOKEN_RE upstream requires len>=3, so this is exercising
    # the whitelist bypass, not real production data.
    wl = {"шк"}
    assert (
        passes_entity_filter("шк", count=10, n_sessions=3, all_zero_positions=False, whitelist=wl)
        is True
    )


def test_slavic_function_words_is_a_frozenset():
    assert isinstance(SLAVIC_FUNCTION_WORDS, frozenset)
    assert "мене" in SLAVIC_FUNCTION_WORDS
    assert "хочу" not in SLAVIC_FUNCTION_WORDS
