# ruff: noqa: RUF001
# Reason: intentional Cyrillic test data.
from __future__ import annotations

from persona_rag.insights.blocklists import (
    ENGLISH_FILLERS,
    URL_PIECES,
    is_sentence_start_only,
    passes_entity_filter,
)


def test_url_pieces_includes_common_noise():
    for token in ("https", "www", "com", "youtube"):
        assert token in URL_PIECES


def test_english_fillers_includes_chat_noise():
    for token in ("yeah", "good", "okay", "know", "think"):
        assert token in ENGLISH_FILLERS


def test_is_sentence_start_only_detects_pure_start():
    # token appears only as sentence-initial → True (treat as not an entity)
    positions = [0, 0, 0]
    assert is_sentence_start_only(positions) is True


def test_is_sentence_start_only_detects_mid_sentence():
    # token appears also at non-zero positions → False (genuine entity)
    positions = [0, 3, 0, 7]
    assert is_sentence_start_only(positions) is False


def test_passes_entity_filter_rejects_urls():
    assert passes_entity_filter("https", count=50, n_sessions=10, all_zero_positions=False) is False


def test_passes_entity_filter_rejects_short_tokens():
    assert passes_entity_filter("ai", count=50, n_sessions=10, all_zero_positions=False) is False


def test_passes_entity_filter_rejects_low_count():
    assert (
        passes_entity_filter("ml_topic", count=8, n_sessions=5, all_zero_positions=False) is False
    )


def test_passes_entity_filter_rejects_single_session_bursts():
    assert (
        passes_entity_filter("ml_topic", count=50, n_sessions=2, all_zero_positions=False) is False
    )


def test_passes_entity_filter_rejects_sentence_starts():
    assert passes_entity_filter("Давай", count=50, n_sessions=10, all_zero_positions=True) is False


def test_passes_entity_filter_accepts_real_entity():
    assert (
        passes_entity_filter("cyberpunk", count=15, n_sessions=4, all_zero_positions=False) is True
    )
