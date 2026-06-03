# ruff: noqa: RUF001
from __future__ import annotations

from datetime import UTC, datetime

from persona_rag.generate.prompt import build_fact_card
from persona_rag.insights.recency import RankedInsight


def _fact(cat, subj, uk, en):
    now = datetime.now(UTC)
    return RankedInsight(
        id=subj,
        text=uk,
        text_en=en,
        category=cat,
        subject=subj,
        confidence=1.0,
        evidence_count=1,
        earliest_date=now,
        latest_date=now,
        trajectory=None,
        source="vault",
        semantic_score=1.0,
        final_score=1.0,
    )


def test_self_desc_card_uses_core_in_query_lang():
    insights = {
        "lane": "self_desc",
        "query_lang": "en",
        "core": [_fact("bio", "school", "Навчається на CS", "Studies CS")],
        "semantic": [],
    }
    card = build_fact_card("who are you", "", insights)
    assert "Studies CS" in card and "Навчається" not in card


def test_self_desc_card_uk_default():
    insights = {
        "lane": "self_desc",
        "query_lang": "uk",
        "core": [_fact("bio", "school", "Навчається на CS", "Studies CS")],
        "semantic": [],
    }
    card = build_fact_card("розкажи про себе", "", insights)
    assert "Навчається на CS" in card


def test_specific_lane_filters_to_identity_categories():
    insights = {
        "lane": "specific",
        "query_lang": "uk",
        "core": [],
        "semantic": [
            _fact("bio", "school", "Навчається на CS", "Studies CS"),
            _fact("interest", "games", "грає", "plays"),
        ],
    }
    card = build_fact_card("де вчишся?", "", insights)
    assert "Навчається на CS" in card and "грає" not in card


def test_none_lane_returns_none():
    insights = {"lane": "none", "query_lang": "uk", "core": [], "semantic": []}
    assert build_fact_card("що по погоді", "", insights) is None


def test_card_respects_400_char_cap():
    core = [_fact("bio", f"s{i}", "x" * 100, "y" * 100) for i in range(10)]
    insights = {"lane": "self_desc", "query_lang": "uk", "core": core, "semantic": []}
    card = build_fact_card("хто ти", "", insights)
    assert card is not None and len(card) <= 400


def test_long_memory_does_not_evict_identity_facts():
    """Regression: identity facts are budgeted first, so a long contact memory can
    never crowd them out of the self_desc card (the bug defeated the whole feature)."""
    insights = {
        "lane": "self_desc",
        "query_lang": "uk",
        "core": [_fact("bio", "school", "Навчається на CS", "Studies CS")],
        "semantic": [],
    }
    card = build_fact_card("розкажи про себе", "M" * 500, insights)
    assert card is not None
    assert "Навчається на CS" in card
    assert len(card) <= 400


def test_build_messages_ollama_injects_card_into_system_turn(monkeypatch):
    """Integration through the LIVE serving entry point: build_messages (ollama
    branch, OLLAMA_FACTS_IN_SYSTEM=true) routes a self_desc insights dict through
    build_fact_card and folds the fact into the THIN system turn — shape stays
    [system, user], no assistant turns (train==serve preserved)."""
    from persona_rag.config import get_settings
    from persona_rag.generate.persona import THIN_SYSTEM
    from persona_rag.generate.prompt import build_messages
    from persona_rag.models import StyleAnchors

    monkeypatch.setenv("GENERATION_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_FACTS_IN_SYSTEM", "true")
    get_settings.cache_clear()
    try:
        anchors = StyleAnchors(
            avg_len_chars=20,
            median_len_chars=18,
            emoji_rate_per_char=0.0,
            lang_distribution={"uk": 1.0},
            top_bigrams=["ok"],
            n_turns=10,
            primary_language="uk",
        )
        insights = {
            "lane": "self_desc",
            "query_lang": "uk",
            "core": [_fact("bio", "school", "Навчається на CS", "Studies CS")],
            "semantic": [],
        }
        msgs = build_messages(
            persona_name="Bohdan",
            persona_description="x",
            style_anchors=anchors,
            user_memory="",
            retrieved=[],
            session=[],
            incoming="розкажи про себе",
            insights=insights,
        )
    finally:
        get_settings.cache_clear()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"].startswith(THIN_SYSTEM)
    assert "Навчається на CS" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert not any(m["role"] == "assistant" for m in msgs)
