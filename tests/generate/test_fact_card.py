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
