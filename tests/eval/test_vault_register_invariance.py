# ruff: noqa: RUF001
"""Construction-level register invariance (spec 2026-06-03 open-Q#6).

Facts-on must add a card ONLY for identity-relevant turns, never for control
turns -- so control-turn prompts are byte-identical facts-off vs facts-on (no
register perturbation possible). The generation-level A/B (shape_js / paren_smiley
/ length on real replies) runs via `make compare-vault` with llama-server up.
"""

from __future__ import annotations

from persona_rag.generate.prompt import build_fact_card

SELF_DESC = ["розкажи про себе", "tell me about yourself", "who are you", "расскажи о себе"]
CONTROL = ["що по погоді", "го грати", "ахах да"]


def _core():
    from datetime import UTC, datetime

    from persona_rag.insights.recency import RankedInsight

    now = datetime.now(UTC)
    return [
        RankedInsight(
            id="b",
            text="Навчається на CS",
            text_en="Studies CS",
            category="bio",
            subject="school",
            confidence=1.0,
            evidence_count=1,
            earliest_date=now,
            latest_date=now,
            trajectory=None,
            source="vault",
            semantic_score=1.0,
            final_score=1.0,
        )
    ]


def test_control_turns_get_no_card():
    for q in CONTROL:
        ins = {"lane": "none", "query_lang": "uk", "core": [], "semantic": []}
        assert build_fact_card(q, "", ins) is None


def test_self_desc_turns_get_card_within_cap():
    for q in SELF_DESC:
        ins = {"lane": "self_desc", "query_lang": "uk", "core": _core(), "semantic": []}
        card = build_fact_card(q, "", ins)
        assert card is not None and len(card) <= 400
        assert "Навчається на CS" in card
