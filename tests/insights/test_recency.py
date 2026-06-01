from __future__ import annotations

from datetime import UTC, datetime, timedelta

from persona_rag.insights.recency import RankedInsight, rerank_with_recency


def _ri(score: float, latest_offset_days: int, _id: str, now: datetime) -> RankedInsight:
    return RankedInsight(
        id=_id,
        text=f"text {_id}",
        category="bio",
        subject="x",
        confidence=0.8,
        evidence_count=3,
        earliest_date=now - timedelta(days=latest_offset_days * 2),
        latest_date=now - timedelta(days=latest_offset_days),
        trajectory=None,
        source="chat",
        semantic_score=score,
    )


def test_recent_beats_old_at_same_semantic_score():
    now = datetime(2026, 5, 1, tzinfo=UTC)
    items = [
        _ri(0.5, latest_offset_days=730, _id="old", now=now),  # 2 years
        _ri(0.5, latest_offset_days=30, _id="new", now=now),
    ]
    out = rerank_with_recency(items, half_life_days=365, now=now)
    assert out[0].id == "new"


def test_score_decays_with_half_life():
    now = datetime(2026, 5, 1, tzinfo=UTC)
    items = [_ri(1.0, latest_offset_days=365, _id="one_year", now=now)]
    out = rerank_with_recency(items, half_life_days=365, now=now)
    # At one half-life, score ≈ 0.5
    assert 0.45 < out[0].final_score < 0.55
