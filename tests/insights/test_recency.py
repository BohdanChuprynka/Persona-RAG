from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from persona_rag.insights.recency import RankedInsight, from_qdrant_point, rerank_with_recency


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


def test_ranked_insight_text_en_optional_default_none():
    now = datetime.now(UTC)
    r = RankedInsight(
        id="a",
        text="навч",
        category="bio",
        subject="school",
        confidence=1.0,
        evidence_count=1,
        earliest_date=now,
        latest_date=now,
        trajectory=None,
        source="vault",
        semantic_score=0.4,
    )
    assert r.text_en is None


def test_from_qdrant_point_reads_text_en():
    now_iso = "2026-06-03T00:00:00+00:00"
    point = MagicMock()
    point.id = "p1"
    point.score = 0.7
    point.payload = {
        "text": "навч",
        "text_en": "studies",
        "category": "bio",
        "subject": "school",
        "confidence": 0.9,
        "evidence_count": 1,
        "earliest_date": now_iso,
        "latest_date": now_iso,
        "source": "vault",
    }
    r = from_qdrant_point(point)
    assert r.text_en == "studies"
