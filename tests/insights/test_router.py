from __future__ import annotations

from datetime import UTC, datetime, timedelta

from persona_rag.insights.consolidator import ConsolidatedInsight
from persona_rag.insights.router import route_insight


def _ci(
    *,
    conf: float,
    evidence: int,
    latest_offset_days: int,
    now: datetime,
) -> ConsolidatedInsight:
    return ConsolidatedInsight(
        id=f"id-{conf}-{evidence}-{latest_offset_days}",
        category="bio",
        canonical_subject="school",
        text="studies CS",
        confidence=conf,
        evidence_count=evidence,
        earliest_date=now - timedelta(days=latest_offset_days * 2),
        latest_date=now - timedelta(days=latest_offset_days),
        trajectory=None,
        source_session_ids=["s1"],
    )


def test_route_auto_when_strong_and_recent():
    now = datetime(2026, 5, 1, tzinfo=UTC)
    ci = _ci(conf=0.85, evidence=4, latest_offset_days=30, now=now)
    assert (
        route_insight(
            ci,
            confidence_threshold=0.7,
            min_evidence=2,
            stale_years=2.0,
            stale_min_evidence=5,
            now=now,
        )
        == "auto"
    )


def test_route_pending_when_weak():
    now = datetime(2026, 5, 1, tzinfo=UTC)
    ci = _ci(conf=0.4, evidence=1, latest_offset_days=10, now=now)
    assert (
        route_insight(
            ci,
            confidence_threshold=0.7,
            min_evidence=2,
            stale_years=2.0,
            stale_min_evidence=5,
            now=now,
        )
        == "pending"
    )


def test_route_pending_when_stale_and_weak_evidence():
    now = datetime(2026, 5, 1, tzinfo=UTC)
    # latest > 2 years ago, evidence < 5 → stale-demote
    ci = _ci(conf=0.9, evidence=3, latest_offset_days=900, now=now)
    assert (
        route_insight(
            ci,
            confidence_threshold=0.7,
            min_evidence=2,
            stale_years=2.0,
            stale_min_evidence=5,
            now=now,
        )
        == "pending"
    )


def test_route_auto_when_stale_but_strong_evidence():
    now = datetime(2026, 5, 1, tzinfo=UTC)
    # latest > 2 years ago BUT evidence ≥ 5 → still auto
    ci = _ci(conf=0.85, evidence=8, latest_offset_days=900, now=now)
    assert (
        route_insight(
            ci,
            confidence_threshold=0.7,
            min_evidence=2,
            stale_years=2.0,
            stale_min_evidence=5,
            now=now,
        )
        == "auto"
    )
