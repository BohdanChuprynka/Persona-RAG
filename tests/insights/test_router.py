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
            min_distinct_partners=0,
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
            min_distinct_partners=0,
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
            min_distinct_partners=0,
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
            min_distinct_partners=0,
            stale_years=2.0,
            stale_min_evidence=5,
            now=now,
        )
        == "auto"
    )


def test_routes_pending_when_partners_below_threshold():
    """Spec §5.7 — insight from too few distinct partners is pending."""
    now = datetime(2026, 1, 1, tzinfo=UTC)
    ci = ConsolidatedInsight(
        id="x",
        category="interest",
        canonical_subject="basketball",
        text="plays basketball",
        confidence=0.9,
        evidence_count=3,
        earliest_date=now,
        latest_date=now,
        trajectory=None,
        source_session_ids=["s1", "s2", "s3"],
        distinct_partners=1,
    )
    result = route_insight(
        ci,
        confidence_threshold=0.7,
        min_evidence=3,
        min_distinct_partners=2,
        stale_years=2.0,
        stale_min_evidence=5,
        now=now,
    )
    assert result == "pending"


def test_routes_auto_when_partners_at_threshold():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    ci = ConsolidatedInsight(
        id="x",
        category="interest",
        canonical_subject="running",
        text="runs",
        confidence=0.9,
        evidence_count=3,
        earliest_date=now,
        latest_date=now,
        trajectory=None,
        source_session_ids=["s1", "s2", "s3"],
        distinct_partners=2,
    )
    result = route_insight(
        ci,
        confidence_threshold=0.7,
        min_evidence=3,
        min_distinct_partners=2,
        stale_years=2.0,
        stale_min_evidence=5,
        now=now,
    )
    assert result == "auto"


def test_min_evidence_default_bumped_from_2_to_3():
    """Insight with ev=2 no longer auto-promotes under new defaults."""
    now = datetime(2026, 1, 1, tzinfo=UTC)
    ci = ConsolidatedInsight(
        id="x",
        category="interest",
        canonical_subject="running",
        text="runs",
        confidence=0.9,
        evidence_count=2,
        earliest_date=now,
        latest_date=now,
        trajectory=None,
        source_session_ids=["s1", "s2"],
        distinct_partners=2,
    )
    result = route_insight(
        ci,
        confidence_threshold=0.7,
        min_evidence=3,
        min_distinct_partners=2,
        stale_years=2.0,
        stale_min_evidence=5,
        now=now,
    )
    assert result == "pending"
