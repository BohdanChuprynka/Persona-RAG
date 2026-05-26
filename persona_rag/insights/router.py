"""Stage E — confidence + staleness routing → review_status."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal

from persona_rag.insights.consolidator import ConsolidatedInsight

ReviewStatus = Literal["auto", "pending", "approved", "rejected"]


def _ensure_aware(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def route_insight(
    ci: ConsolidatedInsight,
    *,
    confidence_threshold: float,
    min_evidence: int,
    stale_years: float,
    stale_min_evidence: int,
    now: datetime | None = None,
) -> ReviewStatus:
    """Decide review_status for a newly-consolidated insight."""
    now = now or datetime.now(UTC)
    age = now - _ensure_aware(ci.latest_date)
    stale_cutoff = timedelta(days=stale_years * 365)

    # Stale + weakly-evidenced → demote
    if age > stale_cutoff and ci.evidence_count < stale_min_evidence:
        return "pending"

    if ci.confidence >= confidence_threshold and ci.evidence_count >= min_evidence:
        return "auto"

    return "pending"
