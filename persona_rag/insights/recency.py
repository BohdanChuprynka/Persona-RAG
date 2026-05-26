"""Retrieval-time recency rerank for self-insights."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel


class RankedInsight(BaseModel):
    id: str
    text: str
    category: str
    subject: str
    confidence: float
    evidence_count: int
    earliest_date: datetime
    latest_date: datetime
    trajectory: str | None
    source: str
    semantic_score: float
    final_score: float = 0.0


def _ensure_aware(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def rerank_with_recency(
    items: list[RankedInsight],
    *,
    half_life_days: int,
    now: datetime | None = None,
) -> list[RankedInsight]:
    """final_score = semantic_score * exp(-age_days * ln2 / half_life_days)."""
    now = now or datetime.now(UTC)
    out: list[RankedInsight] = []
    for it in items:
        age_days = (now - _ensure_aware(it.latest_date)).days
        decay = math.exp(-math.log(2) * age_days / half_life_days)
        out.append(it.model_copy(update={"final_score": it.semantic_score * decay}))
    out.sort(key=lambda x: x.final_score, reverse=True)
    return out


def from_qdrant_point(point: Any) -> RankedInsight:
    """Convert a Qdrant ScoredPoint to a RankedInsight."""
    p = point.payload or {}
    return RankedInsight(
        id=str(point.id),
        text=p.get("text", ""),
        category=p.get("category", "bio"),
        subject=p.get("subject", ""),
        confidence=float(p.get("confidence", 0.5)),
        evidence_count=int(p.get("evidence_count", 1)),
        earliest_date=datetime.fromisoformat(p["earliest_date"]),
        latest_date=datetime.fromisoformat(p["latest_date"]),
        trajectory=p.get("trajectory"),
        source=p.get("source", "chat"),
        semantic_score=float(point.score),
    )
