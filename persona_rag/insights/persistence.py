"""Stage F — persist algo signals + insights to SQLite + Qdrant."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sqlmodel import Session, select

from persona_rag.db.engine import make_engine
from persona_rag.db.models import AlgoSignal, InsightRow
from persona_rag.index.embedder import embed_batch
from persona_rag.index.qdrant_store import to_qdrant_point_id
from persona_rag.insights.consolidator import ConsolidatedInsight
from persona_rag.insights.router import ReviewStatus


def _aware(dt: datetime) -> datetime:
    """Coerce a naive datetime to UTC-aware; leave aware datetimes unchanged."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def persist_algo_signals(signals: dict[str, list[dict[str, Any]]]) -> None:
    """Replace algo_signal table contents with the new run's signals."""
    now = datetime.now(UTC)
    with Session(make_engine()) as s:
        # Clean existing
        for old in s.exec(select(AlgoSignal)).all():
            s.delete(old)
        # Insert new
        for kind, items in signals.items():
            for item in items:
                s.add(
                    AlgoSignal(
                        kind=kind,
                        subject=str(item["subject"]),
                        value_json=json.dumps(
                            {k: v for k, v in item.items() if k != "subject"},
                            default=str,
                        ),
                        first_seen=item["first_seen"],
                        last_seen=item["last_seen"],
                        evidence_count=int(item.get("count", 0)),
                        updated_at=now,
                    )
                )
        s.commit()


async def persist_insights(
    insights: list[ConsolidatedInsight],
    *,
    statuses: dict[str, ReviewStatus],
    qdrant_client: QdrantClient,
    collection: str = "self_insights",
) -> None:
    """Write every insight to SQLite; embed + upsert only auto/approved to Qdrant."""
    now = datetime.now(UTC)

    eligible_for_qdrant: list[ConsolidatedInsight] = []
    eligible_statuses: dict[str, ReviewStatus] = {}

    with Session(make_engine()) as s:
        for ci in insights:
            existing = s.get(InsightRow, ci.id)
            if existing is None:
                # New row — insert with the routed status
                s.add(
                    InsightRow(
                        id=ci.id,
                        category=ci.category,
                        subject=ci.canonical_subject,
                        text=ci.text,
                        confidence=ci.confidence,
                        evidence_count=ci.evidence_count,
                        earliest_date=ci.earliest_date,
                        latest_date=ci.latest_date,
                        trajectory=ci.trajectory,
                        source_session_ids=json.dumps(ci.source_session_ids),
                        distinct_partners=ci.distinct_partners,
                        source="chat",
                        review_status=statuses[ci.id],
                        edited_text=None,
                        created_at=now,
                        updated_at=now,
                    )
                )
                if statuses[ci.id] in ("auto", "approved"):
                    eligible_for_qdrant.append(ci)
                    eligible_statuses[ci.id] = statuses[ci.id]
                continue

            # Existing row — apply user-touch protection. "vault" facts are
            # user-curated (spec 2026-06-03 §7) and must never be overwritten by a
            # chat re-run that happens to derive the same (category, subject) id.
            if (
                existing.source in ("user_verified", "onboarding", "vault")
                or existing.review_status == "rejected"
            ):
                # User has authority on this insight. Only bump evidence freshness
                # signals; do NOT replace text/status/source/confidence.
                existing.evidence_count = ci.evidence_count
                if _aware(ci.latest_date) > _aware(existing.latest_date):
                    existing.latest_date = ci.latest_date
                if _aware(ci.earliest_date) < _aware(existing.earliest_date):
                    existing.earliest_date = ci.earliest_date
                existing.updated_at = now
                s.add(existing)
                # Skip Qdrant — user-verified/onboarding already-correct point stays put;
                # rejected stays out.
                continue

            # Existing row is source=chat AND status in (auto, pending) — full refresh.
            existing.text = ci.text
            existing.confidence = ci.confidence
            existing.evidence_count = ci.evidence_count
            existing.earliest_date = min(_aware(existing.earliest_date), _aware(ci.earliest_date))
            existing.latest_date = max(_aware(existing.latest_date), _aware(ci.latest_date))
            existing.trajectory = ci.trajectory
            existing.source_session_ids = json.dumps(ci.source_session_ids)
            existing.distinct_partners = ci.distinct_partners
            existing.review_status = statuses[ci.id]
            existing.updated_at = now
            s.add(existing)
            if statuses[ci.id] in ("auto", "approved"):
                eligible_for_qdrant.append(ci)
                eligible_statuses[ci.id] = statuses[ci.id]
        s.commit()

    if not eligible_for_qdrant:
        return

    vectors = await embed_batch([ci.text for ci in eligible_for_qdrant])
    points = [
        PointStruct(
            id=to_qdrant_point_id(ci.id),
            vector=vec,
            payload={
                "sqlite_id": ci.id,
                "category": ci.category,
                "subject": ci.canonical_subject,
                "text": ci.text,
                "confidence": ci.confidence,
                "evidence_count": ci.evidence_count,
                "earliest_date": ci.earliest_date.isoformat(),
                "latest_date": ci.latest_date.isoformat(),
                "trajectory": ci.trajectory,
                "source": "chat",
                "review_status": eligible_statuses[ci.id],
            },
        )
        for ci, vec in zip(eligible_for_qdrant, vectors, strict=True)
    ]
    qdrant_client.upsert(collection_name=collection, points=points)
