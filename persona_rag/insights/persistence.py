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
from persona_rag.insights.consolidator import ConsolidatedInsight
from persona_rag.insights.router import ReviewStatus


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

    with Session(make_engine()) as s:
        for ci in insights:
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
                    source="chat",
                    review_status=statuses[ci.id],
                    edited_text=None,
                    created_at=now,
                    updated_at=now,
                )
            )
        s.commit()

    active = [ci for ci in insights if statuses[ci.id] in ("auto", "approved")]
    if not active:
        return

    vectors = await embed_batch([ci.text for ci in active])
    points = [
        PointStruct(
            id=ci.id,
            vector=vec,
            payload={
                "category": ci.category,
                "subject": ci.canonical_subject,
                "text": ci.text,
                "confidence": ci.confidence,
                "evidence_count": ci.evidence_count,
                "earliest_date": ci.earliest_date.isoformat(),
                "latest_date": ci.latest_date.isoformat(),
                "trajectory": ci.trajectory,
                "source": "chat",
                "review_status": statuses[ci.id],
            },
        )
        for ci, vec in zip(active, vectors, strict=True)
    ]
    qdrant_client.upsert(collection_name=collection, points=points)
