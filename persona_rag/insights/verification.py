"""Stage G phase 1 — interactive verification FSM + state mutations.

Pure data layer. Bot handlers in `persona_rag/bot/handlers/admin.py` wire
these to Telegram callbacks.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sqlmodel import Session, select

from persona_rag.db.engine import make_engine
from persona_rag.db.models import InsightRow, VerificationSession
from persona_rag.index.embedder import embed_batch


def start_session(user_id: int) -> VerificationSession:
    now = datetime.now(UTC)
    with Session(make_engine()) as s:
        row = s.get(VerificationSession, user_id)
        if row is None:
            row = VerificationSession(
                user_id=user_id,
                phase="phase1_in_progress",
                started_at=now,
                updated_at=now,
            )
            s.add(row)
        else:
            row.phase = "phase1_in_progress"
            row.updated_at = now
            s.add(row)
        s.commit()
        s.refresh(row)
    return row


def stop_session(user_id: int) -> None:
    with Session(make_engine()) as s:
        row = s.get(VerificationSession, user_id)
        if row is None:
            return
        row.phase = "idle"
        row.updated_at = datetime.now(UTC)
        s.add(row)
        s.commit()


def next_pending_insight(user_id: int) -> InsightRow | None:
    """Return the next chat-source insight that hasn't been verified yet."""
    with Session(make_engine()) as s:
        rows = list(
            s.exec(
                select(InsightRow)
                .where(InsightRow.source == "chat")
                .where(InsightRow.review_status.in_(("auto", "pending")))  # type: ignore[attr-defined]
            ).all()
        )
    if not rows:
        return None
    rows.sort(key=lambda r: (-r.confidence, -r.evidence_count))
    return rows[0]


def _upsert_point(
    client: QdrantClient, collection: str, row: InsightRow, vector: list[float]
) -> None:
    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(
                id=row.id,
                vector=vector,
                payload={
                    "category": row.category,
                    "subject": row.subject,
                    "text": row.text,
                    "confidence": row.confidence,
                    "evidence_count": row.evidence_count,
                    "earliest_date": row.earliest_date.isoformat(),
                    "latest_date": row.latest_date.isoformat(),
                    "trajectory": row.trajectory,
                    "source": row.source,
                    "review_status": row.review_status,
                },
            )
        ],
    )


async def accept_insight(insight_id: str, *, qdrant_client: QdrantClient, collection: str) -> None:
    now = datetime.now(UTC)
    with Session(make_engine()) as s:
        row = s.get(InsightRow, insight_id)
        if row is None:
            return
        row.source = "user_verified"
        row.review_status = "approved"
        row.confidence = 1.0
        row.updated_at = now
        s.add(row)
        s.commit()
        s.refresh(row)

    vec = (await embed_batch([row.text]))[0]
    _upsert_point(qdrant_client, collection, row, vec)


async def edit_insight(
    insight_id: str,
    *,
    new_text: str,
    qdrant_client: QdrantClient,
    collection: str,
) -> None:
    now = datetime.now(UTC)
    with Session(make_engine()) as s:
        row = s.get(InsightRow, insight_id)
        if row is None:
            return
        row.edited_text = row.text
        row.text = new_text
        row.source = "user_verified"
        row.review_status = "approved"
        row.confidence = 1.0
        row.updated_at = now
        s.add(row)
        s.commit()
        s.refresh(row)

    vec = (await embed_batch([row.text]))[0]
    _upsert_point(qdrant_client, collection, row, vec)


def reject_insight(insight_id: str, *, qdrant_client: QdrantClient, collection: str) -> None:
    now = datetime.now(UTC)
    with Session(make_engine()) as s:
        row = s.get(InsightRow, insight_id)
        if row is None:
            return
        row.review_status = "rejected"
        row.updated_at = now
        s.add(row)
        s.commit()

    with contextlib.suppress(Exception):
        qdrant_client.delete(collection_name=collection, points_selector=[insight_id])
