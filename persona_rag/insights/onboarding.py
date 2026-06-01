"""Stage G phase 2 — gap-fill onboarding question runner + answer parser."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sqlmodel import Session

from persona_rag.db.engine import make_engine
from persona_rag.db.models import InsightRow
from persona_rag.index.embedder import embed_batch
from persona_rag.index.qdrant_store import to_qdrant_point_id


class OnboardingQuestion(BaseModel):
    id: str
    category: str
    subject: str
    question: str
    parse: str  # "string" | "multiline_list"
    optional: bool = False


def load_questions(path: Path) -> list[OnboardingQuestion]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [OnboardingQuestion(**item) for item in data]


def parse_answer(text: str, *, parse_kind: str) -> list[str]:
    if parse_kind == "multiline_list":
        return [line.strip() for line in text.split("\n") if line.strip()]
    return [text.strip()] if text.strip() else []


async def save_answer(
    *,
    question: OnboardingQuestion,
    answer_text: str,
    qdrant_client: QdrantClient,
    collection: str,
) -> None:
    items = parse_answer(answer_text, parse_kind=question.parse)
    if not items:
        return

    now = datetime.now(UTC)
    rows: list[InsightRow] = []
    for item in items:
        rows.append(
            InsightRow(
                id=str(uuid.uuid4()),
                category=question.category,
                subject=question.subject,
                text=item,
                confidence=1.0,
                evidence_count=1,
                earliest_date=now,
                latest_date=now,
                trajectory=None,
                source_session_ids="[]",
                source="onboarding",
                review_status="approved",
                edited_text=None,
                created_at=now,
                updated_at=now,
            )
        )

    with Session(make_engine()) as s:
        for r in rows:
            s.add(r)
        s.commit()
        # Refresh inside the session so attributes are loaded before detach.
        for r in rows:
            s.refresh(r)

    vectors = await embed_batch([r.text for r in rows])
    points = [
        PointStruct(
            id=to_qdrant_point_id(r.id),
            vector=vec,
            payload={
                "sqlite_id": r.id,
                "category": r.category,
                "subject": r.subject,
                "text": r.text,
                "confidence": r.confidence,
                "evidence_count": r.evidence_count,
                "earliest_date": r.earliest_date.isoformat(),
                "latest_date": r.latest_date.isoformat(),
                "trajectory": r.trajectory,
                "source": r.source,
                "review_status": r.review_status,
            },
        )
        for r, vec in zip(rows, vectors, strict=True)
    ]
    qdrant_client.upsert(collection_name=collection, points=points)
