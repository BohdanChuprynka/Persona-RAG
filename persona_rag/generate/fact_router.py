# ruff: noqa: RUF001
"""Serving-time fact router: self-description intent + CORE identity loader.

Vague self-description queries ("tell me about yourself") are reached by ROUTE --
a curated CORE of vault identity facts -- not by embedding similarity, which is
unreliable for meta-questions. Specific questions fall through to the existing
semantic retrieval. See spec 2026-06-03 sections 6 and 10.
"""

from __future__ import annotations

import math

from sqlmodel import Session, select

from persona_rag.db.engine import make_engine
from persona_rag.db.models import InsightRow
from persona_rag.index.embedder import embed_batch
from persona_rag.insights.recency import RankedInsight

IDENTITY_CATEGORIES = {"bio", "relationship", "value", "opinion"}
_PRIORITY = {"bio": 0, "relationship": 1, "value": 2, "opinion": 3}

ANCHOR_PHRASES = [
    "розкажи про себе",
    "хто ти",
    "опиши себе",
    "розкажи шось про себе",
    "расскажи о себе",
    "кто ты",
    "tell me about yourself",
    "who are you",
    "tell me about you",
]

_anchor_vecs: list[list[float]] | None = None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def classify_self_description(
    vec: list[float], anchor_vecs: list[list[float]], *, threshold: float
) -> bool:
    """Pure: True if `vec` is within `threshold` cosine of any anchor."""
    return any(_cosine(vec, a) >= threshold for a in anchor_vecs)


async def anchor_vecs() -> list[list[float]]:
    """Embed the anchor phrases once; cache module-level."""
    global _anchor_vecs
    if _anchor_vecs is None:
        _anchor_vecs = await embed_batch(ANCHOR_PHRASES)
    return _anchor_vecs


def load_core_facts(*, limit: int, query_lang: str) -> list[RankedInsight]:
    """Top vault identity facts by (category priority, confidence). Approved only."""
    with Session(make_engine()) as s:
        rows = list(
            s.exec(
                select(InsightRow).where(
                    InsightRow.source == "vault",
                    InsightRow.review_status.in_(("auto", "approved")),  # type: ignore[attr-defined]
                )
            ).all()
        )
    rows = [r for r in rows if r.category in IDENTITY_CATEGORIES]
    rows.sort(key=lambda r: (_PRIORITY.get(r.category, 9), -r.confidence))
    out: list[RankedInsight] = []
    for r in rows[:limit]:
        out.append(
            RankedInsight(
                id=r.id,
                text=r.text,
                text_en=r.text_en,
                category=r.category,
                subject=r.subject,
                confidence=r.confidence,
                evidence_count=r.evidence_count,
                earliest_date=r.earliest_date,
                latest_date=r.latest_date,
                trajectory=r.trajectory,
                source=r.source,
                semantic_score=1.0,
                final_score=1.0,
            )
        )
    return out
