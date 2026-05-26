"""Render the persona description from user_verified + onboarding insights."""

from __future__ import annotations

from sqlmodel import Session, select

from persona_rag.db.engine import make_engine
from persona_rag.db.models import InsightRow


def generate_persona_description(fallback: str, *, cap: int = 5) -> str:
    """Concatenate up to ``cap`` highest-confidence user-confirmed bio insights.

    Returns ``fallback`` if no qualifying insights exist.
    """
    with Session(make_engine()) as s:
        rows = list(
            s.exec(
                select(InsightRow)
                .where(InsightRow.category == "bio")
                .where(InsightRow.source.in_(("user_verified", "onboarding")))  # type: ignore[attr-defined]
                .where(InsightRow.review_status.in_(("auto", "approved")))  # type: ignore[attr-defined]
            ).all()
        )
    if not rows:
        return fallback
    rows.sort(key=lambda r: r.confidence, reverse=True)
    chosen = rows[:cap]
    return ". ".join(r.text.rstrip(".") for r in chosen) + "."
