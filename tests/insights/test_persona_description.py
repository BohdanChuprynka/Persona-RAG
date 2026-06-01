from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import Session

from persona_rag.db.engine import make_engine
from persona_rag.db.models import InsightRow
from persona_rag.insights.persona_description import generate_persona_description


def _seed_insight(engine, source: str, category: str, text: str, conf: float = 0.95) -> None:
    now = datetime.now(UTC)
    with Session(engine) as s:
        s.add(
            InsightRow(
                id=f"{source}-{text[:10]}",
                category=category,
                subject="x",
                text=text,
                confidence=conf,
                evidence_count=3,
                earliest_date=now,
                latest_date=now,
                trajectory=None,
                source_session_ids="[]",
                source=source,
                review_status="approved",
                edited_text=None,
                created_at=now,
                updated_at=now,
            )
        )
        s.commit()


def test_generated_uses_user_verified_and_onboarding(tmp_path, monkeypatch):
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "persona_rag.insights.persona_description.make_engine",
        lambda: make_engine(db_path),
    )
    engine = make_engine(db_path)
    _seed_insight(engine, source="user_verified", category="bio", text="17 years old")
    _seed_insight(engine, source="onboarding", category="bio", text="lives in Kyiv")

    out = generate_persona_description(fallback="(none)")
    assert "17 years old" in out
    assert "lives in Kyiv" in out
    assert out != "(none)"


def test_fallback_when_no_verified_insights(tmp_path, monkeypatch):
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "persona_rag.insights.persona_description.make_engine",
        lambda: make_engine(db_path),
    )
    out = generate_persona_description(fallback="env-fallback")
    assert out == "env-fallback"


def test_skips_chat_only_insights(tmp_path, monkeypatch):
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "persona_rag.insights.persona_description.make_engine",
        lambda: make_engine(db_path),
    )
    engine = make_engine(db_path)
    _seed_insight(engine, source="chat", category="bio", text="unverified")
    out = generate_persona_description(fallback="env-fallback")
    assert out == "env-fallback"
