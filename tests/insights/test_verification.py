from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlmodel import Session

from persona_rag.db.engine import make_engine
from persona_rag.db.models import InsightRow, VerificationSession
from persona_rag.insights.verification import (
    accept_insight,
    edit_insight,
    next_pending_insight,
    reject_insight,
    start_session,
    stop_session,
)


def _seed_pending(engine, insight_id: str, source: str = "chat") -> None:
    now = datetime.now(UTC)
    with Session(engine) as s:
        s.add(
            InsightRow(
                id=insight_id,
                category="bio",
                subject="school",
                text="studies CS",
                confidence=0.85,
                evidence_count=3,
                earliest_date=now,
                latest_date=now,
                trajectory=None,
                source_session_ids="[]",
                source=source,
                review_status="auto",
                edited_text=None,
                created_at=now,
                updated_at=now,
            )
        )
        s.commit()


def test_start_session_creates_fsm_row(tmp_path, monkeypatch):
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "persona_rag.insights.verification.make_engine", lambda: make_engine(db_path)
    )
    engine = make_engine(db_path)
    _seed_pending(engine, "i1")

    sess = start_session(user_id=42)
    assert sess.phase == "phase1_in_progress"
    assert sess.user_id == 42


def test_next_pending_returns_unverified_chat_insight(tmp_path, monkeypatch):
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "persona_rag.insights.verification.make_engine", lambda: make_engine(db_path)
    )
    engine = make_engine(db_path)
    _seed_pending(engine, "i1", source="chat")
    _seed_pending(engine, "i2", source="user_verified")  # already verified

    out = next_pending_insight(user_id=42)
    assert out is not None
    assert out.id == "i1"


@pytest.mark.asyncio
async def test_accept_marks_user_verified_and_embeds(tmp_path, monkeypatch):
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "persona_rag.insights.verification.make_engine", lambda: make_engine(db_path)
    )
    engine = make_engine(db_path)
    _seed_pending(engine, "i1")

    fake_client = MagicMock()
    with patch(
        "persona_rag.insights.verification.embed_batch",
        AsyncMock(return_value=[[0.0] * 1536]),
    ):
        await accept_insight("i1", qdrant_client=fake_client, collection="self_insights")

    with Session(engine) as s:
        row = s.get(InsightRow, "i1")
    assert row.source == "user_verified"
    assert row.review_status == "approved"
    assert row.confidence == 1.0
    fake_client.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_edit_replaces_text_and_preserves_original(tmp_path, monkeypatch):
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "persona_rag.insights.verification.make_engine", lambda: make_engine(db_path)
    )
    engine = make_engine(db_path)
    _seed_pending(engine, "i1")

    fake_client = MagicMock()
    with patch(
        "persona_rag.insights.verification.embed_batch",
        AsyncMock(return_value=[[0.0] * 1536]),
    ):
        await edit_insight(
            "i1",
            new_text="actually studies math",
            qdrant_client=fake_client,
            collection="self_insights",
        )

    with Session(engine) as s:
        row = s.get(InsightRow, "i1")
    assert row.text == "actually studies math"
    assert row.edited_text == "studies CS"
    assert row.source == "user_verified"
    fake_client.upsert.assert_called_once()


def test_reject_sets_rejected_and_removes_from_qdrant(tmp_path, monkeypatch):
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "persona_rag.insights.verification.make_engine", lambda: make_engine(db_path)
    )
    engine = make_engine(db_path)
    _seed_pending(engine, "i1")

    fake_client = MagicMock()
    reject_insight("i1", qdrant_client=fake_client, collection="self_insights")

    with Session(engine) as s:
        row = s.get(InsightRow, "i1")
    assert row.review_status == "rejected"
    fake_client.delete.assert_called_once()


def test_stop_session_persists_state(tmp_path, monkeypatch):
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "persona_rag.insights.verification.make_engine", lambda: make_engine(db_path)
    )
    engine = make_engine(db_path)
    _seed_pending(engine, "i1")

    start_session(user_id=42)
    stop_session(user_id=42)

    with Session(engine) as s:
        row = s.get(VerificationSession, 42)
    assert row.phase == "idle"
