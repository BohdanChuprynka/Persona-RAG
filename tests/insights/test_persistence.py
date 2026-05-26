from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlmodel import Session, select

from persona_rag.db.engine import make_engine
from persona_rag.db.models import AlgoSignal, InsightRow
from persona_rag.insights.consolidator import ConsolidatedInsight
from persona_rag.insights.persistence import (
    persist_algo_signals,
    persist_insights,
)


def test_persist_algo_signals_inserts_rows(tmp_path, monkeypatch):
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "persona_rag.insights.persistence.make_engine",
        lambda: make_engine(db_path),
    )

    now = datetime.now(UTC)
    signals = {
        "entity": [
            {"subject": "python", "count": 12, "n_sessions": 4, "first_seen": now, "last_seen": now}
        ],
        "language": [
            {
                "subject": "uk",
                "count": 100,
                "n_sessions": 0,
                "first_seen": now,
                "last_seen": now,
                "percentage": 0.75,
            }
        ],
    }
    persist_algo_signals(signals)

    with Session(make_engine(db_path)) as s:
        rows = list(s.exec(select(AlgoSignal)).all())
    assert len(rows) == 2
    kinds = {r.kind for r in rows}
    assert kinds == {"entity", "language"}


@pytest.mark.asyncio
async def test_persist_insights_embeds_only_active(tmp_path, monkeypatch):
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "persona_rag.insights.persistence.make_engine",
        lambda: make_engine(db_path),
    )

    now = datetime.now(UTC)
    cis = [
        ConsolidatedInsight(
            id="i1",
            category="bio",
            canonical_subject="school",
            text="studies CS",
            confidence=0.85,
            evidence_count=3,
            earliest_date=now,
            latest_date=now,
            trajectory=None,
            source_session_ids=["s1"],
        ),
        ConsolidatedInsight(
            id="i2",
            category="bio",
            canonical_subject="rare",
            text="one-off",
            confidence=0.4,
            evidence_count=1,
            earliest_date=now,
            latest_date=now,
            trajectory=None,
            source_session_ids=["s2"],
        ),
    ]
    statuses = {"i1": "auto", "i2": "pending"}

    fake_client = MagicMock()
    with patch(
        "persona_rag.insights.persistence.embed_batch",
        AsyncMock(return_value=[[0.0] * 1536]),
    ) as mock_embed:
        await persist_insights(cis, statuses=statuses, qdrant_client=fake_client)

    # Both rows in SQLite
    with Session(make_engine(db_path)) as s:
        rows = list(s.exec(select(InsightRow)).all())
    assert len(rows) == 2

    # Only "auto" got embedded + upserted
    mock_embed.assert_awaited_once()
    fake_client.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_persist_insights_preserves_user_verified_on_rerun(tmp_path, monkeypatch):
    """Re-run with same canonical_subject must NOT overwrite user_verified rows."""
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "persona_rag.insights.persistence.make_engine",
        lambda: make_engine(db_path),
    )

    now = datetime.now(UTC)
    # Seed: user has already verified an insight with id=ID_X
    seed_id = (
        "abc123def4567890"  # stable_id for some (cat, subj) — value doesn't matter for this test
    )
    with Session(make_engine(db_path)) as s:
        s.add(
            InsightRow(
                id=seed_id,
                category="bio",
                subject="school",
                text="user-corrected text",
                confidence=1.0,
                evidence_count=3,
                earliest_date=now,
                latest_date=now,
                trajectory=None,
                source_session_ids="[]",
                source="user_verified",
                review_status="approved",
                edited_text="original text",
                created_at=now,
                updated_at=now,
            )
        )
        s.commit()

    # Re-run produces a ConsolidatedInsight with the same id but different text/status
    ci = ConsolidatedInsight(
        id=seed_id,
        category="bio",
        canonical_subject="school",
        text="freshly extracted text",
        confidence=0.85,
        evidence_count=5,
        earliest_date=now,
        latest_date=now,
        trajectory=None,
        source_session_ids=["s99"],
    )
    fake_client = MagicMock()
    with patch(
        "persona_rag.insights.persistence.embed_batch",
        AsyncMock(return_value=[[0.0] * 1536]),
    ):
        await persist_insights(
            [ci], statuses={seed_id: "auto"}, qdrant_client=fake_client, collection="self_insights"
        )

    with Session(make_engine(db_path)) as s:
        row = s.get(InsightRow, seed_id)
    assert row is not None
    # Identity preserved
    assert row.source == "user_verified"
    assert row.review_status == "approved"
    assert row.text == "user-corrected text"
    assert row.confidence == 1.0
    # Evidence freshness CAN be bumped
    assert row.evidence_count == 5
    # No Qdrant upsert — user's point shouldn't be churned
    fake_client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_persist_insights_preserves_rejected_on_rerun(tmp_path, monkeypatch):
    """Re-run must NOT resurrect previously-rejected insights."""
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "persona_rag.insights.persistence.make_engine",
        lambda: make_engine(db_path),
    )

    now = datetime.now(UTC)
    seed_id = "rejected-id-1234"
    with Session(make_engine(db_path)) as s:
        s.add(
            InsightRow(
                id=seed_id,
                category="interest",
                subject="cyberpunk",
                text="plays cyberpunk",
                confidence=0.6,
                evidence_count=2,
                earliest_date=now,
                latest_date=now,
                trajectory=None,
                source_session_ids="[]",
                source="chat",
                review_status="rejected",
                edited_text=None,
                created_at=now,
                updated_at=now,
            )
        )
        s.commit()

    ci = ConsolidatedInsight(
        id=seed_id,
        category="interest",
        canonical_subject="cyberpunk",
        text="plays cyberpunk again",
        confidence=0.9,
        evidence_count=7,
        earliest_date=now,
        latest_date=now,
        trajectory=None,
        source_session_ids=["s1", "s2"],
    )
    fake_client = MagicMock()
    with patch(
        "persona_rag.insights.persistence.embed_batch",
        AsyncMock(return_value=[[0.0] * 1536]),
    ):
        await persist_insights(
            [ci], statuses={seed_id: "auto"}, qdrant_client=fake_client, collection="self_insights"
        )

    with Session(make_engine(db_path)) as s:
        row = s.get(InsightRow, seed_id)
    assert row is not None
    assert row.review_status == "rejected"
    # Evidence freshness gets bumped (so user can see new mentions if they re-check)
    assert row.evidence_count == 7
    # No Qdrant upsert — rejected stays rejected
    fake_client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_persist_insights_updates_auto_in_place(tmp_path, monkeypatch):
    """Re-run on a chat/auto row updates text/conf/status in place (no duplicate row)."""
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "persona_rag.insights.persistence.make_engine",
        lambda: make_engine(db_path),
    )

    now = datetime.now(UTC)
    seed_id = "auto-row-id-1234"
    with Session(make_engine(db_path)) as s:
        s.add(
            InsightRow(
                id=seed_id,
                category="bio",
                subject="job",
                text="works at acme",
                confidence=0.7,
                evidence_count=2,
                earliest_date=now,
                latest_date=now,
                trajectory=None,
                source_session_ids="[]",
                source="chat",
                review_status="auto",
                edited_text=None,
                created_at=now,
                updated_at=now,
            )
        )
        s.commit()

    ci = ConsolidatedInsight(
        id=seed_id,
        category="bio",
        canonical_subject="job",
        text="works at acme — senior eng",  # refreshed
        confidence=0.9,
        evidence_count=5,
        earliest_date=now,
        latest_date=now,
        trajectory="active 2024 → present",
        source_session_ids=["s1", "s2", "s3"],
    )
    fake_client = MagicMock()
    with patch(
        "persona_rag.insights.persistence.embed_batch",
        AsyncMock(return_value=[[0.0] * 1536]),
    ):
        await persist_insights(
            [ci], statuses={seed_id: "auto"}, qdrant_client=fake_client, collection="self_insights"
        )

    # Still exactly one row with that id
    with Session(make_engine(db_path)) as s:
        rows = list(s.exec(select(InsightRow)).all())
    assert len(rows) == 1
    row = rows[0]
    assert row.text == "works at acme — senior eng"
    assert row.confidence == 0.9
    assert row.evidence_count == 5
    assert row.trajectory == "active 2024 → present"
    # Qdrant upserted (auto stays embeddable)
    fake_client.upsert.assert_called_once()
