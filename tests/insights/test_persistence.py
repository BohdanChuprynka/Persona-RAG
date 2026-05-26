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
