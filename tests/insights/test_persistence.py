from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlmodel import Session, select

from persona_rag.db.engine import make_engine
from persona_rag.db.models import AlgoSignal, InsightRow, InsightRunState, RawInsightRow
from persona_rag.insights.consolidator import ConsolidatedInsight
from persona_rag.insights.extractor import RawInsight
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


# --- Stage C checkpointing: _persist_raws_and_mark ---
# See docs/superpowers/specs/2026-05-26-stage-c-checkpointing-design.md.


def _raw(session_id: str, subject: str = "x", text: str = "t", conf: float = 0.5) -> RawInsight:
    return RawInsight(
        session_id=session_id,
        category="bio",
        subject=subject,
        text=text,
        confidence=conf,
        source_quote="q",
        extracted_at=datetime.now(UTC),
    )


def test_persist_raws_atomic_with_runstate(tmp_path, monkeypatch):
    """One transaction writes raws AND marks InsightRunState — neither lands without the other."""
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "scripts.distill_insights.make_engine",
        lambda: make_engine(db_path),
    )
    from scripts.distill_insights import _persist_raws_and_mark

    raws = [_raw("s1", subject="a"), _raw("s1", subject="b")]
    _persist_raws_and_mark("s1", raws)

    with Session(make_engine(db_path)) as s:
        persisted = list(
            s.exec(select(RawInsightRow).where(RawInsightRow.session_id == "s1")).all()
        )
        state = s.get(InsightRunState, "s1")
    assert len(persisted) == 2
    assert {r.subject for r in persisted} == {"a", "b"}
    assert state is not None
    assert state.failed is False
    assert state.insights_count == 2


def test_persist_raws_replaces_prior_session_raws(tmp_path, monkeypatch):
    """Re-calling for the same session deletes the old raws so Stage D never sees duplicates.

    This is what makes --force-session safe even when raws were persisted on a prior run.
    """
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "scripts.distill_insights.make_engine",
        lambda: make_engine(db_path),
    )
    from scripts.distill_insights import _persist_raws_and_mark

    # First call persists 3 raws
    _persist_raws_and_mark("s1", [_raw("s1", subject=c) for c in ("a", "b", "c")])
    # Second call with 1 raw must REPLACE, not append
    _persist_raws_and_mark("s1", [_raw("s1", subject="z", text="new")])

    with Session(make_engine(db_path)) as s:
        persisted = list(
            s.exec(select(RawInsightRow).where(RawInsightRow.session_id == "s1")).all()
        )
        state = s.get(InsightRunState, "s1")
    assert len(persisted) == 1
    assert persisted[0].subject == "z"
    assert persisted[0].text == "new"
    assert state is not None
    assert state.insights_count == 1


def test_load_resume_state_returns_skip_ids_and_raws(tmp_path, monkeypatch):
    """In incremental mode, _load_resume_state hydrates raws from DB for already-done sessions."""
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "scripts.distill_insights.make_engine",
        lambda: make_engine(db_path),
    )
    from scripts.distill_insights import _load_resume_state, _persist_raws_and_mark

    # Two completed sessions with persisted raws
    _persist_raws_and_mark("s1", [_raw("s1", subject="a"), _raw("s1", subject="b")])
    _persist_raws_and_mark("s2", [_raw("s2", subject="c")])
    # One previously-failed session — should NOT be skipped (so it retries)
    with Session(make_engine(db_path)) as s:
        s.add(
            InsightRunState(
                session_id="s-failed",
                last_extracted_at=datetime.now(UTC),
                insights_count=0,
                failed=True,
                error_message="boom",
            )
        )
        s.commit()

    skip_ids, resumed = _load_resume_state(mode="incremental")
    assert skip_ids == {"s1", "s2"}
    assert "s-failed" not in skip_ids
    assert len(resumed) == 3
    assert {r.subject for r in resumed} == {"a", "b", "c"}


def test_load_resume_state_full_mode_returns_empty(tmp_path, monkeypatch):
    """In --mode full, resume state is bypassed entirely (truncation handles the rest)."""
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "scripts.distill_insights.make_engine",
        lambda: make_engine(db_path),
    )
    from scripts.distill_insights import _load_resume_state, _persist_raws_and_mark

    _persist_raws_and_mark("s1", [_raw("s1")])
    skip_ids, resumed = _load_resume_state(mode="full")
    assert skip_ids == set()
    assert resumed == []


def test_persist_raws_empty_list_still_marks_session(tmp_path, monkeypatch):
    """If extractor returns zero raws (low-signal session), still mark the session done.

    Otherwise it'd get re-extracted on every incremental run forever.
    """
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "scripts.distill_insights.make_engine",
        lambda: make_engine(db_path),
    )
    from scripts.distill_insights import _persist_raws_and_mark

    _persist_raws_and_mark("s-empty", [])

    with Session(make_engine(db_path)) as s:
        rows = list(
            s.exec(select(RawInsightRow).where(RawInsightRow.session_id == "s-empty")).all()
        )
        state = s.get(InsightRunState, "s-empty")
    assert rows == []
    assert state is not None
    assert state.failed is False
    assert state.insights_count == 0
