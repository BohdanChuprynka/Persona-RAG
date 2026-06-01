from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import Session, select

from persona_rag.db.engine import make_engine
from persona_rag.db.models import (
    AlgoSignal,
    InsightRow,
    InsightRunState,
    RawInsightRow,
    VerificationSession,
)
from persona_rag.insights.consolidator import ConsolidatedInsight
from persona_rag.insights.extractor import RawInsight


def test_algo_signal_round_trip(tmp_path, monkeypatch):
    db_path = str(tmp_path / "p.db")
    engine = make_engine(db_path)

    now = datetime.now(UTC)
    row = AlgoSignal(
        kind="entity",
        subject="cyberpunk 2077",
        value_json='{"count": 12}',
        first_seen=now,
        last_seen=now,
        evidence_count=12,
        updated_at=now,
    )
    with Session(engine) as s:
        s.add(row)
        s.commit()

    with Session(engine) as s:
        loaded = list(s.exec(select(AlgoSignal)).all())
    assert len(loaded) == 1
    assert loaded[0].subject == "cyberpunk 2077"


def test_insight_row_round_trip(tmp_path):
    db_path = str(tmp_path / "p.db")
    engine = make_engine(db_path)

    now = datetime.now(UTC)
    row = InsightRow(
        id="abc-123",
        category="bio",
        subject="school",
        text="studies CS",
        confidence=0.85,
        evidence_count=3,
        earliest_date=now,
        latest_date=now,
        trajectory="active 2024-Q1 → present",
        source_session_ids='["s1", "s2", "s3"]',
        source="chat",
        review_status="auto",
        edited_text=None,
        created_at=now,
        updated_at=now,
    )
    with Session(engine) as s:
        s.add(row)
        s.commit()

    with Session(engine) as s:
        loaded = s.get(InsightRow, "abc-123")
    assert loaded is not None
    assert loaded.source == "chat"
    assert loaded.review_status == "auto"


def test_insight_run_state_round_trip(tmp_path):
    db_path = str(tmp_path / "p.db")
    engine = make_engine(db_path)

    now = datetime.now(UTC)
    row = InsightRunState(
        session_id="s-1",
        last_extracted_at=now,
        insights_count=5,
        failed=False,
        error_message=None,
    )
    with Session(engine) as s:
        s.add(row)
        s.commit()

    with Session(engine) as s:
        loaded = s.get(InsightRunState, "s-1")
    assert loaded is not None
    assert loaded.insights_count == 5


def test_verification_session_round_trip(tmp_path):
    db_path = str(tmp_path / "p.db")
    engine = make_engine(db_path)

    now = datetime.now(UTC)
    row = VerificationSession(
        user_id=42,
        phase="phase1_in_progress",
        current_insight_id="abc-123",
        current_question_id=None,
        started_at=now,
        updated_at=now,
    )
    with Session(engine) as s:
        s.add(row)
        s.commit()

    with Session(engine) as s:
        loaded = s.get(VerificationSession, 42)
    assert loaded is not None
    assert loaded.phase == "phase1_in_progress"


def test_raw_insight_has_verification_fields():
    r = RawInsight(
        session_id="s1",
        category="bio",
        subject="school",
        text="goes to school",
        confidence=1.0,
        source_quote="я в школі",
        extracted_at=datetime.now(UTC),
    )
    assert r.source_quote_validated is False
    assert r.verification_verdict is None
    assert r.verification_reason is None


def test_raw_insight_row_has_verification_columns():
    r = RawInsightRow(
        id="x1",
        session_id="s1",
        category="bio",
        subject="school",
        text="goes to school",
        confidence=1.0,
        source_quote="я в школі",
        extracted_at=datetime.now(UTC),
    )
    assert r.source_quote_validated is False
    assert r.verification_verdict is None
    assert r.verification_reason is None


def test_consolidated_insight_has_distinct_partners():
    now = datetime.now(UTC)
    ci = ConsolidatedInsight(
        id="i1",
        category="bio",
        canonical_subject="school",
        text="x",
        confidence=1.0,
        evidence_count=1,
        earliest_date=now,
        latest_date=now,
        trajectory=None,
        source_session_ids=["s1"],
    )
    assert ci.distinct_partners == 0


def test_insight_row_has_distinct_partners():
    now = datetime.now(UTC)
    row = InsightRow(
        id="x",
        category="bio",
        subject="school",
        text="x",
        confidence=1.0,
        evidence_count=1,
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
    assert row.distinct_partners == 0
