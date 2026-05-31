# ruff: noqa: RUF001
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlmodel import Session, select

from persona_rag.db.engine import make_engine
from persona_rag.db.models import (
    AlgoSignal,
    InsightRow,
    PersonaTurnRow,
)
from persona_rag.insights.algo import run_stage_a
from persona_rag.insights.consolidator import consolidate
from persona_rag.insights.extractor import extract_from_session
from persona_rag.insights.persistence import persist_algo_signals, persist_insights
from persona_rag.insights.router import route_insight
from persona_rag.insights.sessions import build_sessions, filter_high_signal


def _seed_corpus(engine, persona_id: str, n_sessions: int = 3) -> None:
    """Create n_sessions x 15-turn synthetic conversations talking about C++ and ML."""
    now = datetime(2026, 1, 1, tzinfo=UTC)
    with Session(engine) as s:
        for sess_i in range(n_sessions):
            base = now + timedelta(days=sess_i * 7)
            for t_i in range(15):
                ts = base + timedelta(minutes=t_i)
                topic = "c++" if sess_i % 2 == 0 else "ml"
                s.add(
                    PersonaTurnRow(
                        id=f"{sess_i}-{t_i}",
                        your_reply=f"я зара кодю {topic} це тренування мозку",
                        incoming_context_json='["що робиш?"]',
                        channel="telegram",
                        chat_id_hash=f"chat{sess_i}",
                        recipient_id_hash="r1",
                        timestamp=ts,
                        language="uk",
                        your_reply_len_chars=40,
                        your_reply_emoji_count=0,
                    )
                )
        s.commit()


@pytest.mark.asyncio
async def test_pipeline_e2e_produces_active_insight(tmp_path, monkeypatch):
    db_path = str(tmp_path / "p.db")
    engine = make_engine(db_path)
    monkeypatch.setattr(
        "persona_rag.insights.persistence.make_engine", lambda: make_engine(db_path)
    )

    _seed_corpus(engine, persona_id="me")

    # Stage A
    with Session(engine) as s:
        all_turns = list(s.exec(select(PersonaTurnRow)).all())
    stage_a = run_stage_a(all_turns)
    persist_algo_signals(stage_a)
    with Session(engine) as s:
        signal_rows = list(s.exec(select(AlgoSignal)).all())
    assert len(signal_rows) > 0

    # Stage B
    sessions = build_sessions(all_turns, gap_hours=6)
    high = filter_high_signal(
        sessions,
        history_years=10,
        min_turns=10,
        min_chars=300,
        max_sessions=10,
        now=datetime(2026, 2, 1, tzinfo=UTC),
    )
    assert len(high) >= 1

    # Stage C — stub the LLM
    canned = (
        '{"insights": [{"category": "interest", "subject": "programming",'
        ' "text": "Studies C++ programming",'
        ' "confidence": 0.85, "source_quote": "я зара кодю"}]}'
    )
    raws = []
    with patch(
        "persona_rag.insights.extractor.chat_complete",
        AsyncMock(return_value=canned),
    ):
        for sess in high:
            raws.extend(
                await extract_from_session(sess, persona_name="TestPersona", entity_hints=[])
            )
    assert raws

    # Stage D — synonyms (none) + 3+ evidence → triggers LLM mock
    canned_merge = "Active C++ learner.\nTrajectory: active 2026-Q1"
    with patch(
        "persona_rag.insights.consolidator.chat_complete",
        AsyncMock(return_value=canned_merge),
    ):
        consolidated = await consolidate(raws, synonyms={})
    assert consolidated

    # Stage E
    now = datetime(2026, 2, 1, tzinfo=UTC)
    statuses = {
        ci.id: route_insight(
            ci,
            confidence_threshold=0.7,
            min_evidence=2,
            min_distinct_partners=0,
            stale_years=2.0,
            stale_min_evidence=5,
            now=now,
        )
        for ci in consolidated
    }

    # Stage F
    fake_client = MagicMock()
    with patch(
        "persona_rag.insights.persistence.embed_batch",
        AsyncMock(return_value=[[0.0] * 1536] * len(consolidated)),
    ):
        await persist_insights(
            consolidated, statuses=statuses, qdrant_client=fake_client, collection="self_insights"
        )

    with Session(engine) as s:
        rows = list(s.exec(select(InsightRow)).all())
    assert len(rows) >= 1
    active = [r for r in rows if r.review_status == "auto"]
    assert len(active) >= 1
    fake_client.upsert.assert_called()
