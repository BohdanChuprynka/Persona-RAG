from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from sqlmodel import Session, select

from persona_rag.db.engine import make_engine
from persona_rag.db.models import InsightRow
from persona_rag.insights.onboarding import (
    OnboardingQuestion,
    load_questions,
    parse_answer,
    save_answer,
)


def test_load_questions_returns_list(tmp_path: Path):
    src = tmp_path / "q.yaml"
    src.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "name",
                    "category": "bio",
                    "subject": "name",
                    "question": "What's your name?",
                    "parse": "string",
                    "optional": False,
                }
            ]
        )
    )
    out = load_questions(src)
    assert len(out) == 1
    assert isinstance(out[0], OnboardingQuestion)
    assert out[0].id == "name"


def test_parse_answer_string():
    out = parse_answer("Bohdan Chuprynka", parse_kind="string")
    assert out == ["Bohdan Chuprynka"]


def test_parse_answer_multiline_list_splits():
    out = parse_answer("a\nb\nc", parse_kind="multiline_list")
    assert out == ["a", "b", "c"]


def test_parse_answer_multiline_drops_blanks():
    out = parse_answer("a\n\n\nb", parse_kind="multiline_list")
    assert out == ["a", "b"]


@pytest.mark.asyncio
async def test_save_answer_creates_insight_rows(tmp_path, monkeypatch):
    db_path = str(tmp_path / "p.db")
    make_engine(db_path)  # initialize tables
    monkeypatch.setattr("persona_rag.insights.onboarding.make_engine", lambda: make_engine(db_path))

    q = OnboardingQuestion(
        id="hot_takes",
        category="opinion",
        subject="hot takes",
        question="?",
        parse="multiline_list",
        optional=True,
    )
    fake_client = MagicMock()
    with patch(
        "persona_rag.insights.onboarding.embed_batch",
        AsyncMock(return_value=[[0.0] * 1536, [0.0] * 1536]),
    ):
        await save_answer(
            question=q,
            answer_text="AI agents will eat SaaS\nweb3 is dead",
            qdrant_client=fake_client,
            collection="self_insights",
        )

    with Session(make_engine(db_path)) as s:
        rows = list(s.exec(select(InsightRow)).all())
    assert len(rows) == 2
    assert all(r.source == "onboarding" for r in rows)
    assert all(r.review_status == "approved" for r in rows)
    fake_client.upsert.assert_called_once()
