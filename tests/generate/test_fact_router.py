from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import Session

from persona_rag.db.engine import make_engine
from persona_rag.db.models import InsightRow
from persona_rag.generate.fact_router import (
    IDENTITY_CATEGORIES,
    classify_self_description,
    load_core_facts,
)


def test_identity_categories():
    assert {"bio", "relationship", "value", "opinion"} == IDENTITY_CATEGORIES


def test_classify_self_description_pure():
    anchors = [[1.0, 0.0], [0.0, 1.0]]
    assert classify_self_description([1.0, 0.0], anchors, threshold=0.9) is True
    assert classify_self_description([0.7, 0.7], anchors, threshold=0.9) is False


def _seed(db, rows):
    now = datetime.now(UTC)
    with Session(make_engine(db)) as s:
        for i, (cat, subj, uk, en, conf, status) in enumerate(rows):
            s.add(
                InsightRow(
                    id=f"v{i}",
                    category=cat,
                    subject=subj,
                    text=uk,
                    text_en=en,
                    confidence=conf,
                    evidence_count=1,
                    earliest_date=now,
                    latest_date=now,
                    trajectory=None,
                    source_session_ids="[]",
                    source="vault",
                    review_status=status,
                    created_at=now,
                    updated_at=now,
                )
            )
        s.commit()


def test_load_core_facts_priority_language_and_status(tmp_path, monkeypatch):
    db = str(tmp_path / "p.db")
    make_engine(db)
    monkeypatch.setattr("persona_rag.generate.fact_router.make_engine", lambda: make_engine(db))
    _seed(
        db,
        [
            ("opinion", "quux", "думка", "opinion-en", 0.9, "approved"),
            ("bio", "school", "навч", "studies", 0.9, "approved"),
            ("value", "directness", "прямота", "directness-en", 0.9, "approved"),
            ("bio", "hidden", "сховано", "hidden", 0.9, "pending"),
        ],
    )
    core = load_core_facts(limit=2, query_lang="en")
    assert [c.category for c in core] == ["bio", "value"]
    assert core[0].text_en == "studies"
    uk = load_core_facts(limit=4, query_lang="uk")
    assert all(c.subject != "hidden" for c in uk)
