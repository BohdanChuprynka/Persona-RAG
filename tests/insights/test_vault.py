from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import Session, select

from persona_rag.config import get_settings
from persona_rag.db.engine import make_engine
from persona_rag.db.models import InsightRow


def test_vault_settings_exist():
    s = get_settings()
    assert s.VAULT_RAW_DIR == "data/raw/vault"
    assert 0.0 < s.VAULT_CONFIDENCE_THRESHOLD <= 1.0
    assert isinstance(s.INSIGHTS_FACTS_ROUTER_ENABLED, bool)
    assert 0.0 < s.INSIGHTS_SELFDESC_ANCHOR_THRESHOLD <= 1.0
    assert s.INSIGHTS_CORE_MAX_FACTS >= 1


def test_synthetic_fixture_present():
    p = Path("tests/fixtures/vault/me.md")
    assert p.exists() and p.read_text(encoding="utf-8").strip()


def test_insightrow_has_text_en(tmp_path):
    db = str(tmp_path / "p.db")
    eng = make_engine(db)
    now = datetime.now(UTC)
    with Session(eng) as s:
        s.add(
            InsightRow(
                id="x1",
                category="bio",
                subject="school",
                text="навчається",
                text_en="studies",
                confidence=1.0,
                evidence_count=1,
                earliest_date=now,
                latest_date=now,
                trajectory=None,
                source_session_ids="[]",
                source="vault",
                review_status="approved",
                created_at=now,
                updated_at=now,
            )
        )
        s.commit()
    with Session(make_engine(db)) as s:
        row = s.exec(select(InsightRow).where(InsightRow.id == "x1")).one()
    assert row.text_en == "studies"
