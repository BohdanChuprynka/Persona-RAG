# ruff: noqa: RUF001
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlmodel import Session, select

from persona_rag.config import get_settings
from persona_rag.db.engine import make_engine
from persona_rag.db.models import InsightRow
from persona_rag.insights.vault import (
    VAULT_EXTRACT_SYSTEM_PROMPT,
    RawVaultFact,
    VaultFact,
    _wipe_vault_rows,
    chunk_markdown,
    consolidate_vault,
    extract_vault_chunk,
    parse_vault_response,
    persist_vault,
    read_vault_files,
    rebuild_vault,
)

# --- Task 1/2: settings, fixture, model ---


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


# --- Task 3: read + chunk ---


def test_chunk_splits_on_headings():
    text = "# A\nintro line\n## B\nbody b\n## C\nbody c"
    chunks = chunk_markdown(text)
    assert len(chunks) == 3
    assert any("intro line" in c for c in chunks)
    assert any("body b" in c for c in chunks)


def test_chunk_subsplits_long_section():
    long = "# H\n" + ("параграф один.\n\n" * 200)
    chunks = chunk_markdown(long, max_chars=1500)
    assert len(chunks) > 1
    assert all(len(c) <= 1700 for c in chunks)


def test_read_vault_files_reads_fixture(tmp_path):
    (tmp_path / "a.md").write_text("# t\nhello", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("ignored", encoding="utf-8")
    docs = read_vault_files(str(tmp_path))
    assert len(docs) == 1
    assert docs[0].relpath == "a.md"
    assert "hello" in docs[0].text


# --- Task 4: prompt + parser ---


def test_vault_prompt_lists_identity_categories_and_dual_lang():
    p = VAULT_EXTRACT_SYSTEM_PROMPT.format(persona_name="TestPersona")
    for cat in ("bio", "relationship", "value", "opinion"):
        assert cat in p
    assert "text_uk" in p and "text_en" in p
    assert "interest" not in p and "behavior" not in p


def test_parse_vault_response_happy():
    resp = (
        '{"facts": [{"category": "bio", "subject": "school",'
        ' "text_uk": "Навчається на CS", "text_en": "Studies CS", "confidence": 0.9}]}'
    )
    out = parse_vault_response(resp, source_file="me.md")
    assert len(out) == 1 and isinstance(out[0], RawVaultFact)
    assert out[0].text_uk == "Навчається на CS"
    assert out[0].text_en == "Studies CS"
    assert out[0].source_file == "me.md"


def test_parse_vault_rejects_unknown_category_and_missing_uk():
    bad_cat = (
        '{"facts": [{"category": "behavior", "subject": "x", "text_uk": "a", "text_en": "b"}]}'
    )
    assert parse_vault_response(bad_cat, source_file="f") == []
    # text_uk is required (the canonical); a fact missing it is dropped.
    missing_uk = '{"facts": [{"category": "bio", "subject": "x", "text_en": "b"}]}'
    assert parse_vault_response(missing_uk, source_file="f") == []


def test_parse_vault_keeps_uk_only_fact():
    """text_en is optional: a uk-only fact is kept with text_en=None (not the string
    'None'); _render_fact falls back to text_uk for any query language."""
    uk_only = '{"facts": [{"category": "bio", "subject": "x", "text_uk": "тільки укр"}]}'
    out = parse_vault_response(uk_only, source_file="f")
    assert len(out) == 1
    assert out[0].text_en is None


def test_parse_vault_strips_fence():
    assert parse_vault_response('```json\n{"facts": []}\n```', source_file="f") == []


def test_parse_vault_assigns_trusted_confidence():
    """Regression: vault facts are user-authored -> trusted. The parser ignores the
    model's confidence (gpt-4o-mini echoed the schema's 0.0 and sank every fact to
    'pending') and assigns a high value so curated facts route to 'approved'."""
    resp = (
        '{"facts": [{"category": "bio", "subject": "school",'
        ' "text_uk": "a", "text_en": "b", "confidence": 0.0}]}'
    )
    out = parse_vault_response(resp, source_file="me.md")
    assert out[0].confidence >= 0.6


# --- Task 5: extract (mocked LLM) ---


@pytest.mark.asyncio
async def test_extract_vault_chunk_uses_json_mode_low_temp():
    canned = (
        '{"facts": [{"category": "value", "subject": "directness",'
        ' "text_uk": "Цінує прямоту", "text_en": "Values directness", "confidence": 0.9}]}'
    )
    with patch("persona_rag.insights.vault.chat_complete", AsyncMock(return_value=canned)) as mock:
        out = await extract_vault_chunk("Ціную прямоту", source_file="me.md")
    assert len(out) == 1 and out[0].category == "value"
    kwargs = mock.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["temperature"] == 0.0  # deterministic extraction (text-stability)


# --- Task 6: consolidate ---


def _raw(cat, subj, uk, en, conf, f="me.md"):
    return RawVaultFact(
        category=cat, subject=subj, text_uk=uk, text_en=en, confidence=conf, source_file=f
    )


def test_consolidate_dedups_by_category_subject():
    raws = [
        _raw("bio", "School", "Навчається на CS", "Studies CS", 0.8),
        _raw("bio", "school", "Вчиться на CS", "Goes to CS", 0.9),
        _raw("value", "directness", "Цінує прямоту", "Values directness", 0.7),
    ]
    out = consolidate_vault(raws)
    assert len(out) == 2
    school = next(f for f in out if f.category == "bio")
    assert school.confidence == 0.9
    assert school.text_en == "Goes to CS"


def test_consolidate_is_idempotent_stable_ids():
    raws = [_raw("bio", "school", "uk", "en", 0.9)]
    a = consolidate_vault(raws)
    b = consolidate_vault(raws)
    assert [f.id for f in a] == [f.id for f in b]


def test_vault_id_namespaced_away_from_chat_collision():
    """Regression (real-db bug): vault facts and chat insights BOTH hash
    (category, subject) via _stable_insight_id, so without a namespace a vault
    'bio/school' fact PK-collides with a chat-learned 'bio/school' insight
    (SQLite IntegrityError; silent Qdrant overwrite). The vault id must differ,
    while the stored subject stays clean."""
    from persona_rag.insights.consolidator import _stable_insight_id, normalize_subject

    chat_id = _stable_insight_id("bio", normalize_subject("school"))
    (fact,) = consolidate_vault([_raw("bio", "school", "навч", "studies", 0.9)])
    assert fact.id != chat_id  # namespaced -> coexists, no collision
    assert fact.subject == "school"  # subject unchanged


# --- Task 7: persist + wipe ---


@pytest.mark.asyncio
async def test_persist_vault_writes_rows_and_qdrant(tmp_path, monkeypatch):
    db = str(tmp_path / "p.db")
    make_engine(db)
    monkeypatch.setattr("persona_rag.insights.vault.make_engine", lambda: make_engine(db))
    facts = [
        VaultFact(
            id="id_hi",
            category="bio",
            subject="school",
            text_uk="навч",
            text_en="studies",
            confidence=0.9,
            source_files=["me.md"],
        ),
        VaultFact(
            id="id_lo",
            category="opinion",
            subject="quux",
            text_uk="думка",
            text_en="opinion",
            confidence=0.3,
            source_files=["me.md"],
        ),
    ]
    fake_q = MagicMock()
    with patch("persona_rag.insights.vault.embed_batch", AsyncMock(return_value=[[0.0] * 1536])):
        await persist_vault(facts, qdrant_client=fake_q, collection="self_insights", threshold=0.6)
    with Session(make_engine(db)) as s:
        rows = {r.id: r for r in s.exec(select(InsightRow)).all()}
    assert rows["id_hi"].source == "vault" and rows["id_hi"].review_status == "approved"
    assert rows["id_hi"].text_en == "studies"
    assert rows["id_lo"].review_status == "pending"
    fake_q.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_rebuild_wipes_prior_vault_rows(tmp_path, monkeypatch):
    db = str(tmp_path / "p.db")
    make_engine(db)
    monkeypatch.setattr("persona_rag.insights.vault.make_engine", lambda: make_engine(db))
    now = datetime.now(UTC)
    with Session(make_engine(db)) as s:
        s.add(
            InsightRow(
                id="stale",
                category="bio",
                subject="old",
                text="old",
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
        s.add(
            InsightRow(
                id="chat_keep",
                category="bio",
                subject="c",
                text="c",
                confidence=1.0,
                evidence_count=1,
                earliest_date=now,
                latest_date=now,
                trajectory=None,
                source_session_ids="[]",
                source="chat",
                review_status="approved",
                created_at=now,
                updated_at=now,
            )
        )
        s.commit()
    fake_q = MagicMock()
    await _wipe_vault_rows(qdrant_client=fake_q, collection="self_insights")
    with Session(make_engine(db)) as s:
        ids = {r.id for r in s.exec(select(InsightRow)).all()}
    assert "stale" not in ids and "chat_keep" in ids
    fake_q.delete.assert_called_once()


# --- Task 9: rebuild orchestrator end-to-end ---


@pytest.mark.asyncio
async def test_rebuild_vault_end_to_end(tmp_path, monkeypatch):
    db = str(tmp_path / "p.db")
    make_engine(db)
    monkeypatch.setattr("persona_rag.insights.vault.make_engine", lambda: make_engine(db))
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    (vault_dir / "me.md").write_text("# Me\nНавчаюсь на CS.", encoding="utf-8")
    canned = (
        '{"facts": [{"category": "bio", "subject": "school",'
        ' "text_uk": "Навчається на CS", "text_en": "Studies CS", "confidence": 0.9}]}'
    )
    fake_q = MagicMock()
    with (
        patch("persona_rag.insights.vault.chat_complete", AsyncMock(return_value=canned)),
        patch("persona_rag.insights.vault.embed_batch", AsyncMock(return_value=[[0.0] * 1536])),
    ):
        n = await rebuild_vault(
            directory=str(vault_dir),
            qdrant_client=fake_q,
            collection="self_insights",
            threshold=0.6,
        )
    assert n == 1
    with Session(make_engine(db)) as s:
        rows = list(s.exec(select(InsightRow).where(InsightRow.source == "vault")).all())
    assert len(rows) == 1 and rows[0].subject == "school"


@pytest.mark.asyncio
async def test_rebuild_preserves_facts_on_total_extraction_failure(tmp_path, monkeypatch):
    """Build-then-swap: if EVERY chunk fails extraction (API outage), rebuild aborts
    WITHOUT wiping, so existing curated vault facts survive — a wipe-first rebuild
    would empty the store, the exact fabrication failure this feature prevents."""
    db = str(tmp_path / "p.db")
    make_engine(db)
    monkeypatch.setattr("persona_rag.insights.vault.make_engine", lambda: make_engine(db))
    now = datetime.now(UTC)
    with Session(make_engine(db)) as s:
        s.add(
            InsightRow(
                id="keep",
                category="bio",
                subject="school",
                text="EXISTING",
                text_en="existing",
                confidence=0.9,
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
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    (vault_dir / "me.md").write_text("# Me\nsome identity prose.", encoding="utf-8")
    fake_q = MagicMock()
    with (
        patch(
            "persona_rag.insights.vault.chat_complete",
            AsyncMock(side_effect=RuntimeError("api down")),
        ),
        patch("persona_rag.insights.vault.embed_batch", AsyncMock(return_value=[])),
        pytest.raises(RuntimeError),
    ):
        await rebuild_vault(
            directory=str(vault_dir),
            qdrant_client=fake_q,
            collection="self_insights",
            threshold=0.6,
        )
    with Session(make_engine(db)) as s:
        rows = list(s.exec(select(InsightRow).where(InsightRow.source == "vault")).all())
    assert len(rows) == 1 and rows[0].text == "EXISTING"  # survived the failed rebuild
    fake_q.delete.assert_not_called()  # the wipe never fired


@pytest.mark.asyncio
async def test_rebuild_twice_yields_identical_rows(tmp_path, monkeypatch):
    """Idempotency across full rebuilds: running rebuild_vault twice on unchanged
    input yields the same id set (no duplicate accumulation, no PK crash)."""
    db = str(tmp_path / "p.db")
    make_engine(db)
    monkeypatch.setattr("persona_rag.insights.vault.make_engine", lambda: make_engine(db))
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    (vault_dir / "me.md").write_text("# Me\nidentity.", encoding="utf-8")
    canned = (
        '{"facts": [{"category": "bio", "subject": "school",'
        ' "text_uk": "навч", "text_en": "studies"}]}'
    )
    fake_q = MagicMock()

    async def _run():
        with (
            patch("persona_rag.insights.vault.chat_complete", AsyncMock(return_value=canned)),
            patch("persona_rag.insights.vault.embed_batch", AsyncMock(return_value=[[0.0] * 1536])),
        ):
            await rebuild_vault(
                directory=str(vault_dir),
                qdrant_client=fake_q,
                collection="self_insights",
                threshold=0.6,
            )

    await _run()
    with Session(make_engine(db)) as s:
        ids1 = sorted(
            r.id for r in s.exec(select(InsightRow).where(InsightRow.source == "vault")).all()
        )
    await _run()
    with Session(make_engine(db)) as s:
        ids2 = sorted(
            r.id for r in s.exec(select(InsightRow).where(InsightRow.source == "vault")).all()
        )
    assert ids1 == ids2 and len(ids1) == 1
