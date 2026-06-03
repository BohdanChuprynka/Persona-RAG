# Vault Fact Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest durable persona facts from an Obsidian drop-folder into the existing insight store, and inject a tiny, query-language, intent-routed fact card into the LoRA serving turn — grounding identity without touching voice.

**Architecture:** Offline pipeline reads `data/raw/vault/*.md` → chunk → extract dual-language facts (gpt-4o-mini) → dedup by stable ID → persist `source="vault"` to SQLite+Qdrant (full-rebuild each run). Serving adds an intent router: self-description queries get a curated CORE card (by route), specific questions get semantic retrieval, everything else gets nothing — rendered in the query's language, capped ≤400 chars.

**Tech Stack:** Python 3.12, SQLModel/SQLite, Qdrant, OpenAI (offline extract+embed, v1), pytest + pytest-asyncio. Reuses `persona_rag.insights.*`, `persona_rag.index.*`, `persona_rag.generate.prompt`.

**Reference spec:** `docs/superpowers/specs/2026-06-03-vault-fact-ingestion-design.md`

---

## File Structure

**Create:**
- `persona_rag/insights/vault.py` — read/chunk/extract/dedup/persist/rebuild (offline).
- `persona_rag/generate/lang_detect.py` — uk/ru/en detector (pure).
- `persona_rag/generate/fact_router.py` — intent classification + CORE loader (serving).
- `scripts/ingest_vault.py` — CLI orchestrator.
- `tests/insights/test_vault.py`, `tests/generate/test_lang_detect.py`,
  `tests/generate/test_fact_router.py`, `tests/generate/test_fact_card.py`,
  `tests/eval/test_vault_register_invariance.py`.
- `tests/fixtures/vault/me.md` (synthetic), `data/raw/vault/.gitkeep`.

**Modify:**
- `persona_rag/db/models.py` — `InsightRow.text_en`.
- `persona_rag/db/engine.py` — idempotent `text_en` migration.
- `persona_rag/insights/persistence.py` — protect `source="vault"`.
- `persona_rag/insights/recency.py` — `RankedInsight.text_en` + `from_qdrant_point`.
- `persona_rag/graph/nodes/retrieve_insights.py` — wire router into state.
- `persona_rag/generate/prompt.py` — `build_fact_card` replacing `_compact_facts`.
- `persona_rag/config.py` — new settings.
- `.gitignore`, `Makefile`.

**Conventions (from existing tests):** temp DB via `make_engine(str(tmp_path/"p.db"))` + `monkeypatch.setattr("module.make_engine", lambda: make_engine(db_path))`; mock IO with `AsyncMock`/`MagicMock`; OpenAI embeddings are 1536-dim; async tests use `@pytest.mark.asyncio`.

---

## Task 1: Scaffolding — settings, gitignore, folder, synthetic fixture

**Files:**
- Modify: `persona_rag/config.py` (near the `INSIGHTS_*` block, ~line 144)
- Modify: `.gitignore` (after the `data/raw/instagram/` block, ~line 18)
- Create: `data/raw/vault/.gitkeep` (empty), `tests/fixtures/vault/me.md`
- Test: `tests/insights/test_vault.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/insights/test_vault.py
from __future__ import annotations

from pathlib import Path

from persona_rag.config import get_settings


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/insights/test_vault.py -v`
Expected: FAIL (`AttributeError: ... VAULT_RAW_DIR` / fixture missing).

- [ ] **Step 3: Add settings + gitignore + fixture**

In `persona_rag/config.py`, add inside the Settings class near the other `INSIGHTS_*` fields:

```python
    # --- Vault fact ingestion (spec 2026-06-03) ---
    VAULT_RAW_DIR: str = "data/raw/vault"
    VAULT_CONFIDENCE_THRESHOLD: float = 0.6
    INSIGHTS_FACTS_ROUTER_ENABLED: bool = True
    INSIGHTS_SELFDESC_ANCHOR_THRESHOLD: float = 0.55
    INSIGHTS_CORE_MAX_FACTS: int = 4
```

In `.gitignore`, after the instagram block add:

```
!data/raw/vault/
data/raw/vault/*
!data/raw/vault/.gitkeep
```

Create empty `data/raw/vault/.gitkeep`. Create `tests/fixtures/vault/me.md` (SYNTHETIC — all facts fake):

```markdown
# Про мене
Мене звати Test Persona. Живу в місті Springfield. Навчаюсь на Computer Science
в Fictional State University. Будую AI-інструменти для персон.

## Цінності
Ціную прямоту і чесність. Не люблю тягнути з рішеннями.

## Люди
Найближчий друг — Sam, з ним раджусь щодо важливого.

## Думки
Вважаю що фреймворк QuuxJS переоцінений.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/insights/test_vault.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona_rag/config.py .gitignore data/raw/vault/.gitkeep tests/fixtures/vault/me.md tests/insights/test_vault.py
git commit -m "feat(vault): settings, gitignore, synthetic fixture scaffolding"
```

---

## Task 2: `InsightRow.text_en` + idempotent migration

**Files:**
- Modify: `persona_rag/db/models.py:92-110` (InsightRow)
- Modify: `persona_rag/db/engine.py:17-37` (after create_all)
- Test: `tests/insights/test_vault.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/insights/test_vault.py
from datetime import UTC, datetime

from sqlmodel import Session, select

from persona_rag.db.engine import make_engine
from persona_rag.db.models import InsightRow


def test_insightrow_has_text_en(tmp_path):
    db = str(tmp_path / "p.db")
    eng = make_engine(db)
    now = datetime.now(UTC)
    with Session(eng) as s:
        s.add(InsightRow(
            id="x1", category="bio", subject="school", text="навчається",
            text_en="studies", confidence=1.0, evidence_count=1,
            earliest_date=now, latest_date=now, trajectory=None,
            source_session_ids="[]", source="vault", review_status="approved",
            created_at=now, updated_at=now,
        ))
        s.commit()
    with Session(make_engine(db)) as s:
        row = s.exec(select(InsightRow).where(InsightRow.id == "x1")).one()
    assert row.text_en == "studies"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/insights/test_vault.py::test_insightrow_has_text_en -v`
Expected: FAIL (`TypeError: unexpected keyword argument 'text_en'`).

- [ ] **Step 3: Add the field + migration**

In `persona_rag/db/models.py`, inside `InsightRow`, after the `text: str` line add:

```python
    text_en: str | None = None
```

In `persona_rag/db/engine.py`, inside the existing `with engine.begin() as conn:` block (after the usermemory rename logic, before `return engine`), add:

```python
        ins_cols = {
            r[1] for r in conn.exec_driver_sql("PRAGMA table_info(insight_row)").fetchall()
        }
        if ins_cols and "text_en" not in ins_cols:
            conn.exec_driver_sql("ALTER TABLE insight_row ADD COLUMN text_en VARCHAR")
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/insights/test_vault.py -v`
Expected: PASS. Also run `uv run pytest tests/insights/test_models.py -v` (no regression).

- [ ] **Step 5: Commit**

```bash
git add persona_rag/db/models.py persona_rag/db/engine.py tests/insights/test_vault.py
git commit -m "feat(vault): add InsightRow.text_en + idempotent sqlite migration"
```

---

## Task 3: `vault.py` — read + chunk markdown

**Files:**
- Create: `persona_rag/insights/vault.py`
- Test: `tests/insights/test_vault.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/insights/test_vault.py
from persona_rag.insights.vault import chunk_markdown, read_vault_files


def test_chunk_splits_on_headings():
    text = "# A\nintro line\n## B\nbody b\n## C\nbody c"
    chunks = chunk_markdown(text)
    assert len(chunks) == 3
    assert any("intro line" in c for c in chunks)
    assert any("body b" in c for c in chunks)


def test_chunk_subsplits_long_section():
    long = "# H\n" + ("параграф один.\n\n" * 200)  # > 1500 chars, many paragraphs
    chunks = chunk_markdown(long, max_chars=1500)
    assert len(chunks) > 1
    assert all(len(c) <= 1700 for c in chunks)  # soft bound (paragraph granularity)


def test_read_vault_files_reads_fixture(tmp_path):
    (tmp_path / "a.md").write_text("# t\nhello", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("ignored", encoding="utf-8")
    docs = read_vault_files(str(tmp_path))
    assert len(docs) == 1
    assert docs[0].relpath == "a.md"
    assert "hello" in docs[0].text
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/insights/test_vault.py -k "chunk or read_vault" -v`
Expected: FAIL (`ModuleNotFoundError: persona_rag.insights.vault`).

- [ ] **Step 3: Implement read + chunk**

```python
# persona_rag/insights/vault.py
"""Vault fact ingestion (spec 2026-06-03): read → chunk → extract → dedup → persist."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_HEADING_RE = re.compile(r"^#{1,2}\s+", re.MULTILINE)


@dataclass
class VaultDoc:
    relpath: str
    text: str


def read_vault_files(directory: str) -> list[VaultDoc]:
    """Read every *.md file under `directory` (recursively)."""
    base = Path(directory)
    out: list[VaultDoc] = []
    for p in sorted(base.rglob("*.md")):
        txt = p.read_text(encoding="utf-8")
        if txt.strip():
            out.append(VaultDoc(relpath=str(p.relative_to(base)), text=txt))
    return out


def chunk_markdown(text: str, *, max_chars: int = 1500) -> list[str]:
    """Split on H1/H2 headings; sub-split oversized sections by paragraph."""
    # Split keeping heading with its body.
    idxs = [m.start() for m in _HEADING_RE.finditer(text)]
    if not idxs:
        sections = [text]
    else:
        bounds = idxs + [len(text)]
        sections = [text[bounds[i]:bounds[i + 1]] for i in range(len(idxs))]
        if idxs[0] > 0:
            sections.insert(0, text[: idxs[0]])
    chunks: list[str] = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        if len(sec) <= max_chars:
            chunks.append(sec)
            continue
        buf = ""
        for para in sec.split("\n\n"):
            if len(buf) + len(para) > max_chars and buf:
                chunks.append(buf.strip())
                buf = ""
            buf += para + "\n\n"
        if buf.strip():
            chunks.append(buf.strip())
    return chunks
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/insights/test_vault.py -k "chunk or read_vault" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona_rag/insights/vault.py tests/insights/test_vault.py
git commit -m "feat(vault): read + heading-aware chunking of md files"
```

---

## Task 4: `vault.py` — dual-language extractor prompt + parser

**Files:**
- Modify: `persona_rag/insights/vault.py`
- Test: `tests/insights/test_vault.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/insights/test_vault.py
from persona_rag.insights.vault import (
    VAULT_CATEGORIES,
    VAULT_EXTRACT_SYSTEM_PROMPT,
    RawVaultFact,
    parse_vault_response,
)


def test_vault_prompt_lists_identity_categories_and_dual_lang():
    p = VAULT_EXTRACT_SYSTEM_PROMPT.format(persona_name="TestPersona")
    for cat in ("bio", "relationship", "value", "opinion"):
        assert cat in p
    assert "text_uk" in p and "text_en" in p
    assert "interest" not in p and "behavior" not in p  # dropped from vault scheme


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


def test_parse_vault_rejects_unknown_category_and_missing_lang():
    bad_cat = '{"facts": [{"category": "behavior", "subject": "x", "text_uk": "a", "text_en": "b", "confidence": 0.5}]}'
    assert parse_vault_response(bad_cat, source_file="f") == []
    missing = '{"facts": [{"category": "bio", "subject": "x", "text_uk": "a", "confidence": 0.5}]}'
    assert parse_vault_response(missing, source_file="f") == []


def test_parse_vault_strips_fence():
    assert parse_vault_response('```json\n{"facts": []}\n```', source_file="f") == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/insights/test_vault.py -k "vault_prompt or parse_vault" -v`
Expected: FAIL (import errors).

- [ ] **Step 3: Implement prompt + model + parser**

Append to `persona_rag/insights/vault.py`:

```python
import json
from typing import Literal

from pydantic import BaseModel, ValidationError

VAULT_CATEGORIES = {"bio", "relationship", "value", "opinion"}

VAULT_EXTRACT_SYSTEM_PROMPT = """\
You extract DURABLE IDENTITY facts about {persona_name} from their own first-person
notes. The text is already written by {persona_name} — treat every statement as theirs
(no speaker attribution needed).

Categories (use exactly these four):
- bio: stable facts — where they live/study/work, age, languages, one-line "what I build".
- relationship: a specific person who matters to them, and that person's role.
- value: a principle, goal, or what drives them.
- opinion: a stable take, preference, or taste.

EXCLUDE everything volatile or non-identity: work/sprint/pipeline/outreach notes,
technical project/code/architecture notes, gym/study/setup logs, to-dos, dated events.

For EACH fact output BOTH languages: text_uk (Ukrainian) and text_en (English). Keep
proper nouns, names, and places verbatim in both. One short declarative sentence each.

Output ONLY valid JSON:
{{
  "facts": [
    {{
      "category": "bio|relationship|value|opinion",
      "subject": "short lowercase noun-phrase, e.g. 'school', 'best friend'",
      "text_uk": "<one Ukrainian sentence>",
      "text_en": "<one English sentence>",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- Only durable identity. If a chunk has none, return {{"facts": []}}.
- Be specific: "studies CS at Fictional State University" beats "is a student".
- Max 8 facts per chunk. Each text ≤ 25 words.
"""


class RawVaultFact(BaseModel):
    category: Literal["bio", "relationship", "value", "opinion"]
    subject: str
    text_uk: str
    text_en: str
    confidence: float
    source_file: str


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def parse_vault_response(text: str, *, source_file: str) -> list[RawVaultFact]:
    m = _FENCE_RE.match(text.strip())
    cleaned = m.group(1) if m else text
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as e:
        preview = (text[:200] + "…") if len(text) > 200 else text
        raise ValueError(f"non-JSON vault output: {e} | preview={preview!r}") from e
    items = payload.get("facts", [])
    if not isinstance(items, list):
        raise ValueError("'facts' is not a list")
    out: list[RawVaultFact] = []
    for item in items:
        if not isinstance(item, dict) or item.get("category") not in VAULT_CATEGORIES:
            continue
        try:
            out.append(RawVaultFact(
                category=item["category"],
                subject=str(item["subject"]),
                text_uk=str(item["text_uk"]),
                text_en=str(item["text_en"]),
                confidence=float(item.get("confidence", 0.5)),
                source_file=source_file,
            ))
        except (KeyError, ValidationError, ValueError):
            continue
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/insights/test_vault.py -k "vault_prompt or parse_vault" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona_rag/insights/vault.py tests/insights/test_vault.py
git commit -m "feat(vault): dual-language identity extractor prompt + JSON parser"
```

---

## Task 5: `vault.py` — extract chunk (LLM call, mocked)

**Files:**
- Modify: `persona_rag/insights/vault.py`
- Test: `tests/insights/test_vault.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/insights/test_vault.py
import pytest
from unittest.mock import AsyncMock, patch

from persona_rag.insights.vault import extract_vault_chunk


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
    assert kwargs["temperature"] == 0.2
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/insights/test_vault.py::test_extract_vault_chunk_uses_json_mode_low_temp -v`
Expected: FAIL (import error).

- [ ] **Step 3: Implement**

Append to `persona_rag/insights/vault.py`:

```python
from persona_rag.config import get_settings
from persona_rag.generate.llm_client import chat_complete


async def extract_vault_chunk(chunk: str, *, source_file: str) -> list[RawVaultFact]:
    s = get_settings()
    messages = [
        {"role": "system", "content": VAULT_EXTRACT_SYSTEM_PROMPT.format(
            persona_name=s.PERSONA_NAME)},
        {"role": "user", "content": chunk},
    ]
    resp = await chat_complete(
        messages,
        model=s.INSIGHTS_EXTRACT_MODEL,
        temperature=0.2,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )
    return parse_vault_response(resp, source_file=source_file)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/insights/test_vault.py::test_extract_vault_chunk_uses_json_mode_low_temp -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona_rag/insights/vault.py tests/insights/test_vault.py
git commit -m "feat(vault): extract_vault_chunk LLM wrapper (json mode, low temp)"
```

---

## Task 6: `vault.py` — consolidate (stable-ID dedup, idempotent)

**Files:**
- Modify: `persona_rag/insights/vault.py`
- Test: `tests/insights/test_vault.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/insights/test_vault.py
from persona_rag.insights.vault import VaultFact, consolidate_vault


def _raw(cat, subj, uk, en, conf, f="me.md"):
    return RawVaultFact(category=cat, subject=subj, text_uk=uk, text_en=en,
                        confidence=conf, source_file=f)


def test_consolidate_dedups_by_category_subject():
    raws = [
        _raw("bio", "School", "Навчається на CS", "Studies CS", 0.8),
        _raw("bio", "school", "Вчиться на CS", "Goes to CS", 0.9),  # same after normalize
        _raw("value", "directness", "Цінує прямоту", "Values directness", 0.7),
    ]
    out = consolidate_vault(raws)
    assert len(out) == 2  # the two school raws merged
    school = next(f for f in out if f.category == "bio")
    assert school.confidence == 0.9  # highest-confidence member kept
    assert school.text_en == "Goes to CS"


def test_consolidate_is_idempotent_stable_ids():
    raws = [_raw("bio", "school", "uk", "en", 0.9)]
    a = consolidate_vault(raws)
    b = consolidate_vault(raws)
    assert [f.id for f in a] == [f.id for f in b]  # deterministic IDs across runs
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/insights/test_vault.py -k consolidate -v`
Expected: FAIL (import error).

- [ ] **Step 3: Implement (reuse dedup primitives, skip LLM-merge — curated source)**

Append to `persona_rag/insights/vault.py`:

```python
from persona_rag.insights.consolidator import _stable_insight_id, normalize_subject


class VaultFact(BaseModel):
    id: str
    category: str
    subject: str  # canonical
    text_uk: str
    text_en: str
    confidence: float
    source_files: list[str]


def consolidate_vault(raws: list[RawVaultFact]) -> list[VaultFact]:
    """Group by (category, normalized subject); keep highest-confidence member.

    Reuses the chat pipeline's dedup primitives (`normalize_subject`,
    `_stable_insight_id`) for cross-run idempotency, but skips the LLM merge —
    a curated vault rarely has 3+ near-duplicate facts.
    """
    groups: dict[tuple[str, str], list[RawVaultFact]] = {}
    for r in raws:
        key = (r.category, normalize_subject(r.subject))
        groups.setdefault(key, []).append(r)
    out: list[VaultFact] = []
    for (category, canon), members in groups.items():
        best = max(members, key=lambda r: r.confidence)
        out.append(VaultFact(
            id=_stable_insight_id(category, canon),
            category=category,
            subject=canon,
            text_uk=best.text_uk,
            text_en=best.text_en,
            confidence=best.confidence,
            source_files=sorted({m.source_file for m in members}),
        ))
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/insights/test_vault.py -k consolidate -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona_rag/insights/vault.py tests/insights/test_vault.py
git commit -m "feat(vault): stable-ID dedup consolidation (idempotent)"
```

---

## Task 7: `vault.py` — persist + full rebuild (SQLite + Qdrant)

**Files:**
- Modify: `persona_rag/insights/vault.py`
- Test: `tests/insights/test_vault.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/insights/test_vault.py
from unittest.mock import MagicMock


@pytest.mark.asyncio
async def test_persist_vault_writes_rows_and_qdrant(tmp_path, monkeypatch):
    db = str(tmp_path / "p.db")
    make_engine(db)
    monkeypatch.setattr("persona_rag.insights.vault.make_engine", lambda: make_engine(db))
    facts = [
        VaultFact(id="id_hi", category="bio", subject="school", text_uk="навч",
                  text_en="studies", confidence=0.9, source_files=["me.md"]),
        VaultFact(id="id_lo", category="opinion", subject="quux", text_uk="думка",
                  text_en="opinion", confidence=0.3, source_files=["me.md"]),
    ]
    fake_q = MagicMock()
    with patch("persona_rag.insights.vault.embed_batch", AsyncMock(return_value=[[0.0] * 1536])):
        await persist_vault(facts, qdrant_client=fake_q, collection="self_insights", threshold=0.6)
    with Session(make_engine(db)) as s:
        rows = {r.id: r for r in s.exec(select(InsightRow)).all()}
    assert rows["id_hi"].source == "vault" and rows["id_hi"].review_status == "approved"
    assert rows["id_hi"].text_en == "studies"
    assert rows["id_lo"].review_status == "pending"  # below threshold
    fake_q.upsert.assert_called_once()  # only approved embedded


@pytest.mark.asyncio
async def test_rebuild_wipes_prior_vault_rows(tmp_path, monkeypatch):
    db = str(tmp_path / "p.db")
    make_engine(db)
    monkeypatch.setattr("persona_rag.insights.vault.make_engine", lambda: make_engine(db))
    now = datetime.now(UTC)
    with Session(make_engine(db)) as s:
        s.add(InsightRow(id="stale", category="bio", subject="old", text="old",
                         confidence=1.0, evidence_count=1, earliest_date=now, latest_date=now,
                         trajectory=None, source_session_ids="[]", source="vault",
                         review_status="approved", created_at=now, updated_at=now))
        s.add(InsightRow(id="chat_keep", category="bio", subject="c", text="c",
                         confidence=1.0, evidence_count=1, earliest_date=now, latest_date=now,
                         trajectory=None, source_session_ids="[]", source="chat",
                         review_status="approved", created_at=now, updated_at=now))
        s.commit()
    fake_q = MagicMock()
    await _wipe_vault_rows(qdrant_client=fake_q, collection="self_insights")
    with Session(make_engine(db)) as s:
        ids = {r.id for r in s.exec(select(InsightRow)).all()}
    assert "stale" not in ids and "chat_keep" in ids  # only vault wiped
    fake_q.delete.assert_called_once()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/insights/test_vault.py -k "persist_vault or rebuild_wipes" -v`
Expected: FAIL (import errors).

- [ ] **Step 3: Implement persist + wipe**

Append to `persona_rag/insights/vault.py`:

```python
from datetime import UTC, datetime

from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
)
from sqlmodel import Session, select

from persona_rag.db.engine import make_engine
from persona_rag.db.models import InsightRow
from persona_rag.index.embedder import embed_batch
from persona_rag.index.qdrant_store import to_qdrant_point_id

_VAULT_FILTER = Filter(must=[FieldCondition(key="source", match=MatchValue(value="vault"))])


async def _wipe_vault_rows(*, qdrant_client, collection: str) -> None:
    """Delete all source='vault' rows from SQLite + Qdrant (full-rebuild step)."""
    with Session(make_engine()) as s:
        for row in s.exec(select(InsightRow).where(InsightRow.source == "vault")).all():
            s.delete(row)
        s.commit()
    qdrant_client.delete(
        collection_name=collection, points_selector=FilterSelector(filter=_VAULT_FILTER)
    )


async def persist_vault(
    facts: list[VaultFact], *, qdrant_client, collection: str, threshold: float
) -> None:
    now = datetime.now(UTC)
    approved: list[VaultFact] = []
    with Session(make_engine()) as s:
        for f in facts:
            status = "approved" if f.confidence >= threshold else "pending"
            s.add(InsightRow(
                id=f.id, category=f.category, subject=f.subject,
                text=f.text_uk, text_en=f.text_en, confidence=f.confidence,
                evidence_count=1, earliest_date=now, latest_date=now, trajectory=None,
                source_session_ids=json.dumps(f.source_files), distinct_partners=0,
                source="vault", review_status=status, edited_text=None,
                created_at=now, updated_at=now,
            ))
            if status == "approved":
                approved.append(f)
        s.commit()
    if not approved:
        return
    vectors = await embed_batch([f.text_uk for f in approved])
    points = [
        PointStruct(
            id=to_qdrant_point_id(f.id),
            vector=vec,
            payload={
                "sqlite_id": f.id, "category": f.category, "subject": f.subject,
                "text": f.text_uk, "text_en": f.text_en, "confidence": f.confidence,
                "evidence_count": 1, "earliest_date": now.isoformat(),
                "latest_date": now.isoformat(), "trajectory": None,
                "source": "vault", "review_status": "approved",
            },
        )
        for f, vec in zip(approved, vectors, strict=True)
    ]
    qdrant_client.upsert(collection_name=collection, points=points)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/insights/test_vault.py -k "persist_vault or rebuild_wipes" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona_rag/insights/vault.py tests/insights/test_vault.py
git commit -m "feat(vault): persist source=vault + full-rebuild wipe (SQLite+Qdrant)"
```

---

## Task 8: Protect `source="vault"` from chat clobbering

**Files:**
- Modify: `persona_rag/insights/persistence.py:97-99`
- Test: `tests/insights/test_persistence.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/insights/test_persistence.py
import pytest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from sqlmodel import Session, select

from persona_rag.db.engine import make_engine
from persona_rag.db.models import InsightRow
from persona_rag.insights.consolidator import ConsolidatedInsight
from persona_rag.insights.persistence import persist_insights


@pytest.mark.asyncio
async def test_chat_rerun_does_not_clobber_vault_row(tmp_path, monkeypatch):
    db = str(tmp_path / "p.db")
    make_engine(db)
    monkeypatch.setattr("persona_rag.insights.persistence.make_engine", lambda: make_engine(db))
    now = datetime.now(UTC)
    with Session(make_engine(db)) as s:
        s.add(InsightRow(id="shared", category="bio", subject="school",
                         text="VAULT TRUTH", text_en="vault truth", confidence=1.0,
                         evidence_count=1, earliest_date=now, latest_date=now, trajectory=None,
                         source_session_ids="[]", source="vault", review_status="approved",
                         created_at=now, updated_at=now))
        s.commit()
    ci = ConsolidatedInsight(id="shared", category="bio", canonical_subject="school",
                             text="CHAT GUESS", confidence=0.5, evidence_count=3,
                             earliest_date=now, latest_date=now, trajectory=None,
                             source_session_ids=["s1"], distinct_partners=2)
    with patch("persona_rag.insights.persistence.embed_batch", AsyncMock(return_value=[])):
        await persist_insights([ci], statuses={"shared": "auto"},
                               qdrant_client=MagicMock(), collection="self_insights")
    with Session(make_engine(db)) as s:
        row = s.exec(select(InsightRow).where(InsightRow.id == "shared")).one()
    assert row.text == "VAULT TRUTH" and row.source == "vault"  # untouched
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/insights/test_persistence.py::test_chat_rerun_does_not_clobber_vault_row -v`
Expected: FAIL (chat run overwrites text to "CHAT GUESS").

- [ ] **Step 3: Implement (one-line guard extension)**

In `persona_rag/insights/persistence.py`, change the protection tuple (~line 98) from:

```python
                existing.source in ("user_verified", "onboarding")
```

to:

```python
                existing.source in ("user_verified", "onboarding", "vault")
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/insights/test_persistence.py -v`
Expected: PASS (all, no regression).

- [ ] **Step 5: Commit**

```bash
git add persona_rag/insights/persistence.py tests/insights/test_persistence.py
git commit -m "feat(vault): protect source=vault rows from chat re-run clobbering"
```

---

## Task 9: `scripts/ingest_vault.py` + Makefile target

**Files:**
- Modify: `persona_rag/insights/vault.py` (add `rebuild_vault` orchestrator)
- Create: `scripts/ingest_vault.py`
- Modify: `Makefile` (after the `insights-dry` target, ~line 96)
- Test: `tests/insights/test_vault.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/insights/test_vault.py
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
        n = await rebuild_vault(directory=str(vault_dir), qdrant_client=fake_q,
                                collection="self_insights", threshold=0.6)
    assert n == 1
    with Session(make_engine(db)) as s:
        rows = list(s.exec(select(InsightRow).where(InsightRow.source == "vault")).all())
    assert len(rows) == 1 and rows[0].subject == "school"
```

Add the import at the top of the test edits: `from persona_rag.insights.vault import rebuild_vault`.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/insights/test_vault.py::test_rebuild_vault_end_to_end -v`
Expected: FAIL (`rebuild_vault` undefined).

- [ ] **Step 3: Implement orchestrator + script + Makefile**

Append to `persona_rag/insights/vault.py`:

```python
from persona_rag._logging import get_logger

log = get_logger()


async def rebuild_vault(*, directory: str, qdrant_client, collection: str, threshold: float) -> int:
    """Full rebuild: wipe prior vault facts, then re-extract the whole folder."""
    await _wipe_vault_rows(qdrant_client=qdrant_client, collection=collection)
    docs = read_vault_files(directory)
    raws: list[RawVaultFact] = []
    for doc in docs:
        for i, chunk in enumerate(chunk_markdown(doc.text)):
            try:
                raws.extend(await extract_vault_chunk(chunk, source_file=f"{doc.relpath}#{i}"))
            except ValueError as e:
                log.warning("vault_chunk_failed", file=doc.relpath, chunk=i, error=str(e)[:200])
    facts = consolidate_vault(raws)
    await persist_vault(facts, qdrant_client=qdrant_client, collection=collection,
                        threshold=threshold)
    log.info("vault_rebuild_done", docs=len(docs), raws=len(raws), facts=len(facts))
    return len(facts)
```

Create `scripts/ingest_vault.py`:

```python
"""Ingest durable persona facts from the Obsidian drop-folder (spec 2026-06-03)."""
from __future__ import annotations

import asyncio

from persona_rag._logging import configure_logging
from persona_rag.config import get_settings
from persona_rag.index.qdrant_store import ensure_insights_collection, make_client
from persona_rag.insights.vault import rebuild_vault


async def _main() -> int:
    s = get_settings()
    client = make_client()
    ensure_insights_collection(client, s.QDRANT_INSIGHTS_COLLECTION)
    n = await rebuild_vault(
        directory=s.VAULT_RAW_DIR,
        qdrant_client=client,
        collection=s.QDRANT_INSIGHTS_COLLECTION,
        threshold=s.VAULT_CONFIDENCE_THRESHOLD,
    )
    print(f"vault facts ingested: {n}")
    return 0


def main() -> None:
    configure_logging()
    raise SystemExit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
```

In `Makefile`, after the `insights-dry` target add (and append `insights-vault` to its `.PHONY` line):

```makefile
insights-vault:
	uv run python scripts/ingest_vault.py
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/insights/test_vault.py -v`
Expected: PASS (whole file).

- [ ] **Step 5: Commit**

```bash
git add persona_rag/insights/vault.py scripts/ingest_vault.py Makefile tests/insights/test_vault.py
git commit -m "feat(vault): rebuild orchestrator + ingest_vault.py + make insights-vault"
```

---

## Task 10: `lang_detect.py` — uk/ru/en detector

**Files:**
- Create: `persona_rag/generate/lang_detect.py`
- Test: `tests/generate/test_lang_detect.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/generate/test_lang_detect.py
from __future__ import annotations

from persona_rag.generate.lang_detect import detect_language


def test_english():
    assert detect_language("tell me about yourself") == "en"
    assert detect_language("where do you study?") == "en"


def test_ukrainian_distinctive_chars():
    assert detect_language("розкажи про себе") == "uk"
    assert detect_language("де ти живеш?") == "uk"  # і present


def test_russian_distinctive_chars():
    assert detect_language("расскажи о себе, кто ты") == "ru"  # ы/ё-class, no uk markers


def test_mixed_defaults_uk():
    assert detect_language("ok розкажи") == "uk"  # cyrillic present → uk default
    assert detect_language("") == "uk"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/generate/test_lang_detect.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# persona_rag/generate/lang_detect.py
# ruff: noqa: RUF001
"""Detect the language/script of an incoming message: uk | ru | en.

Heuristic and dependency-free. Script first (Cyrillic vs Latin); within Cyrillic,
Ukrainian-distinctive letters (і ї є ґ) win over Russian-distinctive (ы ъ э ё).
Defaults to uk (the persona's primary register) when ambiguous.
"""
from __future__ import annotations

from typing import Literal

Lang = Literal["uk", "ru", "en"]

_UK = set("іїєґ")
_RU = set("ыъэё")


def detect_language(text: str) -> Lang:
    t = (text or "").lower()
    cyr = sum(1 for c in t if "Ѐ" <= c <= "ӿ")
    lat = sum(1 for c in t if "a" <= c <= "z")
    if cyr == 0 and lat > 0:
        return "en"
    if cyr == 0 and lat == 0:
        return "uk"
    if any(c in _UK for c in t):
        return "uk"
    if any(c in _RU for c in t) and not any(c in _UK for c in t):
        return "ru"
    return "uk"
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/generate/test_lang_detect.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona_rag/generate/lang_detect.py tests/generate/test_lang_detect.py
git commit -m "feat(serve): uk/ru/en language detector for card rendering"
```

---

## Task 11: `RankedInsight.text_en` + `from_qdrant_point`

**Files:**
- Modify: `persona_rag/insights/recency.py:12-24,48-63`
- Test: `tests/insights/test_recency.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/insights/test_recency.py
from unittest.mock import MagicMock

from persona_rag.insights.recency import RankedInsight, from_qdrant_point


def test_ranked_insight_text_en_optional_default_none():
    from datetime import UTC, datetime
    now = datetime.now(UTC)
    r = RankedInsight(id="a", text="навч", category="bio", subject="school",
                      confidence=1.0, evidence_count=1, earliest_date=now, latest_date=now,
                      trajectory=None, source="vault", semantic_score=0.4)
    assert r.text_en is None


def test_from_qdrant_point_reads_text_en():
    now_iso = "2026-06-03T00:00:00+00:00"
    point = MagicMock()
    point.id = "p1"
    point.score = 0.7
    point.payload = {"text": "навч", "text_en": "studies", "category": "bio",
                     "subject": "school", "confidence": 0.9, "evidence_count": 1,
                     "earliest_date": now_iso, "latest_date": now_iso, "source": "vault"}
    r = from_qdrant_point(point)
    assert r.text_en == "studies"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/insights/test_recency.py -k text_en -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `persona_rag/insights/recency.py`, add to `RankedInsight` after `text: str`:

```python
    text_en: str | None = None
```

In `from_qdrant_point`, add to the constructor kwargs:

```python
        text_en=p.get("text_en"),
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/insights/test_recency.py -v`
Expected: PASS (no regression).

- [ ] **Step 5: Commit**

```bash
git add persona_rag/insights/recency.py tests/insights/test_recency.py
git commit -m "feat(serve): carry text_en through RankedInsight + from_qdrant_point"
```

---

## Task 12: `fact_router.py` — intent classifier + CORE loader

**Files:**
- Create: `persona_rag/generate/fact_router.py`
- Test: `tests/generate/test_fact_router.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/generate/test_fact_router.py
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from unittest.mock import patch
from sqlmodel import Session

from persona_rag.db.engine import make_engine
from persona_rag.db.models import InsightRow
from persona_rag.generate.fact_router import (
    IDENTITY_CATEGORIES,
    classify_self_description,
    load_core_facts,
)


def test_identity_categories():
    assert IDENTITY_CATEGORIES == {"bio", "relationship", "value", "opinion"}


def test_classify_self_description_pure():
    anchors = [[1.0, 0.0], [0.0, 1.0]]
    assert classify_self_description([1.0, 0.0], anchors, threshold=0.9) is True   # cos=1
    assert classify_self_description([0.7, 0.7], anchors, threshold=0.9) is False  # cos≈0.71


def _seed(db, rows):
    now = datetime.now(UTC)
    with Session(make_engine(db)) as s:
        for i, (cat, subj, uk, en, conf, status) in enumerate(rows):
            s.add(InsightRow(id=f"v{i}", category=cat, subject=subj, text=uk, text_en=en,
                             confidence=conf, evidence_count=1, earliest_date=now, latest_date=now,
                             trajectory=None, source_session_ids="[]", source="vault",
                             review_status=status, created_at=now, updated_at=now))
        s.commit()


def test_load_core_facts_priority_language_and_status(tmp_path, monkeypatch):
    db = str(tmp_path / "p.db")
    make_engine(db)
    monkeypatch.setattr("persona_rag.generate.fact_router.make_engine", lambda: make_engine(db))
    _seed(db, [
        ("opinion", "quux", "думка", "opinion-en", 0.9, "approved"),
        ("bio", "school", "навч", "studies", 0.9, "approved"),
        ("value", "directness", "прямота", "directness-en", 0.9, "approved"),
        ("bio", "hidden", "сховано", "hidden", 0.9, "pending"),  # not approved → excluded
    ])
    core = load_core_facts(limit=2, query_lang="en")
    assert [c.category for c in core] == ["bio", "value"]  # priority bio > value > opinion
    assert core[0].text_en == "studies"  # rendered fields available
    uk = load_core_facts(limit=4, query_lang="uk")
    assert all(c.subject != "hidden" for c in uk)  # pending excluded
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/generate/test_fact_router.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# persona_rag/generate/fact_router.py
# ruff: noqa: RUF001
"""Serving-time fact router: self-description intent + CORE identity loader.

Self-description queries (vague: "розкажи про себе") are reached by ROUTE — a
curated CORE of vault identity facts — not by embedding similarity, which is
unreliable for meta-questions. Specific questions fall through to the existing
semantic retrieval. See spec 2026-06-03 §6/§10.
"""
from __future__ import annotations

import math

from sqlmodel import Session, select

from persona_rag.db.engine import make_engine
from persona_rag.db.models import InsightRow
from persona_rag.index.embedder import embed_batch
from persona_rag.insights.recency import RankedInsight

IDENTITY_CATEGORIES = {"bio", "relationship", "value", "opinion"}
_PRIORITY = {"bio": 0, "relationship": 1, "value": 2, "opinion": 3}

ANCHOR_PHRASES = [
    "розкажи про себе", "хто ти", "опиши себе", "розкажи шось про себе",
    "расскажи о себе", "кто ты",
    "tell me about yourself", "who are you", "tell me about you",
]

_anchor_vecs: list[list[float]] | None = None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def classify_self_description(
    vec: list[float], anchor_vecs: list[list[float]], *, threshold: float
) -> bool:
    """Pure: True if `vec` is within `threshold` cosine of any anchor."""
    return any(_cosine(vec, a) >= threshold for a in anchor_vecs)


async def anchor_vecs() -> list[list[float]]:
    """Embed the anchor phrases once; cache module-level."""
    global _anchor_vecs
    if _anchor_vecs is None:
        _anchor_vecs = await embed_batch(ANCHOR_PHRASES)
    return _anchor_vecs


def load_core_facts(*, limit: int, query_lang: str) -> list[RankedInsight]:
    """Top vault identity facts by (category priority, confidence). Approved only."""
    with Session(make_engine()) as s:
        rows = list(s.exec(
            select(InsightRow).where(
                InsightRow.source == "vault",
                InsightRow.review_status.in_(("auto", "approved")),  # type: ignore[attr-defined]
            )
        ).all())
    rows = [r for r in rows if r.category in IDENTITY_CATEGORIES]
    rows.sort(key=lambda r: (_PRIORITY.get(r.category, 9), -r.confidence))
    out: list[RankedInsight] = []
    for r in rows[:limit]:
        out.append(RankedInsight(
            id=r.id, text=r.text, text_en=r.text_en, category=r.category, subject=r.subject,
            confidence=r.confidence, evidence_count=r.evidence_count,
            earliest_date=r.earliest_date, latest_date=r.latest_date,
            trajectory=r.trajectory, source=r.source, semantic_score=1.0, final_score=1.0,
        ))
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/generate/test_fact_router.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona_rag/generate/fact_router.py tests/generate/test_fact_router.py
git commit -m "feat(serve): intent classifier (anchor cosine) + CORE identity loader"
```

---

## Task 13: Wire router into `retrieve_insights`

**Files:**
- Modify: `persona_rag/graph/nodes/retrieve_insights.py:45-59`
- Test: `tests/graph/test_retrieve_insights.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/graph/test_retrieve_insights.py
@pytest.mark.asyncio
async def test_self_desc_query_sets_core_lane(monkeypatch):
    fake_client = MagicMock()
    fake_resp = MagicMock(); fake_resp.points = []
    fake_client.query_points = MagicMock(return_value=fake_resp)
    monkeypatch.setattr(
        "persona_rag.graph.nodes.retrieve_insights.make_client", lambda: fake_client)

    from datetime import UTC, datetime
    from persona_rag.insights.recency import RankedInsight
    now = datetime.now(UTC)
    core = [RankedInsight(id="b", text="навч", text_en="studies", category="bio",
                          subject="school", confidence=1.0, evidence_count=1,
                          earliest_date=now, latest_date=now, trajectory=None,
                          source="vault", semantic_score=1.0, final_score=1.0)]
    monkeypatch.setattr(
        "persona_rag.graph.nodes.retrieve_insights.classify_self_description",
        lambda vec, anchors, threshold: True)
    monkeypatch.setattr(
        "persona_rag.graph.nodes.retrieve_insights.load_core_facts",
        lambda *, limit, query_lang: core)
    with (
        patch("persona_rag.graph.nodes.retrieve_insights.embed_batch",
              AsyncMock(return_value=[[0.0] * 1536])),
        patch("persona_rag.graph.nodes.retrieve_insights.anchor_vecs",
              AsyncMock(return_value=[[0.0] * 1536])),
    ):
        out = await retrieve_insights(
            {"user_id": 1, "chat_id": 1, "incoming": "розкажи про себе"})
    ins = out["insights"]
    assert ins["lane"] == "self_desc"
    assert ins["query_lang"] == "uk"
    assert [c.subject for c in ins["core"]] == ["school"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/graph/test_retrieve_insights.py -k self_desc -v`
Expected: FAIL (no `lane` key).

- [ ] **Step 3: Implement the wiring**

In `persona_rag/graph/nodes/retrieve_insights.py`, add imports at top:

```python
from persona_rag.generate.fact_router import (
    anchor_vecs,
    classify_self_description,
    load_core_facts,
)
from persona_rag.generate.lang_detect import detect_language
```

Replace the `state["insights"] = {"semantic": semantic, "static": static}` line (~line 59) with:

```python
        query_lang = detect_language(state["incoming"])
        lane = "specific"
        core: list = []
        if s.INSIGHTS_FACTS_ROUTER_ENABLED:
            avs = await anchor_vecs()
            if classify_self_description(
                vec, avs, threshold=s.INSIGHTS_SELFDESC_ANCHOR_THRESHOLD
            ):
                lane = "self_desc"
                core = load_core_facts(
                    limit=s.INSIGHTS_CORE_MAX_FACTS, query_lang=query_lang
                )
            elif not semantic:
                lane = "none"
        state["insights"] = {
            "semantic": semantic, "static": static,
            "lane": lane, "core": core, "query_lang": query_lang,
        }
```

Note: the existing `except Exception` fallback (~line 67) must also set the new keys; update it to:

```python
        state["insights"] = {"semantic": [], "static": {},
                             "lane": "none", "core": [], "query_lang": "uk"}
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/graph/test_retrieve_insights.py -v`
Expected: PASS (all — existing tests still assert `semantic`).

- [ ] **Step 5: Commit**

```bash
git add persona_rag/graph/nodes/retrieve_insights.py tests/graph/test_retrieve_insights.py
git commit -m "feat(serve): route self-description queries to CORE lane in retrieve_insights"
```

---

## Task 14: `build_fact_card` — lane + language-aware card

**Files:**
- Modify: `persona_rag/generate/prompt.py:216-250`
- Test: `tests/generate/test_fact_card.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/generate/test_fact_card.py
from __future__ import annotations

from datetime import UTC, datetime

from persona_rag.generate.prompt import build_fact_card
from persona_rag.insights.recency import RankedInsight


def _fact(cat, subj, uk, en):
    now = datetime.now(UTC)
    return RankedInsight(id=subj, text=uk, text_en=en, category=cat, subject=subj,
                         confidence=1.0, evidence_count=1, earliest_date=now, latest_date=now,
                         trajectory=None, source="vault", semantic_score=1.0, final_score=1.0)


def test_self_desc_card_uses_core_in_query_lang():
    insights = {"lane": "self_desc", "query_lang": "en", "core": [
        _fact("bio", "school", "Навчається на CS", "Studies CS"),
    ], "semantic": []}
    card = build_fact_card("who are you", "", insights)
    assert "Studies CS" in card and "Навчається" not in card  # en chosen


def test_self_desc_card_uk_default():
    insights = {"lane": "self_desc", "query_lang": "uk", "core": [
        _fact("bio", "school", "Навчається на CS", "Studies CS")], "semantic": []}
    card = build_fact_card("розкажи про себе", "", insights)
    assert "Навчається на CS" in card


def test_specific_lane_filters_to_identity_categories():
    insights = {"lane": "specific", "query_lang": "uk", "core": [], "semantic": [
        _fact("bio", "school", "Навчається на CS", "Studies CS"),
        _fact("interest", "games", "грає", "plays"),  # non-identity → excluded
    ]}
    card = build_fact_card("де вчишся?", "", insights)
    assert "Навчається на CS" in card and "грає" not in card


def test_none_lane_returns_none():
    insights = {"lane": "none", "query_lang": "uk", "core": [], "semantic": []}
    assert build_fact_card("що по погоді", "", insights) is None


def test_card_respects_400_char_cap():
    core = [_fact("bio", f"s{i}", "x" * 100, "y" * 100) for i in range(10)]
    insights = {"lane": "self_desc", "query_lang": "uk", "core": core, "semantic": []}
    card = build_fact_card("хто ти", "", insights)
    assert len(card) <= 400
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/generate/test_fact_card.py -v`
Expected: FAIL (`build_fact_card` undefined).

- [ ] **Step 3: Implement — replace `_compact_facts`**

In `persona_rag/generate/prompt.py`, add import near the top:

```python
from persona_rag.generate.fact_router import IDENTITY_CATEGORIES
```

Replace the `_compact_facts` function with `build_fact_card`:

```python
def _render_fact(r: Any, query_lang: str) -> str:
    text = getattr(r, "text_en", None) if query_lang == "en" else None
    return text or getattr(r, "text", "")


def build_fact_card(
    incoming: str, user_memory: str, insights: dict[str, Any] | None, *, cap: int = 400
) -> str | None:
    """Lane + language-aware fact card for the thin LoRA path (spec 2026-06-03 §6).

    self_desc → curated CORE; specific → identity-category semantic hits;
    none → nothing. Rendered in the query language, capped. The system turn is
    never in training loss, so a brief in-language addendum is a mild conditioning
    shift — never the full insights block.
    """
    ins = insights or {}
    lane = ins.get("lane", "specific")
    query_lang = ins.get("query_lang", "uk")
    parts: list[str] = []
    if user_memory and user_memory.strip():
        parts.append(user_memory.strip())
    if lane == "self_desc":
        for r in ins.get("core", []):
            parts.append(f"- {_render_fact(r, query_lang)}")
    elif lane == "specific":
        for r in ins.get("semantic", []):
            if getattr(r, "category", None) in IDENTITY_CATEGORIES:
                parts.append(f"- {_render_fact(r, query_lang)}")
    joined = "\n".join(parts).strip()
    return joined[:cap] or None
```

In `build_messages` (the `ollama` branch, ~line 249), change:

```python
        facts = _compact_facts(user_memory, insights) if s.OLLAMA_FACTS_IN_SYSTEM else None
```

to:

```python
        facts = build_fact_card(incoming, user_memory, insights) if s.OLLAMA_FACTS_IN_SYSTEM else None
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/generate/test_fact_card.py tests/generate -v`
Expected: PASS. Also run `uv run pytest -q` for a full-suite regression check.

- [ ] **Step 5: Commit**

```bash
git add persona_rag/generate/prompt.py tests/generate/test_fact_card.py
git commit -m "feat(serve): lane + language-aware build_fact_card replaces _compact_facts"
```

---

## Task 15: Register-invariance validation (open-Q#6)

**Files:**
- Create: `tests/eval/test_vault_register_invariance.py`
- Reference: `persona_rag/eval/distribution.py` (`paren_smiley_rate`, `per_bubble_lengths`),
  `persona_rag/eval/compare.py` (`shape_js_metric`)
- Test: `tests/eval/test_vault_register_invariance.py`

This task has **two layers**: an automated *construction-level* invariant (runs in CI, the
Tier-1 green target) and a documented *generation-level* A/B (needs llama-server, run at the
Tier-1b checkpoint).

- [ ] **Step 1: Write the construction-level test (the automated invariant)**

```python
# tests/eval/test_vault_register_invariance.py
"""Construction-level register invariance: facts-on must add a card ONLY for
identity-relevant turns, never for control turns — so control-turn prompts are
byte-identical facts-off vs facts-on (no register perturbation possible)."""
from __future__ import annotations

from persona_rag.generate.prompt import build_fact_card

SELF_DESC = ["розкажи про себе", "tell me about yourself", "who are you", "расскажи о себе"]
CONTROL = ["що по погоді", "го грати", "ахах да"]


def _core():
    from datetime import UTC, datetime
    from persona_rag.insights.recency import RankedInsight
    now = datetime.now(UTC)
    return [RankedInsight(id="b", text="Навчається на CS", text_en="Studies CS",
                          category="bio", subject="school", confidence=1.0, evidence_count=1,
                          earliest_date=now, latest_date=now, trajectory=None, source="vault",
                          semantic_score=1.0, final_score=1.0)]


def test_control_turns_get_no_card():
    for q in CONTROL:
        ins = {"lane": "none", "query_lang": "uk", "core": [], "semantic": []}
        assert build_fact_card(q, "", ins) is None


def test_self_desc_turns_get_card_within_cap():
    for q in SELF_DESC:
        ins = {"lane": "self_desc", "query_lang": "uk", "core": _core(), "semantic": []}
        card = build_fact_card(q, "", ins)
        assert card is not None and len(card) <= 400
        assert "Навчається на CS" in card
```

- [ ] **Step 2: Run to verify it fails, then passes**

Run: `uv run pytest tests/eval/test_vault_register_invariance.py -v`
Expected: FAIL before Task 14, PASS after (it depends on `build_fact_card`). Since Task 14
is done, it should PASS immediately — if so, that is the green signal for this layer.

- [ ] **Step 3: Add the generation-level A/B as a documented Make target**

In `Makefile`, after `insights-vault` add (append to `.PHONY`):

```makefile
# Generation-level register-invariance A/B. Needs llama-server up (gguf present).
# Runs the probe set facts-OFF then facts-ON; compares shape_js / paren_smiley / length.
compare-vault:
	OLLAMA_FACTS_IN_SYSTEM=false uv run python scripts/compare_persona.py --n 60 --seed 0 --name vault_off
	OLLAMA_FACTS_IN_SYSTEM=true  uv run python scripts/compare_persona.py --n 60 --seed 0 --name vault_on
	@echo "Compare reports/vault_off vs reports/vault_on: shape_js / paren_smiley / per-bubble length must be within noise; self-description replies must become fact-faithful."
```

- [ ] **Step 4: Document the acceptance bar (no code — record in the plan/PR)**

Generation-level PASS criteria (run at Tier-1b checkpoint with llama-server up):
- `shape_js`, `paren_smiley_rate`, mean `per_bubble_lengths`: facts-on vs facts-off delta
  within the facts-off run-to-run noise band (re-run facts-off twice to size the band).
- Self-description (class A) replies contain the fixture/real identity facts; fabricated
  identity claims → ~0.

- [ ] **Step 5: Commit**

```bash
git add tests/eval/test_vault_register_invariance.py Makefile
git commit -m "test(vault): construction-level register invariance + compare-vault A/B target"
```

---

## Self-Review

**1. Spec coverage**

| Spec § | Covered by |
|--------|-----------|
| §5 taxonomy {bio,relationship,value,opinion} | Task 4 (VAULT_CATEGORIES, prompt) |
| §6 intent router (3 lanes) | Tasks 12–14 |
| §7 text_en + source=vault | Tasks 2, 7, 11 |
| §8 TDD probes (A/B/C) | Tasks 12–15 (assertions per class) |
| §9 ingestion + full-rebuild + make target | Tasks 3–9 |
| §10 lang detect + card builder | Tasks 10, 14 |
| §12 gitignore/privacy | Task 1 |
| §13 files add/change | all tasks (file headers) |
| §14 A/B register invariance | Task 15 |
| §16 settings | Task 1 |
| Phase 3 (embedder swap) | **out of scope** — deferred per spec §11 |

No uncovered v1 requirement.

**2. Placeholder scan:** every code/test step contains runnable code; no TBD/TODO. The
generation-level A/B (Task 15 step 4) is intentionally a manual checkpoint (needs
llama-server) — its automated counterpart (step 1) is fully specified.

**3. Type/name consistency:** `RawVaultFact`, `VaultFact`, `build_fact_card`,
`load_core_facts`, `classify_self_description`, `anchor_vecs`, `detect_language`,
`IDENTITY_CATEGORIES`, `_wipe_vault_rows`, `rebuild_vault`, `persist_vault`,
`consolidate_vault`, `extract_vault_chunk`, `parse_vault_response`, `chunk_markdown`,
`read_vault_files` — referenced consistently across tasks. `text_en` field name uniform in
models.py, recency.py, Qdrant payload, and renderers. `insights` state keys
(`lane`/`core`/`query_lang`/`semantic`) consistent between Task 13 (producer) and Task 14
(consumer).

**Deviation from spec note:** spec §9 said "reuse `consolidate()`"; the plan instead reuses
its *primitives* (`normalize_subject`, `_stable_insight_id`) via a thin `consolidate_vault`
— because dual-language (`text_uk`/`text_en`) doesn't thread cleanly through the chat
`consolidate()`'s single-`text` LLM merge. Same outcome (stable-ID dedup), smaller blast
radius, no chat-model changes. Recorded here per the executing-plans deviation protocol.

---

## Execution Handoff

Per spec §15: this runs **autonomous-to-green in a git worktree** (isolated Qdrant test
collection via `QDRANT_INSIGHTS_COLLECTION` env override + synthetic fixture vault), then a
checkpoint for the real-vault faithfulness + voice sign-off.
