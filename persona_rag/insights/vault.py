"""Vault fact ingestion (spec 2026-06-03).

Pipeline: read drop-folder md → heading-aware chunks → dual-language identity
extraction (gpt-4o, offline) → stable-ID dedup → persist source="vault" to
SQLite + Qdrant. Full-rebuild each run so edits/removals in the folder are
reflected. Raw note text NEVER reaches the serving model — only distilled facts.

Reuses the chat pipeline's dedup primitives (`normalize_subject`,
`_stable_insight_id`) and the index/embedder/qdrant helpers; it does NOT reuse
the chat `consolidate()` LLM-merge because dual-language text doesn't thread
through its single-`text` path and a curated vault rarely needs merging.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
)
from sqlmodel import Session, select

from persona_rag._logging import get_logger
from persona_rag.config import get_settings
from persona_rag.db.engine import make_engine
from persona_rag.db.models import InsightRow
from persona_rag.generate.llm_client import chat_complete
from persona_rag.index.embedder import embed_batch
from persona_rag.index.qdrant_store import to_qdrant_point_id
from persona_rag.insights.consolidator import _stable_insight_id, normalize_subject

log = get_logger()

VAULT_CATEGORIES = {"bio", "relationship", "value", "opinion"}

# Vault notes are user-authored, so the facts are trusted (like onboarding). We do
# NOT ask the model to rate confidence — gpt-4o-mini echoed the schema's 0.0
# placeholder on every fact and sank them all to "pending". Assign a fixed high
# value so curated facts route to "approved".
_VAULT_FACT_CONFIDENCE = 0.9

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
      "text_en": "<one English sentence>"
    }}
  ]
}}

Rules:
- Only durable identity. If a chunk has none, return {{"facts": []}}.
- Be specific: "studies CS at Fictional State University" beats "is a student".
- Max 8 facts per chunk. Each text <= 25 words.
"""

_HEADING_RE = re.compile(r"^#{1,2}\s+", re.MULTILINE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)

_VAULT_FILTER = Filter(must=[FieldCondition(key="source", match=MatchValue(value="vault"))])


@dataclass
class VaultDoc:
    relpath: str
    text: str


class RawVaultFact(BaseModel):
    category: Literal["bio", "relationship", "value", "opinion"]
    subject: str
    text_uk: str
    text_en: str
    confidence: float
    source_file: str


class VaultFact(BaseModel):
    id: str
    category: str
    subject: str  # canonical
    text_uk: str
    text_en: str
    confidence: float
    source_files: list[str]


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
    idxs = [m.start() for m in _HEADING_RE.finditer(text)]
    if not idxs:
        sections = [text]
    else:
        bounds = [*idxs, len(text)]
        sections = [text[bounds[i] : bounds[i + 1]] for i in range(len(idxs))]
        if idxs[0] > 0:
            sections.insert(0, text[: idxs[0]])
    chunks: list[str] = []
    for raw_sec in sections:
        sec = raw_sec.strip()
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


def parse_vault_response(text: str, *, source_file: str) -> list[RawVaultFact]:
    """Parse the extractor's dual-language JSON. Raises ValueError on bad JSON."""
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
            out.append(
                RawVaultFact(
                    category=item["category"],
                    subject=str(item["subject"]),
                    text_uk=str(item["text_uk"]),
                    text_en=str(item["text_en"]),
                    confidence=_VAULT_FACT_CONFIDENCE,
                    source_file=source_file,
                )
            )
        except (KeyError, ValidationError, ValueError):
            continue
    return out


async def extract_vault_chunk(chunk: str, *, source_file: str) -> list[RawVaultFact]:
    """Single LLM call: a markdown chunk -> dual-language identity facts."""
    s = get_settings()
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": VAULT_EXTRACT_SYSTEM_PROMPT.format(persona_name=s.PERSONA_NAME),
        },
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


def consolidate_vault(raws: list[RawVaultFact]) -> list[VaultFact]:
    """Group by (category, normalized subject); keep the highest-confidence member.

    Reuses `normalize_subject` + `_stable_insight_id` for cross-run idempotency,
    but skips the chat pipeline's LLM merge — a curated vault rarely has 3+
    near-duplicate facts, and the LLM merge can't carry dual-language text.
    """
    groups: dict[tuple[str, str], list[RawVaultFact]] = {}
    for r in raws:
        key = (r.category, normalize_subject(r.subject))
        groups.setdefault(key, []).append(r)
    out: list[VaultFact] = []
    for (category, canon), members in groups.items():
        best = max(members, key=lambda r: r.confidence)
        out.append(
            VaultFact(
                id=_stable_insight_id(category, canon),
                category=category,
                subject=canon,
                text_uk=best.text_uk,
                text_en=best.text_en,
                confidence=best.confidence,
                source_files=sorted({m.source_file for m in members}),
            )
        )
    return out


async def _wipe_vault_rows(*, qdrant_client: QdrantClient, collection: str) -> None:
    """Delete all source='vault' rows from SQLite + Qdrant (full-rebuild step)."""
    with Session(make_engine()) as s:
        for row in s.exec(select(InsightRow).where(InsightRow.source == "vault")).all():
            s.delete(row)
        s.commit()
    qdrant_client.delete(
        collection_name=collection, points_selector=FilterSelector(filter=_VAULT_FILTER)
    )


async def persist_vault(
    facts: list[VaultFact], *, qdrant_client: QdrantClient, collection: str, threshold: float
) -> None:
    """Write facts to SQLite (source='vault'); embed + upsert only approved to Qdrant."""
    now = datetime.now(UTC)
    approved: list[VaultFact] = []
    with Session(make_engine()) as s:
        for f in facts:
            status = "approved" if f.confidence >= threshold else "pending"
            s.add(
                InsightRow(
                    id=f.id,
                    category=f.category,
                    subject=f.subject,
                    text=f.text_uk,
                    text_en=f.text_en,
                    confidence=f.confidence,
                    evidence_count=1,
                    earliest_date=now,
                    latest_date=now,
                    trajectory=None,
                    source_session_ids=json.dumps(f.source_files),
                    distinct_partners=0,
                    source="vault",
                    review_status=status,
                    edited_text=None,
                    created_at=now,
                    updated_at=now,
                )
            )
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
                "sqlite_id": f.id,
                "category": f.category,
                "subject": f.subject,
                "text": f.text_uk,
                "text_en": f.text_en,
                "confidence": f.confidence,
                "evidence_count": 1,
                "earliest_date": now.isoformat(),
                "latest_date": now.isoformat(),
                "trajectory": None,
                "source": "vault",
                "review_status": "approved",
            },
        )
        for f, vec in zip(approved, vectors, strict=True)
    ]
    qdrant_client.upsert(collection_name=collection, points=points)


async def rebuild_vault(
    *, directory: str, qdrant_client: QdrantClient, collection: str, threshold: float
) -> int:
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
    await persist_vault(
        facts, qdrant_client=qdrant_client, collection=collection, threshold=threshold
    )
    log.info("vault_rebuild_done", docs=len(docs), raws=len(raws), facts=len(facts))
    return len(facts)
