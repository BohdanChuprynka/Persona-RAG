"""Stage D — merge near-duplicate insights via synonym map + optional LLM."""

from __future__ import annotations

import hashlib
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel

from persona_rag._logging import get_logger
from persona_rag.generate.llm_client import chat_complete
from persona_rag.insights.extractor import RawInsight

log = get_logger()

_PUNCT_RE = re.compile(r"[^\w\s+]+")
_WS_RE = re.compile(r"\s+")


def _stable_insight_id(category: str, canonical_subject: str) -> str:
    """Deterministic 16-char hex ID — same (category, canonical_subject) always maps to same ID.

    Lets persist_insights upsert across runs without losing user-touched state.
    """
    payload = f"{category}:{canonical_subject}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def normalize_subject(subject: str, synonyms: dict[str, list[str]] | None = None) -> str:
    """Lowercase, strip punctuation (except +), collapse whitespace, then apply synonym map."""
    s = subject.lower()
    # Preserve '+' (for C++), strip other punct
    s = _PUNCT_RE.sub("", s)
    s = _WS_RE.sub(" ", s).strip()
    if synonyms:
        for canonical, variants in synonyms.items():
            if s == canonical:
                return canonical
            for v in variants:
                vn = _WS_RE.sub(" ", _PUNCT_RE.sub("", v.lower())).strip()
                if s == vn:
                    return canonical
    return s


def load_synonyms(path: Path) -> dict[str, list[str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    return {str(k): list(v) for k, v in data.items() if isinstance(v, list)}


class ConsolidatedInsight(BaseModel):
    id: str
    category: str
    canonical_subject: str
    text: str
    confidence: float
    evidence_count: int
    earliest_date: datetime
    latest_date: datetime
    trajectory: str | None
    source_session_ids: list[str]
    # Spec 2026-05-31 §6.3: count of unique chat partners (recipient_id_hash)
    # contributing source sessions. Used by Stage E to require breadth.
    distinct_partners: int = 0


CONSOLIDATE_PROMPT = """\
Below are {n} similar observations about a person across time.
Each observation has a category and a date.

Write ONE consolidated insight that captures the merged truth, in 1-2 sentences.
Then on a new line starting with "Trajectory:", write a one-line note about
how this changed over time (e.g. "active 2023-Q1 → present", or "one-off in 2024-Q3").

Observations:
{observations}

Output format:
<consolidated insight sentence>
Trajectory: <one-line trajectory>
"""


def _split_consolidation(text: str) -> tuple[str, str | None]:
    lines = text.strip().splitlines()
    body_lines = []
    trajectory: str | None = None
    for line in lines:
        if line.lower().startswith("trajectory:"):
            trajectory = line.split(":", 1)[1].strip()
        else:
            body_lines.append(line)
    return " ".join(body_lines).strip(), trajectory


async def consolidate(
    raws: list[RawInsight],
    *,
    synonyms: dict[str, list[str]],
) -> list[ConsolidatedInsight]:
    """Group raw insights by (category, normalized_subject); merge groups of 3+ via LLM."""
    from persona_rag.config import get_settings

    groups: dict[tuple[str, str], list[RawInsight]] = defaultdict(list)
    for r in raws:
        key = (r.category, normalize_subject(r.subject, synonyms))
        groups[key].append(r)

    total_merges = sum(1 for m in groups.values() if len(m) >= 3)
    log.info(
        "insights_stage_d_start",
        raws_total=len(raws),
        groups_total=len(groups),
        merges_total=total_merges,
    )
    s = get_settings()
    out: list[ConsolidatedInsight] = []
    merge_num = 0
    stage_d_t0 = time.monotonic()
    for (category, canon), members in groups.items():
        members.sort(key=lambda r: r.extracted_at)
        confidence = max(r.confidence for r in members)
        earliest = min(r.extracted_at for r in members)
        latest = max(r.extracted_at for r in members)
        session_ids = list({r.session_id for r in members})

        if len(members) >= 3:
            merge_num += 1
            obs_block = "\n".join(
                f"- [{r.extracted_at:%Y-%m-%d}] (conf={r.confidence:.2f}) {r.text}" for r in members
            )
            response = await chat_complete(
                [
                    {
                        "role": "user",
                        "content": CONSOLIDATE_PROMPT.format(
                            n=len(members), observations=obs_block
                        ),
                    }
                ],
                model=s.INSIGHTS_CONSOLIDATE_MODEL,
                temperature=0.2,
                max_tokens=500,
            )
            text, trajectory = _split_consolidation(response)
            log.info(
                "insights_merge_done",
                merge_num=merge_num,
                total_merges=total_merges,
                category=category,
                subject=canon,
                group_size=len(members),
                stage_d_elapsed_s=round(time.monotonic() - stage_d_t0, 1),
            )
        else:
            best = max(members, key=lambda r: r.confidence)
            text = best.text
            trajectory = None

        out.append(
            ConsolidatedInsight(
                id=_stable_insight_id(category, canon),
                category=category,
                canonical_subject=canon,
                text=text,
                confidence=confidence,
                evidence_count=len(members),
                earliest_date=earliest,
                latest_date=latest,
                trajectory=trajectory,
                source_session_ids=session_ids,
            )
        )
    return out
