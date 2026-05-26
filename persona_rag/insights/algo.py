# ruff: noqa: RUF001, RUF003
"""Stage A — algorithmic signals extracted from PersonaTurnRow corpus.

Each extractor returns a list of dicts shaped for direct conversion to
``AlgoSignal`` rows. Final persistence happens in `algo` orchestrator (Task 6).
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from typing import Any

from persona_rag.db.models import PersonaTurnRow
from persona_rag.insights.blocklists import passes_entity_filter

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЇїІіЄєҐґЁё]{3,}")
SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")


def _tokens_with_positions(text: str) -> list[tuple[str, int]]:
    """Yield (token, position_in_sentence) for every alpha token.

    Sentences are split on punctuation + newline. Position 0 means first
    token of a sentence — used to detect sentence-start capitalization.
    """
    out: list[tuple[str, int]] = []
    for sentence in SENTENCE_SPLIT_RE.split(text):
        for i, m in enumerate(TOKEN_RE.finditer(sentence)):
            out.append((m.group(0), i))
    return out


def extract_entities(rows: list[PersonaTurnRow]) -> list[dict[str, Any]]:
    """Mine candidate entities from persona's own replies.

    Returns a list of dicts ready to become AlgoSignal(kind="entity", ...).
    """
    counts: dict[str, int] = defaultdict(int)
    sessions: dict[str, set[str]] = defaultdict(set)
    positions: dict[str, list[int]] = defaultdict(list)
    first_seen: dict[str, datetime] = {}
    last_seen: dict[str, datetime] = {}

    for r in rows:
        tokens = _tokens_with_positions(r.your_reply)
        seen_in_row: set[str] = set()
        for token, pos in tokens:
            counts[token] += 1
            positions[token].append(pos)
            if token not in seen_in_row:
                sessions[token].add(r.chat_id_hash)
                seen_in_row.add(token)
            if token not in first_seen or r.timestamp < first_seen[token]:
                first_seen[token] = r.timestamp
            if token not in last_seen or r.timestamp > last_seen[token]:
                last_seen[token] = r.timestamp

    candidates: list[dict[str, Any]] = []
    for token, count in counts.items():
        n_sessions = len(sessions[token])
        all_zero = all(p == 0 for p in positions[token])
        if not passes_entity_filter(
            token,
            count=count,
            n_sessions=n_sessions,
            all_zero_positions=all_zero,
        ):
            continue
        candidates.append(
            {
                "subject": token,
                "count": count,
                "n_sessions": n_sessions,
                "first_seen": first_seen[token],
                "last_seen": last_seen[token],
            }
        )

    # Rank by count × distinct_session_count; rewards recurring across-chat mentions.
    candidates.sort(key=lambda c: c["count"] * c["n_sessions"], reverse=True)
    return candidates[:50]
