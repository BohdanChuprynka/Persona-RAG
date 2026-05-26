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


def extract_counterparty_rhythms(rows: list[PersonaTurnRow]) -> list[dict[str, Any]]:
    """Aggregate per-recipient statistics."""
    per: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "total_chars": 0,
            "first_seen": None,
            "last_seen": None,
            "emoji_total": 0,
        }
    )
    for r in rows:
        bucket = per[r.recipient_id_hash]
        bucket["count"] += 1
        bucket["total_chars"] += r.your_reply_len_chars
        bucket["emoji_total"] += r.your_reply_emoji_count
        if bucket["first_seen"] is None or r.timestamp < bucket["first_seen"]:
            bucket["first_seen"] = r.timestamp
        if bucket["last_seen"] is None or r.timestamp > bucket["last_seen"]:
            bucket["last_seen"] = r.timestamp

    out: list[dict[str, Any]] = []
    for recipient, stats in per.items():
        out.append(
            {
                "subject": recipient,
                "count": stats["count"],
                "n_sessions": 0,
                "first_seen": stats["first_seen"],
                "last_seen": stats["last_seen"],
                "avg_chars": stats["total_chars"] / stats["count"] if stats["count"] else 0,
                "emoji_rate": (
                    stats["emoji_total"] / stats["total_chars"] if stats["total_chars"] else 0
                ),
            }
        )
    out.sort(key=lambda x: x["count"], reverse=True)
    return out[:20]


def extract_languages(rows: list[PersonaTurnRow]) -> list[dict[str, Any]]:
    """Language distribution + totals."""
    from collections import Counter

    counter: Counter[str] = Counter()
    first_seen: dict[str, datetime] = {}
    last_seen: dict[str, datetime] = {}
    for r in rows:
        lang = r.language or "unknown"
        counter[lang] += 1
        if lang not in first_seen or r.timestamp < first_seen[lang]:
            first_seen[lang] = r.timestamp
        if lang not in last_seen or r.timestamp > last_seen[lang]:
            last_seen[lang] = r.timestamp

    total = sum(counter.values()) or 1
    out: list[dict[str, Any]] = []
    for lang, n in counter.most_common():
        out.append(
            {
                "subject": lang,
                "count": n,
                "n_sessions": 0,
                "first_seen": first_seen[lang],
                "last_seen": last_seen[lang],
                "percentage": round(n / total, 4),
            }
        )
    return out


def _quarter_label(ts: datetime) -> str:
    return f"{ts.year}-Q{(ts.month - 1) // 3 + 1}"


def extract_phases(rows: list[PersonaTurnRow]) -> list[dict[str, Any]]:
    """Bucket turns by quarter; for each quarter, compute count + top entities."""
    by_quarter: dict[str, list[PersonaTurnRow]] = defaultdict(list)
    for r in rows:
        by_quarter[_quarter_label(r.timestamp)].append(r)

    out: list[dict[str, Any]] = []
    for label, quarter_rows in by_quarter.items():
        top_entities = extract_entities(quarter_rows)[:3]
        out.append(
            {
                "subject": label,
                "count": len(quarter_rows),
                "n_sessions": 0,
                "first_seen": min(r.timestamp for r in quarter_rows),
                "last_seen": max(r.timestamp for r in quarter_rows),
                "top_entities": [e["subject"] for e in top_entities],
                "primary_language": (
                    max(
                        {r.language for r in quarter_rows},
                        key=lambda lg: sum(1 for r in quarter_rows if r.language == lg),
                    )
                ),
            }
        )
    out.sort(key=lambda x: x["subject"])
    return out


def extract_style(rows: list[PersonaTurnRow]) -> list[dict[str, Any]]:
    """All-caps frequency, multi-line burst frequency."""
    all_caps_count = 0
    multi_line_count = 0
    code_switch_count = 0
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    prev_lang: str | None = None

    for r in rows:
        if r.your_reply and r.your_reply == r.your_reply.upper() and len(r.your_reply) > 2:
            all_caps_count += 1
        if "\n" in r.your_reply:
            multi_line_count += 1
        if prev_lang and r.language and prev_lang != r.language:
            code_switch_count += 1
        prev_lang = r.language
        if first_seen is None or r.timestamp < first_seen:
            first_seen = r.timestamp
        if last_seen is None or r.timestamp > last_seen:
            last_seen = r.timestamp

    if first_seen is None:
        return []

    return [
        {
            "subject": "all_caps",
            "count": all_caps_count,
            "n_sessions": 0,
            "first_seen": first_seen,
            "last_seen": last_seen,
        },
        {
            "subject": "multi_line",
            "count": multi_line_count,
            "n_sessions": 0,
            "first_seen": first_seen,
            "last_seen": last_seen,
        },
        {
            "subject": "code_switch",
            "count": code_switch_count,
            "n_sessions": 0,
            "first_seen": first_seen,
            "last_seen": last_seen,
        },
    ]


def run_stage_a(rows: list[PersonaTurnRow]) -> dict[str, list[dict[str, Any]]]:
    """Run all five Stage A extractors. Caller persists results."""
    return {
        "entity": extract_entities(rows),
        "rhythm": extract_counterparty_rhythms(rows),
        "language": extract_languages(rows),
        "phase": extract_phases(rows),
        "style": extract_style(rows),
    }
