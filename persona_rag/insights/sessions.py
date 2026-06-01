"""Stage B — group PersonaTurnRows into sessions, then filter for signal."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from persona_rag.db.models import PersonaTurnRow


@dataclass
class SessionDoc:
    session_id: str
    chat_id_hash: str
    start: datetime
    end: datetime
    n_persona_turns: int
    persona_chars: int
    primary_language: str
    turns: list[PersonaTurnRow] = field(default_factory=list)


def _session_id(chat_id: str, start: datetime) -> str:
    payload = f"{chat_id}:{start.isoformat()}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _ensure_aware(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def _doc(chat_id_hash: str, turns: list[PersonaTurnRow]) -> SessionDoc:
    langs = Counter(t.language for t in turns if t.language)
    primary = langs.most_common(1)[0][0] if langs else "unknown"
    start = turns[0].timestamp
    return SessionDoc(
        session_id=_session_id(chat_id_hash, start),
        chat_id_hash=chat_id_hash,
        start=start,
        end=turns[-1].timestamp,
        n_persona_turns=len(turns),
        persona_chars=sum(t.your_reply_len_chars for t in turns),
        primary_language=primary,
        turns=list(turns),
    )


def build_sessions(rows: Iterable[PersonaTurnRow], *, gap_hours: int = 6) -> list[SessionDoc]:
    """Group rows by (chat_id_hash, time-gap > gap_hours) into sessions."""
    gap = timedelta(hours=gap_hours)
    by_chat: dict[str, list[PersonaTurnRow]] = {}
    for r in rows:
        by_chat.setdefault(r.chat_id_hash, []).append(r)

    sessions: list[SessionDoc] = []
    for chat_id, chat_rows in by_chat.items():
        chat_rows.sort(key=lambda r: r.timestamp)
        current: list[PersonaTurnRow] = []
        for r in chat_rows:
            if current and (r.timestamp - current[-1].timestamp) > gap:
                sessions.append(_doc(chat_id, current))
                current = []
            current.append(r)
        if current:
            sessions.append(_doc(chat_id, current))
    return sessions


def _non_reaction_ratio(turns: list[PersonaTurnRow]) -> float:
    if not turns:
        return 0.0
    n_non_reaction = sum(1 for t in turns if len(t.your_reply) > 5)
    return n_non_reaction / len(turns)


def filter_high_signal(
    sessions: list[SessionDoc],
    *,
    history_years: float,
    min_turns: int,
    min_chars: int,
    max_sessions: int,
    non_reaction_floor: float = 0.20,
    now: datetime | None = None,
) -> list[SessionDoc]:
    """Drop low-signal + out-of-window sessions; cap to max_sessions by chars desc."""
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(days=history_years * 365)
    keep: list[SessionDoc] = []
    for s in sessions:
        if _ensure_aware(s.start) < cutoff:
            continue
        if s.n_persona_turns < min_turns:
            continue
        if s.persona_chars < min_chars:
            continue
        if _non_reaction_ratio(s.turns) < non_reaction_floor:
            continue
        keep.append(s)
    keep.sort(key=lambda s: s.persona_chars, reverse=True)
    return keep[:max_sessions]
