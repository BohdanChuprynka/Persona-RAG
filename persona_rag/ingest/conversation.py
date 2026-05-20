from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import timedelta

from persona_rag.config import get_settings
from persona_rag.models import RawMessage


def collapse_bursts(
    msgs: Iterable[RawMessage], *, burst_seconds: int | None = None
) -> list[RawMessage]:
    """Consecutive same-sender messages within burst_seconds → joined with newline."""
    if burst_seconds is None:
        burst_seconds = get_settings().MESSAGE_BURST_SECONDS
    burst = timedelta(seconds=burst_seconds)
    out: list[RawMessage] = []
    for m in msgs:
        if out and out[-1].sender_id == m.sender_id and (m.timestamp - out[-1].timestamp) <= burst:
            prev = out[-1]
            out[-1] = prev.model_copy(update={"text": f"{prev.text}\n{m.text}"})
        else:
            out.append(m)
    return out


def split_sessions(
    msgs: Iterable[RawMessage], *, gap_hours: int | None = None
) -> Iterator[list[RawMessage]]:
    if gap_hours is None:
        gap_hours = get_settings().SESSION_BREAK_HOURS
    gap = timedelta(hours=gap_hours)
    current: list[RawMessage] = []
    for m in msgs:
        if current and (m.timestamp - current[-1].timestamp) > gap:
            yield current
            current = []
        current.append(m)
    if current:
        yield current
