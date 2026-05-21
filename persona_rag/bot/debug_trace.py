"""In-memory cache of the last graph trace per user.

Lets the admin inspect what RAG actually retrieved + the exact prompt sent to
the model. Stored only in process memory — wiped on bot restart, never written
to disk.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TypedDict

from persona_rag.models import RetrievedTurn


class LastTurn(TypedDict):
    ts: datetime
    incoming: str
    retrieved: list[RetrievedTurn]
    prompt: list[dict[str, str]]
    reply: str


_LAST: dict[int, LastTurn] = {}


def record(
    user_id: int,
    *,
    incoming: str,
    retrieved: list[RetrievedTurn],
    prompt: list[dict[str, str]],
    reply: str,
) -> None:
    _LAST[user_id] = {
        "ts": datetime.now(UTC),
        "incoming": incoming,
        "retrieved": retrieved,
        "prompt": prompt,
        "reply": reply,
    }


def get(user_id: int) -> LastTurn | None:
    return _LAST.get(user_id)


def get_latest_any() -> tuple[int, LastTurn] | None:
    if not _LAST:
        return None
    uid = max(_LAST, key=lambda u: _LAST[u]["ts"])
    return uid, _LAST[uid]
