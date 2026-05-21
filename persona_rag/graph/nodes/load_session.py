from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from persona_rag.config import get_settings
from persona_rag.graph.state import GraphState
from persona_rag.models import ChatMessage


@dataclass
class SessionEntry:
    messages: list[ChatMessage] = field(default_factory=list)
    last_seen: datetime = field(default_factory=lambda: datetime.now(UTC))


# In-memory session store keyed by user_id; replaced with Redis later if needed.
_SESSIONS: dict[int, SessionEntry] = {}


def _is_expired(entry: SessionEntry) -> bool:
    timeout = timedelta(minutes=get_settings().SESSION_TIMEOUT_MINUTES)
    return datetime.now(UTC) - entry.last_seen > timeout


def load_session(state: GraphState) -> GraphState:
    entry = _SESSIONS.get(state["user_id"])
    if entry is None:
        state["session"] = []
        return state
    if _is_expired(entry):
        _SESSIONS.pop(state["user_id"], None)
        state["session"] = []
        return state
    state["session"] = list(entry.messages)
    return state


def get_sessions() -> dict[int, SessionEntry]:
    """Test/debug accessor."""
    return _SESSIONS
