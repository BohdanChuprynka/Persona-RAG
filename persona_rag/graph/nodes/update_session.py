from __future__ import annotations

from datetime import UTC, datetime

from persona_rag.config import get_settings
from persona_rag.graph.nodes.load_session import _SESSIONS, SessionEntry
from persona_rag.graph.state import GraphState
from persona_rag.models import ChatMessage


def update_session(state: GraphState) -> GraphState:
    """Append (incoming, reply) to the user's session ring buffer.

    No-op when reply is empty (auth-blocked, shadow with no reply recorded, or
    guardrail-stripped). Caps history to ``CURRENT_SESSION_WINDOW`` messages.
    """
    reply = state.get("reply", "")
    if not reply:
        return state

    uid = state["user_id"]
    entry = _SESSIONS.get(uid)
    if entry is None:
        entry = SessionEntry()
        _SESSIONS[uid] = entry

    entry.messages.append(ChatMessage(role="user", content=state["incoming"]))
    entry.messages.append(ChatMessage(role="assistant", content=reply))

    window = get_settings().CURRENT_SESSION_WINDOW
    if len(entry.messages) > window:
        entry.messages = entry.messages[-window:]

    entry.last_seen = datetime.now(UTC)
    return state
