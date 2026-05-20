from __future__ import annotations

from persona_rag.graph.state import GraphState
from persona_rag.models import ChatMessage

# In-memory session store keyed by user_id; replaced with Redis later if needed.
_SESSIONS: dict[int, list[ChatMessage]] = {}


def load_session(state: GraphState) -> GraphState:
    state["session"] = list(_SESSIONS.get(state["user_id"], []))
    return state


def get_sessions() -> dict[int, list[ChatMessage]]:
    """Test/debug accessor."""
    return _SESSIONS
