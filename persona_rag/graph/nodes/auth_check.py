from __future__ import annotations

from persona_rag.bot.auth import get_user_state
from persona_rag.graph.state import GraphState


def auth_check(state: GraphState) -> GraphState:
    state["auth_state"] = get_user_state(state["user_id"]).value
    return state
