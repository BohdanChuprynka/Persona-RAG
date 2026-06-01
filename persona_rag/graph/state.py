from __future__ import annotations

from typing import Any, TypedDict

from persona_rag.models import ChatMessage, RetrievedTurn


class GraphState(TypedDict, total=False):
    incoming: str
    user_id: int
    chat_id: int
    session_id: str
    auth_state: str
    retrieved: list[RetrievedTurn]
    memory: str
    insights: dict[str, Any]  # {"semantic": list[RankedInsight], "static": dict}
    session: list[ChatMessage]
    style_anchors_json: str
    prompt: list[dict[str, str]]
    reply: str
    shadow: bool
