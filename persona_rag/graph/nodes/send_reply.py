from __future__ import annotations

from typing import Any

from persona_rag.graph.state import GraphState

_BOT: Any = None  # Bot instance injected at compile time


def attach_bot(bot: Any) -> None:
    global _BOT
    _BOT = bot


async def send_reply(state: GraphState) -> GraphState:
    if _BOT is not None and state.get("reply"):
        await _BOT.send_message(state["chat_id"], state["reply"])
    return state
