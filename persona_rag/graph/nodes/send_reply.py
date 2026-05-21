from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from persona_rag.config import get_settings
from persona_rag.graph.state import GraphState

_BOT: Any = None  # Bot instance injected at compile time


def attach_bot(bot: Any) -> None:
    global _BOT
    _BOT = bot


def _split_reply(reply: str) -> list[str]:
    """Split a reply on newlines so each fragment becomes a separate Telegram
    message — mirrors how Bohdan actually chats. Empty fragments dropped.
    """
    if not get_settings().REPLY_SPLIT_NEWLINES:
        return [reply] if reply else []
    chunks = [c.strip() for c in reply.split("\n")]
    return [c for c in chunks if c]


def _typing_delay_ms(chunk: str) -> int:
    s = get_settings()
    raw = s.REPLY_CHUNK_DELAY_BASE_MS + len(chunk) * s.REPLY_CHUNK_DELAY_PER_CHAR_MS
    return min(raw, s.REPLY_CHUNK_DELAY_MAX_MS)


async def send_reply(state: GraphState) -> GraphState:
    if _BOT is None or not state.get("reply"):
        return state

    chunks = _split_reply(state["reply"])
    if not chunks:
        return state

    chat_id = state["chat_id"]
    for i, chunk in enumerate(chunks):
        if i > 0:
            # show typing between messages so it feels like a real reply
            with contextlib.suppress(Exception):
                await _BOT.send_chat_action(chat_id, "typing")
            await asyncio.sleep(_typing_delay_ms(chunk) / 1000)
        await _BOT.send_message(chat_id, chunk)
    return state
