from __future__ import annotations

import asyncio
import contextlib
import random
from typing import Any

from persona_rag.config import get_settings
from persona_rag.generate.bubbles import split_bubbles
from persona_rag.graph.state import GraphState

_BOT: Any = None  # Bot instance injected at compile time


def attach_bot(bot: Any) -> None:
    global _BOT
    _BOT = bot


def _split_reply(reply: str) -> list[str]:
    """Split a reply into Telegram bubbles via the canonical ``split_bubbles``
    (the same primitive used by measurement and the shape-hint), so delivery,
    eval, and generation all agree on what a "bubble" is. Honours the
    REPLY_SPLIT_NEWLINES kill-switch (everything in one bubble when off)."""
    if not get_settings().REPLY_SPLIT_NEWLINES:
        return [reply] if reply else []
    return split_bubbles(reply)


def _typing_delay_ms(chunk: str) -> int:
    """Per-chunk delay with random jitter so consecutive messages don't pulse
    at machine-precise intervals."""
    s = get_settings()
    raw = s.REPLY_CHUNK_DELAY_BASE_MS + len(chunk) * s.REPLY_CHUNK_DELAY_PER_CHAR_MS
    capped = min(raw, s.REPLY_CHUNK_DELAY_MAX_MS)
    jitter = max(0.0, min(s.REPLY_CHUNK_DELAY_JITTER_PCT, 0.95))
    if jitter > 0:
        capped = int(capped * random.uniform(1 - jitter, 1 + jitter))
    return max(0, capped)


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
