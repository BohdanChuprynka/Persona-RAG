# ruff: noqa: RUF001
# Reason: tests use intentional Cyrillic strings to exercise UA splitter.
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from persona_rag.graph.nodes import send_reply as sr_mod
from persona_rag.graph.nodes.send_reply import _split_reply, attach_bot, send_reply
from persona_rag.graph.state import GraphState


def test_split_single_line():
    assert _split_reply("та норм") == ["та норм"]


def test_split_multiline_drops_empties():
    assert _split_reply("та норм\nсонний трохи\n\nале живий") == [
        "та норм",
        "сонний трохи",
        "але живий",
    ]


def test_split_empty():
    assert _split_reply("") == []


@pytest.mark.asyncio
async def test_send_reply_emits_one_message_per_line(monkeypatch):
    monkeypatch.setattr(sr_mod, "asyncio", _FastAsyncio())  # zero out sleep
    bot = type(
        "B",
        (),
        {
            "send_message": AsyncMock(),
            "send_chat_action": AsyncMock(),
        },
    )()
    attach_bot(bot)
    state: GraphState = {
        "user_id": 1,
        "chat_id": 99,
        "incoming": "?",
        "reply": "та норм\nсонний трохи\nале живий",
    }
    await send_reply(state)
    assert bot.send_message.await_count == 3
    sent = [call.args[1] for call in bot.send_message.await_args_list]
    assert sent == ["та норм", "сонний трохи", "але живий"]


class _FastAsyncio:
    """Stub asyncio module that skips sleep but still has create_task etc."""

    async def sleep(self, _seconds: float) -> None:
        return None
