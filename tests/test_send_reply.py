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


def test_split_literal_backslash_n_is_normalized():
    """Regression: the prompt used to ask for "\\n" so the model output it
    as the two-char escape sequence instead of a real newline. Splitter
    must normalise to actual newlines before splitting."""
    raw = "хз де я\\nдесь валяюсь як завжди"
    assert _split_reply(raw) == ["хз де я", "десь валяюсь як завжди"]


def test_split_crlf_normalized():
    assert _split_reply("a\r\nb") == ["a", "b"]


def test_typing_delay_jitter_within_bounds(monkeypatch):
    """JITTER_PCT=0.5 means delay lands within +/-50% of the deterministic
    base for any random.uniform draw."""
    from persona_rag.graph.nodes.send_reply import _typing_delay_ms

    samples = [_typing_delay_ms("a" * 10) for _ in range(50)]
    # Base = 300 + 10*20 = 500ms, capped at MAX=1800ms (so 500). Jitter 0.5
    # means range [250, 750].
    assert all(250 <= s <= 750 for s in samples)
    # And there is *some* spread (essentially zero chance all 50 collide).
    assert len(set(samples)) > 1


def test_typing_delay_jitter_zero_is_deterministic(monkeypatch):
    """JITTER_PCT=0.0 reproduces the old deterministic behaviour."""
    from persona_rag.graph.nodes.send_reply import _typing_delay_ms

    class FakeSettings:
        REPLY_CHUNK_DELAY_BASE_MS = 300
        REPLY_CHUNK_DELAY_PER_CHAR_MS = 20
        REPLY_CHUNK_DELAY_MAX_MS = 1800
        REPLY_CHUNK_DELAY_JITTER_PCT = 0.0

    monkeypatch.setattr(
        "persona_rag.graph.nodes.send_reply.get_settings",
        lambda: FakeSettings(),
    )
    # 300 + 10*20 = 500, no jitter, no cap hit -> exactly 500
    assert _typing_delay_ms("a" * 10) == 500


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
