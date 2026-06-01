from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from persona_rag.generate.llm_client import chat_complete


def _fake_resp(content: str = "ok"):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.mark.asyncio
async def test_chat_complete_uses_defaults_when_no_overrides():
    fake = AsyncMock(return_value=_fake_resp("hi"))
    with patch("persona_rag.generate.llm_client._client") as mc:
        mc.return_value.chat.completions.create = fake
        out = await chat_complete([{"role": "user", "content": "yo"}])
    assert out == "hi"
    kwargs = fake.call_args.kwargs
    assert "temperature" in kwargs
    assert "max_tokens" in kwargs
    assert "response_format" not in kwargs  # default: omit when not set


@pytest.mark.asyncio
async def test_chat_complete_forwards_overrides():
    fake = AsyncMock(return_value=_fake_resp("{}"))
    with patch("persona_rag.generate.llm_client._client") as mc:
        mc.return_value.chat.completions.create = fake
        await chat_complete(
            [{"role": "user", "content": "json please"}],
            model="gpt-4o",
            temperature=0.2,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
    kwargs = fake.call_args.kwargs
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 2000
    assert kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_chat_complete_returns_empty_string_for_none_content():
    fake = AsyncMock(return_value=_fake_resp(None))
    with patch("persona_rag.generate.llm_client._client") as mc:
        mc.return_value.chat.completions.create = fake
        out = await chat_complete([{"role": "user", "content": "x"}])
    assert out == ""
