from __future__ import annotations

from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from persona_rag.config import get_settings


@lru_cache(maxsize=1)
def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=get_settings().OPENAI_API_KEY)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def chat_complete(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
) -> str:
    s = get_settings()
    kwargs: dict[str, Any] = {
        "model": model or s.OPENAI_CHAT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else s.MAX_REPLY_TOKENS,
        "temperature": temperature if temperature is not None else s.TEMPERATURE,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    resp = await _client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""
