from __future__ import annotations

from functools import lru_cache

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from persona_rag.config import get_settings


@lru_cache(maxsize=1)
def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=get_settings().OPENAI_API_KEY)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def chat_complete(messages: list[dict[str, str]], *, model: str | None = None) -> str:
    s = get_settings()
    resp = await _client().chat.completions.create(
        model=model or s.OPENAI_CHAT_MODEL,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=s.MAX_REPLY_TOKENS,
        temperature=s.TEMPERATURE,
    )
    return resp.choices[0].message.content or ""
