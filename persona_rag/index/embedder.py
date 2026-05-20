from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from persona_rag.config import get_settings


@lru_cache(maxsize=1)
def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=get_settings().OPENAI_API_KEY)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
async def embed_batch(texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    resp = await _client().embeddings.create(
        model=model or get_settings().OPENAI_EMBEDDING_MODEL,
        input=list(texts),
    )
    return [item.embedding for item in resp.data]
