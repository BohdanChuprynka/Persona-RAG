from __future__ import annotations

from functools import lru_cache
from typing import Any

import tiktoken
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from persona_rag.config import get_settings


@lru_cache(maxsize=1)
def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=get_settings().OPENAI_API_KEY)


@lru_cache(maxsize=8)
def _paren_token_ids(model: str) -> tuple[int, ...]:
    """Single-token ids for the paren-smiley tic under ``model``'s encoding.
    Only single-token strings qualify (logit_bias keys are token ids)."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    ids: set[int] = set()
    for tok in (")", "))", " )", ")))", "))))"):
        e = enc.encode(tok)
        if len(e) == 1:
            ids.add(e[0])
    return tuple(sorted(ids))


def paren_logit_bias() -> dict[int, int] | None:
    """Map the ")"/"))" token ids to PAREN_LOGIT_BIAS, or None when disabled.
    Nudges Bohdan's paren-smiley tic that prompt rules don't reach."""
    s = get_settings()
    if not s.PAREN_LOGIT_BIAS:
        return None
    ids = _paren_token_ids(s.OPENAI_CHAT_MODEL)
    if not ids:
        return None
    return {i: int(s.PAREN_LOGIT_BIAS) for i in ids}


def _base_kwargs(
    messages: list[dict[str, str]],
    *,
    model: str | None,
    temperature: float | None,
    max_tokens: int | None,
    logit_bias: dict[int, int] | None,
) -> dict[str, Any]:
    s = get_settings()
    kwargs: dict[str, Any] = {
        "model": model or s.OPENAI_CHAT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else s.MAX_REPLY_TOKENS,
        "temperature": temperature if temperature is not None else s.TEMPERATURE,
    }
    if logit_bias:
        kwargs["logit_bias"] = logit_bias
    return kwargs


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def chat_complete(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    logit_bias: dict[int, int] | None = None,
) -> str:
    kwargs = _base_kwargs(
        messages, model=model, temperature=temperature, max_tokens=max_tokens, logit_bias=logit_bias
    )
    if response_format is not None:
        kwargs["response_format"] = response_format
    resp = await _client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def chat_complete_n(
    messages: list[dict[str, str]],
    *,
    n: int,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    logit_bias: dict[int, int] | None = None,
) -> list[str]:
    """Sample ``n`` candidates in a single request (best-of-N selection)."""
    kwargs = _base_kwargs(
        messages, model=model, temperature=temperature, max_tokens=max_tokens, logit_bias=logit_bias
    )
    kwargs["n"] = n
    resp = await _client().chat.completions.create(**kwargs)
    return [c.message.content or "" for c in resp.choices]
