from __future__ import annotations

from functools import lru_cache
from typing import Any

import tiktoken
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from persona_rag.config import get_settings


@lru_cache(maxsize=2)
def _openai_client(api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key)


@lru_cache(maxsize=2)
def _ollama_client(base_url: str) -> AsyncOpenAI:
    # Ollama exposes an OpenAI-compatible API; the key is ignored but required.
    return AsyncOpenAI(base_url=base_url, api_key="ollama")


def _client() -> AsyncOpenAI:
    """Active chat client. Ollama (local fine-tuned LoRA) when GENERATION_BACKEND
    is 'ollama', else OpenAI."""
    s = get_settings()
    if s.GENERATION_BACKEND == "ollama":
        return _ollama_client(s.OLLAMA_BASE_URL)
    return _openai_client(s.OPENAI_API_KEY)


def active_model() -> str:
    """Model name for the active backend."""
    s = get_settings()
    return s.OLLAMA_MODEL if s.GENERATION_BACKEND == "ollama" else s.OPENAI_CHAT_MODEL


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


@lru_cache(maxsize=8)
def _exclaim_token_ids(model: str) -> tuple[int, ...]:
    """Single-token ids for "!" runs under ``model``'s encoding."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    ids: set[int] = set()
    for tok in ("!", "!!", "!!!", " !"):
        e = enc.encode(tok)
        if len(e) == 1:
            ids.add(e[0])
    return tuple(sorted(ids))


def exclaim_logit_bias() -> dict[int, int] | None:
    """Map the "!" token ids to EXCLAIM_LOGIT_BIAS (negative), or None when
    disabled. Suppresses the model's exclamation habit — Bohdan never uses "!"."""
    s = get_settings()
    if not s.EXCLAIM_LOGIT_BIAS:
        return None
    ids = _exclaim_token_ids(s.OPENAI_CHAT_MODEL)
    if not ids:
        return None
    return {i: int(s.EXCLAIM_LOGIT_BIAS) for i in ids}


def voice_logit_bias() -> dict[int, int] | None:
    """Merged decoding nudges applied at generation: paren tic up, exclamation
    habit down. None when both are off. Paren/exclaim token ids are disjoint."""
    merged: dict[int, int] = {}
    for part in (paren_logit_bias(), exclaim_logit_bias()):
        if part:
            merged.update(part)
    return merged or None


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
        "model": model or active_model(),
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
