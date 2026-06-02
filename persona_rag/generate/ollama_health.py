"""Startup preflight for the local Ollama backend.

When ``GENERATION_BACKEND=ollama`` the bot talks to a locally-served LoRA. If the
Ollama server is down or the model isn't installed, the failure otherwise only
surfaces mid-generation as an opaque 500 from inside the graph. This check runs
once at startup and fails fast with the exact fix command instead.

Deliberately a preflight, NOT a process manager: the bot does not spawn or own
``ollama serve`` — that keeps the dependency explicit and avoids a daemon the
bot would have to babysit (and leak on crash).
"""

from __future__ import annotations

import httpx

from persona_rag._logging import get_logger
from persona_rag.config import Settings, get_settings

log = get_logger()

_SERVE_HINT = "start it with:  ollama serve"


def _model_matches(wanted: str, available: list[str]) -> bool:
    """Ollama tags models as ``name:tag`` (e.g. ``bohdan:latest``) while
    ``OLLAMA_MODEL`` is usually the bare name. Match either form."""
    wanted = wanted.strip()
    return any(tag == wanted or tag.split(":", 1)[0] == wanted for tag in available)


def _missing_model_error(model: str, available: list[str]) -> str:
    have = ", ".join(sorted(available)) or "(none)"
    return (
        f"Ollama is running but model {model!r} is not installed (installed: {have}). "
        f"Build it from the Colab export:\n"
        f"    unzip bohdan-lora-gguf.zip -d bohdan-gguf && cd bohdan-gguf\n"
        f"    ollama create {model} -f Modelfile"
    )


async def _list_models(base_url: str, *, client: httpx.AsyncClient) -> list[str]:
    """Model ids from Ollama's OpenAI-compatible ``/models`` endpoint."""
    resp = await client.get(f"{base_url.rstrip('/')}/models")
    resp.raise_for_status()
    data = resp.json()
    return [m["id"] for m in data.get("data", []) if "id" in m]


async def ensure_ollama_ready(
    settings: Settings | None = None, *, client: httpx.AsyncClient | None = None
) -> None:
    """No-op unless ``GENERATION_BACKEND=ollama``. Otherwise raise ``RuntimeError``
    if the Ollama server is unreachable or ``OLLAMA_MODEL`` is not installed."""
    s = settings or get_settings()
    if s.GENERATION_BACKEND != "ollama":
        return

    owns_client = client is None
    c = client or httpx.AsyncClient(timeout=5.0)
    try:
        try:
            available = await _list_models(s.OLLAMA_BASE_URL, client=c)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            raise RuntimeError(
                f"Ollama server not reachable at {s.OLLAMA_BASE_URL} — {_SERVE_HINT}"
            ) from e
    finally:
        if owns_client:
            await c.aclose()

    if not _model_matches(s.OLLAMA_MODEL, available):
        raise RuntimeError(_missing_model_error(s.OLLAMA_MODEL, available))
    log.info("ollama_ready", model=s.OLLAMA_MODEL, base_url=s.OLLAMA_BASE_URL)
