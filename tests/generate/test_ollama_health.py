"""Preflight check for the local Ollama backend.

Fail fast at startup if the server is down or the model isn't installed, with
the exact fix command — instead of a confusing 500 from inside the graph.
"""

from __future__ import annotations

import httpx
import pytest

from persona_rag.config import get_settings
from persona_rag.generate.ollama_health import (
    _missing_model_error,
    _model_matches,
    ensure_ollama_ready,
)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestModelMatches:
    def test_exact_name(self):
        assert _model_matches("bohdan", ["bohdan"]) is True

    def test_name_tag_form(self):
        # Ollama tags models as name:tag — OLLAMA_MODEL is the bare name.
        assert _model_matches("bohdan", ["bohdan:latest"]) is True

    def test_absent(self):
        assert _model_matches("bohdan", ["llama3:latest", "qwen2.5:3b"]) is False

    def test_empty_list(self):
        assert _model_matches("bohdan", []) is False


def test_missing_model_error_names_the_create_command():
    msg = _missing_model_error("bohdan", ["llama3:latest"])
    assert "ollama create bohdan" in msg
    assert "bohdan" in msg


async def test_noop_for_openai_backend(monkeypatch):
    # Non-ollama backend must not touch the network at all.
    monkeypatch.setenv("GENERATION_BACKEND", "openai")
    get_settings.cache_clear()
    try:
        await ensure_ollama_ready()  # no client passed; must return before any GET
    finally:
        get_settings.cache_clear()


async def test_passes_when_model_installed(monkeypatch):
    monkeypatch.setenv("GENERATION_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "bohdan")
    get_settings.cache_clear()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "bohdan:latest"}]})

    client = _client(handler)
    try:
        await ensure_ollama_ready(client=client)  # no raise
    finally:
        await client.aclose()
        get_settings.cache_clear()


async def test_raises_when_model_missing(monkeypatch):
    monkeypatch.setenv("GENERATION_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "bohdan")
    get_settings.cache_clear()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "llama3:latest"}]})

    client = _client(handler)
    try:
        with pytest.raises(RuntimeError, match="ollama create bohdan"):
            await ensure_ollama_ready(client=client)
    finally:
        await client.aclose()
        get_settings.cache_clear()


async def test_raises_when_server_unreachable(monkeypatch):
    monkeypatch.setenv("GENERATION_BACKEND", "ollama")
    get_settings.cache_clear()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(handler)
    try:
        with pytest.raises(RuntimeError, match="ollama serve"):
            await ensure_ollama_ready(client=client)
    finally:
        await client.aclose()
        get_settings.cache_clear()
