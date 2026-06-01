"""Generation-backend routing: openai (default) vs a local Ollama LoRA."""

from __future__ import annotations

from persona_rag.config import get_settings
from persona_rag.generate.llm_client import _client, active_model


def test_default_backend_uses_openai_model(monkeypatch):
    monkeypatch.delenv("GENERATION_BACKEND", raising=False)
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    get_settings.cache_clear()
    try:
        assert active_model() == "gpt-4o-mini"
    finally:
        get_settings.cache_clear()


def test_ollama_backend_uses_ollama_model_and_base_url(monkeypatch):
    monkeypatch.setenv("GENERATION_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "bohdan")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    get_settings.cache_clear()
    try:
        assert active_model() == "bohdan"
        assert "11434" in str(_client().base_url)
    finally:
        get_settings.cache_clear()
