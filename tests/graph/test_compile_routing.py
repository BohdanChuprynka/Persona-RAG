"""Graph entry routing.

The LoRA (``GENERATION_BACKEND=ollama``) serves the THIN prompt and never uses
retrieved few-shot turns, so the entry router skips ``retrieve_hybrid`` (and its
per-message OpenAI embedding call) and goes straight to the facts layer. The
OpenAI path keeps few-shot retrieval. Non-whitelisted users short-circuit to END.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END

from persona_rag.config import get_settings
from persona_rag.graph.compile import _route_after_auth
from persona_rag.models import UserState


def _whitelisted() -> dict[str, Any]:
    return {"auth_state": UserState.WHITELISTED.value}


def test_non_whitelisted_routes_to_end(monkeypatch):
    monkeypatch.delenv("GENERATION_BACKEND", raising=False)
    get_settings.cache_clear()
    try:
        assert _route_after_auth({"auth_state": "pending"}) == END
    finally:
        get_settings.cache_clear()


def test_openai_backend_routes_through_fewshot_retrieval(monkeypatch):
    monkeypatch.setenv("GENERATION_BACKEND", "openai")
    get_settings.cache_clear()
    try:
        assert _route_after_auth(_whitelisted()) == "retrieve_hybrid"
    finally:
        get_settings.cache_clear()


def test_ollama_backend_skips_fewshot_retrieval(monkeypatch):
    # The LoRA serves the thin prompt — retrieved few-shots are never injected,
    # so retrieve_hybrid is dead weight (and its embedding call hits OpenAI).
    # Route straight to the facts layer (load_memory -> retrieve_insights -> ...).
    monkeypatch.setenv("GENERATION_BACKEND", "ollama")
    get_settings.cache_clear()
    try:
        assert _route_after_auth(_whitelisted()) == "load_memory"
    finally:
        get_settings.cache_clear()
