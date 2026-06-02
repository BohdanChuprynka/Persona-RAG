"""`--local` run profile and startup preflight wiring for the Telegram bot."""

from __future__ import annotations

import persona_rag.bot.main as botmain
from persona_rag.config import get_settings


class TestApplyLocalOverrides:
    """`--local` is the opinionated 'serve the local LoRA with facts' profile."""

    def test_sets_ollama_backend(self):
        env: dict[str, str] = {}
        botmain.apply_local_overrides(env)
        assert env["GENERATION_BACKEND"] == "ollama"

    def test_folds_facts_into_system_by_default(self):
        env: dict[str, str] = {}
        botmain.apply_local_overrides(env)
        assert env["OLLAMA_FACTS_IN_SYSTEM"] == "true"

    def test_existing_facts_value_is_preserved(self):
        # an explicit export wins — the flag only supplies the default
        env = {"OLLAMA_FACTS_IN_SYSTEM": "false"}
        botmain.apply_local_overrides(env)
        assert env["OLLAMA_FACTS_IN_SYSTEM"] == "false"
        assert env["GENERATION_BACKEND"] == "ollama"


async def test_amain_runs_ollama_preflight_before_polling(monkeypatch):
    """On the ollama backend the bot must verify the local model is up BEFORE it
    opens a Telegram connection."""
    monkeypatch.setenv("GENERATION_BACKEND", "ollama")
    get_settings.cache_clear()
    order: list[str] = []

    async def fake_ready(settings=None, **kw):
        order.append("preflight")

    class FakeDispatcher:
        def include_router(self, router):
            pass

        async def start_polling(self, bot):
            order.append("poll")

    monkeypatch.setattr(botmain, "ensure_ollama_ready", fake_ready)
    monkeypatch.setattr(botmain, "Bot", lambda **kw: object())
    monkeypatch.setattr(botmain, "Dispatcher", lambda: FakeDispatcher())
    try:
        await botmain.amain()
    finally:
        get_settings.cache_clear()

    assert order == ["preflight", "poll"]
