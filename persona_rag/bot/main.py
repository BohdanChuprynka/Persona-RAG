from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import MutableMapping

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from persona_rag._logging import configure_logging, get_logger
from persona_rag.config import get_settings
from persona_rag.generate.ollama_health import ensure_ollama_ready

log = get_logger()


def apply_local_overrides(env: MutableMapping[str, str]) -> None:
    """Apply the ``--local`` profile: serve the local fine-tuned LoRA via Ollama
    with contact facts folded into the system turn. Forces the backend; supplies
    facts-on only as a default so an explicit export still wins."""
    env["GENERATION_BACKEND"] = "ollama"
    env.setdefault("OLLAMA_FACTS_IN_SYSTEM", "true")


async def amain() -> None:
    configure_logging()
    s = get_settings()
    # Fail fast if the local model isn't up (no-op on the OpenAI backend), BEFORE
    # opening a Telegram connection.
    await ensure_ollama_ready(s)
    if s.LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = str(s.LANGCHAIN_TRACING_V2).lower()
        os.environ["LANGCHAIN_API_KEY"] = s.LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = s.LANGCHAIN_PROJECT
        log.info("langsmith_enabled", project=s.LANGCHAIN_PROJECT)
    bot = Bot(
        token=s.TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()

    from persona_rag.bot.handlers import admin, chat, onboarding

    dp.include_router(admin.router)
    dp.include_router(onboarding.router)
    dp.include_router(chat.router)

    log.info("bot_starting", admin_id=s.ADMIN_TELEGRAM_ID, persona=s.PERSONA_NAME)
    await dp.start_polling(bot)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="persona-rag-bot", description="Run the persona Telegram bot."
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="serve the local fine-tuned LoRA via Ollama (GENERATION_BACKEND=ollama, "
        "facts folded into the system turn) instead of the OpenAI API",
    )
    args = parser.parse_args()
    if args.local:
        apply_local_overrides(os.environ)
        get_settings.cache_clear()
    asyncio.run(amain())


if __name__ == "__main__":
    main()
