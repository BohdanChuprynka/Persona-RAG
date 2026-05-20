from __future__ import annotations

import asyncio
import os

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from persona_rag._logging import configure_logging, get_logger
from persona_rag.config import get_settings

log = get_logger()


async def amain() -> None:
    configure_logging()
    s = get_settings()
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
    asyncio.run(amain())


if __name__ == "__main__":
    main()
