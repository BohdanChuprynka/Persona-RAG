from __future__ import annotations

import asyncio

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from persona_rag._logging import configure_logging, get_logger
from persona_rag.config import get_settings

log = get_logger()


async def amain() -> None:
    configure_logging()
    s = get_settings()
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
