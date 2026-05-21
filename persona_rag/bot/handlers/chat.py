from __future__ import annotations

from typing import Any

from aiogram import F, Router
from aiogram.types import Message

from persona_rag._logging import get_logger
from persona_rag.bot import debug_trace
from persona_rag.bot.auth import ensure_user
from persona_rag.bot.handlers.onboarding import request_admin_approval
from persona_rag.bot.rate_limit import TokenBucket
from persona_rag.config import get_settings
from persona_rag.graph.compile import build_graph
from persona_rag.graph.nodes.send_reply import attach_bot
from persona_rag.models import UserState

router = Router(name="chat")
log = get_logger()
_graph: Any = None
_bucket = TokenBucket(rate_per_minute=get_settings().MAX_MESSAGES_PER_MINUTE)


def _get_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


@router.message(F.text & ~F.text.startswith("/"))
async def on_message(message: Message) -> None:
    user = message.from_user
    if user is None or message.bot is None:
        return
    state = ensure_user(user.id, user.username, user.first_name)
    if state == UserState.UNKNOWN:
        await request_admin_approval(message, message.bot)
        return
    if state == UserState.PENDING:
        await message.answer("Still awaiting approval.")
        return
    if state == UserState.BLOCKED:
        return
    if not _bucket.allow(user.id):
        await message.answer("Slow down a sec.")
        return

    attach_bot(message.bot)
    graph = _get_graph()
    incoming = message.text or ""
    final = await graph.ainvoke(
        {
            "user_id": user.id,
            "chat_id": message.chat.id,
            "incoming": incoming,
        }
    )
    debug_trace.record(
        user.id,
        incoming=incoming,
        retrieved=final.get("retrieved", []),
        prompt=final.get("prompt", []),
        reply=final.get("reply", ""),
    )
    log.info("message_processed", user_id=user.id, reply_len=len(final.get("reply", "")))
