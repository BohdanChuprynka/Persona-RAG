from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message
from sqlmodel import Session, select

from persona_rag.bot.auth import approve_user, block_user, get_pending
from persona_rag.config import get_settings
from persona_rag.db.engine import make_engine
from persona_rag.db.models import User
from persona_rag.models import UserState

router = Router(name="admin")


@router.message(Command("users"))
async def handle_users(message: Message) -> None:
    if message.from_user is None or message.from_user.id != get_settings().ADMIN_TELEGRAM_ID:
        return
    with Session(make_engine()) as s:
        users = list(s.exec(select(User).where(User.state == UserState.WHITELISTED.value)).all())
    if not users:
        await message.answer("No whitelisted users.")
        return
    lines = [
        f"• @{u.username or '<no-username>'} ({u.telegram_id}) — last {u.last_interaction or '—'}"
        for u in users
    ]
    await message.answer("Whitelisted:\n" + "\n".join(lines))


@router.message(Command("pending"))
async def handle_pending(message: Message) -> None:
    if message.from_user is None or message.from_user.id != get_settings().ADMIN_TELEGRAM_ID:
        return
    pending = get_pending()
    if not pending:
        await message.answer("No pending requests.")
        return
    lines = [f"• @{u.username or '?'} ({u.telegram_id})" for u in pending]
    await message.answer("Pending:\n" + "\n".join(lines))


@router.message(Command("approve"))
async def handle_approve(message: Message) -> None:
    if message.from_user is None or message.from_user.id != get_settings().ADMIN_TELEGRAM_ID:
        return
    parts = (message.text or "").split()
    if len(parts) < 2:
        await message.answer("Usage: /approve <telegram_id>")
        return
    try:
        uid = int(parts[1])
    except ValueError:
        await message.answer("Invalid id.")
        return
    approve_user(uid, admin_id=get_settings().ADMIN_TELEGRAM_ID)
    await message.answer(f"Approved {uid}.")


@router.message(Command("block"))
async def handle_block(message: Message) -> None:
    if message.from_user is None or message.from_user.id != get_settings().ADMIN_TELEGRAM_ID:
        return
    parts = (message.text or "").split()
    if len(parts) < 2:
        await message.answer("Usage: /block <telegram_id>")
        return
    try:
        uid = int(parts[1])
    except ValueError:
        await message.answer("Invalid id.")
        return
    block_user(uid, admin_id=get_settings().ADMIN_TELEGRAM_ID)
    await message.answer(f"Blocked {uid}.")
