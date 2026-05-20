from __future__ import annotations

from aiogram import Bot, F, Router
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from persona_rag.bot.auth import (
    approve_user,
    block_user,
    buffer_pending_message,
    set_user_state,
)
from persona_rag.config import get_settings
from persona_rag.models import UserState

router = Router(name="onboarding")


def _admin_kb(target_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Approve", callback_data=f"approve:{target_id}"),
                InlineKeyboardButton(text="🚫 Block", callback_data=f"block:{target_id}"),
            ]
        ]
    )


async def request_admin_approval(message: Message, bot: Bot) -> None:
    user = message.from_user
    if user is None:
        return
    set_user_state(user.id, UserState.PENDING)
    buffer_pending_message(user.id, message.text or "")
    await message.answer("Awaiting approval. You'll hear back soon.")
    await bot.send_message(
        chat_id=get_settings().ADMIN_TELEGRAM_ID,
        text=(
            f"🔐 New user request\n"
            f"User: @{user.username or '?'} (id={user.id})\n"
            f"Name: {user.full_name}\n"
            f"First msg:\n> {message.text}"
        ),
        reply_markup=_admin_kb(user.id),
    )


@router.callback_query(F.data.startswith("approve:"))
async def cb_approve(cb: CallbackQuery) -> None:
    if cb.from_user.id != get_settings().ADMIN_TELEGRAM_ID:
        await cb.answer("Not admin.")
        return
    target = int((cb.data or "").split(":")[1])
    approve_user(target, admin_id=get_settings().ADMIN_TELEGRAM_ID)
    if isinstance(cb.message, Message):
        await cb.message.edit_text(f"Approved {target}.")
    if cb.bot is not None:
        await cb.bot.send_message(target, "✅ Authorized. I'm online.")
    await cb.answer()


@router.callback_query(F.data.startswith("block:"))
async def cb_block(cb: CallbackQuery) -> None:
    if cb.from_user.id != get_settings().ADMIN_TELEGRAM_ID:
        await cb.answer("Not admin.")
        return
    target = int((cb.data or "").split(":")[1])
    block_user(target, admin_id=get_settings().ADMIN_TELEGRAM_ID)
    if isinstance(cb.message, Message):
        await cb.message.edit_text(f"Blocked {target}.")
    await cb.answer()
