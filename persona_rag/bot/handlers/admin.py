from __future__ import annotations

import html
from datetime import UTC, datetime

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlmodel import Session, select

from persona_rag.bot import debug_trace
from persona_rag.bot.auth import approve_user, block_user, get_pending
from persona_rag.config import get_settings
from persona_rag.db.engine import make_engine
from persona_rag.db.models import InsightRow, User
from persona_rag.index.qdrant_store import make_client as make_qdrant_client
from persona_rag.insights import verification
from persona_rag.models import UserState

router = Router(name="admin")


def _fmt_age(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    days = (datetime.now(UTC) - ts).days
    if days < 1:
        return "<1d"
    if days < 30:
        return f"{days}d"
    if days < 365:
        return f"{days // 30}mo"
    return f"{days // 365}y"


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


@router.message(Command("debug"))
async def handle_debug(message: Message) -> None:
    """Dump retrieval context, system prompt header, and reply from the most
    recent graph run. Admin only.

    Usage:
      /debug          → latest turn across all users
      /debug me       → latest turn from admin's own user_id
      /debug <uid>    → latest turn from that user_id
    """
    if message.from_user is None or message.from_user.id != get_settings().ADMIN_TELEGRAM_ID:
        return

    parts = (message.text or "").split()
    if len(parts) == 1:
        latest = debug_trace.get_latest_any()
        if latest is None:
            await message.answer("No traces yet. Send a message to the bot first.")
            return
        target_uid, trace = latest
    elif parts[1] == "me":
        target_uid = message.from_user.id
        maybe = debug_trace.get(target_uid)
        if maybe is None:
            await message.answer("No trace recorded for you yet.")
            return
        trace = maybe
    else:
        try:
            target_uid = int(parts[1])
        except ValueError:
            await message.answer("Usage: /debug | /debug me | /debug <uid>")
            return
        maybe = debug_trace.get(target_uid)
        if maybe is None:
            await message.answer(f"No trace for uid {target_uid}.")
            return
        trace = maybe

    retrieved = trace["retrieved"]
    lines: list[str] = []
    lines.append(f"<b>Trace</b> uid={target_uid}  ts={trace['ts'].strftime('%H:%M:%S')}")
    lines.append(f"<b>Incoming:</b> {html.escape(trace['incoming'][:200])}")
    lines.append(f"<b>Retrieved:</b> {len(retrieved)} turns")
    for i, r in enumerate(retrieved, 1):
        preview = html.escape(r.turn.your_reply.replace("\n", " ")[:120])
        lines.append(
            f"  {i}. score={r.score:.3f} "
            f"(dense={r.score_dense:.2f} bm25={r.score_bm25:.2f}) "
            f"lang={r.turn.language} age={_fmt_age(r.turn.timestamp)}\n"
            f"     reply: {preview}"
        )
    lines.append(f"<b>Reply:</b> {html.escape(trace['reply'][:300])}")

    out = "\n".join(lines)
    # Telegram message limit: 4096
    for i in range(0, len(out), 3800):
        await message.answer(out[i : i + 3800])

    # Send the system prompt as a second message so it's easy to skim.
    prompt = trace["prompt"]
    if prompt and prompt[0].get("role") == "system":
        sys_msg = html.escape(prompt[0]["content"][:3500])
        await message.answer(f"<b>System prompt:</b>\n<pre>{sys_msg}</pre>")


def _verify_keyboard(insight_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✓ Yes", callback_data=f"insights:yes:{insight_id}"),
                InlineKeyboardButton(text="✏️ Fix", callback_data=f"insights:fix:{insight_id}"),
                InlineKeyboardButton(text="✗ No", callback_data=f"insights:no:{insight_id}"),
            ],
            [
                InlineKeyboardButton(text="⏭ Skip", callback_data=f"insights:skip:{insight_id}"),
                InlineKeyboardButton(text="🛑 Stop", callback_data=f"insights:stop:{insight_id}"),
            ],
        ]
    )


def _render_insight_for_review(insight: InsightRow) -> str:
    return (
        f"<b>Insight</b>\n"
        f"<i>I think you {html.escape(insight.text)}</i>\n\n"
        f"category: {insight.category}  |  confidence: {insight.confidence:.2f}\n"
        f"seen in {insight.evidence_count} session(s),"
        f" last in {insight.latest_date:%Y-%m}\n"
    )


@router.message(Command("insights"))
async def handle_insights(message: Message) -> None:
    if message.from_user is None or message.from_user.id != get_settings().ADMIN_TELEGRAM_ID:
        return
    parts = (message.text or "").split()
    if len(parts) < 2 or parts[1] != "verify":
        await message.answer("Usage: /insights verify")
        return

    verification.start_session(message.from_user.id)
    nxt = verification.next_pending_insight(message.from_user.id)
    if nxt is None:
        await message.answer("No pending insights to verify.")
        return
    await message.answer(
        _render_insight_for_review(nxt),
        reply_markup=_verify_keyboard(nxt.id),
    )


@router.callback_query(F.data.startswith("insights:"))
async def handle_insights_callback(callback: CallbackQuery) -> None:
    if callback.from_user.id != get_settings().ADMIN_TELEGRAM_ID:
        return
    if callback.data is None:
        return
    _, action, insight_id = callback.data.split(":", 2)
    s = get_settings()
    client = make_qdrant_client()

    if action == "yes":
        await verification.accept_insight(
            insight_id, qdrant_client=client, collection=s.QDRANT_INSIGHTS_COLLECTION
        )
        await callback.answer("✓ verified")
    elif action == "no":
        verification.reject_insight(
            insight_id, qdrant_client=client, collection=s.QDRANT_INSIGHTS_COLLECTION
        )
        await callback.answer("✗ rejected")
    elif action == "skip":
        await callback.answer("⏭ skipped")
    elif action == "stop":
        verification.stop_session(callback.from_user.id)
        await callback.answer("🛑 stopped")
        if callback.message is not None:
            await callback.message.answer("Stopped. Resume any time with /insights verify.")
        return
    elif action == "fix":
        # Two-message flow: ask for new text. The next non-command message becomes the new text.
        # For the v1 cut, just acknowledge and tell user to use a more detailed flow later.
        await callback.answer(
            "Type the corrected version as a reply. (Manual flow for v1.)", show_alert=True
        )
        return

    nxt = verification.next_pending_insight(callback.from_user.id)
    if callback.message is None:
        return
    if nxt is None:
        await callback.message.answer("All done! Run /insights stats to see counts.")
        verification.stop_session(callback.from_user.id)
        return
    await callback.message.answer(
        _render_insight_for_review(nxt),
        reply_markup=_verify_keyboard(nxt.id),
    )
