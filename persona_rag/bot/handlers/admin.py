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
        await callback.message.answer(
            "Phase 1 done.\n\n"
            "Phase 2 (optional): I'll ask you ~10 questions about things your "
            "chats don't reveal. Type /insights_onboarding to start, or /insights "
            "skip to finish here."
        )
        verification.stop_session(callback.from_user.id)
        return
    await callback.message.answer(
        _render_insight_for_review(nxt),
        reply_markup=_verify_keyboard(nxt.id),
    )


# Slash commands can't have a space in aiogram routing — use underscore.
@router.message(Command("insights_onboarding"))
async def handle_insights_onboarding(message: Message) -> None:
    if message.from_user is None or message.from_user.id != get_settings().ADMIN_TELEGRAM_ID:
        return
    # For v1: simplified linear flow — bot asks each question; user replies; bot
    # parses + stores. The user MUST reply to each prompt before the next is sent.
    from pathlib import Path as _Path

    from persona_rag.insights.onboarding import load_questions

    s = get_settings()
    qpath = s.INSIGHTS_ONBOARDING_PATH or (
        _Path(__file__).parent.parent.parent / "insights" / "onboarding_questions.yaml"
    )
    if not qpath.exists():
        await message.answer(f"Question file missing: {qpath}")
        return
    questions = load_questions(qpath)
    await message.answer(
        f"Will ask {len(questions)} questions. Reply to each one in plain text.\n\n"
        "Type /insights skip to stop early. First question: " + questions[0].question
    )


@router.message(Command("insights_stats"))
async def handle_insights_stats(message: Message) -> None:
    if message.from_user is None or message.from_user.id != get_settings().ADMIN_TELEGRAM_ID:
        return
    with Session(make_engine()) as s:
        rows = list(s.exec(select(InsightRow)).all())

    by_source: dict[str, int] = {}
    by_status: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for r in rows:
        by_source[r.source] = by_source.get(r.source, 0) + 1
        by_status[r.review_status] = by_status.get(r.review_status, 0) + 1
        by_category[r.category] = by_category.get(r.category, 0) + 1

    lines = [
        "<b>Insights stats</b>",
        f"total: {len(rows)}",
        "",
        "<b>by source:</b>  " + ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())),
        "<b>by status:</b>  " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())),
        "<b>by category:</b>  " + ", ".join(f"{k}={v}" for k, v in sorted(by_category.items())),
    ]
    await message.answer("\n".join(lines))


@router.message(Command("insights_search"))
async def handle_insights_search(message: Message) -> None:
    if message.from_user is None or message.from_user.id != get_settings().ADMIN_TELEGRAM_ID:
        return
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Usage: /insights_search <query>")
        return
    query = parts[1].strip()

    from persona_rag.index.embedder import embed_batch

    s = get_settings()
    vec = (await embed_batch([query]))[0]
    client = make_qdrant_client()
    resp = client.query_points(
        collection_name=s.QDRANT_INSIGHTS_COLLECTION,
        query=vec,
        limit=5,
        with_payload=True,
    )
    if not resp.points:
        await message.answer("No matches.")
        return
    lines = [f"<b>Search:</b> {html.escape(query)}", ""]
    for p in resp.points:
        payload = p.payload or {}
        lines.append(
            f"• [{payload.get('source', '?')}] {payload.get('text', '')[:120]} "
            f"(conf={payload.get('confidence', 0):.2f}, score={float(p.score):.2f})"
        )
    await message.answer("\n".join(lines))


@router.message(Command("insights_delete"))
async def handle_insights_delete(message: Message) -> None:
    if message.from_user is None or message.from_user.id != get_settings().ADMIN_TELEGRAM_ID:
        return
    parts = (message.text or "").split()
    if len(parts) < 2:
        await message.answer("Usage: /insights_delete <insight_id>")
        return
    iid = parts[1].strip()
    s = get_settings()
    client = make_qdrant_client()
    verification.reject_insight(iid, qdrant_client=client, collection=s.QDRANT_INSIGHTS_COLLECTION)
    await message.answer(f"Marked rejected: {iid}")


@router.message(Command("insights_show"))
async def handle_insights_show(message: Message) -> None:
    if message.from_user is None or message.from_user.id != get_settings().ADMIN_TELEGRAM_ID:
        return
    parts = (message.text or "").split()
    if len(parts) < 2:
        await message.answer("Usage: /insights_show <insight_id>")
        return
    iid = parts[1].strip()
    with Session(make_engine()) as s:
        row = s.get(InsightRow, iid)
    if row is None:
        await message.answer(f"No insight with id {iid}")
        return
    lines = [
        f"<b>Insight {iid}</b>",
        f"text: {html.escape(row.text)}",
        f"category: {row.category}, subject: {row.subject}",
        f"source: {row.source}, status: {row.review_status}",
        f"confidence: {row.confidence:.2f}, evidence: {row.evidence_count}",
        f"earliest: {row.earliest_date:%Y-%m-%d}, latest: {row.latest_date:%Y-%m-%d}",
    ]
    if row.trajectory:
        lines.append(f"trajectory: {row.trajectory}")
    if row.edited_text:
        lines.append(f"\n<i>pre-edit text:</i> {html.escape(row.edited_text)}")
    await message.answer("\n".join(lines))
