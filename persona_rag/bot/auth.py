from __future__ import annotations

from datetime import UTC, datetime

from sqlmodel import Session, select

from persona_rag.db.engine import make_engine
from persona_rag.db.models import AuditLog, PendingMessage, User
from persona_rag.models import UserState


def ensure_user(telegram_id: int, username: str | None, first_name: str | None) -> UserState:
    with Session(make_engine()) as s:
        u = s.get(User, telegram_id)
        if u is None:
            s.add(
                User(
                    telegram_id=telegram_id,
                    username=username,
                    first_name=first_name,
                    state=UserState.UNKNOWN.value,
                    first_seen=datetime.now(UTC),
                )
            )
            s.commit()
            return UserState.UNKNOWN
        return UserState(u.state)


def get_user_state(telegram_id: int) -> UserState:
    with Session(make_engine()) as s:
        u = s.get(User, telegram_id)
        return UserState(u.state) if u else UserState.UNKNOWN


def set_user_state(telegram_id: int, state: UserState) -> None:
    with Session(make_engine()) as s:
        u = s.get(User, telegram_id)
        if u is not None:
            u.state = state.value
            s.add(u)
            s.commit()


def approve_user(telegram_id: int, *, admin_id: int) -> None:
    now = datetime.now(UTC)
    with Session(make_engine()) as s:
        u = s.get(User, telegram_id)
        if u is not None:
            u.state = UserState.WHITELISTED.value
            u.approved_by = admin_id
            u.approved_at = now
            s.add(u)
            s.add(
                AuditLog(timestamp=now, actor_id=admin_id, action="approve", target_id=telegram_id)
            )
            s.commit()


def block_user(telegram_id: int, *, admin_id: int) -> None:
    set_user_state(telegram_id, UserState.BLOCKED)
    with Session(make_engine()) as s:
        s.add(
            AuditLog(
                timestamp=datetime.now(UTC),
                actor_id=admin_id,
                action="block",
                target_id=telegram_id,
            )
        )
        s.commit()


def get_pending() -> list[User]:
    with Session(make_engine()) as s:
        return list(s.exec(select(User).where(User.state == UserState.PENDING.value)).all())


def buffer_pending_message(user_id: int, text: str) -> None:
    with Session(make_engine()) as s:
        s.add(PendingMessage(user_id=user_id, text=text, timestamp=datetime.now(UTC)))
        s.commit()
