from datetime import UTC, datetime

from sqlmodel import Session, SQLModel, create_engine, select

from persona_rag.db.models import (  # noqa: F401
    ContactMemory,
    Conversation,
    Message,
    PersonaTurnRow,
    User,
)
from persona_rag.models import UserState


def _engine():
    e = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(e)
    return e


def test_user_crud():
    e = _engine()
    with Session(e) as s:
        s.add(
            User(
                telegram_id=42,
                state=UserState.WHITELISTED.value,
                first_seen=datetime.now(UTC),
            )
        )
        s.commit()
        u = s.exec(select(User).where(User.telegram_id == 42)).one()
        assert u.state == "whitelisted"


def test_persona_turn_row_roundtrip():
    e = _engine()
    with Session(e) as s:
        s.add(
            PersonaTurnRow(
                id="abc",
                your_reply="hi",
                incoming_context_json='["q"]',
                channel="telegram",
                chat_id_hash="x",
                recipient_id_hash="y",
                timestamp=datetime.now(UTC),
                language="en",
                your_reply_len_chars=2,
                your_reply_emoji_count=0,
                eval_split=False,
            )
        )
        s.commit()
        assert s.exec(select(PersonaTurnRow)).one().your_reply == "hi"
