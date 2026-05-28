from datetime import UTC, datetime

from sqlmodel import Session, SQLModel, create_engine, select

from persona_rag.db.models import (  # noqa: F401
    ContactMemory,
    Conversation,
    Message,
    PersonaTurnRow,
    RawInsightRow,
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


def test_raw_insight_row_roundtrip():
    e = _engine()
    now = datetime.now(UTC)
    with Session(e) as s:
        s.add(
            RawInsightRow(
                session_id="sess-1",
                category="interest",
                subject="cyberpunk 2077",
                text="Plays Cyberpunk 2077",
                confidence=0.9,
                source_quote="грав вчора cyberpunk",
                extracted_at=now,
            )
        )
        s.commit()
        row = s.exec(select(RawInsightRow)).one()
        assert row.session_id == "sess-1"
        assert row.subject == "cyberpunk 2077"
        assert row.confidence == 0.9
        assert isinstance(row.id, str) and len(row.id) > 0  # auto-generated UUID hex


def test_raw_insight_row_indexed_by_session_id():
    e = _engine()
    now = datetime.now(UTC)
    with Session(e) as s:
        for sid in ("a", "a", "b"):
            s.add(
                RawInsightRow(
                    session_id=sid,
                    category="bio",
                    subject="x",
                    text="t",
                    confidence=0.5,
                    source_quote="q",
                    extracted_at=now,
                )
            )
        s.commit()
        a_rows = list(s.exec(select(RawInsightRow).where(RawInsightRow.session_id == "a")).all())
        assert len(a_rows) == 2
        # IDs must differ — each row gets its own UUID
        assert a_rows[0].id != a_rows[1].id


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
