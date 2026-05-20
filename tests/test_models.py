from datetime import UTC, datetime

from persona_rag.models import PersonaTurn, RawMessage, UserState


def test_persona_turn_roundtrip():
    t = PersonaTurn(
        id="abc-123",
        your_reply="hi there",
        incoming_context=["how are you?"],
        channel="telegram",
        chat_id_hash="x" * 16,
        recipient_id_hash="y" * 16,
        timestamp=datetime.now(UTC),
        language="en",
        your_reply_len_chars=8,
        your_reply_emoji_count=0,
        eval_split=False,
    )
    d = t.model_dump()
    assert PersonaTurn.model_validate(d) == t


def test_raw_message_required_fields():
    m = RawMessage(
        channel="instagram",
        chat_id="c1",
        sender_id="s1",
        sender_name="Alice",
        text="hey",
        timestamp=datetime.now(UTC),
        is_group=False,
    )
    assert m.channel == "instagram"


def test_user_state_enum():
    assert UserState.WHITELISTED.value == "whitelisted"
