from datetime import UTC, datetime, timedelta

from persona_rag.ingest.turns import extract_persona_turns
from persona_rag.models import RawMessage


def _m(s, t, txt="x"):
    return RawMessage(
        channel="telegram",
        chat_id="c1",
        sender_id=s,
        sender_name=s,
        text=txt,
        timestamp=t,
        is_group=False,
    )


def test_extract_yields_turn_per_persona_reply():
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    session = [
        _m("friend", t0, "how are you?"),
        _m("PERSONA", t0 + timedelta(minutes=1), "good thx"),
        _m("friend", t0 + timedelta(minutes=2), "what r u up to"),
        _m("PERSONA", t0 + timedelta(minutes=3), "coding"),
    ]
    turns = list(extract_persona_turns(session, persona_sender_id="PERSONA", context_turns=10))
    assert len(turns) == 2
    assert turns[0].your_reply == "good thx"
    assert turns[0].incoming_context == ["how are you?"]
    assert turns[1].your_reply == "coding"
    assert turns[1].incoming_context == ["how are you?", "good thx", "what r u up to"]
