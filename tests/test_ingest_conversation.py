from datetime import UTC, datetime, timedelta

from persona_rag.ingest.conversation import collapse_bursts, split_sessions
from persona_rag.models import RawMessage


def _msg(sender: str, t: datetime, text: str = "x") -> RawMessage:
    return RawMessage(
        channel="telegram",
        chat_id="c",
        sender_id=sender,
        sender_name=sender,
        text=text,
        timestamp=t,
        is_group=False,
    )


def test_collapse_same_sender_within_burst():
    t0 = datetime(2025, 1, 1, 10, 0, tzinfo=UTC)
    msgs = [
        _msg("A", t0, "hi"),
        _msg("A", t0 + timedelta(seconds=30), "there"),
        _msg("B", t0 + timedelta(seconds=60), "hey"),
    ]
    out = collapse_bursts(msgs, burst_seconds=300)
    assert len(out) == 2
    assert out[0].text == "hi\nthere"


def test_split_sessions_by_gap():
    t0 = datetime(2025, 1, 1, 10, 0, tzinfo=UTC)
    msgs = [
        _msg("A", t0),
        _msg("B", t0 + timedelta(minutes=1)),
        _msg("A", t0 + timedelta(hours=10)),  # > 6h gap
        _msg("B", t0 + timedelta(hours=10, minutes=1)),
    ]
    sessions = list(split_sessions(msgs, gap_hours=6))
    assert len(sessions) == 2
