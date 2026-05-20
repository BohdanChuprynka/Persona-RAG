from datetime import UTC, datetime, timedelta

from persona_rag.models import PersonaTurn, RetrievedTurn
from persona_rag.retrieval.rerank import recency_decay


def _r(score: float, days_old: int) -> RetrievedTurn:
    ts = datetime.now(UTC) - timedelta(days=days_old)
    return RetrievedTurn(
        turn=PersonaTurn(
            id=str(days_old),
            your_reply="x",
            incoming_context=[],
            channel="telegram",
            chat_id_hash="a",
            recipient_id_hash="b",
            timestamp=ts,
            language="en",
            your_reply_len_chars=1,
            your_reply_emoji_count=0,
        ),
        score=score,
        score_dense=score,
    )


def test_recent_beats_old_at_same_base():
    items = [_r(1.0, 365), _r(1.0, 7)]
    out = recency_decay(items, half_life_days=180)
    assert out[0].turn.id == "7"
    assert out[0].score > out[1].score
