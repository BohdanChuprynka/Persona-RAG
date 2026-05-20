from datetime import UTC, datetime

from persona_rag.models import PersonaTurn, RetrievedTurn
from persona_rag.retrieval.hybrid import fuse_scores


def _r(_id: str, dense: float, bm25: float) -> RetrievedTurn:
    return RetrievedTurn(
        turn=PersonaTurn(
            id=_id,
            your_reply=_id,
            incoming_context=[],
            channel="telegram",
            chat_id_hash="x",
            recipient_id_hash="y",
            timestamp=datetime.now(UTC),
            language="en",
            your_reply_len_chars=1,
            your_reply_emoji_count=0,
        ),
        score=0.0,
        score_dense=dense,
        score_bm25=bm25,
    )


def test_fuse_alpha_one_is_dense_only():
    dense = [_r("a", 0.9, 0), _r("b", 0.5, 0)]
    bm25 = [_r("b", 0, 10), _r("a", 0, 0)]
    out = fuse_scores(dense, bm25, alpha=1.0, top_k=2)
    assert out[0].turn.id == "a"


def test_fuse_blends_when_alpha_half():
    dense = [_r("a", 1.0, 0), _r("b", 0.0, 0)]
    bm25 = [_r("b", 0, 1.0), _r("a", 0, 0.0)]
    out = fuse_scores(dense, bm25, alpha=0.5, top_k=2)
    ids = {x.turn.id for x in out}
    assert ids == {"a", "b"}
