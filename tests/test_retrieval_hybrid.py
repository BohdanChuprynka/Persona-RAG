from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from persona_rag.models import PersonaTurn, RetrievedTurn
from persona_rag.retrieval import retrieve
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


def _scored(_id: str, score: float) -> RetrievedTurn:
    """Build a RetrievedTurn with a fused/reranked score baked in."""
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
        score=score,
        score_dense=score,
        score_bm25=0.0,
    )


@pytest.mark.asyncio
async def test_retrieve_drops_below_hybrid_score_floor(monkeypatch):
    """Past turns below HYBRID_SCORE_FLOOR are dropped before the top-k slice.

    Regression: a 0.098-score 'office' past-turn fed vocabulary the model
    parroted out of context. Floor 0.15 drops it.
    """
    # Inject the floor via monkeypatched settings
    from persona_rag import retrieval as retr_mod

    real_get_settings = retr_mod.get_settings

    class FakeSettings:
        TOP_K = 4
        HYBRID_SCORE_FLOOR = 0.15
        HYBRID_DENSE_ALPHA = 0.7

    monkeypatch.setattr(retr_mod, "get_settings", lambda: FakeSettings())

    high = _scored("strong", 0.50)
    mid = _scored("ok", 0.20)
    weak = _scored("noise", 0.098)  # the 'office' turn score from the trace

    # Bypass real retrieval — return our seeded list as the reranked output
    monkeypatch.setattr(retr_mod, "retrieve_dense", AsyncMock(return_value=[]))
    monkeypatch.setattr(retr_mod, "retrieve_bm25", lambda *a, **kw: [])
    monkeypatch.setattr(retr_mod, "fuse_scores", lambda *a, **kw: [high, mid, weak])
    monkeypatch.setattr(retr_mod, "recency_decay", lambda items: items)

    fake_client = MagicMock()
    out = await retrieve("где я?", client=fake_client)

    ids = [r.turn.id for r in out]
    assert "strong" in ids
    assert "ok" in ids
    assert "noise" not in ids  # below floor — must be dropped

    # Restore for other tests in same module run
    monkeypatch.setattr(retr_mod, "get_settings", real_get_settings)


@pytest.mark.asyncio
async def test_retrieve_floor_zero_keeps_everything(monkeypatch):
    """HYBRID_SCORE_FLOOR=0 disables the filter (backward compat)."""
    from persona_rag import retrieval as retr_mod

    class FakeSettings:
        TOP_K = 4
        HYBRID_SCORE_FLOOR = 0.0
        HYBRID_DENSE_ALPHA = 0.7

    monkeypatch.setattr(retr_mod, "get_settings", lambda: FakeSettings())

    weak1 = _scored("a", 0.05)
    weak2 = _scored("b", 0.01)

    monkeypatch.setattr(retr_mod, "retrieve_dense", AsyncMock(return_value=[]))
    monkeypatch.setattr(retr_mod, "retrieve_bm25", lambda *a, **kw: [])
    monkeypatch.setattr(retr_mod, "fuse_scores", lambda *a, **kw: [weak1, weak2])
    monkeypatch.setattr(retr_mod, "recency_decay", lambda items: items)

    fake_client = MagicMock()
    out = await retrieve("q", client=fake_client)

    ids = {r.turn.id for r in out}
    assert ids == {"a", "b"}
