from __future__ import annotations

from datetime import UTC, datetime

import pytest

from persona_rag.models import PersonaTurn, RetrievedTurn
from persona_rag.retrieval.mmr import cosine, mmr_rerank


def _rt(_id: str, score: float, embedding: list[float] | None = None) -> RetrievedTurn:
    return RetrievedTurn(
        turn=PersonaTurn(
            id=_id,
            your_reply="x",
            incoming_context=["y"],
            channel="telegram",
            chat_id_hash="c1",
            recipient_id_hash="r1",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            language="uk",
            your_reply_len_chars=1,
            your_reply_emoji_count=0,
        ),
        score=score,
        embedding=embedding,
    )


def test_mmr_returns_all_when_pool_le_k():
    pool = [_rt("a", 0.9, [1.0, 0.0]), _rt("b", 0.8, [0.0, 1.0])]
    out = mmr_rerank(pool, k=5, lambda_param=0.6)
    assert [r.turn.id for r in out] == ["a", "b"]


def test_mmr_picks_top_relevance_first():
    pool = [
        _rt("a", 0.9, [1.0, 0.0]),
        _rt("b", 0.5, [0.0, 1.0]),
        _rt("c", 0.7, [0.5, 0.5]),
    ]
    out = mmr_rerank(pool, k=3, lambda_param=0.6)
    assert out[0].turn.id == "a"


def test_mmr_diversifies_when_top_candidates_are_duplicates():
    # Five near-duplicates with similar high scores + one outlier with lower score.
    # With low λ, MMR should prefer the outlier over yet another duplicate.
    pool = [
        _rt("dup1", 0.95, [1.0, 0.0]),
        _rt("dup2", 0.94, [1.0, 0.0]),
        _rt("dup3", 0.93, [1.0, 0.0]),
        _rt("dup4", 0.92, [1.0, 0.0]),
        _rt("dup5", 0.91, [1.0, 0.0]),
        _rt("outlier", 0.60, [0.0, 1.0]),
    ]
    out = mmr_rerank(pool, k=2, lambda_param=0.3)
    ids = [r.turn.id for r in out]
    assert ids[0] == "dup1"
    assert ids[1] == "outlier"


def test_mmr_reduces_to_top_k_when_lambda_is_one():
    pool = [
        _rt("a", 0.9, [1.0, 0.0]),
        _rt("b", 0.8, [1.0, 0.0]),
        _rt("c", 0.7, [0.0, 1.0]),
    ]
    out = mmr_rerank(pool, k=3, lambda_param=1.0)
    assert [r.turn.id for r in out] == ["a", "b", "c"]


def test_mmr_pure_diversity_when_lambda_zero():
    # After "a" is picked (highest relevance, always first), with λ=0 the next
    # pick must be the candidate maximally distant from "a". "c" (orthogonal)
    # beats "b" (identical to a) even though "b" has higher score.
    pool = [
        _rt("a", 0.9, [1.0, 0.0]),
        _rt("b", 0.8, [1.0, 0.0]),
        _rt("c", 0.5, [0.0, 1.0]),
    ]
    out = mmr_rerank(pool, k=2, lambda_param=0.0)
    assert [r.turn.id for r in out] == ["a", "c"]


def test_mmr_handles_empty_pool():
    out = mmr_rerank([], k=4, lambda_param=0.6)
    assert out == []


def test_mmr_handles_zero_diversity_pool():
    # All identical embeddings → diversity penalty equal for every candidate →
    # MMR falls back to relevance ordering.
    pool = [
        _rt("a", 0.9, [1.0, 0.0]),
        _rt("b", 0.7, [1.0, 0.0]),
        _rt("c", 0.5, [1.0, 0.0]),
    ]
    out = mmr_rerank(pool, k=3, lambda_param=0.6)
    assert [r.turn.id for r in out] == ["a", "b", "c"]


def test_mmr_output_is_subset_no_duplicates_size_k():
    pool = [_rt(f"i{i}", 0.9 - 0.05 * i, [float(i), 1.0 - float(i)]) for i in range(8)]
    out = mmr_rerank(pool, k=4, lambda_param=0.5)
    out_ids = [r.turn.id for r in out]
    pool_ids = {r.turn.id for r in pool}
    assert set(out_ids).issubset(pool_ids)
    assert len(out_ids) == len(set(out_ids))
    assert len(out) == 4


def test_mmr_first_pick_is_top_relevance_when_lambda_positive():
    pool = [
        _rt("low_score_diverse", 0.10, [0.0, 1.0]),
        _rt("top_score", 0.99, [1.0, 0.0]),
        _rt("mid_score", 0.50, [0.7, 0.7]),
    ]
    for lam in (0.01, 0.3, 0.6, 0.9):
        out = mmr_rerank(pool, k=3, lambda_param=lam)
        assert out[0].turn.id == "top_score", f"lambda={lam} broke first-pick invariant"


def test_mmr_includes_outlier_that_plain_top_k_misses():
    """The canonical 'MMR works for our use case' proof.

    Plain top-K (sorted by score) returns 4 near-duplicates. MMR with mild
    diversity bias swaps one of those duplicates for the lower-scoring outlier.
    """
    pool = [
        _rt("dup1", 0.95, [1.0, 0.0]),
        _rt("dup2", 0.94, [1.0, 0.0]),
        _rt("dup3", 0.93, [1.0, 0.0]),
        _rt("dup4", 0.92, [1.0, 0.0]),
        _rt("outlier", 0.60, [0.0, 1.0]),
    ]
    plain_top_k = sorted(pool, key=lambda r: r.score, reverse=True)[:4]
    plain_ids = {r.turn.id for r in plain_top_k}
    mmr_out = mmr_rerank(pool, k=4, lambda_param=0.5)
    mmr_ids = {r.turn.id for r in mmr_out}
    assert "outlier" not in plain_ids
    assert "outlier" in mmr_ids


def test_cosine_matches_numpy_reference():
    """Sanity-check the cosine helper against hand-computed values."""
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert cosine([1.0, 1.0], [1.0, 0.0]) == pytest.approx(1.0 / (2**0.5))
    # Zero-vector edge case: defined to return 0.0 (avoids divide-by-zero).
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
