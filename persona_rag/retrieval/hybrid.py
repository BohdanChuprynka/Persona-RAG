from __future__ import annotations

from persona_rag.config import get_settings
from persona_rag.models import RetrievedTurn


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-9:
        return dict.fromkeys(scores, 0.0)
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def fuse_scores(
    dense: list[RetrievedTurn],
    bm25: list[RetrievedTurn],
    *,
    alpha: float | None = None,
    top_k: int | None = None,
) -> list[RetrievedTurn]:
    s = get_settings()
    a = alpha if alpha is not None else s.HYBRID_DENSE_ALPHA
    k = top_k or s.TOP_K

    dense_map = {x.turn.id: x.score_dense for x in dense}
    bm25_map = {x.turn.id: x.score_bm25 for x in bm25}

    dense_norm = _minmax(dense_map)
    bm25_norm = _minmax(bm25_map)

    all_ids = set(dense_map) | set(bm25_map)
    turn_by_id = {x.turn.id: x.turn for x in (*dense, *bm25)}
    # Dense hits carry embeddings; BM25-only hits do not. Prefer the dense
    # entry's embedding when both exist.
    embedding_by_id: dict[str, list[float] | None] = {x.turn.id: x.embedding for x in dense}
    for x in bm25:
        embedding_by_id.setdefault(x.turn.id, x.embedding)

    fused: list[RetrievedTurn] = []
    for _id in all_ids:
        d = dense_norm.get(_id, 0.0)
        b = bm25_norm.get(_id, 0.0)
        s_score = a * d + (1 - a) * b
        fused.append(
            RetrievedTurn(
                turn=turn_by_id[_id],
                score=s_score,
                score_dense=dense_map.get(_id, 0.0),
                score_bm25=bm25_map.get(_id, 0.0),
                embedding=embedding_by_id.get(_id),
            )
        )
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused[:k]
