from __future__ import annotations

from qdrant_client import QdrantClient

from persona_rag.config import get_settings
from persona_rag.models import RetrievedTurn
from persona_rag.retrieval.bm25 import retrieve_bm25
from persona_rag.retrieval.dense import retrieve_dense
from persona_rag.retrieval.hybrid import fuse_scores
from persona_rag.retrieval.mmr import mmr_rerank
from persona_rag.retrieval.rerank import recency_decay


async def retrieve(
    query: str,
    *,
    client: QdrantClient,
    language: str | None = None,
    top_k: int | None = None,
    alpha: float | None = None,
    exclude_ids: set[str] | None = None,
) -> list[RetrievedTurn]:
    s = get_settings()
    k = top_k or s.TOP_K

    # Pull a wider pool so MMR (or the floor) has material to work with.
    # MMR_POOL_SIZE is the cap on candidates that go INTO MMR; we ask the
    # retrievers for at least that many. Falls back to k*4 (legacy behaviour)
    # when MMR is disabled.
    pool = s.MMR_POOL_SIZE if s.MMR_ENABLED else k * 4

    dense = await retrieve_dense(
        client, query, top_k=pool, language=language, exclude_ids=exclude_ids
    )
    bm25 = retrieve_bm25(query, top_k=pool, exclude_ids=exclude_ids)
    fused = fuse_scores(dense, bm25, alpha=alpha, top_k=pool)
    reranked = recency_decay(fused)

    floor = s.HYBRID_SCORE_FLOOR
    if floor > 0:
        reranked = [r for r in reranked if r.score >= floor]

    if not s.MMR_ENABLED:
        return reranked[:k]

    # MMR rerank on the post-floor pool, then slice to k. Reverse so the
    # most-relevant content match lands at the END of the few-shot — LLMs
    # weight the last few-shot most heavily.
    mmr_pool = reranked[: s.MMR_POOL_SIZE]
    picked = mmr_rerank(mmr_pool, k=k, lambda_param=s.MMR_LAMBDA)

    # Spec §8 structured logging — keep it cheap (no embeddings in the line).
    from persona_rag._logging import get_logger

    log = get_logger()
    mean_div = _mean_pairwise_distance([p.embedding for p in picked if p.embedding])
    log.info(
        "mmr_pick_done",
        mmr_enabled=True,
        mmr_pool_size=len(mmr_pool),
        mmr_lambda=s.MMR_LAMBDA,
        mmr_picked_ids=[p.turn.id for p in picked],
        mmr_diversity_score=mean_div,
    )

    picked.reverse()
    return picked


def _mean_pairwise_distance(embs: list[list[float]]) -> float:
    """Mean of (1 - cosine) across all unique pairs. 0.0 if <2 embeddings."""
    if len(embs) < 2:
        return 0.0
    from persona_rag.retrieval.mmr import cosine

    total = 0.0
    count = 0
    for i, a in enumerate(embs):
        for b in embs[i + 1 :]:
            total += 1.0 - cosine(a, b)
            count += 1
    return total / count if count else 0.0
