from __future__ import annotations

from qdrant_client import QdrantClient

from persona_rag.config import get_settings
from persona_rag.models import RetrievedTurn
from persona_rag.retrieval.bm25 import retrieve_bm25
from persona_rag.retrieval.dense import retrieve_dense
from persona_rag.retrieval.hybrid import fuse_scores
from persona_rag.retrieval.rerank import recency_decay


async def retrieve(
    query: str,
    *,
    client: QdrantClient,
    language: str | None = None,
    top_k: int | None = None,
    alpha: float | None = None,
) -> list[RetrievedTurn]:
    s = get_settings()
    k = top_k or s.TOP_K
    pool = k * 4
    dense = await retrieve_dense(client, query, top_k=pool, language=language)
    bm25 = retrieve_bm25(query, top_k=pool)
    fused = fuse_scores(dense, bm25, alpha=alpha, top_k=pool)
    reranked = recency_decay(fused)
    return reranked[:k]
