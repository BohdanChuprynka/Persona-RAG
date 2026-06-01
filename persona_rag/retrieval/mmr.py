"""Maximal Marginal Relevance reranking for the few-shot pool.

Per spec docs/superpowers/specs/2026-05-31-mmr-retrieval-design.md §5.1.
"""

from __future__ import annotations

import math

from persona_rag.models import RetrievedTurn


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Returns 0.0 if either input is the zero vector."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def mmr_rerank(
    candidates: list[RetrievedTurn],
    *,
    k: int,
    lambda_param: float = 0.6,
) -> list[RetrievedTurn]:
    """Re-rank candidates balancing relevance (their hybrid `score`) and
    diversity (cosine distance between their `embedding` vectors).

    Greedy algorithm:
        1. Pick the highest-`score` candidate first.
        2. For each subsequent slot, pick the candidate that maximises
               lam * relevance(c) - (1 - lam) * max_sim(c, already_picked)
           where relevance is `score` min-max-normalised to [0, 1] over the
           input pool, and max_sim is the maximum cosine similarity between
           `c.embedding` and any already-picked embedding.

    Candidates without an `embedding` are treated as max-distance from every
    other candidate (their diversity penalty contribution is 0). This lets
    BM25-only hits rank purely by relevance without crashing the math.

    Inputs:
        candidates: pool of candidates with `score` set. Order does not
                    matter - the seed pick is chosen by max score.
        k:          number of items to return.
        lambda_param: 1.0 -> pure relevance; 0.0 -> pure diversity.

    Returns:
        Up to `k` candidates, ordered by greedy pick (most relevant first).
    """
    if not candidates:
        return []
    if len(candidates) <= k:
        # Even in the trivial case, honour the "highest relevance first"
        # invariant - callers may pass an unsorted pool.
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    scores = [c.score for c in candidates]
    s_min, s_max = min(scores), max(scores)
    # 1e-9 floor guards against div-by-zero when every candidate has identical score
    # (e.g. test_mmr_handles_zero_diversity_pool); relevance then becomes uniform 0
    # for all picks and the diversity penalty alone determines order.
    s_range = max(s_max - s_min, 1e-9)

    def relevance(c: RetrievedTurn) -> float:
        return (c.score - s_min) / s_range

    def max_sim_to_picked(c: RetrievedTurn, picked: list[RetrievedTurn]) -> float:
        if c.embedding is None:
            return 0.0
        best = 0.0
        for p in picked:
            if p.embedding is None:
                continue
            sim = cosine(c.embedding, p.embedding)
            if sim > best:
                best = sim
        return best

    # Seed: highest-relevance item, regardless of input order.
    seed_idx = max(range(len(candidates)), key=lambda i: candidates[i].score)
    picked: list[RetrievedTurn] = [candidates[seed_idx]]
    pool: list[RetrievedTurn] = [c for i, c in enumerate(candidates) if i != seed_idx]

    while len(picked) < k and pool:
        best_i = 0
        best_mmr = -float("inf")
        for i, c in enumerate(pool):
            mmr = lambda_param * relevance(c) - (1.0 - lambda_param) * max_sim_to_picked(c, picked)
            if mmr > best_mmr:
                best_mmr = mmr
                best_i = i
        picked.append(pool.pop(best_i))

    return picked
