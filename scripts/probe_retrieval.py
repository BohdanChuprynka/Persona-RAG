"""Diagnostic probe for retrieval quality.

Runs dense + BM25 + hybrid retrieval against the live indexes for a set of
realistic Ukrainian queries and prints the top hits side by side so you can
judge whether the matches are actually meaningful.

Usage:
    uv run python scripts/probe_retrieval.py
    uv run python scripts/probe_retrieval.py --query "своя фраза тут"
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from persona_rag.config import get_settings
from persona_rag.index.embedder import embed_batch
from persona_rag.index.qdrant_store import make_client, search_dense
from persona_rag.retrieval.bm25 import retrieve_bm25
from persona_rag.retrieval.hybrid import fuse_scores

DEFAULT_PROBES: list[str] = [
    "Привіт, як ти?",
    "що зара робиш?",
    "чим займаєшся сьогодні?",
    "розкажи про машинне навчання",
    "коли тренування?",
    "плани на вихідних",
    "скільки тобі років?",
    "що читаєш зараз",
]


def _fmt_reply(text: str, n: int = 80) -> str:
    t = text.replace("\n", " ⏎ ")
    return t if len(t) <= n else t[: n - 1] + "…"


def _fmt_ctx(ctx: list[str], n_last: int = 2, char_limit: int = 100) -> str:
    if not ctx:
        return "(no context)"
    last = " | ".join(c.replace("\n", " ") for c in ctx[-n_last:])
    return last if len(last) <= char_limit else last[: char_limit - 1] + "…"


async def probe_one(client: Any, query: str, top_k: int = 5) -> None:
    s = get_settings()
    print(f"\n{'=' * 78}")
    print(f"QUERY: {query!r}")
    print("=" * 78)

    vec = (await embed_batch([query]))[0]
    dense = search_dense(client, s.QDRANT_COLLECTION, vec, top_k=top_k, exclude_eval=True)
    bm25 = retrieve_bm25(query, top_k=top_k)
    fused = fuse_scores(dense, bm25, alpha=s.HYBRID_DENSE_ALPHA, top_k=top_k)

    print("\n-- DENSE (semantic, embeddings of past replies) --")
    if not dense:
        print("  (no results)")
    for i, r in enumerate(dense, 1):
        print(f"  {i}. score={r.score:.3f}  lang={r.turn.language}")
        print(f"     incoming: {_fmt_ctx(r.turn.incoming_context)}")
        print(f"     reply:    {_fmt_reply(r.turn.your_reply)}")

    print("\n-- BM25 (lexical, token overlap on past replies) --")
    if not bm25:
        print("  (no results — bm25 store may be empty)")
    for i, r in enumerate(bm25, 1):
        print(f"  {i}. score={r.score:.3f}  lang={r.turn.language}")
        print(f"     incoming: {_fmt_ctx(r.turn.incoming_context)}")
        print(f"     reply:    {_fmt_reply(r.turn.your_reply)}")

    print(f"\n-- HYBRID (alpha={s.HYBRID_DENSE_ALPHA:.2f}, dense + bm25 normalized + fused) --")
    for i, r in enumerate(fused, 1):
        print(
            f"  {i}. fused={r.score:.3f}  (dense={r.score_dense:.2f} bm25={r.score_bm25:.2f}) "
            f"lang={r.turn.language}"
        )
        print(f"     reply: {_fmt_reply(r.turn.your_reply)}")


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--query",
        action="append",
        help="Custom probe (repeatable). Overrides defaults if provided.",
    )
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    probes = args.query if args.query else DEFAULT_PROBES
    client = make_client()
    for q in probes:
        await probe_one(client, q, top_k=args.top_k)


if __name__ == "__main__":
    asyncio.run(main())
