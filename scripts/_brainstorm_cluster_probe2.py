"""Second clustering probe — test two hypotheses:
H1: longer replies (>50 chars) cluster better than the full population
H2: clustering on (your_reply || last_incoming_context) gives topical groups
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter

import numpy as np
from sklearn.cluster import HDBSCAN
from sqlmodel import Session, select

from persona_rag.db.engine import make_engine
from persona_rag.db.models import PersonaTurnRow
from persona_rag.index.embedder import embed_batch


def _print_clusters(labels: list[int], texts: list[str], langs: list[str], n_top: int = 8) -> None:
    sizes = Counter(labels)
    if -1 in sizes:
        n_noise = sizes.pop(-1)
        print(f"  noise: {n_noise} ({100 * n_noise / len(labels):.1f}%)")
    print(f"  clusters: {len(sizes)}")
    if not sizes:
        return
    for cid, csize in sizes.most_common(n_top):
        idxs = [i for i, lab in enumerate(labels) if lab == cid][:4]
        primary_lang = Counter(langs[i] for i in idxs).most_common(1)[0][0]
        print(f"\n  CLUSTER {cid} (n={csize}, lang={primary_lang}):")
        for i in idxs:
            preview = texts[i].replace("\n", " | ")[:150]
            print(f"    {preview}")


async def main() -> None:
    random.seed(42)
    print("loading persona turns from SQLite…")
    with Session(make_engine()) as s:
        rows = list(
            s.exec(select(PersonaTurnRow).where(PersonaTurnRow.eval_split == False)).all()  # noqa: E712
        )
    print(f"  total non-eval rows: {len(rows)}")

    # H1: long-reply subset
    long_rows = [r for r in rows if len(r.your_reply) >= 50]
    print(f"  rows with reply >= 50 chars: {len(long_rows)}")
    random.shuffle(long_rows)
    sample_long = long_rows[:1500]

    print("\nembedding 1500 long-reply turns…")
    long_texts = [r.your_reply for r in sample_long]
    long_vecs = np.array(await embed_batch(long_texts), dtype=np.float32)
    print("=== H1: HDBSCAN on long-reply only (cosine, min_cluster_size=8) ===")
    labels_h1 = HDBSCAN(min_cluster_size=8, metric="cosine").fit_predict(long_vecs)
    _print_clusters(labels_h1.tolist(), long_texts, [r.language for r in sample_long])

    # H2: your_reply || last_incoming_context concat
    import json

    rich_rows = []
    rich_texts = []
    for r in random.sample(rows, min(1500, len(rows))):
        try:
            ctx = json.loads(r.incoming_context_json)
        except Exception:
            ctx = []
        last_in = ctx[-1] if ctx else ""
        combined = f"{r.your_reply} || ctx: {last_in}"
        rich_rows.append(r)
        rich_texts.append(combined)

    print("\nembedding 1500 (reply || context) pairs…")
    rich_vecs = np.array(await embed_batch(rich_texts), dtype=np.float32)
    print("=== H2: HDBSCAN on reply||context (cosine, min_cluster_size=8) ===")
    labels_h2 = HDBSCAN(min_cluster_size=8, metric="cosine").fit_predict(rich_vecs)
    _print_clusters(labels_h2.tolist(), rich_texts, [r.language for r in rich_rows])


if __name__ == "__main__":
    asyncio.run(main())
