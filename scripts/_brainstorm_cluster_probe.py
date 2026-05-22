"""Throwaway validation script — checks whether HDBSCAN produces meaningful
clusters on real PersonaTurn embeddings. Used during brainstorming to decide
whether Flavor 3 (cluster-first + temporal sampling) is viable for this
corpus. Delete after design is finalized.

Run:  uv run python scripts/_brainstorm_cluster_probe.py
"""

# ruff: noqa: E501
# Reason: throwaway research script; long debug print lines are fine.
from __future__ import annotations

import random
from collections import Counter

import numpy as np
from qdrant_client import QdrantClient
from sklearn.cluster import HDBSCAN

from persona_rag.config import get_settings


def main() -> None:
    s = get_settings()
    client = QdrantClient(url=s.QDRANT_URL)

    # Pull a random sample of points with vectors + payload
    sample_size = 2000
    print(f"sampling {sample_size} points from Qdrant collection '{s.QDRANT_COLLECTION}'…")
    # Scroll with no filter — Qdrant returns results in insertion order; randomize after.
    points: list = []
    offset: int | None = None
    while len(points) < 10_000:
        batch, offset = client.scroll(
            collection_name=s.QDRANT_COLLECTION,
            limit=2048,
            with_vectors=True,
            with_payload=True,
            offset=offset,
        )
        points.extend(batch)
        if offset is None:
            break
    random.seed(42)
    random.shuffle(points)
    sample = points[:sample_size]
    print(f"pulled {len(sample)} points, vector dim = {len(sample[0].vector)}")

    vecs = np.array([p.vector for p in sample], dtype=np.float32)
    texts = [(p.payload or {}).get("your_reply", "") for p in sample]
    langs = [(p.payload or {}).get("language", "?") for p in sample]
    timestamps = [(p.payload or {}).get("timestamp", "") for p in sample]

    print("\nlanguage distribution in sample:")
    for lang, n in Counter(langs).most_common(10):
        print(f"  {lang}: {n}")

    print("\nreply length distribution (chars):")
    lens = [len(t) for t in texts]
    print(
        f"  min={min(lens)} p25={np.percentile(lens, 25):.0f} "
        f"median={np.median(lens):.0f} p75={np.percentile(lens, 75):.0f} "
        f"p95={np.percentile(lens, 95):.0f} max={max(lens)}"
    )

    # Try a few HDBSCAN configurations
    for min_cluster_size in (8, 15, 30):
        print(f"\n=== HDBSCAN min_cluster_size={min_cluster_size} (cosine) ===")
        # sklearn HDBSCAN supports metric="cosine" via sklearn 1.4+
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric="cosine")
        labels = clusterer.fit_predict(vecs)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        print(f"  clusters: {n_clusters}")
        print(f"  noise:    {n_noise} ({100 * n_noise / len(labels):.1f}%)")

        sizes = Counter(labels)
        del sizes[-1]
        if sizes:
            top = sizes.most_common(10)
            print(
                f"  cluster size distribution: largest={top[0][1]} top10={sum(n for _, n in top)}"
            )
            print("\n  --- top 10 clusters: 5 sample texts each ---")
            for cid, csize in top:
                idxs = [i for i, lab in enumerate(labels) if lab == cid][:5]
                print(
                    f"\n  CLUSTER {cid} (n={csize}, primary_lang={Counter(langs[i] for i in idxs).most_common(1)[0][0]}):"
                )
                for i in idxs:
                    preview = texts[i].replace("\n", " | ")[:80]
                    print(f"    [{timestamps[i][:10]}] {preview}")


if __name__ == "__main__":
    main()
