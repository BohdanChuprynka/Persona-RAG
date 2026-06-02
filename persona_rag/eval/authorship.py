"""Style-embedding "is this me?" scorer.

A content-independent voice-fidelity signal: embed replies with an authorship/
style encoder (StyleDistance by default), build a reference centroid from
Bohdan's real replies, and score generated replies by mean cosine to it. Higher
= more like Bohdan's surface voice. Survives code-switch and topic drift in a
way the deterministic distributional metrics can't, and doubles as a best-of-N
selector / reward (research item 1).

The encoder needs torch + a model download; everything model-facing is lazy and
behind try/except at the call sites, so the eval degrades gracefully (returns
None for style_self_sim) when the model isn't installed. The centroid + cosine
math is pure and unit-tested.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

DEFAULT_STYLE_MODEL = "StyleDistance/styledistance"


def _l2_normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm else list(v)


def centroid(vectors: list[list[float]]) -> list[float]:
    """Mean of the vectors, L2-normalized. [] for empty input."""
    if not vectors:
        return []
    dim = len(vectors[0])
    mean = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
    return _l2_normalize(mean)


def mean_cosine_to_ref(vectors: list[list[float]], ref: list[float]) -> float:
    """Mean cosine similarity of each vector to ref. 0.0 for empty inputs."""
    if not vectors or not ref:
        return 0.0
    rn = math.sqrt(sum(x * x for x in ref))
    if rn == 0.0:
        return 0.0
    ref_n = [x / rn for x in ref]
    total = 0.0
    for v in vectors:
        vn = math.sqrt(sum(x * x for x in v))
        if vn == 0.0:
            continue
        total += sum(a * b for a, b in zip(v, ref_n, strict=False)) / vn
    return total / len(vectors)


@lru_cache(maxsize=1)
def _model():  # type: ignore[no-untyped-def]
    """Load the style encoder once. Raises if sentence-transformers/model is
    unavailable — callers guard with try/except."""
    from sentence_transformers import SentenceTransformer

    name = os.environ.get("STYLE_EMBED_MODEL", DEFAULT_STYLE_MODEL)
    return SentenceTransformer(name)


def _encode(texts: list[str]) -> list[list[float]]:
    model = _model()
    embs = model.encode(list(texts), normalize_embeddings=False, show_progress_bar=False)
    return [[float(x) for x in e] for e in embs]


def reference_vector(real_texts: list[str]) -> list[float]:
    """Bohdan's voice centroid from his real replies."""
    texts = [t for t in real_texts if t and t.strip()]
    return centroid(_encode(texts))


def self_similarity(gen_texts: list[str], ref: list[float]) -> float:
    """Mean style cosine of generated replies to the reference voice centroid."""
    texts = [t for t in gen_texts if t and t.strip()]
    return mean_cosine_to_ref(_encode(texts), ref)


@lru_cache(maxsize=1)
def cached_reference_vector(sample: int = 300, seed: int = 0) -> tuple[float, ...]:
    """Bohdan's voice centroid from a random sample of his real (training-split)
    replies, cached for the process. Used by best-of-N selection in the live
    generate path. Excludes eval_split rows so it never leaks into eval metrics."""
    import random

    from sqlmodel import Session, select

    from persona_rag.db.engine import make_engine
    from persona_rag.db.models import PersonaTurnRow

    with Session(make_engine()) as s:
        rows = list(
            s.exec(
                select(PersonaTurnRow.your_reply).where(
                    PersonaTurnRow.eval_split == False  # noqa: E712
                )
            ).all()
        )
    replies = [r for r in rows if r and r.strip()]
    rng = random.Random(seed)
    rng.shuffle(replies)
    return tuple(reference_vector(replies[:sample]))
