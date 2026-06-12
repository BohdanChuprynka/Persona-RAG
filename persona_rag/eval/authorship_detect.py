"""Validated stylometric author-detector (review fix #2 — construct validity).

The headline distances (shape, length, tics) are *surface* proxies for voice. This
module adds a metric that is both (a) literature-grounded — char-n-gram authorship
attribution in the tradition of Stamatatos — and (b) *validated*: before it is read
as a voice score, the detector is checked against ground truth by held-out ROC-AUC at
telling the owner's real replies from his correspondents' messages. Only once it is
shown to be a working author detector is its P(owner) applied to generated replies as
a calibrated "does this pass as the person?" score — a far better proxy for voice than
counting exclamation marks, and the only one here checked against a known answer.

Privacy-safe and dependency-light: trains and scores in-process; only aggregate AUC /
mean-probability numbers ever leave. sklearn-only — no torch, no model download, no GPU.
"""

from __future__ import annotations

import math
import random
from typing import Any


def train_detector(
    owner_texts: list[str],
    other_texts: list[str],
    *,
    ngram: tuple[int, int] = (2, 5),
    c: float = 4.0,
) -> Any:
    """Fit a char-n-gram TF-IDF + logistic-regression detector of the owner's
    authorship (owner = 1, others = 0). char_wb n-grams capture script mix, casing,
    punctuation and sub-word lexical style; TF-IDF damps the raw length signal so the
    detector is not merely a length classifier. Returns the fitted sklearn pipeline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, min_df=2, sublinear_tf=True),
            ),
            ("clf", LogisticRegression(C=c, max_iter=2000, class_weight="balanced")),
        ]
    )
    labels = [1] * len(owner_texts) + [0] * len(other_texts)
    pipe.fit(owner_texts + other_texts, labels)
    return pipe


def p_owner(model: Any, texts: list[str]) -> list[float]:
    """Detector probability that each text was authored by the owner."""
    if not texts:
        return []
    probs = model.predict_proba(texts)
    return [float(row[1]) for row in probs]


def _roc_auc(labels: list[int], scores: list[float]) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def validate_auc(
    model: Any,
    owner_eval: list[str],
    other_eval: list[str],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Held-out ROC-AUC (owner vs others) with a stratified bootstrap 95% CI. This is
    the validation step: an AUC well above 0.5 certifies the detector is a real author
    detector before its scores are used as a voice metric."""
    o = p_owner(model, owner_eval)
    n = p_owner(model, other_eval)
    point = _roc_auc([1] * len(o) + [0] * len(n), o + n)
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        ob = [o[rng.randrange(len(o))] for _ in o]
        nb = [n[rng.randrange(len(n))] for _ in n]
        boots.append(_roc_auc([1] * len(ob) + [0] * len(nb), ob + nb))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[min(len(boots) - 1, int(0.975 * len(boots)))]
    return {
        "auc": point,
        "auc_ci": [lo, hi],
        "n_owner": len(owner_eval),
        "n_other": len(other_eval),
        "owner_mean_p": (sum(o) / len(o)) if o else math.nan,
        "other_mean_p": (sum(n) / len(n)) if n else math.nan,
    }


def acceptance(model: Any, texts: list[str]) -> dict[str, Any]:
    """Apply the validated detector to a backend's generations: mean P(owner) and the
    share the detector accepts as the owner (P > 0.5). Blank generations are dropped."""
    ps = p_owner(model, [t for t in texts if t and t.strip()])
    if not ps:
        return {"n": 0, "mean_p_owner": math.nan, "accept_rate": math.nan}
    return {
        "n": len(ps),
        "mean_p_owner": sum(ps) / len(ps),
        "accept_rate": sum(1 for p in ps if p > 0.5) / len(ps),
    }
