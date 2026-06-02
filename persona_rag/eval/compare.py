# Reason: Cyrillic literals appear in docstrings/tests for the code-switch metrics.
"""Pure logic for a *fair*, paired API-vs-LoRA persona comparison.

This module is the trustworthy core prescribed by the 2026-06-02 eval-architecture
audit. It does NOT generate text or touch the network/DB — it takes two aligned
lists of generated replies (one per backend) plus the real replies they should
match, and produces a paired scorecard with uncertainty and anti-gaming guards.

Design rules enforced here (audit R4/R6/R7):
- **Paired bootstrap CIs** on the per-backend distance deltas — a difference
  counts only if its 95% CI excludes 0.
- **NaN, never 0.0**, on degenerate (empty) input, so an all-blank backend can
  never score an artificially perfect distance.
- **Anti-gaming guards**: copy/leak rate vs the training replies, distinct-reply
  rate (mode-collapse), empty/failed rate per arm.

All distances reuse the audited-as-sound primitives in ``eval/distribution.py``
and the canonical bubble splitter, so measurement cannot drift from production.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from collections.abc import Callable
from difflib import SequenceMatcher
from statistics import median
from typing import Any

from persona_rag.eval.distribution import (
    js_divergence,
    per_bubble_lengths,
    shape_histogram,
    wasserstein_1d,
)
from persona_rag.generate.bubbles import split_bubbles

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

MetricFn = Callable[[list[str], list[str]], float]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    """Whitespace-collapsed, lowercased form for exact/near duplicate detection."""
    return _WS.sub(" ", text).strip().lower()


def _nonempty(texts: list[str]) -> bool:
    return any(t and t.strip() for t in texts)


def _all_bubbles(texts: list[str]) -> list[str]:
    return [b for t in texts for b in split_bubbles(t)]


# --------------------------------------------------------------------------- #
# distance metrics over (real, gen) — NaN-safe (audit R6)
# --------------------------------------------------------------------------- #
def shape_js_metric(real: list[str], gen: list[str]) -> float:
    """JS divergence of the bubble-count shape histograms (lower = closer).

    Returns NaN if either side has no non-empty bubbles, so an all-empty
    generation cannot score an artificial 0.0.
    """
    if not _nonempty(real) or not _nonempty(gen):
        return math.nan
    return js_divergence(shape_histogram(real), shape_histogram(gen))


def len_wasserstein_metric(real: list[str], gen: list[str]) -> float:
    """Earth-mover distance between per-bubble character-length distributions."""
    rl = [float(x) for x in per_bubble_lengths(real)]
    gl = [float(x) for x in per_bubble_lengths(gen)]
    if not rl or not gl:
        return math.nan
    return wasserstein_1d(rl, gl)


def len_wasserstein_norm_metric(real: list[str], gen: list[str]) -> float:
    """``len_wasserstein`` normalized by the median real bubble length, so a few
    very long bubbles don't dominate the absolute EMD (audit §4.2)."""
    raw = len_wasserstein_metric(real, gen)
    rl = [x for x in per_bubble_lengths(real)]
    if math.isnan(raw) or not rl:
        return math.nan
    med = median(rl)
    return raw / med if med else math.nan


# --------------------------------------------------------------------------- #
# single-corpus descriptive rates — NaN-safe
# --------------------------------------------------------------------------- #
def exclaim_rate(texts: list[str]) -> float:
    """Fraction of bubbles containing ``!`` — the rule is *Bohdan never uses it*,
    so lower is more like him (the API base model tends to add them)."""
    bubbles = _all_bubbles(texts)
    if not bubbles:
        return math.nan
    return sum(1 for b in bubbles if "!" in b) / len(bubbles)


def opener_entropy(texts: list[str]) -> float:
    """Shannon entropy (bits) of the first-word distribution across replies — a
    companion to ``opener_top_share`` that a 2-4 opener rotation can't game."""
    openers: list[str] = []
    for x in texts:
        bubbles = split_bubbles(x)
        if not bubbles:
            continue
        toks = _WORD.findall(bubbles[0].lower())
        if toks:
            openers.append(toks[0])
    if not openers:
        return math.nan
    total = len(openers)
    return -sum((c / total) * math.log2(c / total) for c in Counter(openers).values())


def distinct_reply_rate(texts: list[str]) -> float:
    """Unique / total over non-empty normalized replies — a mode-collapse guard
    (a backend that emits the same canned reply scores low)."""
    norms = [_norm(t) for t in texts if t and t.strip()]
    if not norms:
        return math.nan
    return len(set(norms)) / len(norms)


def empty_rate(texts: list[str]) -> float:
    """Fraction of generations that are empty/blank (audit R6 — reported, not
    silently dropped)."""
    if not texts:
        return math.nan
    return sum(1 for t in texts if not (t and t.strip())) / len(texts)


def language_bucket(text: str) -> str:
    """Dominant-script bucket of a reply: ``latin`` / ``cyrillic`` / ``mixed`` /
    ``other`` — for per-language breakdowns (the audit's R10)."""
    toks = _WORD.findall(text.lower())
    lat = sum(1 for t in toks if re.search(r"[a-z]", t))
    cyr = sum(1 for t in toks if re.search(r"[Ѐ-ӿ]", t))
    total = lat + cyr
    if total == 0:
        return "other"
    lat_frac = lat / total
    if lat_frac >= 0.8:
        return "latin"
    if lat_frac <= 0.2:
        return "cyrillic"
    return "mixed"


# --------------------------------------------------------------------------- #
# anti-gaming: copy / leak rate vs training replies (audit R7, critical under R1)
# --------------------------------------------------------------------------- #
def copy_leak_rate(
    gens: list[str],
    train_replies: list[str],
    *,
    near_threshold: float = 0.9,
    max_candidates: int = 40,
) -> dict[str, float]:
    """Fraction of generations that reproduce a training reply.

    ``exact`` = normalized string equals some training reply.
    ``near`` = best ``SequenceMatcher`` ratio against token-sharing candidates
    is >= ``near_threshold``. Candidates are gathered via a word -> indices index
    and capped at ``max_candidates`` for speed (a documented proxy, not exhaustive).
    """
    norm_train = [_norm(t) for t in train_replies if t and t.strip()]
    train_set = set(norm_train)
    index: dict[str, list[int]] = {}
    for i, t in enumerate(norm_train):
        for w in set(_WORD.findall(t)):
            index.setdefault(w, []).append(i)

    norm_gens = [_norm(g) for g in gens if g and g.strip()]
    if not norm_gens:
        return {"exact": math.nan, "near": math.nan, "n": 0.0}

    exact = 0
    near = 0
    for g in norm_gens:
        if g in train_set:
            exact += 1
            near += 1
            continue
        words = set(_WORD.findall(g))
        cand_ids: list[int] = []
        for w in words:
            cand_ids.extend(index.get(w, ()))
        if not cand_ids:
            continue
        # most-overlapping candidates first
        ranked = [i for i, _ in Counter(cand_ids).most_common(max_candidates)]
        best = max((SequenceMatcher(None, g, norm_train[i]).ratio() for i in ranked), default=0.0)
        if best >= near_threshold:
            near += 1
    n = len(norm_gens)
    return {"exact": exact / n, "near": near / n, "n": float(n)}


# --------------------------------------------------------------------------- #
# paired bootstrap CI on the per-backend distance delta (audit R4)
# --------------------------------------------------------------------------- #
def paired_bootstrap_delta_ci(
    real: list[str],
    gen_a: list[str],
    gen_b: list[str],
    metric_fn: MetricFn,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap 95% CI for ``metric(real, gen_a) - metric(real, gen_b)``.

    Paired: the same resampled item indices are applied to real / a / b in each
    iteration, so shared sampling noise cancels. ``metric_fn`` is a distance
    where LOWER = closer to real; a NEGATIVE delta therefore means A is closer.
    A delta is significant iff its CI excludes 0.
    """
    n = len(real)
    if not (len(gen_a) == n and len(gen_b) == n):
        raise ValueError("real, gen_a, gen_b must be aligned and equal length")
    point = metric_fn(real, gen_a) - metric_fn(real, gen_b)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        rs = [real[i] for i in idx]
        asub = [gen_a[i] for i in idx]
        bsub = [gen_b[i] for i in idx]
        d = metric_fn(rs, asub) - metric_fn(rs, bsub)
        if not math.isnan(d):
            deltas.append(d)
    if not deltas:
        return {"delta": point, "ci_lo": math.nan, "ci_hi": math.nan, "excludes_zero": False}
    deltas.sort()
    lo_q = (1.0 - ci) / 2.0
    lo = deltas[int(lo_q * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int((1.0 - lo_q) * len(deltas)))]
    return {
        "delta": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "favored": ("a" if point < 0 else "b") if (lo > 0 or hi < 0) else "tie",
    }


# --------------------------------------------------------------------------- #
# scorecard assembly
# --------------------------------------------------------------------------- #
def arm_summary(real: list[str], gen: list[str]) -> dict[str, float]:
    """All single-arm numbers for one backend: distances to real + tic panel +
    guards. ``real`` is the shared reference set."""
    return {
        "shape_js_vs_real": shape_js_metric(real, gen),
        "len_wasserstein_vs_real": len_wasserstein_metric(real, gen),
        "len_wasserstein_norm_vs_real": len_wasserstein_norm_metric(real, gen),
        "exclaim_rate": exclaim_rate(gen),
        "opener_entropy": opener_entropy(gen),
        "distinct_reply_rate": distinct_reply_rate(gen),
        "empty_rate": empty_rate(gen),
    }


def compare_scorecard(
    real: list[str],
    gen_api: list[str],
    gen_lora: list[str],
    *,
    train_replies: list[str] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Full paired scorecard for one arm (default: the controlled thin-prompt arm).

    ``gen_api`` / ``gen_lora`` must be aligned with ``real`` item-for-item. A
    negative ``delta`` favors the API (it is closer to real); positive favors
    the LoRA.
    """
    card: dict[str, Any] = {
        "n_items": len(real),
        "arms": {
            "api": arm_summary(real, gen_api),
            "lora": arm_summary(real, gen_lora),
        },
        "deltas_api_minus_lora": {
            "shape_js": paired_bootstrap_delta_ci(
                real, gen_api, gen_lora, shape_js_metric, n_boot=n_boot, seed=seed
            ),
            "len_wasserstein": paired_bootstrap_delta_ci(
                real, gen_api, gen_lora, len_wasserstein_metric, n_boot=n_boot, seed=seed
            ),
        },
    }
    if train_replies is not None:
        card["copy_leak"] = {
            "api": copy_leak_rate(gen_api, train_replies),
            "lora": copy_leak_rate(gen_lora, train_replies),
            # Natural floor: how often the REAL held-out replies near-match train.
            # Short casual texts reuse phrases, so a backend near this floor is not
            # overfitting — interpret lora/api copy rates relative to this.
            "baseline_real_vs_train": copy_leak_rate(real, train_replies),
        }
    return card


# --------------------------------------------------------------------------- #
# blind human-preference scoring (audit §4.4) — the verdict on the real target
# --------------------------------------------------------------------------- #
def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion ``wins/n``; (0.5, 0.5) if n==0."""
    if n == 0:
        return (0.5, 0.5)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def score_preferences(choices: dict[str, str], key: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Resolve blind A/B/tie choices into a LoRA-vs-API win-rate + Wilson CI.

    ``choices``: item_id -> "A" | "B" | "tie". ``key``: item_id -> {"A": backend, "B": backend}.
    Verdict = LoRA win-rate among DECISIVE (non-tie) items; a CI excluding 0.5 is
    a real preference (the audit's definition of "better").
    """
    lora_wins = api_wins = ties = unknown = 0
    for iid, ch in choices.items():
        k = key.get(iid)
        if k is None:
            unknown += 1
        elif ch == "tie":
            ties += 1
        elif ch in ("A", "B"):
            winner = k[ch]
            if winner == "lora":
                lora_wins += 1
            elif winner == "api":
                api_wins += 1
    decisive = lora_wins + api_wins
    lo, hi = wilson_ci(lora_wins, decisive)
    win_rate = lora_wins / decisive if decisive else math.nan
    if decisive and (lo > 0.5 or hi < 0.5):
        verdict = "lora" if win_rate > 0.5 else "api"
    else:
        verdict = "tie"
    return {
        "lora_wins": lora_wins,
        "api_wins": api_wins,
        "ties": ties,
        "unknown": unknown,
        "decisive": decisive,
        "lora_win_rate": win_rate,
        "wilson_95ci": [lo, hi],
        "verdict": verdict,
    }
