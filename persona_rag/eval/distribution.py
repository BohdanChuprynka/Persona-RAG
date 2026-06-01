"""Distributional persona-accuracy metrics.

The legacy ``eval/stylometry.py`` reduces each side to a corpus *mean*, so a bot
that always emits 45-char, 3-bubble replies can score ~0 against a bimodal real
speaker. These metrics compare full *distributions* — message-shape histogram,
per-bubble length, punctuation — so failure #1 (shape uniformity) is visible.

Pure functions only; no I/O. The runner in ``scripts/eval_persona.py`` feeds
real held-out replies and freshly generated replies through ``persona_distance``.
"""

from __future__ import annotations

import bisect
import math
import re
from statistics import median
from typing import Any

from persona_rag.eval.stylometry import compute_features
from persona_rag.generate.bubbles import count_bubbles, split_bubbles

# Paren-smiley: a ) or )) used as a smiley — a non-space/non-open-paren char
# followed by a close paren not followed by a word char, or a bare )) run.
_PAREN_SMILEY = re.compile(r"\)\)+|[^\s(]\)(?!\w)")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_LATIN = re.compile(r"[a-z]")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def _word_tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def latin_script_rate(texts: list[str]) -> float:
    """Share of alphabetic tokens written in Latin vs Cyrillic script — the
    code-switch signal. Real Bohdan is ~46% Latin; a bot pinned to one language
    drops near 0."""
    toks = [t for x in texts for t in _word_tokens(x)]
    lat = sum(1 for t in toks if _LATIN.search(t))
    cyr = sum(1 for t in toks if _CYRILLIC.search(t))
    total = lat + cyr
    return lat / total if total else 0.0


def opener_top_share(texts: list[str]) -> float:
    """Share of replies that open with the single most common first word — the
    opener-monotony signal (the bot opens 'та' >50% vs the real ~6%)."""
    openers = []
    for x in texts:
        bubbles = split_bubbles(x)
        if not bubbles:
            continue
        toks = _word_tokens(bubbles[0])
        if toks:
            openers.append(toks[0])
    if not openers:
        return 0.0
    from collections import Counter

    return Counter(openers).most_common(1)[0][1] / len(openers)


def _bubbles(text: str) -> list[str]:
    """Split a reply into its Telegram messages (canonical splitter)."""
    return split_bubbles(text)


def bubble_count(text: str) -> int:
    """Number of separate Telegram messages this reply becomes."""
    return count_bubbles(text)


def paren_smiley_rate(texts: list[str]) -> float:
    """Fraction of bubbles containing a paren-smiley — Bohdan's signature tic
    that the emoji-codepoint metric is blind to."""
    bubbles = [b for t in texts for b in split_bubbles(t)]
    if not bubbles:
        return 0.0
    return sum(1 for b in bubbles if _PAREN_SMILEY.search(b)) / len(bubbles)


def per_bubble_lengths(texts: list[str]) -> list[int]:
    """Flat list of per-bubble character lengths across all texts."""
    out: list[int] = []
    for t in texts:
        out.extend(len(b) for b in _bubbles(t))
    return out


def shape_histogram(texts: list[str], max_bucket: int = 6) -> dict[int, float]:
    """Normalized distribution of bubble-count per reply, bucketed 1..max_bucket.

    Replies with more than ``max_bucket`` bubbles fall in the top bucket. Empty
    (zero-bubble) texts are excluded. Returns every bucket 1..max_bucket so two
    histograms always share support.
    """
    counts: dict[int, int] = {}
    n = 0
    for t in texts:
        c = bubble_count(t)
        if c == 0:
            continue
        b = min(c, max_bucket)
        counts[b] = counts.get(b, 0) + 1
        n += 1
    if n == 0:
        return {}
    return {b: counts.get(b, 0) / n for b in range(1, max_bucket + 1)}


def js_divergence(p: dict[int, float], q: dict[int, float]) -> float:
    """Jensen-Shannon divergence (log base 2 → bounded [0, 1]) between two
    discrete distributions keyed identically. 0 = identical, 1 = disjoint."""
    keys = set(p) | set(q)
    m = {k: (p.get(k, 0.0) + q.get(k, 0.0)) / 2 for k in keys}

    def _kl(x: dict[int, float]) -> float:
        s = 0.0
        for k in keys:
            xi = x.get(k, 0.0)
            mi = m.get(k, 0.0)
            if xi > 0 and mi > 0:
                s += xi * math.log2(xi / mi)
        return s

    return 0.5 * _kl(p) + 0.5 * _kl(q)


def wasserstein_1d(a: list[float], b: list[float]) -> float:
    """1-D Wasserstein (earth-mover) distance between two empirical samples,
    integrating |CDF_a - CDF_b| over the merged support. Handles unequal sizes."""
    if not a or not b:
        return 0.0
    sa, sb = sorted(a), sorted(b)
    na, nb = len(sa), len(sb)
    pts = sorted(set(sa) | set(sb))
    total = 0.0
    for i in range(len(pts) - 1):
        x0, x1 = pts[i], pts[i + 1]
        ca = bisect.bisect_right(sa, x0) / na
        cb = bisect.bisect_right(sb, x0) / nb
        total += abs(ca - cb) * (x1 - x0)
    return total


def ks_statistic(a: list[float], b: list[float]) -> float:
    """Kolmogorov-Smirnov statistic: max gap between empirical CDFs."""
    if not a or not b:
        return 0.0
    sa, sb = sorted(a), sorted(b)
    na, nb = len(sa), len(sb)
    pts = sorted(set(sa) | set(sb))
    return max(abs(bisect.bisect_right(sa, x) / na - bisect.bisect_right(sb, x) / nb) for x in pts)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(texts: list[str]) -> dict[str, float | dict[int, float]]:
    """Distributional fingerprint of a corpus of replies."""
    bub_counts = [c for c in (bubble_count(t) for t in texts) if c > 0]
    n = len(bub_counts)
    lens = per_bubble_lengths(texts)
    feats = [compute_features(b) for t in texts for b in _bubbles(t)]
    return {
        "n_texts": len(texts),
        "n_bubbles": len(lens),
        "shape_hist": shape_histogram(texts),
        "pct_single": (sum(1 for c in bub_counts if c == 1) / n) if n else 0.0,
        "bubble_len_median": float(median(lens)) if lens else 0.0,
        "bubble_len_mean": _mean([float(x) for x in lens]),
        "caps_ratio_mean": _mean([f["caps_ratio"] for f in feats]),
        "punct_density_mean": _mean([f["punct_density"] for f in feats]),
        "emoji_rate_mean": _mean([f["emoji_rate"] for f in feats]),
        "paren_smiley_rate": paren_smiley_rate(texts),
        "latin_script_rate": latin_script_rate(texts),
        "opener_top_share": opener_top_share(texts),
    }


def persona_distance(real: list[str], gen: list[str]) -> dict[str, Any]:
    """Headline persona-accuracy distances of generated vs real replies.

    ``shape_js`` (message-shape divergence) and ``len_wasserstein`` (per-bubble
    length earth-mover) are the primary numbers — lower is closer to the real
    speaker. ``real``/``gen`` carry the full summaries for the scorecard.
    """
    rs = summarize(real)
    gs = summarize(gen)
    rl = per_bubble_lengths(real)
    gl = per_bubble_lengths(gen)
    return {
        "shape_js": js_divergence(rs["shape_hist"], gs["shape_hist"]),  # type: ignore[arg-type]
        "len_wasserstein": wasserstein_1d([float(x) for x in rl], [float(x) for x in gl]),
        "len_ks": ks_statistic([float(x) for x in rl], [float(x) for x in gl]),
        "pct_single_real": rs["pct_single"],
        "pct_single_gen": gs["pct_single"],
        "real": rs,
        "gen": gs,
    }
