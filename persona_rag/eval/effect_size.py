# Reason: Cyrillic literals appear in sibling eval tests for the metrics.
"""Per-item effect sizes for the persona comparison (audit R4 follow-up).

The paired bootstrap in ``compare.py`` gives a corpus-level CI on the
length-distance delta; this adds the *per-item* complement the audit prescribed:
a standardized effect size (Cliff's delta + matched-pairs rank-biserial) plus
assumption-light significance (sign test + Wilcoxon signed-rank) on per-item
reply-length error. No new generation — reads the same aligned (real, gen_api,
gen_lora) triples the scorecard already scores. Hand-rolled (no scipy) to match
the eval-core house style.

Convention: ``diff_i = err_api_i - err_lora_i``; a POSITIVE diff means the LoRA
is closer to the real reply on item i. ``cliffs_delta(err_api, err_lora)`` is
+1 when API errors dominate (i.e. the LoRA is closer everywhere).
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Any

from persona_rag.generate.bubbles import split_bubbles

# Cliff's delta magnitude thresholds (Romano et al. 2006).
_CLIFF_SMALL, _CLIFF_MEDIUM, _CLIFF_LARGE = 0.147, 0.33, 0.474


def mean_bubble_len(text: str) -> float:
    """Mean character length of a reply's non-empty bubbles; 0.0 if none."""
    bubbles = [b for b in split_bubbles(text) if b.strip()]
    return mean(len(b) for b in bubbles) if bubbles else 0.0


def per_item_length_errors(real: list[str], gen: list[str]) -> list[float]:
    """|mean-bubble-length(gen_i) - mean-bubble-length(real_i)| per aligned item."""
    return [abs(mean_bubble_len(g) - mean_bubble_len(r)) for r, g in zip(real, gen, strict=True)]


def cliffs_delta(xs: list[float], ys: list[float]) -> float:
    """Cliff's delta = P(x>y) - P(x<y) over all pairs. +1 => xs dominate."""
    if not xs or not ys:
        return float("nan")
    gt = sum(1 for x in xs for y in ys if x > y)
    lt = sum(1 for x in xs for y in ys if x < y)
    return (gt - lt) / (len(xs) * len(ys))


def cliffs_magnitude(delta: float) -> str:
    d = abs(delta)
    if math.isnan(d):
        return "undefined"
    if d < _CLIFF_SMALL:
        return "negligible"
    if d < _CLIFF_MEDIUM:
        return "small"
    if d < _CLIFF_LARGE:
        return "medium"
    return "large"


def _avg_ranks(values: list[float]) -> list[float]:
    """1-based average ranks; ties share the mean of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def wilcoxon_signed_rank(diffs: list[float]) -> dict[str, float]:
    """Wilcoxon signed-rank on paired diffs (zeros dropped). Normal approx with
    continuity correction (exact is overkill at n~300). Two-sided p via erfc."""
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    if n == 0:
        return {
            "n": 0,
            "w_plus": float("nan"),
            "w_minus": float("nan"),
            "z": float("nan"),
            "p": float("nan"),
        }
    ranks = _avg_ranks([abs(d) for d in nz])
    w_plus = sum(r for r, d in zip(ranks, nz, strict=True) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nz, strict=True) if d < 0)
    mean_w = n * (n + 1) / 4
    sd_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if not sd_w:
        return {"n": n, "w_plus": w_plus, "w_minus": w_minus, "z": float("nan"), "p": float("nan")}
    # Two-sided, continuity-corrected so the deviation shrinks toward 0 (perfectly
    # symmetric ranks => z=0, p=1); large W+/W- imbalance => large z, tiny p.
    z = max(0.0, abs(w_plus - mean_w) - 0.5) / sd_w
    p = math.erfc(z / math.sqrt(2))
    return {"n": n, "w_plus": w_plus, "w_minus": w_minus, "z": z, "p": p}


def matched_pairs_rank_biserial(diffs: list[float]) -> float:
    """r = (W+ - W-) / (W+ + W-) from the signed-rank sums."""
    w = wilcoxon_signed_rank(diffs)
    denom = w["w_plus"] + w["w_minus"]
    if math.isnan(denom) or denom == 0:
        return float("nan")
    return (w["w_plus"] - w["w_minus"]) / denom


def sign_test(diffs: list[float]) -> dict[str, float]:
    """Two-sided exact binomial sign test (zeros = ties dropped)."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return {"n": 0, "pos": 0, "neg": 0, "p": float("nan")}
    k = max(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) * (0.5**n)
    return {"n": n, "pos": pos, "neg": neg, "p": min(1.0, 2 * tail)}


def length_effect_sizes(real: list[str], gen_api: list[str], gen_lora: list[str]) -> dict[str, Any]:
    """Bundle of per-item length-error effect sizes, API vs LoRA."""
    err_api = per_item_length_errors(real, gen_api)
    err_lora = per_item_length_errors(real, gen_lora)
    diffs = [a - b for a, b in zip(err_api, err_lora, strict=True)]
    delta = cliffs_delta(err_api, err_lora)
    sign = sign_test(diffs)
    return {
        "n_items": len(real),
        "cliffs_delta": delta,
        "cliffs_magnitude": cliffs_magnitude(delta),
        "lora_closer": sign["pos"],
        "api_closer": sign["neg"],
        "ties": len(real) - sign["n"],
        "sign_test_p": sign["p"],
        "rank_biserial": matched_pairs_rank_biserial(diffs),
        "wilcoxon": wilcoxon_signed_rank(diffs),
    }
