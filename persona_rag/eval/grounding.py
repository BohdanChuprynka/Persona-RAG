"""Factual-grounding probe scoring (spec 2026-06-08).

Pure, unit-tested core for the bare-vs-grounded grounding probe:

- ``parse_judge_label`` — the LLM judge's reply -> one of {correct, hallucinated,
  deflected}, or ``None`` when unrecognized (the caller counts parse failures
  rather than silently miscounting).
- ``rate_with_ci`` / ``aggregate_labels`` — per-class counts and Wilson 95%
  intervals. The hallucination and correct rates are the headline, so they carry
  intervals; the register metrics below are descriptive (the report's tic-metric
  altitude).
- ``register_profile`` — voice metrics on the generations themselves, for the
  register-preservation check (the fact card must add facts without moving voice).

The generation + judge I/O lives in ``scripts/probe_grounding.py``; this module is
deliberately free of network and config so it stays trivially testable.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from persona_rag.eval.compare import exclaim_rate, wilson_ci
from persona_rag.eval.distribution import (
    latin_script_rate,
    paren_smiley_rate,
    per_bubble_lengths,
)

LABELS: tuple[str, ...] = ("correct", "hallucinated", "deflected")

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def parse_judge_label(raw: str) -> str | None:
    """Map the judge's reply to one of LABELS, or ``None`` if unrecognized.

    Accepts a JSON object with a ``"label"`` key (optionally fenced in a code
    block), or loose text mentioning exactly one label keyword. Anything else
    returns ``None`` so the caller can count and handle parse failures explicitly.
    """
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            lab = str(payload.get("label", "")).strip().lower()
            if lab in LABELS:
                return lab
    except json.JSONDecodeError:
        pass
    # Loose-text fallback: accept only an UNAMBIGUOUS single-label mention.
    low = raw.lower()
    hits = {lab for lab in LABELS if lab in low}
    if len(hits) == 1:
        return next(iter(hits))
    return None


@dataclass(frozen=True)
class RateCI:
    """A proportion k/n with its Wilson 95% interval."""

    k: int
    n: int
    rate: float
    lo: float
    hi: float


def rate_with_ci(k: int, n: int) -> RateCI:
    """``k`` of ``n`` as a proportion with a Wilson 95% interval.

    ``n == 0`` returns an all-zero record (no data) rather than dividing by zero.
    """
    if n <= 0:
        return RateCI(0, 0, 0.0, 0.0, 0.0)
    lo, hi = wilson_ci(k, n)
    # A proportion interval lives in [0, 1]; clamp the Wilson bounds so
    # floating-point noise (e.g. lo = -1.4e-17 at k=0) never prints a faint
    # negative bound in a figure or table.
    lo = min(1.0, max(0.0, lo))
    hi = min(1.0, max(0.0, hi))
    return RateCI(k, n, k / n, lo, hi)


def aggregate_labels(labels: list[str]) -> dict[str, Any]:
    """Per-class counts and Wilson-interval rates over a list of probe labels.

    Entries not in LABELS are counted as ``unparsed`` and excluded from ``n``, so
    a malformed judge output can never inflate any rate.
    """
    counts = {lab: 0 for lab in LABELS}
    unparsed = 0
    for lab in labels:
        if lab in counts:
            counts[lab] += 1
        else:
            unparsed += 1
    n = sum(counts.values())
    return {
        "n": n,
        "unparsed": unparsed,
        "counts": counts,
        "correct": asdict(rate_with_ci(counts["correct"], n)),
        "hallucinated": asdict(rate_with_ci(counts["hallucinated"], n)),
        "deflected": asdict(rate_with_ci(counts["deflected"], n)),
    }


def register_profile(texts: list[str]) -> dict[str, Any]:
    """Voice metrics on a set of generations — the register-preservation check.

    Descriptive (the report's tic-metric altitude), so the fact card can be shown
    to add facts *without* moving the voice: mean per-bubble character length,
    Latin-script (code-switch) rate, exclamation rate, and paren-smiley rate.
    """
    lengths = per_bubble_lengths(texts)
    mean_len = sum(lengths) / len(lengths) if lengths else 0.0
    return {
        "n": len(texts),
        "mean_bubble_len": mean_len,
        "latin_rate": latin_script_rate(texts),
        "exclaim_rate": exclaim_rate(texts) if texts else 0.0,
        "paren_smiley_rate": paren_smiley_rate(texts),
    }
