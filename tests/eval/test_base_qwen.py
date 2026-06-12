# ruff: noqa: RUF001
"""Unit test for the base-Qwen ablation scoring core (review fix #9).

Only the pure ``compute_base_scorecard`` is exercised — the live generation needs a
hosted Qwen endpoint and is run via ``make compare-base``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from base_qwen_arm import compute_base_scorecard

_REAL = ["ок\nнорм", "та нема за що)", "ще побачимось пізніше", "ага точно"]
_GEN = ["Okay, that sounds good!", "No problem at all.", "See you later.", "Yes exactly."]


def test_compute_base_scorecard_shape() -> None:
    sc = compute_base_scorecard(_REAL, _GEN)
    assert set(sc) == {"base_vs_real", "real_reference"}
    base = sc["base_vs_real"]
    for k in (
        "shape_js_vs_real",
        "len_wasserstein_vs_real",
        "exclaim_rate",
        "latin_script_rate",
        "opener_entropy",
        "paren_smiley_rate",
    ):
        assert k in base
    # The English generations are mostly Latin; the real Cyrillic replies are not.
    assert base["latin_script_rate"] > sc["real_reference"]["latin_script_rate"]
    assert 0.0 <= sc["real_reference"]["paren_smiley_rate"] <= 1.0
