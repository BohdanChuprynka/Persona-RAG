"""Unit tests for the fair API-vs-LoRA comparison core (persona_rag/eval/compare.py)."""

from __future__ import annotations

import math

from persona_rag.eval.compare import (
    arm_summary,
    compare_scorecard,
    copy_leak_rate,
    distinct_reply_rate,
    empty_rate,
    exclaim_rate,
    language_bucket,
    len_wasserstein_metric,
    opener_entropy,
    paired_bootstrap_delta_ci,
    score_preferences,
    shape_js_metric,
    wilson_ci,
)


def test_shape_js_identical_is_zero() -> None:
    real = ["a\nb", "c", "d\ne\nf"]
    assert shape_js_metric(real, list(real)) == 0.0


def test_shape_js_empty_is_nan_not_zero() -> None:
    # The audit's R6: an all-empty generation must NOT score a perfect 0.0.
    assert math.isnan(shape_js_metric(["a", "b"], ["", "   "]))
    assert math.isnan(shape_js_metric([], ["a"]))


def test_len_wasserstein_identical_is_zero_empty_is_nan() -> None:
    real = ["hello", "yo\nsup"]
    assert len_wasserstein_metric(real, list(real)) == 0.0
    assert math.isnan(len_wasserstein_metric(real, ["", ""]))


def test_exclaim_rate() -> None:
    assert exclaim_rate(["hey!", "yo"]) == 0.5
    assert exclaim_rate(["calm", "cool"]) == 0.0
    assert math.isnan(exclaim_rate(["", "  "]))


def test_opener_entropy_bounds() -> None:
    # one opener -> 0 bits; two equally-likely openers -> 1 bit.
    assert opener_entropy(["та ладно", "та норм", "та все"]) == 0.0
    assert opener_entropy(["та раз", "ок два"]) == 1.0


def test_distinct_reply_rate() -> None:
    assert distinct_reply_rate(["a", "a", "b"]) == 2 / 3
    assert distinct_reply_rate(["x", "x", "x"]) == 1 / 3
    assert math.isnan(distinct_reply_rate(["", " "]))


def test_empty_rate() -> None:
    assert empty_rate(["a", "", "  ", "b"]) == 0.5
    assert empty_rate(["a", "b"]) == 0.0


def test_language_bucket() -> None:
    assert language_bucket("hello there friend") == "latin"
    assert language_bucket("привіт як справи") == "cyrillic"
    assert language_bucket("yo привіт bro як") == "mixed"
    assert language_bucket("123 !!! ???") == "other"


def test_copy_leak_exact_and_near() -> None:
    train = ["привіт як справи", "та норм брат", "що по плану"]
    # exact normalized copy
    exact = copy_leak_rate(["Привіт   як справи"], train)
    assert exact["exact"] == 1.0
    # near copy (trailing tic) — shares words, ratio >= 0.9
    near = copy_leak_rate(["привіт як справи)"], train, near_threshold=0.9)
    assert near["near"] == 1.0
    # unrelated reply — no shared words, no copy
    clean = copy_leak_rate(["zzz qqq www"], train)
    assert clean["exact"] == 0.0 and clean["near"] == 0.0


def test_bootstrap_detects_clear_winner() -> None:
    # gen_a == real (distance 0); gen_b is a constant single-bubble blob (far in shape).
    real = ["a\nb\nc", "d\ne", "f", "g\nh\ni\nj", "k\nl"] * 6  # n=30, varied shapes
    gen_a = list(real)
    gen_b = ["x"] * len(real)
    res = paired_bootstrap_delta_ci(real, gen_a, gen_b, shape_js_metric, n_boot=500, seed=0)
    assert res["delta"] < 0  # A (==real) is closer
    assert res["excludes_zero"] is True
    assert res["favored"] == "a"


def test_bootstrap_identical_arms_is_tie() -> None:
    real = ["a\nb", "c", "d\ne\nf"] * 5
    gen = ["a", "b\nc", "d"] * 5
    res = paired_bootstrap_delta_ci(real, gen, list(gen), shape_js_metric, n_boot=300, seed=1)
    assert res["delta"] == 0.0
    assert res["excludes_zero"] is False
    assert res["favored"] == "tie"


def test_bootstrap_alignment_error() -> None:
    try:
        paired_bootstrap_delta_ci(["a"], ["b", "c"], ["d"], shape_js_metric)
    except ValueError:
        return
    raise AssertionError("expected ValueError on misaligned inputs")


def test_compare_scorecard_structure() -> None:
    real = ["a\nb", "c\nd", "e"] * 5
    gen_api = ["a", "b\nc", "d"] * 5
    gen_lora = ["a\nb", "c\nd", "e"] * 5  # == real
    card = compare_scorecard(real, gen_api, gen_lora, train_replies=["a\nb"], n_boot=200, seed=0)
    assert card["n_items"] == len(real)
    assert set(card["arms"]) == {"api", "lora"}
    assert "shape_js" in card["deltas_api_minus_lora"]
    assert "copy_leak" in card
    # lora == real here, so lora distance ~0 and the delta should favor lora (positive).
    assert card["arms"]["lora"]["shape_js_vs_real"] == 0.0


def test_arm_summary_keys() -> None:
    s = arm_summary(["a\nb", "c"], ["a", "b"])
    for k in (
        "shape_js_vs_real",
        "len_wasserstein_vs_real",
        "exclaim_rate",
        "opener_entropy",
        "distinct_reply_rate",
        "empty_rate",
    ):
        assert k in s


def test_wilson_ci() -> None:
    lo, _hi = wilson_ci(18, 20)
    assert lo > 0.5  # 90% wins -> CI clears 0.5
    lo2, hi2 = wilson_ci(5, 10)
    assert lo2 < 0.5 < hi2  # 50/50 -> spans 0.5
    assert wilson_ci(0, 0) == (0.5, 0.5)


def test_score_preferences() -> None:
    key = {str(i): {"A": "lora", "B": "api"} for i in range(10)}
    choices = {**{str(i): "A" for i in range(7)}, "7": "B", "8": "tie", "9": "tie"}
    res = score_preferences(choices, key)
    assert res["lora_wins"] == 7
    assert res["api_wins"] == 1
    assert res["ties"] == 2
    assert res["decisive"] == 8
    assert abs(res["lora_win_rate"] - 7 / 8) < 1e-9
    assert res["verdict"] in ("lora", "tie")
    # unknown item_ids are counted, not crashed on
    assert score_preferences({"999": "A"}, key)["unknown"] == 1
