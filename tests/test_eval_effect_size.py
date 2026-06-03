"""Tests for the per-item length effect-size primitives."""

from __future__ import annotations

import math

from persona_rag.eval.effect_size import (
    cliffs_delta,
    cliffs_magnitude,
    length_effect_sizes,
    mean_bubble_len,
    per_item_length_errors,
    sign_test,
    wilcoxon_signed_rank,
)


def test_mean_bubble_len_multibubble() -> None:
    # bubbles "a" (1) and "bb" (2) -> mean 1.5
    assert mean_bubble_len("a\nbb") == 1.5


def test_mean_bubble_len_empty_is_zero() -> None:
    assert mean_bubble_len("   ") == 0.0


def test_per_item_length_errors_absolute() -> None:
    real = ["abcd", "ab"]  # lens 4, 2
    gen = ["a", "abcdef"]  # lens 1, 6 -> errors 3, 4
    assert per_item_length_errors(real, gen) == [3.0, 4.0]


def test_cliffs_delta_full_dominance() -> None:
    # every x > every y -> +1
    assert cliffs_delta([3, 4, 5], [1, 2]) == 1.0
    # every x < every y -> -1
    assert cliffs_delta([1, 2], [3, 4, 5]) == -1.0


def test_cliffs_magnitude_thresholds() -> None:
    assert cliffs_magnitude(0.10) == "negligible"
    assert cliffs_magnitude(0.20) == "small"
    assert cliffs_magnitude(0.40) == "medium"
    assert cliffs_magnitude(0.95) == "large"


def test_sign_test_all_positive() -> None:
    out = sign_test([1.0, 2.0, 0.5, 3.0])
    assert out["pos"] == 4 and out["neg"] == 0 and out["n"] == 4
    assert out["p"] < 0.13  # 2 * 0.5^4 = 0.125


def test_sign_test_drops_ties() -> None:
    out = sign_test([1.0, 0.0, -1.0, 0.0])
    assert out["n"] == 2 and out["pos"] == 1 and out["neg"] == 1


def test_wilcoxon_symmetric_is_nonsignificant() -> None:
    out = wilcoxon_signed_rank([1.0, -1.0, 2.0, -2.0])
    assert out["w_plus"] == out["w_minus"]
    assert out["p"] > 0.9  # z≈0


def test_length_effect_sizes_lora_closer() -> None:
    # LoRA matches real exactly; API is far off on every item.
    real = ["ab", "abcd", "abcdef"]
    gen_api = ["abcdefghijkl", "x", "abcdefghijklmnop"]
    gen_lora = ["ab", "abcd", "abcdef"]
    out = length_effect_sizes(real, gen_api, gen_lora)
    assert out["lora_closer"] == 3 and out["api_closer"] == 0
    assert out["cliffs_delta"] == 1.0  # API errors dominate => LoRA closer
    assert out["cliffs_magnitude"] == "large"
    assert not math.isnan(out["sign_test_p"])
