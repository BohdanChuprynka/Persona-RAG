# ruff: noqa: RUF001, RUF003
# Reason: Cyrillic sample replies/comments exercise the real persona distribution.
"""Tests for distributional persona-accuracy metrics.

These metrics judge generated replies against the real corpus by *distribution*
(message-shape, per-bubble length, punctuation) rather than mean-of-means, so
they can actually see failure #1 (shape uniformity). The mean-MAD eval cannot.
"""

from __future__ import annotations

from persona_rag.eval.distribution import (
    bubble_count,
    js_divergence,
    ks_statistic,
    latin_script_rate,
    opener_top_share,
    paren_smiley_rate,
    per_bubble_lengths,
    persona_distance,
    shape_histogram,
    summarize,
    wasserstein_1d,
)


class TestBubbleCount:
    """bubble_count must mirror send_reply._split_reply exactly: a reply is
    split into Telegram messages on newlines, blanks dropped."""

    def test_empty_is_zero(self) -> None:
        assert bubble_count("") == 0

    def test_whitespace_only_is_zero(self) -> None:
        assert bubble_count("   \n  ") == 0

    def test_single_line(self) -> None:
        assert bubble_count("ага") == 1

    def test_two_real_newlines(self) -> None:
        assert bubble_count("ага\nок") == 2

    def test_blank_lines_dropped(self) -> None:
        assert bubble_count("ага\n\nок") == 2

    def test_literal_backslash_n_counts_as_split(self) -> None:
        # _split_reply normalizes the two-char sequence \n that some models emit
        assert bubble_count("ага\\nок") == 2

    def test_whitespace_stripped_around_chunks(self) -> None:
        assert bubble_count("  ага \n ок ") == 2


class TestShapeHistogram:
    def test_probabilities_sum_to_one(self) -> None:
        h = shape_histogram(["a", "a\nb", "a\nb\nc"])
        assert abs(sum(h.values()) - 1.0) < 1e-9

    def test_fractions(self) -> None:
        h = shape_histogram(["a", "a", "a\nb"])  # two single, one double
        assert abs(h[1] - 2 / 3) < 1e-9
        assert abs(h[2] - 1 / 3) < 1e-9

    def test_caps_at_max_bucket(self) -> None:
        eight = "\n".join(["x"] * 8)
        h = shape_histogram([eight], max_bucket=6)
        assert h[6] == 1.0

    def test_empty_texts_excluded(self) -> None:
        h = shape_histogram(["", "a"])
        assert h[1] == 1.0

    def test_all_buckets_present(self) -> None:
        h = shape_histogram(["a"], max_bucket=6)
        assert set(h.keys()) == {1, 2, 3, 4, 5, 6}


class TestJSDivergence:
    def test_identical_is_zero(self) -> None:
        p = {1: 0.5, 2: 0.5}
        assert js_divergence(p, p) == 0.0

    def test_disjoint_is_one(self) -> None:
        assert abs(js_divergence({1: 1.0}, {2: 1.0}) - 1.0) < 1e-9

    def test_symmetric(self) -> None:
        p, q = {1: 0.7, 2: 0.3}, {1: 0.2, 2: 0.8}
        assert abs(js_divergence(p, q) - js_divergence(q, p)) < 1e-12

    def test_bounded_zero_to_one(self) -> None:
        d = js_divergence({1: 0.9, 2: 0.1}, {1: 0.1, 2: 0.9})
        assert 0.0 <= d <= 1.0


class TestWasserstein:
    def test_identical_is_zero(self) -> None:
        assert wasserstein_1d([1, 2, 3], [1, 2, 3]) == 0.0

    def test_uniform_shift_equals_shift(self) -> None:
        assert abs(wasserstein_1d([0, 0, 0], [5, 5, 5]) - 5.0) < 1e-9

    def test_known_value(self) -> None:
        # matches scipy.stats.wasserstein_distance([0,1],[0,2]) == 0.5
        assert abs(wasserstein_1d([0, 1], [0, 2]) - 0.5) < 1e-9

    def test_empty_guarded(self) -> None:
        assert wasserstein_1d([], [1, 2]) == 0.0


class TestKS:
    def test_identical_is_zero(self) -> None:
        assert ks_statistic([1, 2, 3], [1, 2, 3]) == 0.0

    def test_disjoint_is_one(self) -> None:
        assert abs(ks_statistic([0, 0], [1, 1]) - 1.0) < 1e-9


class TestLexicalMetrics:
    """Code-switch (Latin/Cyrillic share) and opener monotony — the two biggest
    measured voice gaps the shape metrics are blind to."""

    def test_latin_share_all_cyrillic(self) -> None:
        assert latin_script_rate(["привіт як справи"]) == 0.0

    def test_latin_share_all_latin(self) -> None:
        assert latin_script_rate(["hello how are you"]) == 1.0

    def test_latin_share_mixed_half(self) -> None:
        assert abs(latin_script_rate(["ok cool так норм"]) - 0.5) < 1e-9

    def test_latin_share_empty(self) -> None:
        assert latin_script_rate([]) == 0.0

    def test_opener_top_share(self) -> None:
        assert abs(opener_top_share(["та норм", "та ок", "привіт всім"]) - 2 / 3) < 1e-9

    def test_opener_top_share_empty(self) -> None:
        assert opener_top_share([]) == 0.0


class TestParenSmiley:
    """Bohdan's signature tic is the paren-smiley ) / )) — the emoji metric is
    blind to it (it isn't an emoji codepoint), so it needs its own detector."""

    def test_single_paren_after_word(self) -> None:
        assert paren_smiley_rate(["норм)"]) == 1.0

    def test_double_paren(self) -> None:
        assert paren_smiley_rate(["ахах))"]) == 1.0

    def test_no_smiley(self) -> None:
        assert paren_smiley_rate(["норм", "ок"]) == 0.0

    def test_fraction_over_bubbles(self) -> None:
        assert abs(paren_smiley_rate(["норм)", "ок", "хз"]) - 1 / 3) < 1e-9

    def test_empty(self) -> None:
        assert paren_smiley_rate([]) == 0.0


class TestPerBubbleLengths:
    def test_flattens_across_texts(self) -> None:
        # "ага"(3) + "ок"(2) from first, "хеллоу"(6) from second
        assert sorted(per_bubble_lengths(["ага\nок", "хеллоу"])) == [2, 3, 6]


class TestSummarize:
    def test_pct_single(self) -> None:
        s = summarize(["a", "a\nb"])
        assert abs(s["pct_single"] - 0.5) < 1e-9

    def test_counts(self) -> None:
        s = summarize(["a\nb", "c"])
        assert s["n_texts"] == 2
        assert s["n_bubbles"] == 3


class TestPersonaDistance:
    def test_identical_corpora_have_zero_distance(self) -> None:
        real = ["ага", "ок\nдавай", "хз бро", "\n".join(["го"] * 4)]
        d = persona_distance(real, real)
        assert d["shape_js"] < 1e-9
        assert d["len_wasserstein"] < 1e-9
        assert abs(d["pct_single_real"] - d["pct_single_gen"]) < 1e-9

    def test_shape_gap_is_detected(self) -> None:
        # real: half single-message; gen: always 3 messages (the actual bug)
        real = ["ага", "ок", "хз\nбро\nдавай"]
        gen = ["a\nb\nc", "x\ny\nz", "p\nq\nr"]
        d = persona_distance(real, gen)
        assert d["shape_js"] > 0.3
        assert d["pct_single_real"] > d["pct_single_gen"]
