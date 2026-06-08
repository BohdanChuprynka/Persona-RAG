# ruff: noqa: RUF001
"""Unit tests for the factual-grounding probe scoring core (spec 2026-06-08).

Pure functions only — no network. The generation + judge I/O is exercised by the
live run (`make compare-vault`), not here.
"""

from __future__ import annotations

from persona_rag.eval.grounding import (
    aggregate_labels,
    parse_judge_label,
    rate_with_ci,
    register_profile,
)


class TestParseJudgeLabel:
    def test_clean_json(self):
        assert parse_judge_label('{"label": "correct", "reason": "x"}') == "correct"

    def test_fenced_json(self):
        assert parse_judge_label('```json\n{"label": "hallucinated"}\n```') == "hallucinated"

    def test_case_insensitive(self):
        assert parse_judge_label('{"label": "Deflected"}') == "deflected"

    def test_loose_keyword(self):
        assert parse_judge_label("verdict: the answer is correct") == "correct"

    def test_unknown_is_none(self):
        assert parse_judge_label("no idea what this is") is None

    def test_empty_is_none(self):
        assert parse_judge_label("") is None

    def test_ambiguous_two_labels_is_none(self):
        # loose text naming two distinct labels is ambiguous -> None (caller counts it)
        assert parse_judge_label("not correct, more like hallucinated") is None


class TestRateWithCI:
    def test_zero_successes(self):
        r = rate_with_ci(0, 20)
        assert r.k == 0 and r.n == 20 and r.rate == 0.0
        assert r.lo == 0.0 and 0.0 < r.hi < 0.3  # Wilson upper bound above 0

    def test_bounds_bracket_rate(self):
        r = rate_with_ci(15, 20)
        assert r.rate == 0.75 and r.lo < r.rate < r.hi
        assert r.lo >= 0.0 and r.hi <= 1.0

    def test_zero_n_is_all_zero(self):
        r = rate_with_ci(0, 0)
        assert r.n == 0 and r.rate == 0.0 and r.lo == 0.0 and r.hi == 0.0


class TestAggregateLabels:
    def test_counts_and_rates(self):
        labels = ["correct"] * 14 + ["hallucinated"] + ["deflected"] * 5
        agg = aggregate_labels(labels)
        assert agg["n"] == 20
        assert agg["counts"] == {"correct": 14, "hallucinated": 1, "deflected": 5}
        assert agg["correct"]["rate"] == 0.7
        assert agg["hallucinated"]["rate"] == 0.05
        assert agg["unparsed"] == 0

    def test_unparsed_excluded_from_n(self):
        labels = ["correct", "correct", "??", "garbage"]
        agg = aggregate_labels(labels)
        assert agg["n"] == 2 and agg["unparsed"] == 2
        assert agg["correct"]["rate"] == 1.0  # 2/2, unparsed never inflate n


class TestRegisterProfile:
    def test_keys_and_count(self):
        p = register_profile(["hello", "ok"])
        assert p["n"] == 2
        assert {"mean_bubble_len", "latin_rate", "exclaim_rate", "paren_smiley_rate"} <= set(p)

    def test_latin_vs_cyrillic_direction(self):
        assert register_profile(["hello world"])["latin_rate"] == 1.0
        assert register_profile(["привіт світ"])["latin_rate"] == 0.0

    def test_exclaim_and_paren_tics(self):
        assert register_profile(["yes!"])["exclaim_rate"] == 1.0
        assert register_profile(["ок"])["exclaim_rate"] == 0.0
        assert register_profile(["ага)"])["paren_smiley_rate"] == 1.0

    def test_mean_bubble_len(self):
        assert register_profile(["abcde"])["mean_bubble_len"] == 5.0

    def test_empty_is_safe(self):
        p = register_profile([])
        assert p["n"] == 0 and p["mean_bubble_len"] == 0.0
