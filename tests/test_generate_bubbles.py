# ruff: noqa: RUF001
# Reason: Cyrillic sample replies mirror the real persona corpus.
"""Tests for the canonical bubble splitter + shape-target helper.

One source of truth for "how a reply becomes separate Telegram messages",
shared by send_reply (delivery), eval/distribution (measurement), and the
prompt shape-hint (generation). target_bubbles reads the shape of the moment
off the retrieved examples so generation can be conditioned on it.
"""

from __future__ import annotations

from persona_rag.generate.bubbles import count_bubbles, split_bubbles, target_bubbles


class TestSplitBubbles:
    def test_empty(self) -> None:
        assert split_bubbles("") == []

    def test_single(self) -> None:
        assert split_bubbles("норм") == ["норм"]

    def test_real_newlines(self) -> None:
        assert split_bubbles("та\nок") == ["та", "ок"]

    def test_literal_backslash_n(self) -> None:
        assert split_bubbles("та\\nок") == ["та", "ок"]

    def test_blank_and_whitespace_dropped(self) -> None:
        assert split_bubbles("  та \n\n ок ") == ["та", "ок"]


class TestCountBubbles:
    def test_counts(self) -> None:
        assert count_bubbles("a\nb\nc") == 3
        assert count_bubbles("") == 0


class TestTargetBubbles:
    def test_none_when_empty(self) -> None:
        assert target_bubbles([]) is None

    def test_all_single_gives_one(self) -> None:
        assert target_bubbles(["норм", "ок", "хз"]) == 1

    def test_median_of_shapes(self) -> None:
        # shapes 1,1,3,3 -> median 2
        assert target_bubbles(["a", "b", "a\nb\nc", "x\ny\nz"]) == 2

    def test_clamped_to_max_4(self) -> None:
        big = "\n".join(["x"] * 9)  # 9 bubbles
        assert target_bubbles([big, big]) == 4

    def test_ignores_empty_replies(self) -> None:
        assert target_bubbles(["", "  ", "норм"]) == 1
