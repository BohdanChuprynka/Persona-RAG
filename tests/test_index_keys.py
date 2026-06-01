"""Tests for the retrieval-key selector.

The key is the text a turn is indexed/embedded under. The original system keyed
on ``your_reply`` (the answer) while queries are the friend's incoming message —
an asymmetric match that starves the example pool. ``incoming`` keys on the
context so a question matches past questions.
"""

from __future__ import annotations

import pytest

from persona_rag.index.keys import retrieval_key


class TestRetrievalKey:
    def test_reply_mode_returns_reply(self) -> None:
        assert retrieval_key(["q1", "q2"], "my reply", mode="reply") == "my reply"

    def test_incoming_mode_joins_full_context(self) -> None:
        assert retrieval_key(["q1", "q2"], "r", mode="incoming") == "q1\nq2"

    def test_incoming_last_mode_takes_final_context_line(self) -> None:
        assert retrieval_key(["q1", "q2"], "r", mode="incoming_last") == "q2"

    def test_empty_context_incoming_is_empty(self) -> None:
        assert retrieval_key([], "r", mode="incoming") == ""

    def test_empty_context_incoming_last_is_empty(self) -> None:
        assert retrieval_key([], "r", mode="incoming_last") == ""

    def test_blank_context_lines_dropped_in_join(self) -> None:
        assert retrieval_key(["", "q2", "  "], "r", mode="incoming") == "q2"

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            retrieval_key(["q"], "r", mode="bogus")

    def test_long_key_truncated_to_tail(self) -> None:
        # Embedding model caps input at 8192 tokens; an over-long merged-blob
        # context must be truncated. Keep the tail — recent context matters most.
        long = "x" * 5000
        key = retrieval_key([long], "r", mode="incoming")
        assert len(key) <= 2000
        assert key.endswith("x")

    def test_long_reply_key_also_truncated(self) -> None:
        key = retrieval_key([], "y" * 5000, mode="reply")
        assert len(key) <= 2000
