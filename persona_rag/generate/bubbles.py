"""Canonical bubble splitting + shape target.

One reply becomes several Telegram messages on newlines. This is the single
source of truth used by delivery (send_reply), measurement (eval.distribution),
and generation (the prompt shape-hint), so all three agree on what a "bubble"
is. ``target_bubbles`` reads the typical shape of the moment off the retrieved
examples, so generation can be conditioned on it instead of left to the model's
elaborate-everything instinct.
"""

from __future__ import annotations

from statistics import median

MAX_TARGET_BUBBLES = 4


def split_bubbles(text: str) -> list[str]:
    """Split a reply into Telegram messages: normalize the literal two-char
    ``\\n`` and CRLF, split on newline, strip, drop blanks."""
    if not text:
        return []
    normalized = text.replace("\\n", "\n").replace("\r\n", "\n")
    return [c.strip() for c in normalized.split("\n") if c.strip()]


def count_bubbles(text: str) -> int:
    """How many separate Telegram messages this reply becomes."""
    return len(split_bubbles(text))


def target_bubbles(replies: list[str]) -> int | None:
    """Typical bubble count across the retrieved example replies, clamped to
    [1, MAX_TARGET_BUBBLES]. Returns None when there are no usable replies, so
    the caller can skip the shape hint and let the model decide."""
    counts = [c for c in (count_bubbles(r) for r in replies) if c > 0]
    if not counts:
        return None
    return max(1, min(MAX_TARGET_BUBBLES, round(median(counts))))
