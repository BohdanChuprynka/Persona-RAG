"""Selects the text a persona turn is indexed/embedded under.

Retrieval should answer "when someone said something like THIS, how did I
reply?" — so the index key is the incoming context, not the reply. The legacy
``reply`` mode (key on ``your_reply``) is kept for comparison only; it makes the
query (a question) match past answers, which scores poorly and empties the pool.
"""

from __future__ import annotations

VALID_MODES = ("incoming", "incoming_last", "reply")

# Cap key length so a giant merged-blob context never blows the embedding
# model's 8192-token input limit. Cyrillic worst-case ~3 tokens/char, so 2000
# chars stays safely under. Keep the tail — recent context matters most.
MAX_KEY_CHARS = 2000


def retrieval_key(incoming_context: list[str], your_reply: str, *, mode: str) -> str:
    """Return the text to embed/index for a turn under the given key mode.

    - ``incoming``      → all context messages joined (the full situation)
    - ``incoming_last`` → only the final context line (the message replied to)
    - ``reply``         → the reply text itself (legacy, asymmetric)

    Returns ``""`` when there is no context to key on (caller skips indexing).
    Over-long keys are truncated to their last ``MAX_KEY_CHARS`` characters.
    """
    if mode == "reply":
        return your_reply[-MAX_KEY_CHARS:]
    ctx = [c.strip() for c in incoming_context if c and c.strip()]
    if mode == "incoming":
        return "\n".join(ctx)[-MAX_KEY_CHARS:]
    if mode == "incoming_last":
        return ctx[-1][-MAX_KEY_CHARS:] if ctx else ""
    raise ValueError(f"unknown retrieval key mode: {mode!r} (valid: {VALID_MODES})")
