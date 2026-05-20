from __future__ import annotations

import uuid
from collections.abc import Iterable, Iterator

from persona_rag.config import get_settings
from persona_rag.ingest.normalize import detect_language, hash_id
from persona_rag.models import PersonaTurn, RawMessage


def _count_emojis(text: str) -> int:
    return sum(1 for c in text if 0x1F300 <= ord(c) <= 0x1FAFF or 0x2600 <= ord(c) <= 0x27BF)


def extract_persona_turns(
    session: Iterable[RawMessage],
    *,
    persona_sender_id: str,
    context_turns: int | None = None,
) -> Iterator[PersonaTurn]:
    if context_turns is None:
        context_turns = get_settings().CONTEXT_TURNS
    history: list[RawMessage] = []
    for msg in session:
        if msg.sender_id == persona_sender_id:
            ctx = [m.text for m in history[-context_turns:]]
            yield PersonaTurn(
                id=str(uuid.uuid4()),
                your_reply=msg.text,
                incoming_context=ctx,
                channel=msg.channel,
                chat_id_hash=hash_id(msg.chat_id),
                recipient_id_hash=hash_id(
                    next(
                        (
                            h.sender_id
                            for h in reversed(history)
                            if h.sender_id != persona_sender_id
                        ),
                        "",
                    )
                ),
                timestamp=msg.timestamp,
                language=detect_language(msg.text),
                your_reply_len_chars=len(msg.text),
                your_reply_emoji_count=_count_emojis(msg.text),
                eval_split=False,
            )
        history.append(msg)


def mark_eval_split(turns: list[PersonaTurn], frac: float = 0.1) -> list[PersonaTurn]:
    """Tag last `frac` of turns by timestamp as eval=True."""
    if not turns:
        return turns
    sorted_turns = sorted(turns, key=lambda t: t.timestamp)
    cutoff = int(len(sorted_turns) * (1 - frac))
    return [t.model_copy(update={"eval_split": i >= cutoff}) for i, t in enumerate(sorted_turns)]
