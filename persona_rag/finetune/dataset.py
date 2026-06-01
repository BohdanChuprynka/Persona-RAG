# ruff: noqa: RUF001
"""Export Bohdan's real turn-pairs to ShareGPT JSONL for LoRA fine-tuning.

The generator that closes the lexical-voice gap (code-switch, opener variety,
the ")" tic) wants to LEARN from the raw (context -> reply) pairs, not be told
about them. We emit ShareGPT records that Unsloth's standardize_sharegpt +
train_on_responses_only consume directly:

    {"conversations": [{"from": "human", "value": ctx}, {"from": "gpt", "value": reply}]}

Reply newlines are preserved on purpose — they are Bohdan's multi-bubble bursts,
and the bubble-splitter re-splits on exactly the same newlines at serving time,
so the model learns to emit the burst shape. No "Name:" transcript prefix: we
want the raw voice, not a narrated transcript.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# A minimal persona anchor. Kept short and in-language so it nudges register
# without drowning the style signal. Override/disable via the CLI.
DEFAULT_SYSTEM = "Ти Богдан. Пиши так, як ти зазвичай пишеш у телеграмі."


def to_sharegpt(
    incoming_context: list[str],
    your_reply: str,
    *,
    system: str | None = None,
    max_ctx_chars: int = 2000,
) -> dict[str, Any]:
    """One (context -> reply) pair as a ShareGPT record. Blank context lines are
    dropped; context is tail-truncated; reply newlines (bursts) are preserved."""
    ctx = "\n".join(c for c in incoming_context if c and c.strip())[-max_ctx_chars:]
    convo: list[dict[str, str]] = []
    if system:
        convo.append({"from": "system", "value": system})
    convo.append({"from": "human", "value": ctx})
    convo.append({"from": "gpt", "value": your_reply})
    return {"conversations": convo}


def iter_records(
    *,
    eval_split: bool,
    system: str | None,
    min_reply_chars: int,
    max_ctx_chars: int,
) -> Iterator[dict[str, Any]]:
    """Yield ShareGPT records from the DB for the requested split."""
    from sqlmodel import Session, select

    from persona_rag.db.engine import make_engine
    from persona_rag.db.models import PersonaTurnRow

    with Session(make_engine()) as s:
        rows = list(
            s.exec(select(PersonaTurnRow).where(PersonaTurnRow.eval_split == eval_split)).all()
        )
    for r in rows:
        reply = (r.your_reply or "").strip()
        if len(reply) < min_reply_chars:
            continue
        ctx = json.loads(r.incoming_context_json)
        if not any(c.strip() for c in ctx):
            continue
        yield to_sharegpt(ctx, r.your_reply, system=system, max_ctx_chars=max_ctx_chars)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)
