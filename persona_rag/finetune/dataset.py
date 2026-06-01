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

import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from persona_rag.generate.persona import THIN_SYSTEM

_URL = re.compile(r"https?://\S+|www\.\S+|t\.me/\S+", re.IGNORECASE)
_MULTISPACE = re.compile(r" {2,}")

# A minimal persona anchor. Kept short and in-language so it nudges register
# without drowning the style signal. Override/disable via the CLI. Shared with
# the ollama serving path (generate/prompt.build_thin_messages) so the LoRA is
# trained and served under the BYTE-IDENTICAL system turn.
DEFAULT_SYSTEM = THIN_SYSTEM


def clean_reply(reply: str) -> str | None:
    """Sanitize a reply for training, or return None to DROP the row.

    Two leak classes the LoRA must never learn to emit (audit finding D3):
    - ``<REDACTED>`` scrubber scars — drop the whole reply (a scar mid-sentence
      would be reproduced verbatim).
    - URLs — strip the link, keep the voice around it; drop the row only if
      nothing survives (a bare-link reply). Multi-bubble newlines are preserved.
    """
    if "<REDACTED>" in reply:
        return None
    cleaned = _URL.sub("", reply)
    lines = [_MULTISPACE.sub(" ", ln).strip() for ln in cleaned.split("\n")]
    out = "\n".join(ln for ln in lines if ln).strip()
    return out or None


def eval_split_for(turn_id: str, frac: float = 0.1) -> bool:
    """Deterministic per-turn hold-out flag (True = eval). A uniform hash of the
    turn id splits every recipient ~``frac``/90 independently, so train and eval
    share the SAME recipient mix — and therefore the same code-switch register.
    This replaces the temporal-tail split that put the English-heavy recent
    months entirely in eval and made ``latin_script_rate`` unreachable (D1)."""
    h = hashlib.sha1(turn_id.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) % 1000) < frac * 1000


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
    eval_frac: float = 0.1,
) -> Iterator[dict[str, Any]]:
    """Yield cleaned ShareGPT records for the requested split.

    The split is computed by ``eval_split_for`` (recipient-stratified hash), NOT
    the DB ``eval_split`` column (a temporal tail that skewed the target). Every
    reply is run through ``clean_reply`` first.
    """
    from sqlmodel import Session, select

    from persona_rag.db.engine import make_engine
    from persona_rag.db.models import PersonaTurnRow

    with Session(make_engine()) as s:
        rows = list(s.exec(select(PersonaTurnRow)).all())
    for r in rows:
        if eval_split_for(r.id, eval_frac) != eval_split:
            continue
        reply = clean_reply((r.your_reply or "").strip())
        if reply is None or len(reply) < min_reply_chars:
            continue
        ctx = json.loads(r.incoming_context_json)
        if not any(c.strip() for c in ctx):
            continue
        yield to_sharegpt(ctx, reply, system=system, max_ctx_chars=max_ctx_chars)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)
