from __future__ import annotations

import json
from itertools import groupby
from pathlib import Path
from typing import Any

import tiktoken
from sqlmodel import Session

from persona_rag._logging import get_logger
from persona_rag.config import get_settings
from persona_rag.db.engine import make_engine
from persona_rag.db.models import PersonaTurnRow
from persona_rag.index.bm25_store import build_bm25
from persona_rag.index.bm25_store import save as save_bm25
from persona_rag.index.embedder import embed_batch
from persona_rag.index.qdrant_store import ensure_collection, make_client, upsert_turns
from persona_rag.ingest.conversation import collapse_bursts, split_sessions
from persona_rag.ingest.instagram_parser import walk_instagram_folder
from persona_rag.ingest.pii import redact
from persona_rag.ingest.stylometry import compute_anchors
from persona_rag.ingest.telegram_parser import parse_telegram_export
from persona_rag.ingest.turns import extract_persona_turns, mark_eval_split
from persona_rag.models import PersonaTurn, RawMessage

log = get_logger()

# OpenAI embedding pricing per 1M input tokens (2026-05 rates).
_EMBED_PRICE_PER_M = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


def _estimate_embedding_cost(turns: list[PersonaTurn], model: str) -> dict[str, float]:
    """Count tokens in your_reply corpus and convert to USD."""
    if not turns:
        return {"total_tokens": 0, "estimated_usd": 0.0, "per_1m_usd": 0.0}
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    total_tokens = sum(len(enc.encode(t.your_reply)) for t in turns)
    per_m = _EMBED_PRICE_PER_M.get(model, _EMBED_PRICE_PER_M["text-embedding-3-small"])
    return {
        "total_tokens": float(total_tokens),
        "per_1m_usd": per_m,
        "estimated_usd": round(total_tokens / 1_000_000 * per_m, 6),
    }


async def run_ingest(
    *,
    telegram_path: Path | None = None,
    ig_root: Path | None = None,
    dry_run_embeddings: bool = False,
    estimate_only: bool = False,
    max_messages: int | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    log.info(
        "ingest_start",
        tg=str(telegram_path),
        ig=str(ig_root),
        estimate_only=estimate_only,
        max_messages=max_messages,
    )

    all_turns: list[PersonaTurn] = []
    raw_msgs: list[RawMessage] = []
    if telegram_path:
        raw_msgs.extend(parse_telegram_export(telegram_path))
    if ig_root:
        raw_msgs.extend(walk_instagram_folder(ig_root))

    if max_messages is not None:
        raw_msgs = raw_msgs[:max_messages]

    log.info("raw_messages_parsed", count=len(raw_msgs))

    # PII-redact text
    raw_msgs = [m.model_copy(update={"text": redact(m.text)}) for m in raw_msgs]
    # Sort by chat then time
    raw_msgs.sort(key=lambda m: (m.chat_id, m.timestamp))

    persona_id = str(settings.ADMIN_TELEGRAM_ID)
    persona_msg_count = sum(1 for m in raw_msgs if m.sender_id == persona_id)
    log.info(
        "persona_messages_seen",
        persona_id=persona_id,
        count=persona_msg_count,
        total=len(raw_msgs),
    )
    if persona_msg_count == 0 and raw_msgs:
        sample = list({m.sender_id for m in raw_msgs[:200]})[:5]
        log.warning(
            "no_persona_messages",
            hint=(
                "ADMIN_TELEGRAM_ID doesn't match any sender_id in the export. "
                "Check your numeric Telegram user id matches one of the sample ids."
            ),
            sample_sender_ids=sample,
        )

    for _chat_id, chat_msgs in groupby(raw_msgs, key=lambda m: m.chat_id):
        chat_list = list(chat_msgs)
        collapsed = collapse_bursts(chat_list)
        for session in split_sessions(collapsed):
            if len(session) < settings.MIN_SESSION_TURNS:
                continue
            turns = list(
                extract_persona_turns(
                    session,
                    persona_sender_id=persona_id,
                )
            )
            all_turns.extend(turns)

    all_turns = mark_eval_split(all_turns)
    log.info("turns_extracted", count=len(all_turns))

    cost = _estimate_embedding_cost(all_turns, settings.OPENAI_EMBEDDING_MODEL)
    log.info(
        "embedding_cost_estimate",
        model=settings.OPENAI_EMBEDDING_MODEL,
        total_tokens=int(cost["total_tokens"]),
        per_1m_usd=cost["per_1m_usd"],
        estimated_usd=cost["estimated_usd"],
    )

    if estimate_only:
        log.info("estimate_only_done", turns=len(all_turns), estimated_usd=cost["estimated_usd"])
        return {
            "turns_written": 0,
            "vectors_written": 0,
            "primary_language": "n/a",
            "estimate_only": True,
            **{f"cost_{k}": v for k, v in cost.items()},
        }

    # Stylometric anchors → cached for runtime prompt
    anchors = compute_anchors(all_turns)
    Path("data").mkdir(exist_ok=True)
    Path("data/style_anchors.json").write_text(anchors.model_dump_json(indent=2))

    # Write SQLite
    engine = make_engine()
    with Session(engine) as s:
        for t in all_turns:
            s.merge(
                PersonaTurnRow(
                    id=t.id,
                    your_reply=t.your_reply,
                    incoming_context_json=json.dumps(t.incoming_context, ensure_ascii=False),
                    channel=t.channel,
                    chat_id_hash=t.chat_id_hash,
                    recipient_id_hash=t.recipient_id_hash,
                    timestamp=t.timestamp,
                    language=t.language,
                    your_reply_len_chars=t.your_reply_len_chars,
                    your_reply_emoji_count=t.your_reply_emoji_count,
                    eval_split=t.eval_split,
                )
            )
        s.commit()

    # Embed + Qdrant
    written = 0
    if not dry_run_embeddings and all_turns:
        client = make_client()
        ensure_collection(client, settings.QDRANT_COLLECTION)
        batch_size = 128
        for i in range(0, len(all_turns), batch_size):
            batch = all_turns[i : i + batch_size]
            vecs = await embed_batch([t.your_reply for t in batch])
            upsert_turns(
                client,
                settings.QDRANT_COLLECTION,
                list(zip(batch, vecs, strict=True)),
            )
            written += len(batch)

    # BM25
    corpus = [t.your_reply for t in all_turns if not t.eval_split]
    ids = [t.id for t in all_turns if not t.eval_split]
    if corpus:
        bm25 = build_bm25(corpus)
        save_bm25(bm25, ids, Path("data/bm25.pkl"))

    summary: dict[str, Any] = {
        "turns_written": len(all_turns),
        "vectors_written": written,
        "primary_language": anchors.primary_language,
        "estimated_usd": cost["estimated_usd"],
    }
    log.info("ingest_done", **summary)
    return summary
