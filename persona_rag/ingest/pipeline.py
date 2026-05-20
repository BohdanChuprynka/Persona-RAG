from __future__ import annotations

import json
from itertools import groupby
from pathlib import Path
from typing import Any

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
from persona_rag.models import RawMessage

log = get_logger()


async def run_ingest(
    *,
    telegram_path: Path | None = None,
    ig_root: Path | None = None,
    dry_run_embeddings: bool = False,
) -> dict[str, Any]:
    settings = get_settings()
    log.info("ingest_start", tg=str(telegram_path), ig=str(ig_root))

    all_turns = []
    raw_msgs: list[RawMessage] = []
    if telegram_path:
        raw_msgs.extend(parse_telegram_export(telegram_path))
    if ig_root:
        raw_msgs.extend(walk_instagram_folder(ig_root))

    # PII-redact text
    raw_msgs = [m.model_copy(update={"text": redact(m.text)}) for m in raw_msgs]
    # Sort by chat then time
    raw_msgs.sort(key=lambda m: (m.chat_id, m.timestamp))

    for _chat_id, chat_msgs in groupby(raw_msgs, key=lambda m: m.chat_id):
        chat_list = list(chat_msgs)
        collapsed = collapse_bursts(chat_list)
        for session in split_sessions(collapsed):
            if len(session) < settings.MIN_SESSION_TURNS:
                continue
            turns = list(
                extract_persona_turns(
                    session,
                    persona_sender_id=str(settings.ADMIN_TELEGRAM_ID),
                )
            )
            all_turns.extend(turns)

    all_turns = mark_eval_split(all_turns)
    log.info("turns_extracted", count=len(all_turns))

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
    }
    log.info("ingest_done", **summary)
    return summary
