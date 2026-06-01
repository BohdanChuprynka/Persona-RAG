"""Re-embed + re-index existing persona turns under a chosen retrieval key.

Reads PersonaTurnRow straight from SQLite (no re-parsing the Telegram export),
re-embeds each turn under the selected key (default: the incoming context, so a
query matches past *situations* not past *answers*), rebuilds Qdrant + BM25.

    uv run python scripts/reindex.py --key incoming
    uv run python scripts/reindex.py --key incoming_last
    uv run python scripts/reindex.py --key reply         # legacy, for comparison

The collection is dropped and recreated so no stale vectors linger.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
from pathlib import Path

from sqlmodel import Session, select

from persona_rag._logging import configure_logging, get_logger
from persona_rag.config import get_settings
from persona_rag.db.engine import make_engine
from persona_rag.db.models import PersonaTurnRow
from persona_rag.index.bm25_store import build_bm25
from persona_rag.index.bm25_store import save as save_bm25
from persona_rag.index.embedder import embed_batch
from persona_rag.index.keys import retrieval_key
from persona_rag.index.qdrant_store import ensure_collection, make_client, upsert_turns
from persona_rag.models import PersonaTurn

log = get_logger()


def _row_to_turn(row: PersonaTurnRow) -> PersonaTurn:
    return PersonaTurn(
        id=row.id,
        your_reply=row.your_reply,
        incoming_context=json.loads(row.incoming_context_json),
        channel=row.channel,  # type: ignore[arg-type]
        chat_id_hash=row.chat_id_hash,
        recipient_id_hash=row.recipient_id_hash,
        timestamp=row.timestamp,
        language=row.language,
        your_reply_len_chars=row.your_reply_len_chars,
        your_reply_emoji_count=row.your_reply_emoji_count,
        eval_split=row.eval_split,
    )


async def reindex(key_mode: str, collection: str | None = None) -> None:
    s = get_settings()
    coll = collection or s.QDRANT_COLLECTION

    with Session(make_engine()) as sess:
        rows = list(sess.exec(select(PersonaTurnRow)).all())
    turns = [_row_to_turn(r) for r in rows]

    # Pair each turn with its key text; drop turns with no key (empty context).
    keyed: list[tuple[PersonaTurn, str]] = []
    skipped = 0
    for t in turns:
        kt = retrieval_key(t.incoming_context, t.your_reply, mode=key_mode)
        if kt.strip():
            keyed.append((t, kt))
        else:
            skipped += 1
    log.info("reindex_plan", key=key_mode, total=len(turns), indexed=len(keyed), skipped=skipped)

    client = make_client()
    # Clean re-point: drop the collection so no stale reply-vectors survive.
    with contextlib.suppress(Exception):
        client.delete_collection(coll)
    ensure_collection(client, coll)

    written = 0
    batch_size = 128
    for i in range(0, len(keyed), batch_size):
        batch = keyed[i : i + batch_size]
        vecs = await embed_batch([kt for _, kt in batch])
        upsert_turns(client, coll, [(t, v) for (t, _), v in zip(batch, vecs, strict=True)])
        written += len(batch)
        if written % 2560 == 0:
            log.info("reindex_progress", written=written, total=len(keyed))

    # BM25 over the same key text, excluding held-out turns.
    corpus = [kt for t, kt in keyed if not t.eval_split]
    ids = [t.id for t, _ in keyed if not t.eval_split]
    if corpus:
        bm25 = build_bm25(corpus)
        save_bm25(bm25, ids, Path("data/bm25.pkl"))

    log.info(
        "reindex_done",
        key=key_mode,
        collection=coll,
        vectors=written,
        bm25_docs=len(corpus),
    )
    print(
        f"reindexed {written} vectors + {len(corpus)} bm25 docs "
        f"under key='{key_mode}' (skipped {skipped} empty-context)"
    )


def main() -> None:
    configure_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--key", choices=["incoming", "incoming_last", "reply"], default="incoming")
    p.add_argument("--collection", default=None)
    args = p.parse_args()
    asyncio.run(reindex(args.key, args.collection))


if __name__ == "__main__":
    main()
