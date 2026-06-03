from __future__ import annotations

import json
from pathlib import Path

from sqlmodel import Session, select

from persona_rag.db.engine import make_engine
from persona_rag.db.models import PersonaTurnRow
from persona_rag.index.bm25_store import load, score_bm25
from persona_rag.models import PersonaTurn, RetrievedTurn


def retrieve_bm25(
    query: str, *, top_k: int, exclude_ids: set[str] | None = None
) -> list[RetrievedTurn]:
    bm25_path = Path("data/bm25.pkl")
    if not bm25_path.exists():
        return []
    bm25, ids = load(bm25_path)
    scores = score_bm25(bm25, query)
    ranked = sorted(zip(ids, scores, strict=True), key=lambda x: x[1], reverse=True)
    if exclude_ids:
        # Drop excluded ids BEFORE the top_k slice so top_k stays honest.
        ranked = [p for p in ranked if p[0] not in exclude_ids]
    pairs = ranked[:top_k]
    if not pairs:
        return []
    selected_ids = [p[0] for p in pairs]
    with Session(make_engine()) as s:
        rows = list(s.exec(select(PersonaTurnRow).where(PersonaTurnRow.id.in_(selected_ids))).all())  # type: ignore[attr-defined]
    row_by_id = {r.id: r for r in rows}
    out: list[RetrievedTurn] = []
    for _id, score in pairs:
        row = row_by_id.get(_id)
        if row is None:
            continue
        turn = PersonaTurn(
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
        out.append(RetrievedTurn(turn=turn, score=score, score_bm25=score))
    return out
