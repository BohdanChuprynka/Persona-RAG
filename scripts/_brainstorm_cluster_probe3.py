"""Third probe — does session-level aggregation cluster topically?
Concatenate persona's replies per (chat, 6hr-session-gap), embed the result,
cluster. If clusters look like real topics, Flavor 4 is viable.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter
from datetime import timedelta

import numpy as np
from sklearn.cluster import HDBSCAN
from sqlmodel import Session, select

from persona_rag.db.engine import make_engine
from persona_rag.db.models import PersonaTurnRow
from persona_rag.index.embedder import embed_batch


def _build_sessions(rows: list[PersonaTurnRow], gap_hours: int = 6) -> list[dict]:
    """Group turns by (chat_id, time-gap) into sessions."""
    gap = timedelta(hours=gap_hours)
    by_chat: dict[str, list[PersonaTurnRow]] = {}
    for r in rows:
        by_chat.setdefault(r.chat_id_hash, []).append(r)

    sessions: list[dict] = []
    for chat_id, chat_rows in by_chat.items():
        chat_rows.sort(key=lambda r: r.timestamp)
        current: list[PersonaTurnRow] = []
        for r in chat_rows:
            if current and (r.timestamp - current[-1].timestamp) > gap:
                sessions.append(_session_doc(chat_id, current))
                current = []
            current.append(r)
        if current:
            sessions.append(_session_doc(chat_id, current))
    return sessions


def _session_doc(chat_id: str, turns: list[PersonaTurnRow]) -> dict:
    text = "\n".join(t.your_reply for t in turns)
    return {
        "chat_id": chat_id,
        "start": turns[0].timestamp,
        "end": turns[-1].timestamp,
        "n_turns": len(turns),
        "text": text,
        "primary_lang": Counter(t.language for t in turns).most_common(1)[0][0],
    }


async def main() -> None:
    random.seed(42)
    print("loading rows…")
    with Session(make_engine()) as s:
        rows = list(
            s.exec(select(PersonaTurnRow).where(PersonaTurnRow.eval_split == False)).all()  # noqa: E712
        )
    print(f"  {len(rows)} non-eval turns")

    print("building sessions (6hr gap)…")
    sessions = _build_sessions(rows, gap_hours=6)
    print(f"  {len(sessions)} sessions")

    print("session length distribution (chars):")
    lens = [len(s["text"]) for s in sessions]
    print(
        f"  min={min(lens)} p25={np.percentile(lens, 25):.0f} "
        f"median={np.median(lens):.0f} p75={np.percentile(lens, 75):.0f} "
        f"p95={np.percentile(lens, 95):.0f} max={max(lens)}"
    )

    # Filter: drop sessions with <100 chars of persona content
    high = [s for s in sessions if len(s["text"]) >= 100]
    print(f"  high-signal sessions (>=100 chars persona): {len(high)}")
    random.shuffle(high)
    sample = high[:1200]
    print(f"  embedding {len(sample)} sessions…")

    # Cap each session at ~6000 tokens (text-embedding-3-small max is 8192/input),
    # then batch under 250K tokens total.
    import tiktoken

    enc = tiktoken.encoding_for_model("text-embedding-3-small")

    def truncate(text: str, max_tokens: int) -> str:
        toks = enc.encode(text)
        if len(toks) <= max_tokens:
            return text
        return enc.decode(toks[:max_tokens])

    texts_to_embed = [truncate(s["text"], 6000) for s in sample]
    token_counts = [len(enc.encode(t)) for t in texts_to_embed]
    print(
        f"  token stats: total={sum(token_counts)} max={max(token_counts)} "
        f"median={int(np.median(token_counts))}"
    )

    all_vecs: list[list[float]] = []
    batch: list[str] = []
    batch_tokens = 0
    for t, n in zip(texts_to_embed, token_counts, strict=True):
        if batch_tokens + n > 250_000 and batch:
            all_vecs.extend(await embed_batch(batch))
            batch = []
            batch_tokens = 0
        batch.append(t)
        batch_tokens += n
    if batch:
        all_vecs.extend(await embed_batch(batch))
    vecs = np.array(all_vecs, dtype=np.float32)
    print(f"  vectors: {vecs.shape}")

    for mcs in (8, 15, 25):
        print(f"\n=== HDBSCAN min_cluster_size={mcs} (cosine) ===")
        labels = HDBSCAN(min_cluster_size=mcs, metric="cosine").fit_predict(vecs)
        sizes = Counter(labels)
        n_noise = sizes.pop(-1, 0)
        print(f"  noise: {n_noise} ({100 * n_noise / len(labels):.1f}%)")
        print(f"  clusters: {len(sizes)}")
        if not sizes:
            continue
        for cid, csize in sizes.most_common(10):
            idxs = [i for i, lab in enumerate(labels) if lab == cid][:3]
            primary_lang = Counter(sample[i]["primary_lang"] for i in idxs).most_common(1)[0][0]
            print(f"\n  CLUSTER {cid} (n={csize}, lang={primary_lang}):")
            for i in idxs:
                preview = sample[i]["text"].replace("\n", " | ")[:180]
                print(f"    [{sample[i]['start'].strftime('%Y-%m-%d')}] {preview}")


if __name__ == "__main__":
    asyncio.run(main())
