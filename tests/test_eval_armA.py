"""Arm A (production-realism) — leak-safe retrieval exclusion, guard, runner."""

from __future__ import annotations

import asyncio
import datetime as _dt
from types import SimpleNamespace
from unittest.mock import MagicMock

from qdrant_client.models import Filter, HasIdCondition

import persona_rag.retrieval as retr
from persona_rag.index.qdrant_store import search_dense


# --- Task 1: dense exclude_ids -------------------------------------------------
def test_search_dense_excludes_ids_via_must_not() -> None:
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(points=[])
    search_dense(client, "c", [0.1, 0.2], top_k=4, exclude_ids={"gold-id"})
    flt = client.query_points.call_args.kwargs["query_filter"]
    assert isinstance(flt, Filter)
    assert flt.must_not is not None
    hid = flt.must_not[0]
    assert isinstance(hid, HasIdCondition)
    assert "gold-id" in list(hid.has_id)


# --- Task 2: bm25 exclude_ids --------------------------------------------------
def test_retrieve_bm25_excludes_ids_before_topk(monkeypatch) -> None:
    import persona_rag.retrieval.bm25 as bm25mod

    monkeypatch.setattr(bm25mod.Path, "exists", lambda self: True)
    monkeypatch.setattr(bm25mod, "load", lambda p: (object(), ["a", "b", "c"]))
    monkeypatch.setattr(bm25mod, "score_bm25", lambda b, q: [0.9, 0.5, 0.4])

    class _Row:
        def __init__(self, _id: str) -> None:
            self.id = _id
            self.your_reply = "r"
            self.incoming_context_json = '["x"]'
            self.channel = "telegram"
            self.chat_id_hash = "h"
            self.recipient_id_hash = "h"
            self.timestamp = _dt.datetime(2026, 1, 1)
            self.language = "en"
            self.your_reply_len_chars = 1
            self.your_reply_emoji_count = 0
            self.eval_split = False

    class _Sess:
        def __enter__(self) -> _Sess:
            return self

        def __exit__(self, *a: object) -> bool:
            return False

        def exec(self, *_a: object) -> object:
            return SimpleNamespace(all=lambda: [_Row("b"), _Row("c")])

    monkeypatch.setattr(bm25mod, "Session", lambda e: _Sess())
    monkeypatch.setattr(bm25mod, "make_engine", lambda: None)

    out = bm25mod.retrieve_bm25("q", top_k=2, exclude_ids={"a"})
    ids = [r.turn.id for r in out]
    assert "a" not in ids
    assert ids == ["b", "c"]  # excluded id did not shrink the result below top_k


# --- Task 3: retrieve() fans exclude_ids to both -------------------------------
def test_retrieve_forwards_exclude_ids_to_both(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def fake_dense(client, query, *, top_k, language=None, exclude_ids=None):
        seen["dense"] = exclude_ids
        return []

    def fake_bm25(query, *, top_k, exclude_ids=None):
        seen["bm25"] = exclude_ids
        return []

    monkeypatch.setattr(retr, "retrieve_dense", fake_dense)
    monkeypatch.setattr(retr, "retrieve_bm25", fake_bm25)
    monkeypatch.setattr(retr, "fuse_scores", lambda d, b, alpha=None, top_k=0: [])
    monkeypatch.setattr(retr, "mmr_rerank", lambda pool, k, lambda_param=0.5: [])

    asyncio.run(retr.retrieve("q", client=object(), exclude_ids={"gold"}))
    assert seen["dense"] == {"gold"}
    assert seen["bm25"] == {"gold"}


# --- Task 4: _gen_all forwards logit_bias --------------------------------------
def test_gen_all_forwards_logit_bias() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path("scripts").resolve()))
    import compare_persona as cp

    captured: dict[str, object] = {}

    class _Client:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs: object) -> object:
                    captured.update(kwargs)
                    return SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="ok"),
                                finish_reason="stop",
                            )
                        ],
                        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                    )

    asyncio.run(
        cp._gen_all(
            _Client(),
            "m",
            [[{"role": "user", "content": "hi"}]],
            temperature=0.8,
            max_tokens=10,
            concurrency=1,
            logit_bias={123: 2},
        )
    )
    assert captured.get("logit_bias") == {123: 2}
