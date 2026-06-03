"""Arm A (production-realism) — leak-safe retrieval exclusion, guard, runner."""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client.models import Filter, HasIdCondition

import persona_rag.retrieval as retr
from persona_rag.config import get_settings
from persona_rag.eval.compare import LeakError, leak_guard
from persona_rag.index.qdrant_store import search_dense


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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


# --- Task 5: two-leg leak_guard ------------------------------------------------
def _rt(_id: str, reply: str, ctx: list[str], score: float) -> SimpleNamespace:
    return SimpleNamespace(
        turn=SimpleNamespace(id=_id, your_reply=reply, incoming_context=ctx),
        score=score,
    )


def test_leak_guard_id_leg_raises() -> None:
    retrieved = [_rt("gold", "anything", ["c"], 0.9)]
    try:
        leak_guard(gold_turn_id="gold", gold_reply="x", gold_ctx=["c"], retrieved=retrieved)
        raise AssertionError("expected LeakError")
    except LeakError:
        pass


def test_leak_guard_exact_text_same_context_raises() -> None:
    retrieved = [_rt("other", "да", ["ctx line"], 0.8)]
    try:
        leak_guard(gold_turn_id="gold", gold_reply="да", gold_ctx=["ctx line"], retrieved=retrieved)
        raise AssertionError("expected LeakError")
    except LeakError:
        pass


def test_leak_guard_exact_text_diff_context_counts_not_raises() -> None:
    retrieved = [_rt("other", "да", ["unrelated"], 0.7)]
    out = leak_guard(
        gold_turn_id="gold", gold_reply="да", gold_ctx=["the real ctx"], retrieved=retrieved
    )
    assert out["exact_text_dup_diff_context"] == 1
    assert out["top_sim"] == 0.7


def test_leak_guard_clean_returns_top_sim() -> None:
    retrieved = [_rt("o1", "hello", ["a"], 0.6), _rt("o2", "world", ["b"], 0.3)]
    out = leak_guard(gold_turn_id="gold", gold_reply="да", gold_ctx=["c"], retrieved=retrieved)
    assert out["exact_text_dup_diff_context"] == 0
    assert out["top_sim"] == 0.6


def test_leak_guard_non_strict_counts_id_leak() -> None:
    retrieved = [_rt("gold", "да", ["c"], 0.9)]
    out = leak_guard(
        gold_turn_id="gold", gold_reply="да", gold_ctx=["c"], retrieved=retrieved, strict=False
    )
    assert out["id_leak"] == 1


# --- Task 6: arm-A runner ------------------------------------------------------
def test_armA_load_holdout_keeps_ids(monkeypatch) -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path("scripts").resolve()))
    import compare_persona_armA as a

    class _Row:
        def __init__(self, _id: str, reply: str, ctx: list[str]) -> None:
            self.id = _id
            self.your_reply = reply
            self.incoming_context_json = json.dumps(ctx)
            self.recipient_id_hash = "rh"
            self.language = "en"
            self.timestamp = _dt.datetime(2026, 1, 1)

    rows = [_Row("keep-me", "a real reply here", ["hi", "how are you"])]

    class _Sess:
        def __enter__(self) -> _Sess:
            return self

        def __exit__(self, *x: object) -> bool:
            return False

        def exec(self, *_a: object) -> object:
            return SimpleNamespace(all=lambda: rows)

    monkeypatch.setattr(a, "Session", lambda e: _Sess())
    monkeypatch.setattr(a, "make_engine", lambda: None)
    monkeypatch.setattr(a, "eval_split_for", lambda _id, frac=0.1: True)

    items = a.load_holdout(min_reply_chars=3)
    assert items[0].turn_id == "keep-me"
    assert items[0].ctx == ["hi", "how are you"]


def test_armA_builds_rich_prompt_not_thin(monkeypatch) -> None:
    import sys
    from pathlib import Path

    monkeypatch.setenv("GENERATION_BACKEND", "openai")
    monkeypatch.setenv("INSIGHTS_USE_GENERATED_PERSONA_DESCRIPTION", "false")
    get_settings.cache_clear()

    sys.path.insert(0, str(Path("scripts").resolve()))
    import compare_persona_armA as a

    msgs = a.build_api_messages(ctx=["hi", "ok so"], retrieved=[], insights=None)
    sys_text = next(m["content"] for m in msgs if m["role"] == "system")
    # Rich SYSTEM_TEMPLATE markers (absent from the one-line THIN_SYSTEM).
    assert "You are texting" in sys_text
    assert len(sys_text) > 200
