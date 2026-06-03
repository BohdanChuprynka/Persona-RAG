# Arm A (production-realism comparison) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leak-safe production-realism comparison ("arm A") of the shipped OpenAI product (rich prompt + retrieval + decode levers) vs the LoRA, scored on the recipient-stratified hold-out.

**Architecture:** Add a per-item `exclude_ids` filter threaded through the live retrieval stack (default `None` = zero prod change), a two-leg leak guard in `eval/compare.py`, a `logit_bias` forward in `_gen_all`, and a sibling runner `scripts/compare_persona_armA.py` that reuses production code paths verbatim (`retrieve`, `build_messages`, `retrieve_insights`) plus arm B's scorer/scaffold.

**Tech Stack:** Python 3.12, uv, pytest, mypy/ruff (strict on `persona_rag`), qdrant-client, openai (Async), sqlmodel.

**Spec:** `docs/superpowers/specs/2026-06-02-arm-a-production-realism-design.md`

**Refinement vs spec:** exclusion is **per-item `{turn_id}`** (honors §4's "remove only the answer key" + most production-faithful; production keeps every *other* past turn), not the whole hold-out. Item-set comparison is **corpus-level** (the spec's approved fallback — both arms on the same recipient-stratified hold-out; within-item text-join alignment is a fragile nice-to-have, deferred).

---

## File Structure

- **Modify (additive, default-`None`, zero prod-behaviour change):**
  - `persona_rag/index/qdrant_store.py` — `search_dense` gains `exclude_ids` -> `must_not=[HasIdCondition]`.
  - `persona_rag/retrieval/dense.py` — `retrieve_dense` forwards `exclude_ids`.
  - `persona_rag/retrieval/bm25.py` — `retrieve_bm25` drops `exclude_ids` before `[:top_k]`.
  - `persona_rag/retrieval/__init__.py` — `retrieve` forwards `exclude_ids` to both.
  - `scripts/compare_persona.py` — `_gen_all` gains `logit_bias`.
- **New:**
  - `persona_rag/eval/compare.py` — add `LeakError` + `leak_guard` (pure, testable).
  - `scripts/compare_persona_armA.py` — the runner.
  - `tests/test_eval_armA.py` — exclusion + guard + `_gen_all` + prompt-assembly tests.
- **Outputs (git-ignored):** `data/eval/compare/armA*/`.
- **Makefile:** `compare-arma`.

---

### Task 1: `exclude_ids` on the dense path

**Files:**
- Modify: `persona_rag/index/qdrant_store.py:106-120` (`search_dense`)
- Modify: `persona_rag/retrieval/dense.py:11-25` (`retrieve_dense`)
- Test: `tests/test_eval_armA.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_armA.py
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from qdrant_client.models import Filter, HasIdCondition

from persona_rag.index.qdrant_store import search_dense


def test_search_dense_excludes_ids_via_must_not():
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(points=[])
    search_dense(client, "c", [0.1, 0.2], top_k=4, exclude_ids={"gold-id"})
    flt = client.query_points.call_args.kwargs["query_filter"]
    assert isinstance(flt, Filter)
    assert flt.must_not is not None
    hid = flt.must_not[0]
    assert isinstance(hid, HasIdCondition)
    assert "gold-id" in list(hid.has_id)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_armA.py::test_search_dense_excludes_ids_via_must_not -v`
Expected: FAIL — `search_dense() got an unexpected keyword argument 'exclude_ids'`.

- [ ] **Step 3: Implement `exclude_ids` in `search_dense`**

In `persona_rag/index/qdrant_store.py`, change the signature and the filter build:

```python
def search_dense(
    client: QdrantClient,
    collection: str,
    vector: list[float],
    *,
    top_k: int,
    language: str | None = None,
    exclude_eval: bool = True,
    exclude_ids: set[str] | None = None,
) -> list[RetrievedTurn]:
    conditions: list[_Condition] = []
    if exclude_eval:
        conditions.append(FieldCondition(key="eval_split", match=MatchValue(value=False)))
    if language:
        conditions.append(FieldCondition(key="language", match=MatchValue(value=language)))
    must_not: list[_Condition] = []
    if exclude_ids:
        must_not.append(HasIdCondition(has_id=sorted(exclude_ids)))
    flt = (
        Filter(must=conditions or None, must_not=must_not or None)
        if (conditions or must_not)
        else None
    )
    response = client.query_points(
        collection_name=collection,
        query=vector,
        limit=top_k,
        query_filter=flt,
        with_payload=True,
        with_vectors=True,
    )
    # ... (unchanged body below)
```

(`HasIdCondition` is already imported at `qdrant_store.py:11`; `_Condition` union already includes it at line 42. Persona-turn Qdrant point id IS `turn.id`, so `HasIdCondition` matches the gold directly.)

Then forward from `retrieve_dense` in `persona_rag/retrieval/dense.py`:

```python
async def retrieve_dense(
    client: QdrantClient,
    query: str,
    *,
    top_k: int,
    language: str | None = None,
    exclude_ids: set[str] | None = None,
) -> list[RetrievedTurn]:
    vec = (await embed_batch([query]))[0]
    return search_dense(
        client,
        get_settings().QDRANT_COLLECTION,
        vec,
        top_k=top_k,
        language=language,
        exclude_ids=exclude_ids,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_armA.py::test_search_dense_excludes_ids_via_must_not -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona_rag/index/qdrant_store.py persona_rag/retrieval/dense.py tests/test_eval_armA.py
git commit -m "feat(eval): exclude_ids on the dense retrieval path (arm A leak guard)"
```

---

### Task 2: `exclude_ids` on the BM25 path

**Files:**
- Modify: `persona_rag/retrieval/bm25.py:14-22` (`retrieve_bm25`)
- Test: `tests/test_eval_armA.py`

- [ ] **Step 1: Write the failing test**

```python
def test_retrieve_bm25_excludes_ids_before_topk(monkeypatch, tmp_path):
    import pickle
    import persona_rag.retrieval.bm25 as bm25mod

    # Three candidates; excluding the top-scoring id must NOT consume a top_k slot.
    monkeypatch.setattr(bm25mod.Path, "exists", lambda self: True)
    monkeypatch.setattr(bm25mod, "load", lambda p: (object(), ["a", "b", "c"]))
    monkeypatch.setattr(bm25mod, "score_bm25", lambda b, q: [0.9, 0.5, 0.4])

    class _Row:
        def __init__(self, _id):
            self.id = _id
            self.your_reply = "r"
            self.incoming_context_json = '["x"]'
            self.channel = "telegram"
            self.chat_id_hash = "h"
            self.recipient_id_hash = "h"
            self.timestamp = __import__("datetime").datetime(2026, 1, 1)
            self.language = "en"
            self.your_reply_len_chars = 1
            self.your_reply_emoji_count = 0
            self.eval_split = False

    class _Sess:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def exec(self, *_a):
            return SimpleNamespace(all=lambda: [_Row("b"), _Row("c")])

    monkeypatch.setattr(bm25mod, "Session", lambda e: _Sess())
    monkeypatch.setattr(bm25mod, "make_engine", lambda: None)

    out = bm25mod.retrieve_bm25("q", top_k=2, exclude_ids={"a"})
    ids = [r.turn.id for r in out]
    assert "a" not in ids
    assert ids == ["b", "c"]  # excluded id did not shrink the result below top_k
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_armA.py::test_retrieve_bm25_excludes_ids_before_topk -v`
Expected: FAIL — `retrieve_bm25() got an unexpected keyword argument 'exclude_ids'`.

- [ ] **Step 3: Implement `exclude_ids` in `retrieve_bm25`**

In `persona_rag/retrieval/bm25.py`, change the signature and the slice (lines 14-23):

```python
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
        ranked = [p for p in ranked if p[0] not in exclude_ids]
    pairs = ranked[:top_k]
    if not pairs:
        return []
    # ... (unchanged body below)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_armA.py::test_retrieve_bm25_excludes_ids_before_topk -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona_rag/retrieval/bm25.py tests/test_eval_armA.py
git commit -m "feat(eval): exclude_ids on the BM25 retrieval path (pre-top_k drop)"
```

---

### Task 3: thread `exclude_ids` through `retrieve()`

**Files:**
- Modify: `persona_rag/retrieval/__init__.py:14-33` (`retrieve`)
- Test: `tests/test_eval_armA.py`

- [ ] **Step 1: Write the failing test**

```python
import asyncio
import persona_rag.retrieval as retr


def test_retrieve_forwards_exclude_ids_to_both(monkeypatch):
    seen = {}

    async def fake_dense(client, query, *, top_k, language=None, exclude_ids=None):
        seen["dense"] = exclude_ids
        return []

    def fake_bm25(query, *, top_k, exclude_ids=None):
        seen["bm25"] = exclude_ids
        return []

    monkeypatch.setattr(retr, "retrieve_dense", fake_dense)
    monkeypatch.setattr(retr, "retrieve_bm25", fake_bm25)
    monkeypatch.setattr(retr, "fuse_scores", lambda d, b, alpha=None, top_k=0: [])

    asyncio.run(retr.retrieve("q", client=object(), exclude_ids={"gold"}))
    assert seen["dense"] == {"gold"}
    assert seen["bm25"] == {"gold"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_armA.py::test_retrieve_forwards_exclude_ids_to_both -v`
Expected: FAIL — unexpected keyword argument `exclude_ids`.

- [ ] **Step 3: Implement forwarding in `retrieve()`**

In `persona_rag/retrieval/__init__.py`, add the param and pass it to both retrievers (lines 14-32):

```python
async def retrieve(
    query: str,
    *,
    client: QdrantClient,
    language: str | None = None,
    top_k: int | None = None,
    alpha: float | None = None,
    exclude_ids: set[str] | None = None,
) -> list[RetrievedTurn]:
    s = get_settings()
    k = top_k or s.TOP_K
    pool = s.MMR_POOL_SIZE if s.MMR_ENABLED else k * 4
    dense = await retrieve_dense(client, query, top_k=pool, language=language, exclude_ids=exclude_ids)
    bm25 = retrieve_bm25(query, top_k=pool, exclude_ids=exclude_ids)
    fused = fuse_scores(dense, bm25, alpha=alpha, top_k=pool)
    # ... (unchanged body below)
```

- [ ] **Step 4: Run test to verify it passes + full retrieval suite still green**

Run: `uv run pytest tests/test_eval_armA.py -k exclude -v && uv run pytest tests -k retriev -q`
Expected: PASS; no regressions (default `exclude_ids=None` keeps prod behaviour).

- [ ] **Step 5: Commit**

```bash
git add persona_rag/retrieval/__init__.py tests/test_eval_armA.py
git commit -m "feat(eval): thread exclude_ids through retrieve() (dense+sparse fan-out)"
```

---

### Task 4: `_gen_all` forwards `logit_bias` to OpenAI

**Files:**
- Modify: `scripts/compare_persona.py:80-118` (`_gen_all`)
- Test: `tests/test_eval_armA.py`

- [ ] **Step 1: Write the failing test**

```python
def test_gen_all_forwards_logit_bias(monkeypatch):
    import asyncio
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path("scripts").resolve()))
    import compare_persona as cp

    captured = {}

    class _Client:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    captured.update(kwargs)
                    return SimpleNamespace(
                        choices=[SimpleNamespace(
                            message=SimpleNamespace(content="ok"),
                            finish_reason="stop")],
                        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                    )

    asyncio.run(cp._gen_all(
        _Client(), "m", [[{"role": "user", "content": "hi"}]],
        temperature=0.8, max_tokens=10, concurrency=1, logit_bias={123: 2},
    ))
    assert captured.get("logit_bias") == {123: 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_armA.py::test_gen_all_forwards_logit_bias -v`
Expected: FAIL — unexpected keyword argument `logit_bias`.

- [ ] **Step 3: Implement `logit_bias` forwarding**

In `scripts/compare_persona.py`, add the kwarg and pass it conditionally (so `None` omits it):

```python
async def _gen_all(
    client: AsyncOpenAI,
    model: str,
    messages_list: list[list[dict[str, str]]],
    *,
    temperature: float,
    max_tokens: int,
    concurrency: int,
    logit_bias: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)

    async def one(messages: list[dict[str, str]]) -> dict[str, Any]:
        async with sem:
            t0 = time.perf_counter()
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": cast(Any, messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if logit_bias:
                kwargs["logit_bias"] = logit_bias
            try:
                r = await client.chat.completions.create(**kwargs)
                # ... (unchanged result/except body)
```

- [ ] **Step 4: Run test to verify it passes + arm B unaffected**

Run: `uv run pytest tests/test_eval_armA.py::test_gen_all_forwards_logit_bias -v`
Expected: PASS (default `logit_bias=None` -> kwarg omitted -> arm B byte-identical).

- [ ] **Step 5: Commit**

```bash
git add scripts/compare_persona.py tests/test_eval_armA.py
git commit -m "feat(eval): _gen_all forwards logit_bias to the API (was a silent no-op)"
```

---

### Task 5: two-leg `leak_guard` + `LeakError`

**Files:**
- Modify: `persona_rag/eval/compare.py` (add near `_norm`, ~line 49)
- Test: `tests/test_eval_armA.py`

- [ ] **Step 1: Write the failing tests**

```python
from persona_rag.eval.compare import LeakError, leak_guard


def _rt(_id, reply, ctx, score):
    return SimpleNamespace(
        turn=SimpleNamespace(id=_id, your_reply=reply, incoming_context=ctx),
        score=score,
    )


def test_leak_guard_id_leg_raises():
    retrieved = [_rt("gold", "anything", ["c"], 0.9)]
    try:
        leak_guard(gold_turn_id="gold", gold_reply="x", gold_ctx=["c"], retrieved=retrieved)
        raise AssertionError("expected LeakError")
    except LeakError:
        pass


def test_leak_guard_exact_text_same_context_raises():
    retrieved = [_rt("other", "да", ["ctx line"], 0.8)]
    try:
        leak_guard(gold_turn_id="gold", gold_reply="да", gold_ctx=["ctx line"], retrieved=retrieved)
        raise AssertionError("expected LeakError")
    except LeakError:
        pass


def test_leak_guard_exact_text_diff_context_counts_not_raises():
    retrieved = [_rt("other", "да", ["unrelated"], 0.7)]
    out = leak_guard(gold_turn_id="gold", gold_reply="да", gold_ctx=["the real ctx"], retrieved=retrieved)
    assert out["exact_text_dup_diff_context"] == 1
    assert out["top_sim"] == 0.7


def test_leak_guard_clean_returns_top_sim():
    retrieved = [_rt("o1", "hello", ["a"], 0.6), _rt("o2", "world", ["b"], 0.3)]
    out = leak_guard(gold_turn_id="gold", gold_reply="да", gold_ctx=["c"], retrieved=retrieved)
    assert out["exact_text_dup_diff_context"] == 0
    assert out["top_sim"] == 0.6


def test_leak_guard_non_strict_counts_id_leak():
    retrieved = [_rt("gold", "да", ["c"], 0.9)]
    out = leak_guard(gold_turn_id="gold", gold_reply="да", gold_ctx=["c"], retrieved=retrieved, strict=False)
    assert out["id_leak"] == 1  # leak-ON validation: count, don't raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval_armA.py -k leak_guard -v`
Expected: FAIL — `cannot import name 'LeakError'`.

- [ ] **Step 3: Implement `LeakError` + `leak_guard`**

In `persona_rag/eval/compare.py`, after `_norm` (line 49), add:

```python
class LeakError(RuntimeError):
    """Raised when retrieval surfaces the gold answer-key for the item being scored."""


def leak_guard(
    *,
    gold_turn_id: str,
    gold_reply: str,
    gold_ctx: list[str],
    retrieved: list[Any],
    strict: bool = True,
) -> dict[str, Any]:
    """Two-leg per-item leak guard for arm A (API arm only).

    ID leg: the gold turn itself was retrieved -> a true answer-key leak.
    Exact-text leg: the gold reply text appears under a DIFFERENT turn id; this
    is a leak ONLY if that turn shares the gold's incoming context (a re-minted
    duplicate of THIS turn) -- a generic short reply ('ок') elsewhere is a fair
    neighbour and is only counted. ``strict=False`` (leak-ON validation) counts
    every leak instead of raising. Returns per-item telemetry incl. ``top_sim``.
    """
    ids = {r.turn.id for r in retrieved}
    gold_norm = _norm(gold_reply)
    gold_ctx_norm = _norm("\n".join(c for c in gold_ctx if c and c.strip()))
    id_leak = 1 if gold_turn_id in ids else 0
    if id_leak and strict:
        raise LeakError(f"gold turn_id retrieved: {gold_turn_id}")
    same_ctx_dup = 0
    diff_ctx_dup = 0
    for r in retrieved:
        if _norm(r.turn.your_reply) != gold_norm:
            continue
        r_ctx_norm = _norm("\n".join(c for c in r.turn.incoming_context if c and c.strip()))
        if r_ctx_norm == gold_ctx_norm:
            same_ctx_dup += 1
        else:
            diff_ctx_dup += 1
    if same_ctx_dup and strict:
        raise LeakError(f"exact gold text under a re-minted id w/ same context (gold={gold_turn_id})")
    top_sim = max((r.score for r in retrieved), default=float("nan"))
    return {
        "top_sim": top_sim,
        "id_leak": id_leak,
        "exact_text_dup_same_context": same_ctx_dup,
        "exact_text_dup_diff_context": diff_ctx_dup,
        "n_retrieved": len(retrieved),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval_armA.py -k leak_guard -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add persona_rag/eval/compare.py tests/test_eval_armA.py
git commit -m "feat(eval): two-leg leak_guard + LeakError (id hard-fail; conditional exact-text)"
```

---

### Task 6: the runner `scripts/compare_persona_armA.py`

**Files:**
- Create: `scripts/compare_persona_armA.py`
- Test: `tests/test_eval_armA.py`

- [ ] **Step 1: Write the failing test (DB loader keeps turn-ids; prompt is rich)**

```python
def test_armA_load_holdout_keeps_ids(monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path("scripts").resolve()))
    import compare_persona_armA as a

    class _Row:
        def __init__(self, _id, reply, ctx):
            self.id = _id
            self.your_reply = reply
            self.incoming_context_json = __import__("json").dumps(ctx)
            self.recipient_id_hash = "rh"
            self.language = "en"
            self.timestamp = __import__("datetime").datetime(2026, 1, 1)
    rows = [_Row("keep-me", "a real reply here", ["hi", "how are you"])]

    class _Sess:
        def __enter__(self): return self
        def __exit__(self, *x): return False
        def exec(self, *_a): return SimpleNamespace(all=lambda: rows)
    monkeypatch.setattr(a, "Session", lambda e: _Sess())
    monkeypatch.setattr(a, "make_engine", lambda: None)
    monkeypatch.setattr(a, "eval_split_for", lambda _id, frac=0.1: True)

    items = a.load_holdout(min_reply_chars=3)
    assert items[0].turn_id == "keep-me"
    assert items[0].ctx == ["hi", "how are you"]


def test_armA_builds_rich_prompt_not_thin(monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path("scripts").resolve()))
    import compare_persona_armA as a
    # build_api_messages must produce the rich SYSTEM_TEMPLATE, not THIN_SYSTEM.
    msgs = a.build_api_messages(ctx=["hi", "ok so"], retrieved=[], insights=None)
    sys_text = msgs[0]["content"]
    assert "Богдан" in sys_text or "Bohdan" in sys_text
    assert len(sys_text) > 200  # rich template, not the one-line THIN_SYSTEM
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval_armA.py -k armA -v`
Expected: FAIL — `No module named 'compare_persona_armA'`.

- [ ] **Step 3: Implement the runner**

Create `scripts/compare_persona_armA.py`:

```python
"""Arm A: production-realism comparison (shipped API vs LoRA) on the
recipient-stratified hold-out, with per-item retrieval leak exclusion + guard.

Prereqs: Qdrant up (make up) + index built (make ingest); llama-server serving
the LoRA on OLLAMA_BASE_URL. See docs/superpowers/specs/2026-06-02-arm-a-*.

    uv run python scripts/compare_persona_armA.py --n 300 --seed 0 --name armA
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from compare_persona import (  # noqa: E402  (sibling-script reuse, keeps arm B pristine)
    API_PRICE_IN,
    API_PRICE_OUT,
    _gen_all,
    _latency_cost,
    _print_summary,
)

from persona_rag._logging import configure_logging, get_logger  # noqa: E402
from persona_rag.config import get_settings  # noqa: E402
from persona_rag.db.engine import make_engine  # noqa: E402
from persona_rag.db.models import PersonaTurnRow  # noqa: E402
from persona_rag.eval.compare import (  # noqa: E402
    LeakError,
    compare_scorecard,
    language_bucket,
    leak_guard,
)
from persona_rag.finetune.dataset import clean_reply, eval_split_for  # noqa: E402
from persona_rag.generate.llm_client import voice_logit_bias  # noqa: E402
from persona_rag.generate.prompt import build_messages, build_thin_messages  # noqa: E402
from persona_rag.graph.nodes.build_prompt import _load_anchors  # noqa: E402
from persona_rag.graph.nodes.retrieve_insights import retrieve_insights  # noqa: E402
from persona_rag.index.qdrant_store import make_client  # noqa: E402
from persona_rag.models import ChatMessage  # noqa: E402
from persona_rag.retrieval import retrieve  # noqa: E402
from sqlmodel import Session, select  # noqa: E402

log = get_logger()
OUT_ROOT = Path("data/eval/compare")


@dataclass
class Item:
    turn_id: str
    recipient_id_hash: str
    ctx: list[str]
    reply: str


def load_holdout(*, min_reply_chars: int = 1, eval_frac: float = 0.1) -> list[Item]:
    """The recipient-stratified hold-out, mirroring finetune.dataset.iter_records'
    filters (clean_reply, min_reply_chars, non-empty ctx) but KEEPING turn-ids."""
    with Session(make_engine()) as s:
        rows = list(s.exec(select(PersonaTurnRow)).all())
    out: list[Item] = []
    for r in rows:
        if not eval_split_for(r.id, eval_frac):
            continue
        reply = clean_reply((r.your_reply or "").strip())
        if reply is None or len(reply) < min_reply_chars:
            continue
        ctx = json.loads(r.incoming_context_json)
        if not any(c.strip() for c in ctx):
            continue
        out.append(Item(r.id, r.recipient_id_hash, ctx, reply))
    out.sort(key=lambda it: it.turn_id)  # deterministic order
    return out


def build_api_messages(
    *, ctx: list[str], retrieved: list[Any], insights: dict[str, Any] | None
) -> list[dict[str, str]]:
    """The SHIPPED rich prompt, assembled exactly as build_prompt_node does:
    incoming = ctx[-1]; session = ctx[:-1] as user turns (mirrors eval_persona
    _seed_context)."""
    s = get_settings()
    session = [ChatMessage(role="user", content=c) for c in ctx[:-1] if c.strip()]
    return build_messages(
        persona_name=s.PERSONA_NAME,
        persona_description=s.PERSONA_DESCRIPTION,
        style_anchors=_load_anchors(),
        user_memory="",
        retrieved=retrieved,
        session=session,
        incoming=ctx[-1],
        insights=insights,
    )


async def _assemble_api(items: list[Item], *, strict: bool, leak_on: bool) -> tuple[list, list]:
    """Per item: retrieve (excluding own id unless leak_on) -> leak_guard -> insights
    -> rich messages. Returns (messages_list, per_item_guard_telemetry)."""
    client = make_client()
    msgs_list: list[list[dict[str, str]]] = []
    guard_rows: list[dict[str, Any]] = []
    for it in items:
        q = it.ctx[-1]
        exclude = None if leak_on else {it.turn_id}
        retrieved = await retrieve(q, client=client, exclude_ids=exclude)
        guard = leak_guard(
            gold_turn_id=it.turn_id,
            gold_reply=it.reply,
            gold_ctx=it.ctx,
            retrieved=retrieved,
            strict=strict,
        )
        guard_rows.append(guard)
        state: dict[str, Any] = {"incoming": q}
        await retrieve_insights(state)  # self-wraps in try/except -> empty
        msgs = build_api_messages(ctx=it.ctx, retrieved=retrieved, insights=state.get("insights"))
        assert any(
            len(m["content"]) > 200 for m in msgs if m["role"] == "system"
        ), "API arm produced a thin prompt -- GENERATION_BACKEND not 'openai'?"
        msgs_list.append(msgs)
    return msgs_list, guard_rows


async def run(
    *, name: str, n: int, seed: int, temperature: float, max_tokens: int,
    n_boot: int, learned: bool, leak_on: bool,
) -> None:
    # Pin the API backend so build_messages takes the rich branch + voice_logit_bias resolves.
    os.environ["GENERATION_BACKEND"] = "openai"
    if learned:
        os.environ["PAREN_LOGIT_BIAS"] = "0"
        os.environ["EXCLAIM_LOGIT_BIAS"] = "0"
    get_settings.cache_clear()
    s = get_settings()
    assert s.GENERATION_BACKEND == "openai"

    holdout = load_holdout()
    rng = random.Random(seed)
    items = rng.sample(holdout, min(n, len(holdout)))
    log.info("loaded", holdout=len(holdout), sampled=len(items))

    # Train replies for the copy/leak baseline (reuse arm B's train split).
    train_replies = [
        json.loads(line)["conversations"][-1]["value"]
        for line in Path("data/finetune/train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    api_msgs, guard_rows = await _assemble_api(items, strict=not leak_on, leak_on=leak_on)
    lora_msgs = [
        build_thin_messages(
            incoming=it.ctx[-1],
            session=[ChatMessage(role="user", content=c) for c in it.ctx[:-1] if c.strip()],
        )
        for it in items
    ]

    api = AsyncOpenAI(api_key=s.OPENAI_API_KEY)
    lora = AsyncOpenAI(base_url=s.OLLAMA_BASE_URL, api_key="local")
    bias = voice_logit_bias()  # resolves under GENERATION_BACKEND=openai

    log.info("generating", backend="api", model=s.OPENAI_CHAT_MODEL, bias=bias)
    api_res = await _gen_all(
        api, s.OPENAI_CHAT_MODEL, api_msgs,
        temperature=temperature, max_tokens=max_tokens, concurrency=8, logit_bias=bias,
    )
    log.info("generating", backend="lora", model=s.OLLAMA_MODEL)
    lora_res = await _gen_all(
        lora, s.OLLAMA_MODEL, lora_msgs,
        temperature=temperature, max_tokens=max_tokens, concurrency=4, logit_bias=None,
    )

    real = [it.reply for it in items]
    gen_api = [r["text"] for r in api_res]
    gen_lora = [r["text"] for r in lora_res]
    card = compare_scorecard(real, gen_api, gen_lora, train_replies=train_replies, n_boot=n_boot, seed=seed)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / (name or ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        {
            "item_id": i, "turn_id": items[i].turn_id, "incoming": items[i].ctx[-1],
            "real": real[i], "gen_api": gen_api[i], "gen_lora": gen_lora[i],
            "lang": language_bucket(real[i]), "top_sim": guard_rows[i]["top_sim"],
        }
        for i in range(len(items))
    ]
    (out_dir / "pairs.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8"
    )
    sims = [g["top_sim"] for g in guard_rows if g["top_sim"] == g["top_sim"]]  # drop nan
    results = {
        "name": name, "ts": ts,
        "arm": "A-production (shipped API rich+retrieval+levers vs LoRA thin)",
        "params": {
            "n": len(items), "seed": seed, "temperature": temperature, "max_tokens": max_tokens,
            "n_boot": n_boot, "api_model": s.OPENAI_CHAT_MODEL, "lora_model": s.OLLAMA_MODEL,
            "retrieval_query": "ctx[-1] (runtime-faithful)",
            "levers": {
                "paren_logit_bias": s.PAREN_LOGIT_BIAS, "exclaim_logit_bias": s.EXCLAIM_LOGIT_BIAS,
                "best_of_n": s.BEST_OF_N, "resolved_bias": bias, "pass": "learned" if learned else "shipped",
            },
            "leak_on": leak_on,
            "style_anchors_n_turns": _load_anchors().n_turns,
        },
        "retrieval_leak_guard": {
            "id_leaks": sum(g["id_leak"] for g in guard_rows),
            "exact_text_same_context": sum(g["exact_text_dup_same_context"] for g in guard_rows),
            "exact_text_diff_context": sum(g["exact_text_dup_diff_context"] for g in guard_rows),
            "top_sim_mean": round(sum(sims) / len(sims), 4) if sims else None,
            "top_sim_ge_0_9": sum(1 for x in sims if x >= 0.9),
        },
        "scorecard": card,
        "operational": {
            "api": _latency_cost(api_res, priced=True),
            "lora": _latency_cost(lora_res, priced=False),
        },
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _print_summary(results)
    log.info("wrote", dir=str(out_dir), leak_guard=results["retrieval_leak_guard"])


def main() -> None:
    configure_logging()
    p = argparse.ArgumentParser(description="Arm A: production-realism comparison.")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--name", type=str, default="armA")
    p.add_argument("--learned", action="store_true", help="force levers 0/0 (isolate learned tics)")
    p.add_argument("--leak-on", action="store_true", help="DISABLE exclusion (validation: measure the leak)")
    a = p.parse_args()
    asyncio.run(run(
        name=a.name, n=a.n, seed=a.seed, temperature=a.temp, max_tokens=a.max_tokens,
        n_boot=a.n_boot, learned=a.learned, leak_on=a.leak_on,
    ))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + lint/type gates**

Run: `uv run pytest tests/test_eval_armA.py -v && uv run ruff check persona_rag tests scripts && uv run mypy persona_rag`
Expected: all PASS. (mypy only checks `persona_rag`, so the script's sibling-import is fine; ensure `persona_rag/eval/compare.py` types clean.)

- [ ] **Step 5: Commit**

```bash
git add scripts/compare_persona_armA.py tests/test_eval_armA.py
git commit -m "feat(eval): arm-A runner (rich API vs LoRA, per-item leak exclusion + guard)"
```

---

### Task 7: Makefile target + leak-OFF/ON validation + real run

**Files:**
- Modify: `Makefile` (add `compare-arma` to `.PHONY` + a target)
- No new tests (this is the live validation run).

- [ ] **Step 1: Add the Makefile target**

Add `compare-arma` to the `.PHONY` line and append:

```make
# Arm A: production-realism (shipped API rich+retrieval vs LoRA). Needs Qdrant up
# + index built + llama-server serving the LoRA. See docs/superpowers/specs.
compare-arma:
	uv run python scripts/compare_persona_armA.py --n 300 --seed 0 --name armA
```

- [ ] **Step 2: Confirm prerequisites are up**

Run: `curl -s localhost:6333/healthz >/dev/null && echo "qdrant up" || echo "run: make up"` and confirm `data/bm25.pkl` + `data/style_anchors.json` exist and `llama-server` answers on `OLLAMA_BASE_URL`.
Expected: qdrant up; index artifacts present. If not: `make up` then `make ingest`.

- [ ] **Step 3: Leak-OFF vs leak-ON validation (small n, proves the guard)**

```bash
uv run python scripts/compare_persona_armA.py --n 60 --seed 0 --name armA_leakoff
uv run python scripts/compare_persona_armA.py --n 60 --seed 0 --name armA_leakon --leak-on
```
Expected: `armA_leakoff/results.json` `retrieval_leak_guard.id_leaks == 0` (exclusion works; the run completes without `LeakError`). `armA_leakon` shows `id_leaks > 0` and a visibly better (fake) API copy-rate / lower `len_wasserstein` — the leak quantified. Record the delta.

- [ ] **Step 4: The real headline run (shipped levers) + learned diagnostic**

```bash
make compare-arma                                                   # shipped levers (2/-5)
uv run python scripts/compare_persona_armA.py --n 300 --seed 0 --name armA_learned --learned
```
Expected: `data/eval/compare/armA/results.json` with `id_leaks==0`, scorecard populated, `usd_total < 0.30`. Eyeball the `len_wasserstein` / `exclaim_rate` deltas vs arm B (`data/eval/compare/main/`).

- [ ] **Step 5: Commit the Makefile + findings note**

```bash
git add Makefile
git commit -m "chore(eval): make compare-arma target for the production-realism run"
```

(Findings doc + memory update happen after the run, in a follow-up commit.)

---

## Self-Review

**Spec coverage:**
- §3 leak guard (dense/sparse/exact-text/assertion) -> Tasks 1-3, 5. ✓
- §4 decisions (exclude_ids mechanism, sibling runner, ctx[-1] query, shipped-lever headline) -> Tasks 1-3, 6; ctx[-1] in `build_api_messages`/`_assemble_api`; levers in `run()`. ✓ (item-set = corpus-level, the approved fallback — noted at top.)
- §5.4 `_gen_all` logit_bias -> Task 4. ✓
- §5.5 fidelity (get_settings, generated persona inside build_messages, 2 embeds, anchors provenance) -> Task 6. ✓
- §7 leak-ON/OFF validation -> Task 7 Step 3. ✓
- §8 tests (exclusion, guard, _gen_all, rich-prompt) -> Tasks 1-6. ✓
- §9 cost/knobs (--learned, --leak-on, --n) -> Task 6 `main()`. ✓

**Placeholder scan:** no TBD/TODO; every code step has real code; commands have expected output. ✓

**Type/name consistency:** `exclude_ids: set[str] | None` consistent across `search_dense`/`retrieve_dense`/`retrieve_bm25`/`retrieve`; `leak_guard(... strict=...)` keys (`top_sim`, `id_leak`, `exact_text_dup_same_context`, `exact_text_dup_diff_context`) match the runner's aggregation; `_gen_all(... logit_bias=...)` matches Task 4 + Task 6 calls. ✓

**Verify-during-impl (TDD will catch):** the unchanged tails of `search_dense`/`retrieve_bm25` (Steps show only the changed head); `build_messages` system-message length >200 as the rich-prompt assertion (adjust threshold if the template is shorter); `make_client`/Qdrant availability at run time.
