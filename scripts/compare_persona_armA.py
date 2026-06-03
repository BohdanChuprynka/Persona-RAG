"""Arm A: production-realism comparison (shipped API vs LoRA) on the
recipient-stratified hold-out, with per-item retrieval leak exclusion + guard.

The SHIPPED OpenAI product (rich SYSTEM_TEMPLATE prompt + hybrid retrieval +
register/shape directives + decode levers) vs the LoRA in its real thin serving
config. Reuses production code paths verbatim (retrieve, build_messages,
retrieve_insights) + arm B's scorer/scaffold. The ONLY new risk is the per-item
exclude_ids filter + the fail-the-run leak guard.

Prereqs: Qdrant up (make up) + index built (make ingest) + llama-server serving
the LoRA on OLLAMA_BASE_URL. See docs/superpowers/specs/2026-06-02-arm-a-*.

    uv run python scripts/compare_persona_armA.py --n 300 --seed 0 --name armA
    uv run python scripts/compare_persona_armA.py --n 60 --leak-on --name armA_leakon
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from sqlmodel import Session, select

from persona_rag._logging import configure_logging, get_logger
from persona_rag.config import get_settings
from persona_rag.db.engine import make_engine
from persona_rag.db.models import PersonaTurnRow
from persona_rag.eval.compare import compare_scorecard, language_bucket, leak_guard
from persona_rag.finetune.dataset import clean_reply, eval_split_for
from persona_rag.generate.llm_client import voice_logit_bias
from persona_rag.generate.prompt import build_messages, build_thin_messages
from persona_rag.graph.nodes.build_prompt import _load_anchors
from persona_rag.graph.nodes.retrieve_insights import retrieve_insights
from persona_rag.index.qdrant_store import make_client
from persona_rag.models import ChatMessage
from persona_rag.retrieval import retrieve

# compare_persona is a sibling SCRIPT (not a package). Reuse its scaffold/scorer
# to keep arm B's runner pristine. Needs scripts/ on the path.
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from compare_persona import (
    TRAIN_PATH,
    _gen_all,
    _latency_cost,
    _load_sharegpt,
    _print_summary,
)

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


def _session_from_ctx(ctx: list[str]) -> list[ChatMessage]:
    # Mirror eval_persona._seed_context: lead-up lines become user-role session.
    return [ChatMessage(role="user", content=c) for c in ctx[:-1] if c.strip()]


def build_api_messages(
    *, ctx: list[str], retrieved: list[Any], insights: dict[str, Any] | None
) -> list[dict[str, str]]:
    """The SHIPPED rich prompt, assembled exactly as build_prompt_node does:
    incoming = ctx[-1]; session = ctx[:-1] as user turns."""
    s = get_settings()
    return build_messages(
        persona_name=s.PERSONA_NAME,
        persona_description=s.PERSONA_DESCRIPTION,
        style_anchors=_load_anchors(),
        user_memory="",
        retrieved=retrieved,
        session=_session_from_ctx(ctx),
        incoming=ctx[-1],
        insights=insights,
    )


async def _assemble_api(
    items: list[Item], *, strict: bool, leak_on: bool
) -> tuple[list[list[dict[str, str]]], list[dict[str, Any]]]:
    """Per item: retrieve (excluding own id unless leak_on) -> leak_guard ->
    insights -> rich messages. Returns (messages_list, per-item guard telemetry)."""
    client = make_client()
    msgs_list: list[list[dict[str, str]]] = []
    guard_rows: list[dict[str, Any]] = []
    for it in items:
        q = it.ctx[-1]
        exclude = None if leak_on else {it.turn_id}
        retrieved = await retrieve(q, client=client, exclude_ids=exclude)
        guard_rows.append(
            leak_guard(
                gold_turn_id=it.turn_id,
                gold_reply=it.reply,
                gold_ctx=it.ctx,
                retrieved=retrieved,
                strict=strict,
            )
        )
        state: dict[str, Any] = {"incoming": q}
        await retrieve_insights(state)  # self-wraps in try/except -> empty
        msgs = build_api_messages(ctx=it.ctx, retrieved=retrieved, insights=state.get("insights"))
        rich = any(len(m["content"]) > 200 for m in msgs if m["role"] == "system")
        assert rich, "API arm produced a thin prompt -- GENERATION_BACKEND not 'openai'?"
        msgs_list.append(msgs)
    return msgs_list, guard_rows


async def run(
    *,
    name: str,
    n: int,
    seed: int,
    temperature: float,
    max_tokens: int,
    n_boot: int,
    learned: bool,
    leak_on: bool,
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
    train_replies = [r["gpt"] for r in _load_sharegpt(TRAIN_PATH)]
    log.info("loaded", holdout=len(holdout), sampled=len(items), train_replies=len(train_replies))

    api_msgs, guard_rows = await _assemble_api(items, strict=not leak_on, leak_on=leak_on)
    lora_msgs = [
        build_thin_messages(incoming=it.ctx[-1], session=_session_from_ctx(it.ctx)) for it in items
    ]

    api = AsyncOpenAI(api_key=s.OPENAI_API_KEY)
    lora = AsyncOpenAI(base_url=s.OLLAMA_BASE_URL, api_key="local")
    bias = voice_logit_bias()  # resolves under GENERATION_BACKEND=openai

    log.info("generating", backend="api", model=s.OPENAI_CHAT_MODEL, bias=bias)
    api_res = await _gen_all(
        api,
        s.OPENAI_CHAT_MODEL,
        api_msgs,
        temperature=temperature,
        max_tokens=max_tokens,
        concurrency=4,
        logit_bias=bias,
    )
    log.info("generating", backend="lora", model=s.OLLAMA_MODEL)
    lora_res = await _gen_all(
        lora,
        s.OLLAMA_MODEL,
        lora_msgs,
        temperature=temperature,
        max_tokens=max_tokens,
        concurrency=4,
        logit_bias=None,
    )
    api_errs = [r["err"] for r in api_res if r["err"]]
    if api_errs:
        log.warning(
            "api_errors", n=len(api_errs), top=Counter(e[:60] for e in api_errs).most_common(3)
        )

    real = [it.reply for it in items]
    gen_api = [r["text"] for r in api_res]
    gen_lora = [r["text"] for r in lora_res]
    card = compare_scorecard(
        real, gen_api, gen_lora, train_replies=train_replies, n_boot=n_boot, seed=seed
    )

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / (name or ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        {
            "item_id": i,
            "turn_id": items[i].turn_id,
            "incoming": items[i].ctx[-1],
            "real": real[i],
            "gen_api": gen_api[i],
            "gen_lora": gen_lora[i],
            "lang": language_bucket(real[i]),
            "top_sim": guard_rows[i]["top_sim"],
        }
        for i in range(len(items))
    ]
    (out_dir / "pairs.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8"
    )
    sims = [g["top_sim"] for g in guard_rows if g["top_sim"] == g["top_sim"]]  # drop nan
    results = {
        "name": name,
        "ts": ts,
        "arm": "A-production (shipped API rich+retrieval+levers vs LoRA thin)",
        "params": {
            "n": len(items),
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n_boot": n_boot,
            "api_model": s.OPENAI_CHAT_MODEL,
            "lora_model": s.OLLAMA_MODEL,
            "retrieval_query": "ctx[-1] (runtime-faithful)",
            "levers": {
                "paren_logit_bias": s.PAREN_LOGIT_BIAS,
                "exclaim_logit_bias": s.EXCLAIM_LOGIT_BIAS,
                "best_of_n": s.BEST_OF_N,
                "resolved_bias": bias,
                "pass": "learned" if learned else "shipped",
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
    p.add_argument(
        "--leak-on", action="store_true", help="DISABLE exclusion (validation: measure the leak)"
    )
    a = p.parse_args()
    asyncio.run(
        run(
            name=a.name,
            n=a.n,
            seed=a.seed,
            temperature=a.temp,
            max_tokens=a.max_tokens,
            n_boot=a.n_boot,
            learned=a.learned,
            leak_on=a.leak_on,
        )
    )


if __name__ == "__main__":
    main()
