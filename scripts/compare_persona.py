# Reason: Cyrillic appears in logged sample replies.
"""Fair, paired API-vs-LoRA persona comparison — the CONTROLLED (arm B) runner.

Scores BOTH backends on the LoRA-disjoint hold-out (``data/finetune/eval.jsonl``,
the recipient-stratified ``eval_split_for`` split the LoRA trained disjoint from),
under the IDENTICAL thin prompt — no retrieval, no directives, no logit-bias,
matched temperature / max_tokens / n=1. This isolates *weights* (gpt-4o-mini vs
the fine-tuned Qwen2.5-3B), the only attributable + leak-free comparison today
(see docs/superpowers/2026-06-02-eval-architecture-audit.md, R1/R2/R3).

Outputs under ``data/eval/compare/<ts>/``:
  - results.json   scorecard (with bootstrap CIs + guards) + params + latency/cost
  - pairs.jsonl    {item_id, incoming, real, gen_api, gen_lora, lang} for the blind kit

    uv run python scripts/compare_persona.py --n 300 --seed 0
    uv run python scripts/compare_persona.py --n 150 --seed 1 --name seed1

The LoRA backend must be served on OLLAMA_BASE_URL (llama-server -a bohdan ...);
the API backend uses OPENAI_API_KEY + OPENAI_CHAT_MODEL. Each backend native,
NOT routed through llm_client (we deliberately bypass the production scaffold).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from openai import AsyncOpenAI

from persona_rag._logging import configure_logging, get_logger
from persona_rag.config import get_settings
from persona_rag.eval.compare import compare_scorecard, language_bucket

log = get_logger()

EVAL_PATH = Path("data/finetune/eval.jsonl")
TRAIN_PATH = Path("data/finetune/train.jsonl")
OUT_ROOT = Path("data/eval/compare")

# gpt-4o-mini list price (USD / 1M tokens), approximate — for a ballpark $/1k replies.
API_PRICE_IN = 0.15
API_PRICE_OUT = 0.60


def _load_sharegpt(path: Path) -> list[dict[str, str]]:
    """Parse a ShareGPT jsonl into {system, human, gpt} triples."""
    out: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            conv = json.loads(line)["conversations"]
            by_role = {m["from"]: m["value"] for m in conv}
            if "human" in by_role and "gpt" in by_role:
                out.append(
                    {
                        "system": by_role.get("system", ""),
                        "human": by_role["human"],
                        "gpt": by_role["gpt"],
                    }
                )
    return out


def _sample(records: list[dict[str, str]], n: int, seed: int) -> list[dict[str, str]]:
    import random

    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)
    return pool[:n]


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
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": cast(Any, messages),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if logit_bias:
                    kwargs["logit_bias"] = logit_bias
                r = await client.chat.completions.create(**kwargs)
                dt = time.perf_counter() - t0
                txt = r.choices[0].message.content or ""
                u = r.usage
                pt = u.prompt_tokens if u else 0
                ct = u.completion_tokens if u else 0
                fin = r.choices[0].finish_reason
                return {"text": txt, "latency": dt, "in": pt, "out": ct, "finish": fin, "err": None}
            except Exception as e:
                return {
                    "text": "",
                    "latency": time.perf_counter() - t0,
                    "in": 0,
                    "out": 0,
                    "finish": "error",
                    "err": str(e)[:200],
                }

    return await asyncio.gather(*(one(m) for m in messages_list))


def _pctile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def _latency_cost(results: list[dict[str, Any]], *, priced: bool) -> dict[str, Any]:
    lats = [r["latency"] for r in results]
    n = len(results) or 1
    tot_in = sum(r["in"] for r in results)
    tot_out = sum(r["out"] for r in results)
    trunc = sum(1 for r in results if r["finish"] == "length") / n
    errs = sum(1 for r in results if r["err"]) / n
    block: dict[str, Any] = {
        "p50_latency_s": round(_pctile(lats, 0.5), 3),
        "p95_latency_s": round(_pctile(lats, 0.95), 3),
        "mean_in_tokens": round(tot_in / n, 1),
        "mean_out_tokens": round(tot_out / n, 1),
        "truncation_rate": round(trunc, 3),
        "error_rate": round(errs, 3),
    }
    if priced:
        cost = (tot_in * API_PRICE_IN + tot_out * API_PRICE_OUT) / 1_000_000
        block["usd_total"] = round(cost, 4)
        block["usd_per_1k_replies"] = round(cost / n * 1000, 4)
    return block


async def run(
    name: str, n: int, seed: int, temperature: float, max_tokens: int, n_boot: int
) -> None:
    s = get_settings()
    if not EVAL_PATH.exists():
        log.error("no_eval_file", path=str(EVAL_PATH), hint="run export_finetune_data.py")
        return
    records = _sample(_load_sharegpt(EVAL_PATH), n, seed)
    train_replies = [r["gpt"] for r in _load_sharegpt(TRAIN_PATH)]
    log.info("loaded", eval_items=len(records), train_replies=len(train_replies))

    messages = [
        [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["human"]}]
        for r in records
    ]

    api = AsyncOpenAI(api_key=s.OPENAI_API_KEY)
    lora = AsyncOpenAI(base_url=s.OLLAMA_BASE_URL, api_key="local")

    log.info("generating", backend="api", model=s.OPENAI_CHAT_MODEL, n=len(messages))
    api_res = await _gen_all(
        api,
        s.OPENAI_CHAT_MODEL,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        concurrency=8,
    )
    log.info("generating", backend="lora", model=s.OLLAMA_MODEL, n=len(messages))
    lora_res = await _gen_all(
        lora,
        s.OLLAMA_MODEL,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        concurrency=4,
    )

    real = [r["gpt"] for r in records]
    gen_api = [r["text"] for r in api_res]
    gen_lora = [r["text"] for r in lora_res]

    card = compare_scorecard(
        real, gen_api, gen_lora, train_replies=train_replies, n_boot=n_boot, seed=seed
    )

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / (name or ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for i, r in enumerate(records):
        pairs.append(
            {
                "item_id": i,
                "incoming": r["human"],
                "real": r["gpt"],
                "gen_api": gen_api[i],
                "gen_lora": gen_lora[i],
                "lang": language_bucket(r["gpt"]),
            }
        )
    (out_dir / "pairs.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8"
    )

    results = {
        "name": name,
        "ts": ts,
        "arm": "B-controlled (identical thin prompt, no retrieval/directives/logit-bias)",
        "params": {
            "n": len(records),
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n_boot": n_boot,
            "api_model": s.OPENAI_CHAT_MODEL,
            "lora_model": s.OLLAMA_MODEL,
            "lora_base_url": s.OLLAMA_BASE_URL,
            "eval_split": "eval_split_for (recipient-stratified, LoRA-disjoint)",
        },
        "scorecard": card,
        "operational": {
            "api": _latency_cost(api_res, priced=True),
            "lora": _latency_cost(lora_res, priced=False),
        },
        "lang_distribution": {
            b: sum(1 for p in pairs if p["lang"] == b)
            for b in ("latin", "cyrillic", "mixed", "other")
        },
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _print_summary(results)
    log.info("wrote", dir=str(out_dir))


def _fmt(x: Any) -> str:
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def _print_summary(results: dict[str, Any]) -> None:
    card = results["scorecard"]
    a, lo = card["arms"]["api"], card["arms"]["lora"]
    d = card["deltas_api_minus_lora"]
    print("\n" + "=" * 64)
    print(f"  CONTROLLED A/B — n={card['n_items']}  (lower distance = closer to Bohdan)")
    print("=" * 64)
    print(f"  {'metric':28s} {'API':>10s} {'LoRA':>10s}")
    for k in (
        "shape_js_vs_real",
        "len_wasserstein_vs_real",
        "exclaim_rate",
        "opener_entropy",
        "distinct_reply_rate",
        "empty_rate",
    ):
        print(f"  {k:28s} {_fmt(a[k]):>10s} {_fmt(lo[k]):>10s}")
    print("-" * 64)
    for k, dd in d.items():
        verdict = dd["favored"] if dd["excludes_zero"] else "tie (CI spans 0)"
        print(
            f"  Δ{k:18s} {_fmt(dd['delta']):>10s}  "
            f"CI[{_fmt(dd['ci_lo'])},{_fmt(dd['ci_hi'])}]  -> {verdict}"
        )
    if "copy_leak" in card:
        cl = card["copy_leak"]
        print("-" * 64)
        print(
            f"  copy/leak (exact|near)  API {_fmt(cl['api']['exact'])}|{_fmt(cl['api']['near'])}"
            f"   LoRA {_fmt(cl['lora']['exact'])}|{_fmt(cl['lora']['near'])}"
        )
    op = results["operational"]
    print("-" * 64)
    print(
        f"  API   p50={op['api']['p50_latency_s']}s p95={op['api']['p95_latency_s']}s"
        f"  ${op['api'].get('usd_per_1k_replies', '?')}/1k  err={op['api']['error_rate']}"
    )
    print(
        f"  LoRA  p50={op['lora']['p50_latency_s']}s p95={op['lora']['p95_latency_s']}s"
        f"  $0/1k (local)  err={op['lora']['error_rate']}"
    )
    print("=" * 64 + "\n")


def main() -> None:
    configure_logging()
    p = argparse.ArgumentParser(description="Fair controlled A/B: API vs fine-tuned LoRA.")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--name", type=str, default="")
    a = p.parse_args()
    asyncio.run(run(a.name, a.n, a.seed, a.temp, a.max_tokens, a.n_boot))


if __name__ == "__main__":
    main()
