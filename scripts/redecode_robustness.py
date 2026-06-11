"""Arm-A decode-variance robustness check (review fix #4).

The headline arms decode each item once at temperature 0.8, and the paired
bootstrap resamples *item indices* only -- so it captures which-items noise but
not decode stochasticity. Arm A's sole non-tie is a ~3.6-char *distributional*
reply-length edge for the LoRA; this script asks whether that edge survives
re-decoding.

Method (free, local-only -- the LoRA path skips retrieval, so no Qdrant/API):
reuse the EXACT Arm-A items (``load_holdout`` + ``random.Random(0).sample``),
hold the API generations FIXED (read back from ``data/eval/compare/armA/
pairs.jsonl``), and re-decode the LoRA arm K times via the local llama-server.
Each pass is scored with the same ``compare_scorecard`` the paper uses, so the
length delta + paired-bootstrap CI are computed identically. We then report
whether the API-minus-LoRA length-Wasserstein CI excludes zero in every pass.

    uv run python scripts/redecode_robustness.py --passes 3 --n 300 --seed 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI

from persona_rag._logging import configure_logging, get_logger
from persona_rag.config import get_settings
from persona_rag.eval.compare import compare_scorecard
from persona_rag.generate.prompt import build_thin_messages

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from compare_persona import TRAIN_PATH, _gen_all, _load_sharegpt
from compare_persona_armA import _session_from_ctx, load_holdout

log = get_logger()
OUT_ROOT = Path("data/eval/compare")
ARMA_PAIRS = OUT_ROOT / "armA" / "pairs.jsonl"


def _served_model(base_url: str, fallback: str) -> str:
    """Use whatever the local server actually serves (the GGUF name can drift
    from config OLLAMA_MODEL between runs)."""
    try:
        r = httpx.get(base_url.rstrip("/") + "/models", timeout=5.0)
        ids = [m["id"] for m in r.json().get("data", r.json().get("models", []))]
        ids = [i for i in ids if i]
        if fallback in ids:
            return fallback
        if ids:
            log.info("served_model_override", configured=fallback, using=ids[0])
            return ids[0]
    except Exception as e:
        log.warning("models_probe_failed", err=str(e)[:80])
    return fallback


async def run(
    *, name: str, n: int, seed: int, passes: int, temperature: float, max_tokens: int, n_boot: int
) -> None:
    s = get_settings()
    import random

    holdout = load_holdout()
    items = random.Random(seed).sample(holdout, min(n, len(holdout)))
    real = [it.reply for it in items]
    lora_msgs = [
        build_thin_messages(incoming=it.ctx[-1], session=_session_from_ctx(it.ctx)) for it in items
    ]

    # FIXED API generations from the committed Arm-A run, aligned by turn_id.
    api_by_tid: dict[str, str] = {}
    for line in ARMA_PAIRS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            api_by_tid[p["turn_id"]] = p["gen_api"]
    gen_api = [api_by_tid.get(it.turn_id, "") for it in items]
    missing = sum(1 for it in items if it.turn_id not in api_by_tid)
    if missing:
        log.warning("api_alignment_gaps", missing=missing, of=len(items))

    train_replies = [r["gpt"] for r in _load_sharegpt(TRAIN_PATH)]
    model = _served_model(s.OLLAMA_BASE_URL, s.OLLAMA_MODEL)
    lora = AsyncOpenAI(base_url=s.OLLAMA_BASE_URL, api_key="local")

    rows: list[dict[str, Any]] = []
    for k in range(passes):
        log.info("redecode_pass", k=k + 1, of=passes, model=model)
        res = await _gen_all(
            lora,
            model,
            lora_msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            concurrency=4,
            logit_bias=None,
        )
        gen_lora = [r["text"] for r in res]
        errs = sum(1 for r in res if r.get("err"))
        card = compare_scorecard(
            real, gen_api, gen_lora, train_replies=train_replies, n_boot=n_boot, seed=seed + k
        )
        d = card["deltas_api_minus_lora"]["len_wasserstein"]
        shp = card["deltas_api_minus_lora"]["shape_js"]
        rows.append(
            {
                "pass": k + 1,
                "errors": errs,
                "lora_len_wasserstein_vs_real": card["arms"]["lora"]["len_wasserstein_vs_real"],
                "api_len_wasserstein_vs_real": card["arms"]["api"]["len_wasserstein_vs_real"],
                "len_delta_api_minus_lora": d["delta"],
                "len_ci": [d["ci_lo"], d["ci_hi"]],
                "len_excludes_zero": d["excludes_zero"],
                "len_favored": d["favored"],
                "shape_delta": shp["delta"],
                "shape_excludes_zero": shp["excludes_zero"],
                "lora_mean_bubble_len": card["arms"]["lora"].get("mean_bubble_len"),
            }
        )
        log.info("pass_done", **rows[-1])

    excl = [r["len_excludes_zero"] for r in rows]
    summary = {
        "name": name,
        "arm": "A re-decode robustness (LoRA re-decoded K times, API fixed)",
        "params": {
            "n": len(items),
            "seed": seed,
            "passes": passes,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n_boot": n_boot,
            "lora_model": model,
            "api_fixed_from": str(ARMA_PAIRS),
        },
        "passes": rows,
        "verdict": {
            "len_delta_excludes_zero_all_passes": all(excl),
            "len_delta_excludes_zero_count": f"{sum(excl)}/{len(excl)}",
            "lora_len_dist_range": [
                min(r["lora_len_wasserstein_vs_real"] for r in rows),
                max(r["lora_len_wasserstein_vs_real"] for r in rows),
            ],
            "len_delta_range": [
                min(r["len_delta_api_minus_lora"] for r in rows),
                max(r["len_delta_api_minus_lora"] for r in rows),
            ],
        },
    }
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["verdict"], indent=2))
    for r in rows:
        lo, hi = r["len_ci"]
        print(
            f"  pass {r['pass']}: LoRA W1={r['lora_len_wasserstein_vs_real']:.2f}  "
            f"delta={r['len_delta_api_minus_lora']:+.2f} CI[{lo:.2f},{hi:.2f}] "
            f"excl0={r['len_excludes_zero']} (errors={r['errors']})"
        )
    log.info("wrote", dir=str(out_dir))


def main() -> None:
    configure_logging()
    p = argparse.ArgumentParser(description="Arm-A decode-variance robustness check.")
    p.add_argument("--name", default="armA_redecode")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args()
    asyncio.run(
        run(
            name=a.name,
            n=a.n,
            seed=a.seed,
            passes=a.passes,
            temperature=a.temp,
            max_tokens=a.max_tokens,
            n_boot=a.n_boot,
        )
    )


if __name__ == "__main__":
    main()
