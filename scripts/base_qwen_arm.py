"""Base-Qwen no-LoRA ablation (review fix #9) — the missing within-family control.

Arm B compares the LoRA against gpt-4o-mini (a *different* family); it never shows the
LoRA against its OWN base, so "the tics had to be learned" is asserted, not isolated.
This scores base Qwen2.5-3B-Instruct (no adapter) on the IDENTICAL thin controlled-arm
prompt — same items, same decode params as ``make compare`` — so the only thing that
changed between this column and the LoRA column is the fine-tune.

No GPU, no Mac: point it at any OpenAI-compatible endpoint that serves the EXACT base
model. Qwen2.5-3B is small, so the big aggregators (DeepInfra / Together / OpenRouter)
mostly carry 7B+; hosts with the 3B include Alibaba Model Studio (DashScope — the
official Qwen API), Featherless.ai and Runpod, or a free Colab/Kaggle GPU + vLLM.
Pass env vars INLINE (the script reads the process env, not the pydantic .env):

    # Alibaba DashScope (official, OpenAI-compatible, exact 3B, free trial quota):
    BASE_QWEN_BASE_URL=https://dashscope-intl.aliyun.com/compatible-mode/v1 \
    BASE_QWEN_MODEL=qwen2.5-3b-instruct \
    BASE_QWEN_API_KEY=<key> uv run python scripts/base_qwen_arm.py

BASE_QWEN_MODEL defaults to the HF id Qwen/Qwen2.5-3B-Instruct; override per provider
(DashScope wants ``qwen2.5-3b-instruct``). If only 7B is available, that is a looser
control (different size) — note it. Output: data/eval/compare/base_qwen/.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from persona_rag._logging import configure_logging, get_logger
from persona_rag.eval.compare import (
    arm_summary,
    exclaim_rate,
    language_bucket,
    opener_entropy,
)
from persona_rag.eval.distribution import latin_script_rate, paren_smiley_rate

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from compare_persona import (
    EVAL_PATH,
    _gen_all,
    _latency_cost,
    _load_sharegpt,
    _sample,
)

log = get_logger()
OUT_DIR = Path("data/eval/compare/base_qwen")
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def compute_base_scorecard(real: list[str], gen_base: list[str]) -> dict[str, Any]:
    """Table-relevant metrics for the base arm vs the real held-out replies, plus the
    person's own reference values (the targets) — a pure function so it is unit-testable
    without a live endpoint."""
    base = arm_summary(real, gen_base)
    base["latin_script_rate"] = latin_script_rate(gen_base)
    base["paren_smiley_rate"] = paren_smiley_rate(gen_base)
    real_ref = {
        "exclaim_rate": exclaim_rate(real),
        "latin_script_rate": latin_script_rate(real),
        "opener_entropy": opener_entropy(real),
        "paren_smiley_rate": paren_smiley_rate(real),
    }
    return {"base_vs_real": base, "real_reference": real_ref}


async def run(*, n: int, seed: int, temperature: float, max_tokens: int) -> None:
    base_url = os.environ.get("BASE_QWEN_BASE_URL")
    api_key = os.environ.get("BASE_QWEN_API_KEY")
    model = os.environ.get("BASE_QWEN_MODEL", DEFAULT_MODEL)
    if not base_url or not api_key:
        log.error(
            "base_qwen_unconfigured",
            hint="set BASE_QWEN_BASE_URL + BASE_QWEN_API_KEY to an OpenAI-compatible "
            "endpoint serving Qwen2.5-3B-Instruct (DeepInfra/Together/Fireworks/...).",
        )
        print(__doc__)
        return

    records = _sample(_load_sharegpt(EVAL_PATH), n, seed)  # SAME items as `make compare`
    messages = [
        [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["human"]}]
        for r in records
    ]
    real = [r["gpt"] for r in records]
    log.info("generating", backend="base_qwen", model=model, n=len(messages))
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    res = await _gen_all(
        client, model, messages, temperature=temperature, max_tokens=max_tokens, concurrency=4
    )
    gen_base = [r["text"] for r in res]
    errs = [r["err"] for r in res if r["err"]]
    if errs:
        log.warning("base_qwen_errors", n=len(errs), first=errs[0][:120])

    scorecard = compute_base_scorecard(real, gen_base)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [
        {
            "item_id": i,
            "incoming": records[i]["human"],
            "real": real[i],
            "gen_base": gen_base[i],
            "lang": language_bucket(real[i]),
        }
        for i in range(len(records))
    ]
    (OUT_DIR / "pairs.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8"
    )
    results = {
        "name": "base_qwen",
        "arm": "B-controlled base Qwen2.5-3B-Instruct (no LoRA), identical thin prompt",
        "params": {
            "n": len(records),
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "model": model,
            "endpoint": base_url,
        },
        "scorecard": scorecard,
        "operational": {"base": _latency_cost(res, priced=False)},
    }
    (OUT_DIR / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _print_base(scorecard, len(records))


def _print_base(scorecard: dict[str, Any], n: int) -> None:
    b = scorecard["base_vs_real"]
    ref = scorecard["real_reference"]
    print(f"\n=== base Qwen2.5-3B-Instruct (no LoRA), thin prompt, n={n} ===")
    print(
        f"  shape_js={b['shape_js_vs_real']:.3f}  len_W1={b['len_wasserstein_vs_real']:.2f}  "
        f"exclaim={b['exclaim_rate']:.3f}  latin={b['latin_script_rate']:.3f}  "
        f"opener_H={b['opener_entropy']:.2f}  paren={b['paren_smiley_rate']:.3f}"
    )
    print(
        f"  (real ref: exclaim={ref['exclaim_rate']:.3f}  latin={ref['latin_script_rate']:.3f}  "
        f"opener_H={ref['opener_entropy']:.2f}  paren={ref['paren_smiley_rate']:.3f})"
    )
    print(f"  wrote {OUT_DIR / 'results.json'}")


def score_from_pairs(path: Path) -> None:
    """Score a Colab-generated pairs file (no endpoint needed): each line is
    ``{"real": ..., "gen_base": ...}``. Writes the same results.json a live run would,
    using the exact repo metric functions so the numbers match the paper."""
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    real = [r["real"] for r in rows]
    gen_base = [r["gen_base"] for r in rows]
    scorecard = compute_base_scorecard(real, gen_base)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "name": "base_qwen",
        "arm": "B-controlled base Qwen2.5-3B-Instruct (no LoRA), identical thin prompt",
        "params": {"n": len(rows), "source": str(path)},
        "scorecard": scorecard,
    }
    (OUT_DIR / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _print_base(scorecard, len(rows))


def main() -> None:
    configure_logging()
    import argparse

    p = argparse.ArgumentParser(description="Base-Qwen no-LoRA ablation (Arm B control).")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument(
        "--from-pairs",
        type=str,
        default="",
        help="score a Colab-generated pairs.jsonl ({real, gen_base}) instead of generating",
    )
    a = p.parse_args()
    if a.from_pairs:
        score_from_pairs(Path(a.from_pairs))
        return
    asyncio.run(run(n=a.n, seed=a.seed, temperature=a.temp, max_tokens=a.max_tokens))


if __name__ == "__main__":
    main()
