"""Length-cap baseline (review challenge): does a cheap post-processor close the gap?

The controlled-arm wins reduce to two surface features: reply length and the absence
of "!". A skeptic asks the fair question: hand those to the bare API for free (strip
"!", cap length) and does the LoRA still win? This post-processes the Arm-B API
generations that way and re-scores every arm against the real held-out replies with
the exact repo metrics.

Honest test. If the capped API matches the LoRA everywhere, the fine-tune's marginal
value over post-processing is thin. If the LoRA still wins on register the cap can't
fake (code-switching, the paren tic, bubble shape), it earns its keep. Reads the local
gitignored Arm-B pairs; prints a table only.

    uv run python scripts/length_cap_baseline.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from persona_rag.eval.compare import arm_summary, exclaim_rate, opener_entropy
from persona_rag.eval.distribution import latin_script_rate, paren_smiley_rate

PAIRS = Path("data/eval/compare/main/pairs.jsonl")


def cap_no_excl(text: str, char_cap: int) -> str:
    """The cheap baseline: drop every "!", then truncate to char_cap at a word
    boundary. A global cap (no per-item real length leaks in)."""
    stripped = text.replace("!", "")
    if len(stripped) <= char_cap:
        return stripped.strip()
    cut = stripped[:char_cap]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.strip()


def scorecard(real: list[str], gen: list[str]) -> dict[str, float]:
    s = arm_summary(real, gen)
    return {
        "shape_js": s["shape_js_vs_real"],
        "len_W1": s["len_wasserstein_vs_real"],
        "exclaim": s["exclaim_rate"],
        "latin": latin_script_rate(gen),
        "paren": paren_smiley_rate(gen),
        "opener_H": s["opener_entropy"],
    }


def main() -> None:
    text = PAIRS.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    real = [r["real"] for r in rows]
    api = [r["gen_api"] for r in rows]
    lora = [r["gen_lora"] for r in rows]

    cap = int(statistics.median(len(x) for x in real))
    api_capped = [cap_no_excl(x, cap) for x in api]

    cards = {
        "API-raw": scorecard(real, api),
        "API-capped": scorecard(real, api_capped),
        "LoRA": scorecard(real, lora),
    }
    real_ref = {
        "exclaim": exclaim_rate(real),
        "latin": latin_script_rate(real),
        "paren": paren_smiley_rate(real),
        "opener_H": opener_entropy(real),
    }

    metrics = [
        ("shape_js", "dist to real, lower=better"),
        ("len_W1", "dist to real, lower=better"),
        ("exclaim", "rate, match real"),
        ("latin", "rate, match real"),
        ("paren", "rate, match real"),
        ("opener_H", "entropy, match real"),
    ]
    print(f"\nn={len(rows)}  cap={cap} chars (median real reply), '!' stripped\n")
    print(f"{'metric':<10}{'API-raw':>11}{'API-capped':>13}{'LoRA':>9}{'real':>9}   note")
    print("-" * 72)
    for m, note in metrics:
        rv = real_ref.get(m)
        rv_s = f"{rv:>9.3f}" if rv is not None else f"{0.0:>9.3f}"
        print(
            f"{m:<10}{cards['API-raw'][m]:>11.3f}{cards['API-capped'][m]:>13.3f}"
            f"{cards['LoRA'][m]:>9.3f}{rv_s}   {note}"
        )
    print("\nRead: where API-capped reaches LoRA, the cap was enough. Where the LoRA")
    print("still lands closer to real, that gap is what the fine-tune actually buys.\n")


if __name__ == "__main__":
    main()
