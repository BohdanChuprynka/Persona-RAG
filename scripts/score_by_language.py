"""Per-language breakdown of a comparison run (NO new generation).

Re-scores ``data/eval/compare/<name>/pairs.jsonl`` per language bucket
(latin / cyrillic / mixed / other) with the same ``compare_scorecard``, so we can
see whether the API closes the voice gap in some scripts but not others — your
persona code-switches uk/ru/en, and gpt-4o-mini is strongest in English.

    uv run python scripts/score_by_language.py --name armA
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from persona_rag.eval.compare import compare_scorecard


def _train_replies() -> list[str]:
    p = Path("data/finetune/train.jsonl")
    if not p.exists():
        return []
    out: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line)["conversations"][-1]["value"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-language breakdown of a comparison run.")
    ap.add_argument("--name", default="armA")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--min-n", type=int, default=20, help="skip buckets smaller than this")
    a = ap.parse_args()

    base = Path("data/eval/compare") / a.name
    pairs = [
        json.loads(line)
        for line in (base / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train = _train_replies()

    buckets: dict[str, list[dict[str, Any]]] = {}
    for p in pairs:
        buckets.setdefault(p.get("lang", "other"), []).append(p)

    print("=" * 72)
    print(f"  PER-LANGUAGE — {a.name}   (lower distance = closer to Bohdan)")
    print("=" * 72)
    print(
        f"  {'lang':9s} {'n':>4s}  {'shape A/L':>13s}  {'len_emd A/L':>15s}  {'excl A/L':>10s}  len"
    )
    out: dict[str, Any] = {}
    for lang in ("latin", "cyrillic", "mixed", "other"):
        sub = buckets.get(lang, [])
        if len(sub) < a.min_n:
            print(f"  {lang:9s} {len(sub):>4d}   (n < {a.min_n}, skipped)")
            continue
        real = [p["real"] for p in sub]
        gen_api = [p["gen_api"] for p in sub]
        gen_lora = [p["gen_lora"] for p in sub]
        card = compare_scorecard(
            real, gen_api, gen_lora, train_replies=train, n_boot=a.n_boot, seed=0
        )
        api, lo = card["arms"]["api"], card["arms"]["lora"]
        d = card["deltas_api_minus_lora"]["len_wasserstein"]
        verdict = d["favored"] if d["excludes_zero"] else "tie"
        print(
            f"  {lang:9s} {len(sub):>4d}  "
            f"{api['shape_js_vs_real']:.3f}/{lo['shape_js_vs_real']:.3f}  "
            f"{api['len_wasserstein_vs_real']:6.2f}/{lo['len_wasserstein_vs_real']:5.2f}  "
            f"{api['exclaim_rate']:.2f}/{lo['exclaim_rate']:.2f}  {verdict}"
        )
        out[lang] = card
    (base / "by_language.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
