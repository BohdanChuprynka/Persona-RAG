# Reason: reads pairs.jsonl whose replies contain Cyrillic.
"""Compute per-item length effect sizes for a comparison run (no new generation).

    uv run python scripts/compute_effect_sizes.py --name armA

Reads ``data/eval/compare/<name>/pairs.jsonl`` and writes ``effect_sizes.json``
(overall + per language bucket) next to it. Complements the corpus-level
bootstrap CI already in results.json with the per-item Cliff's delta + sign /
Wilcoxon tests the audit (R4) prescribed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from persona_rag.eval.effect_size import length_effect_sizes


def _load(name: str) -> list[dict[str, str]]:
    p = Path("data/eval/compare") / name / "pairs.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-item length effect sizes for a run.")
    ap.add_argument("--name", default="armA")
    ap.add_argument("--min-n", type=int, default=20)
    a = ap.parse_args()
    pairs = _load(a.name)
    real = [p["real"] for p in pairs]
    api = [p["gen_api"] for p in pairs]
    lora = [p["gen_lora"] for p in pairs]
    out: dict[str, Any] = {"overall": length_effect_sizes(real, api, lora)}
    buckets: dict[str, list[dict[str, str]]] = {}
    for p in pairs:
        buckets.setdefault(p.get("lang", "other"), []).append(p)
    out["by_language"] = {
        lang: length_effect_sizes(
            [p["real"] for p in sub], [p["gen_api"] for p in sub], [p["gen_lora"] for p in sub]
        )
        for lang, sub in buckets.items()
        if len(sub) >= a.min_n
    }
    dst = Path("data/eval/compare") / a.name / "effect_sizes.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    o = out["overall"]
    print(
        f"{a.name}: Cliff's d={o['cliffs_delta']:.3f} ({o['cliffs_magnitude']}), "
        f"LoRA closer {o['lora_closer']}/{o['lora_closer'] + o['api_closer']}, "
        f"sign p={o['sign_test_p']:.2e} -> {dst}"
    )


if __name__ == "__main__":
    main()
