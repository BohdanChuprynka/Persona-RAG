"""Score a completed blind human-eval into a LoRA-vs-API win-rate + Wilson CI.

Run after rating in ``reports/<name>/human_eval/rater.html`` and downloading
``choices.json`` into that folder. Joins choices against ``key.json`` (and, when
present, ``data/eval/compare/<name>/pairs.jsonl`` for a per-language breakdown).

    uv run python scripts/score_human_eval.py --name main

Verdict: a Wilson 95% CI on the LoRA win-rate that excludes 0.5 = a real
preference; otherwise a tie (the audit's definition of "better").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from persona_rag.eval.compare import score_preferences


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a blind human-eval.")
    ap.add_argument("--name", default="main")
    ap.add_argument("--choices", default="", help="path to choices.json (default: in the kit dir)")
    a = ap.parse_args()

    kit = Path("reports") / a.name / "human_eval"
    key = json.loads((kit / "key.json").read_text(encoding="utf-8"))
    choices_path = Path(a.choices) if a.choices else kit / "choices.json"
    if not choices_path.exists():
        print(f"no choices file at {choices_path} — rate in rater.html and download choices.json")
        return
    choices = json.loads(choices_path.read_text(encoding="utf-8"))

    overall = score_preferences(choices, key)

    # Optional per-language breakdown via the run's pairs.jsonl (item_id -> lang).
    by_lang: dict[str, dict[str, object]] = {}
    pairs_path = Path("data/eval/compare") / a.name / "pairs.jsonl"
    if pairs_path.exists():
        lang = {
            str(json.loads(line)["item_id"]): json.loads(line)["lang"]
            for line in pairs_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        for bucket in ("latin", "cyrillic", "mixed", "other"):
            sub = {i: c for i, c in choices.items() if lang.get(i) == bucket}
            if sub:
                by_lang[bucket] = score_preferences(sub, key)

    result = {"name": a.name, "overall": overall, "by_language": by_lang}
    (kit / "human_scorecard.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lo, hi = overall["wilson_95ci"]
    print("=" * 56)
    print(f"  BLIND HUMAN PANEL — {a.name}")
    print("=" * 56)
    print(f"  rated: {len(choices)}   decisive: {overall['decisive']}   ties: {overall['ties']}")
    print(f"  LoRA wins: {overall['lora_wins']}   API wins: {overall['api_wins']}")
    wr = overall["lora_win_rate"]
    wr_s = f"{wr:.3f}" if isinstance(wr, float) else str(wr)
    print(f"  LoRA win-rate: {wr_s}   Wilson 95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  VERDICT: {overall['verdict']}")
    if by_lang:
        print("-" * 56)
        for b, r in by_lang.items():
            w = r["lora_win_rate"]
            w_s = f"{w:.3f}" if isinstance(w, float) else str(w)
            print(f"  {b:9s} n={r['decisive']:<3} lora_win_rate={w_s} verdict={r['verdict']}")
    print("=" * 56)


if __name__ == "__main__":
    main()
