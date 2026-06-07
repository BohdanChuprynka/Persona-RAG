"""Score a completed blind human-eval.

Two modes:
  --mode ab      (default) LoRA-vs-API preference -> LoRA win-rate + Wilson CI.
                 Kit dir: reports/<name>/human_eval/
  --mode turing  real-vs-LoRA detection ("which is the bot?") -> detection-rate
                 + Wilson CI + verdict + a voice-vs-knowledge split of the
                 "tells". Kit dir: reports/<name>/turing/

Run after rating in the kit's rater.html and downloading choices.json into it.

    uv run python scripts/score_human_eval.py --name main                # A/B
    uv run python scripts/score_human_eval.py --name main --mode turing  # Turing

Verdict (ab): a Wilson 95% CI on the LoRA win-rate that excludes 0.5 = a real
preference; otherwise a tie. Verdict (turing): a CI on the detection-rate that
INCLUDES 0.5 = the LoRA is statistically indistinguishable from Bohdan (it
passes); a CI strictly above 0.5 = a rater can still tell.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from persona_rag.eval.compare import score_detection, score_preferences

_LANG_BUCKETS = ("latin", "cyrillic", "mixed", "other")


def _lang_map(name: str) -> dict[str, str]:
    """item_id -> language bucket from the run's pairs.jsonl (empty if absent)."""
    pairs_path = Path("data/eval/compare") / name / "pairs.jsonl"
    if not pairs_path.exists():
        return {}
    return {
        str(json.loads(line)["item_id"]): json.loads(line)["lang"]
        for line in pairs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _fmt(x: object) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def _report_ab(name: str, choices: dict[str, Any], key: dict[str, Any], kit: Path) -> None:
    overall = score_preferences(choices, key)
    lang = _lang_map(name)
    by_lang: dict[str, dict[str, object]] = {}
    for bucket in _LANG_BUCKETS:
        sub = {i: c for i, c in choices.items() if lang.get(i) == bucket}
        if sub:
            by_lang[bucket] = score_preferences(sub, key)

    (kit / "human_scorecard.json").write_text(
        json.dumps(
            {"name": name, "overall": overall, "by_language": by_lang},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    lo, hi = overall["wilson_95ci"]
    print("=" * 56)
    print(f"  BLIND A/B PANEL (LoRA vs API) - {name}")
    print("=" * 56)
    print(f"  rated {len(choices)}  decisive {overall['decisive']}  ties {overall['ties']}")
    print(f"  LoRA wins {overall['lora_wins']}  API wins {overall['api_wins']}")
    print(f"  LoRA win-rate {_fmt(overall['lora_win_rate'])}  Wilson95 [{lo:.3f}, {hi:.3f}]")
    print(f"  VERDICT: {overall['verdict']}")
    if by_lang:
        print("-" * 56)
        for b, r in by_lang.items():
            wr = _fmt(r["lora_win_rate"])
            print(f"  {b:9s} n={r['decisive']:<3} win-rate={wr} -> {r['verdict']}")
    print("=" * 56)


def _report_turing(name: str, choices: dict[str, Any], key: dict[str, Any], kit: Path) -> None:
    overall = score_detection(choices, key)
    lang = _lang_map(name)
    by_lang: dict[str, dict[str, object]] = {}
    for bucket in _LANG_BUCKETS:
        sub = {i: c for i, c in choices.items() if lang.get(i) == bucket}
        if sub:
            by_lang[bucket] = score_detection(sub, key)

    (kit / "turing_scorecard.json").write_text(
        json.dumps(
            {"name": name, "overall": overall, "by_language": by_lang},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    lo, hi = overall["wilson_95ci"]
    tells = overall["tells"]
    print("=" * 56)
    print(f"  TURING PANEL (real vs LoRA) - {name}")
    print("=" * 56)
    print(f"  rated {len(choices)}  decisive {overall['decisive']}  unsure {overall['unsure']}")
    print(f"  bot caught {overall['machine_caught']}  fooled {overall['human_mistaken']}")
    print(f"  detection-rate {_fmt(overall['detection_rate'])}  Wilson95 [{lo:.3f}, {hi:.3f}]")
    print(f"  VERDICT: {overall['verdict']}  (indistinguishable = passes as you)")
    if tells["n"]:
        v, kn, o = tells["voice"], tells["knowledge"], tells["other"]
        print("-" * 56)
        print(f"  caught by: voice {v} / knowledge {kn} / other {o}")
        print(f"  tells: {tells['by_tag']}")
        print("  voice -> decode/training fix; knowledge -> Obsidian fact-card fix")
    if by_lang:
        print("-" * 56)
        for b, r in by_lang.items():
            dr = _fmt(r["detection_rate"])
            print(f"  {b:9s} n={r['decisive']:<3} detect={dr} -> {r['verdict']}")
    print("=" * 56)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a blind human-eval (A/B or Turing).")
    ap.add_argument("--name", default="main")
    ap.add_argument("--mode", choices=["ab", "turing"], default="ab")
    ap.add_argument("--choices", default="", help="path to choices.json (default: in the kit dir)")
    a = ap.parse_args()

    sub = "turing" if a.mode == "turing" else "human_eval"
    kit = Path("reports") / a.name / sub
    key = json.loads((kit / "key.json").read_text(encoding="utf-8"))
    choices_path = Path(a.choices) if a.choices else kit / "choices.json"
    if not choices_path.exists():
        print(f"no choices file at {choices_path} — rate in rater.html and download choices.json")
        return
    choices = json.loads(choices_path.read_text(encoding="utf-8"))

    if a.mode == "turing":
        _report_turing(a.name, choices, key, kit)
    else:
        _report_ab(a.name, choices, key, kit)


if __name__ == "__main__":
    main()
