"""Score a completed blind LoRA-vs-real (Turing) panel.

Run after rating in ``reports/<name>/turing/rater.html`` and downloading
``choices.json`` into that folder. Joins choices against ``key.json`` (and, when
present, ``data/eval/compare/<name>/pairs.jsonl`` for a per-language split).

    uv run python scripts/score_turing_eval.py --name main

Verdict: a Wilson 95% CI on the machine-detection rate that INCLUDES 0.5 means
the LoRA is statistically indistinguishable from Bohdan (it passes the Turing
test); a CI above 0.5 means he can tell. Tells from the correct catches are split
voice-vs-knowledge to localize the gap (knowledge -> RAG; voice -> decode/train).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from persona_rag.eval.compare import score_detection


def _fmt(x: object) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a blind Turing (LoRA-vs-real) panel.")
    ap.add_argument("--name", default="main")
    ap.add_argument("--choices", default="", help="path to choices.json (default: in the kit dir)")
    a = ap.parse_args()

    kit = Path("reports") / a.name / "turing"
    key = json.loads((kit / "key.json").read_text(encoding="utf-8"))
    choices_path = Path(a.choices) if a.choices else kit / "choices.json"
    if not choices_path.exists():
        print(f"no choices file at {choices_path} — rate in rater.html and download choices.json")
        return
    choices = json.loads(choices_path.read_text(encoding="utf-8"))

    overall = score_detection(choices, key)

    # Optional per-language split via the run's pairs.jsonl (item_id -> lang).
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
                by_lang[bucket] = score_detection(sub, key)

    result = {"name": a.name, "overall": overall, "by_language": by_lang}
    (kit / "turing_scorecard.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lo, hi = overall["wilson_95ci"]
    t = overall["tells"]
    dr = _fmt(overall["detection_rate"])
    rated = overall["decisive"] + overall["unsure"]
    print("=" * 60)
    print(f"  TURING PANEL (LoRA vs real Bohdan) -- {a.name}")
    print("=" * 60)
    print(f"  rated {rated}   decisive {overall['decisive']}   unsure {overall['unsure']}")
    print(f"  machine caught {overall['machine_caught']}   mistaken {overall['human_mistaken']}")
    print(f"  detection rate {dr}   Wilson 95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  VERDICT: {overall['verdict']}")
    print("  (detection CI that includes 0.5 = indistinguishable = LoRA passes as Bohdan)")
    print("-" * 60)
    print(f"  catch tells: voice {t['voice']}  knowledge {t['knowledge']}  other {t['other']}")
    if t["voice"] or t["knowledge"]:
        vf, kf = _fmt(t["voice_frac"]), _fmt(t["knowledge_frac"])
        print(f"    of categorized: voice {vf} | knowledge {kf}")
    if t["by_tag"]:
        ranked = sorted(t["by_tag"].items(), key=lambda kv: -kv[1])
        print("    by tag: " + "  ".join(f"{k}={v}" for k, v in ranked))
    if by_lang:
        print("-" * 60)
        for b, r in by_lang.items():
            d = _fmt(r["detection_rate"])
            print(f"  {b:9s} n={r['decisive']:<3} detect={d} -> {r['verdict']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
