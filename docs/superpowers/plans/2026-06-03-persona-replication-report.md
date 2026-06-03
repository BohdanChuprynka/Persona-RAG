# Persona-Replication Report — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a comprehensive, paper-formal Typst→PDF report telling the *replication story* — how we built and honestly evaluated a fine-tuned model that texts like one specific person (Qwen2.5-3B LoRA vs. the shipped gpt-4o-mini RAG product) — backed by all existing evidence plus three computable rigor additions.

**Architecture:** Reuse everything already built. Add (1) a hand-rolled per-item effect-size module + script, (2) one new figure-rendering script for cross-arm/new figures, (3) cetz diagrams, (4) a Typst report package under `report/`. Render existing per-run charts by re-running the existing plotter. No new model generation, no network calls beyond Typst package fetch.

**Tech Stack:** Python 3.12 + uv; pytest; hand-rolled stats (no scipy); matplotlib (already in env); Typst (+ `cetz` diagrams, `arkheion`-style template); hayagriva/BibTeX for refs.

**Spec:** `docs/superpowers/specs/2026-06-03-persona-replication-report-design.md` — read it first; it holds the frame, per-section content map, figure table, and the metrics ADD/PRESENT/AVOID lists.

**Privacy gate (every task):** aggregate-only output. Never write a raw message into `report/`. Figures are distributional. `report/` is committable; `data/` and `reports/` stay git-ignored.

**Source-of-truth numbers** live in the findings docs (`docs/superpowers/2026-06-02-{comparison-findings,arm-a-findings,eval-architecture-audit,turing-test-design}.md`) and the run JSONs under `data/eval/compare/{main,seed1,armA,armA_learned,armA_leakon,armA_leakoff}/`. Prose tasks cite these; do not invent numbers.

---

## Task 0: Prerequisites & baseline render

**Files:** none (environment + existing scripts).

- [ ] **Step 1: Install Typst**

Run: `brew install typst`
Expected: `typst --version` prints a version (≥ 0.12).

- [ ] **Step 2: Confirm matplotlib + the runs are present**

Run: `uv run python -c "import matplotlib; print(matplotlib.__version__)"`
Expected: a version prints (no ImportError).
Run: `ls data/eval/compare/{main,seed1,armA,armA_learned,armA_leakon,armA_leakoff}/results.json data/eval/compare/armA/by_language.json data/eval/compare/{main,armA}/pairs.jsonl`
Expected: all paths exist.

- [ ] **Step 3: Create the report skeleton dirs**

Run: `mkdir -p report/parts report/fig report/diagrams`
Expected: directories exist (empty).

- [ ] **Step 4: Commit the scaffold**

```bash
git add report/.gitkeep 2>/dev/null; touch report/.gitkeep && git add report/.gitkeep
git commit -m "chore(report): scaffold report/ dirs for the replication writeup"
```

---

## Task 1: Per-item effect-size module (TDD)

**Files:**
- Create: `persona_rag/eval/effect_size.py`
- Test: `tests/test_eval_effect_size.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eval_effect_size.py
"""Tests for the per-item length effect-size primitives."""

from __future__ import annotations

import math

from persona_rag.eval.effect_size import (
    cliffs_delta,
    cliffs_magnitude,
    length_effect_sizes,
    mean_bubble_len,
    per_item_length_errors,
    sign_test,
    wilcoxon_signed_rank,
)


def test_mean_bubble_len_multibubble() -> None:
    # bubbles "a" (1) and "bb" (2) -> mean 1.5
    assert mean_bubble_len("a\nbb") == 1.5


def test_mean_bubble_len_empty_is_zero() -> None:
    assert mean_bubble_len("   ") == 0.0


def test_per_item_length_errors_absolute() -> None:
    real = ["abcd", "ab"]      # lens 4, 2
    gen = ["a", "abcdef"]      # lens 1, 6 -> errors 3, 4
    assert per_item_length_errors(real, gen) == [3.0, 4.0]


def test_cliffs_delta_full_dominance() -> None:
    # every x > every y -> +1
    assert cliffs_delta([3, 4, 5], [1, 2]) == 1.0
    # every x < every y -> -1
    assert cliffs_delta([1, 2], [3, 4, 5]) == -1.0


def test_cliffs_magnitude_thresholds() -> None:
    assert cliffs_magnitude(0.10) == "negligible"
    assert cliffs_magnitude(0.20) == "small"
    assert cliffs_magnitude(0.40) == "medium"
    assert cliffs_magnitude(0.95) == "large"


def test_sign_test_all_positive() -> None:
    out = sign_test([1.0, 2.0, 0.5, 3.0])
    assert out["pos"] == 4 and out["neg"] == 0 and out["n"] == 4
    assert out["p"] < 0.13  # 2 * 0.5^4 = 0.125


def test_sign_test_drops_ties() -> None:
    out = sign_test([1.0, 0.0, -1.0, 0.0])
    assert out["n"] == 2 and out["pos"] == 1 and out["neg"] == 1


def test_wilcoxon_symmetric_is_nonsignificant() -> None:
    out = wilcoxon_signed_rank([1.0, -1.0, 2.0, -2.0])
    assert out["w_plus"] == out["w_minus"]
    assert out["p"] > 0.9  # z≈0


def test_length_effect_sizes_lora_closer() -> None:
    # LoRA matches real exactly; API is far off on every item.
    real = ["ab", "abcd", "abcdef"]
    gen_api = ["abcdefghijkl", "x", "abcdefghijklmnop"]
    gen_lora = ["ab", "abcd", "abcdef"]
    out = length_effect_sizes(real, gen_api, gen_lora)
    assert out["lora_closer"] == 3 and out["api_closer"] == 0
    assert out["cliffs_delta"] == 1.0  # API errors dominate => LoRA closer
    assert out["cliffs_magnitude"] == "large"
    assert not math.isnan(out["sign_test_p"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval_effect_size.py -q`
Expected: FAIL — `ModuleNotFoundError: persona_rag.eval.effect_size`.

- [ ] **Step 3: Implement the module**

```python
# persona_rag/eval/effect_size.py
# Reason: Cyrillic literals appear in sibling eval tests for the metrics.
"""Per-item effect sizes for the persona comparison (audit R4 follow-up).

The paired bootstrap in ``compare.py`` gives a corpus-level CI on the
length-distance delta; this adds the *per-item* complement the audit prescribed:
a standardized effect size (Cliff's delta + matched-pairs rank-biserial) plus
assumption-light significance (sign test + Wilcoxon signed-rank) on per-item
reply-length error. No new generation — reads the same aligned (real, gen_api,
gen_lora) triples the scorecard already scores. Hand-rolled (no scipy) to match
the eval-core house style.

Convention: ``diff_i = err_api_i - err_lora_i``; a POSITIVE diff means the LoRA
is closer to the real reply on item i. ``cliffs_delta(err_api, err_lora)`` is
+1 when API errors dominate (i.e. the LoRA is closer everywhere).
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Any

from persona_rag.generate.bubbles import split_bubbles

# Cliff's delta magnitude thresholds (Romano et al. 2006).
_CLIFF_SMALL, _CLIFF_MEDIUM, _CLIFF_LARGE = 0.147, 0.33, 0.474


def mean_bubble_len(text: str) -> float:
    """Mean character length of a reply's non-empty bubbles; 0.0 if none."""
    bubbles = [b for b in split_bubbles(text) if b.strip()]
    return mean(len(b) for b in bubbles) if bubbles else 0.0


def per_item_length_errors(real: list[str], gen: list[str]) -> list[float]:
    """|mean-bubble-length(gen_i) − mean-bubble-length(real_i)| per aligned item."""
    return [abs(mean_bubble_len(g) - mean_bubble_len(r)) for r, g in zip(real, gen, strict=True)]


def cliffs_delta(xs: list[float], ys: list[float]) -> float:
    """Cliff's delta = P(x>y) − P(x<y) over all pairs. +1 ⇒ xs dominate."""
    if not xs or not ys:
        return float("nan")
    gt = sum(1 for x in xs for y in ys if x > y)
    lt = sum(1 for x in xs for y in ys if x < y)
    return (gt - lt) / (len(xs) * len(ys))


def cliffs_magnitude(delta: float) -> str:
    d = abs(delta)
    if math.isnan(d):
        return "undefined"
    if d < _CLIFF_SMALL:
        return "negligible"
    if d < _CLIFF_MEDIUM:
        return "small"
    if d < _CLIFF_LARGE:
        return "medium"
    return "large"


def _avg_ranks(values: list[float]) -> list[float]:
    """1-based average ranks; ties share the mean of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def wilcoxon_signed_rank(diffs: list[float]) -> dict[str, float]:
    """Wilcoxon signed-rank on paired diffs (zeros dropped). Normal approx with
    continuity correction (exact is overkill at n≈300). Two-sided p via erfc."""
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    if n == 0:
        return {"n": 0, "w_plus": float("nan"), "w_minus": float("nan"), "z": float("nan"), "p": float("nan")}
    ranks = _avg_ranks([abs(d) for d in nz])
    w_plus = sum(r for r, d in zip(ranks, nz, strict=True) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nz, strict=True) if d < 0)
    mean_w = n * (n + 1) / 4
    sd_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if not sd_w:
        return {"n": n, "w_plus": w_plus, "w_minus": w_minus, "z": float("nan"), "p": float("nan")}
    z = (min(w_plus, w_minus) - mean_w + 0.5) / sd_w
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"n": n, "w_plus": w_plus, "w_minus": w_minus, "z": z, "p": p}


def matched_pairs_rank_biserial(diffs: list[float]) -> float:
    """r = (W+ − W−) / (W+ + W−) from the signed-rank sums."""
    w = wilcoxon_signed_rank(diffs)
    denom = w["w_plus"] + w["w_minus"]
    if math.isnan(denom) or denom == 0:
        return float("nan")
    return (w["w_plus"] - w["w_minus"]) / denom


def sign_test(diffs: list[float]) -> dict[str, float]:
    """Two-sided exact binomial sign test (zeros = ties dropped)."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return {"n": 0, "pos": 0, "neg": 0, "p": float("nan")}
    k = max(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) * (0.5**n)
    return {"n": n, "pos": pos, "neg": neg, "p": min(1.0, 2 * tail)}


def length_effect_sizes(
    real: list[str], gen_api: list[str], gen_lora: list[str]
) -> dict[str, Any]:
    """Bundle of per-item length-error effect sizes, API vs LoRA."""
    err_api = per_item_length_errors(real, gen_api)
    err_lora = per_item_length_errors(real, gen_lora)
    diffs = [a - b for a, b in zip(err_api, err_lora, strict=True)]
    delta = cliffs_delta(err_api, err_lora)
    sign = sign_test(diffs)
    return {
        "n_items": len(real),
        "cliffs_delta": delta,
        "cliffs_magnitude": cliffs_magnitude(delta),
        "lora_closer": sign["pos"],
        "api_closer": sign["neg"],
        "ties": len(real) - sign["n"],
        "sign_test_p": sign["p"],
        "rank_biserial": matched_pairs_rank_biserial(diffs),
        "wilcoxon": wilcoxon_signed_rank(diffs),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval_effect_size.py -q`
Expected: PASS (9 tests).

- [ ] **Step 5: Lint, type, commit**

Run: `uv run ruff format persona_rag/eval/effect_size.py tests/test_eval_effect_size.py && uv run ruff check persona_rag tests && uv run mypy persona_rag && uv run pre-commit run ruff-format --files persona_rag/eval/effect_size.py tests/test_eval_effect_size.py`
Expected: all clean (both ruff versions agree).

```bash
git add persona_rag/eval/effect_size.py tests/test_eval_effect_size.py
git commit -m "feat(eval): per-item length effect sizes (Cliff's delta, sign, Wilcoxon)"
```

---

## Task 2: Effect-size CLI over a run's pairs.jsonl

**Files:**
- Create: `scripts/compute_effect_sizes.py`

- [ ] **Step 1: Write the script**

```python
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
```

- [ ] **Step 2: Run it on the three n=300 arms; sanity-check against the spec's claim**

Run: `uv run python scripts/compute_effect_sizes.py --name main && uv run python scripts/compute_effect_sizes.py --name armA && uv run python scripts/compute_effect_sizes.py --name armA_learned`
Expected: each prints a Cliff's delta + counts; `main` should report **Cliff's d ≈ 0.95, LoRA closer ≈ 292/300** (the spec's pre-verified figure). If it diverges materially, STOP and reconcile before writing prose around it.

- [ ] **Step 3: Lint + commit (effect_sizes.json files are git-ignored under data/)**

Run: `uv run ruff format scripts/compute_effect_sizes.py && uv run ruff check scripts && uv run pre-commit run ruff-format --files scripts/compute_effect_sizes.py`
```bash
git add scripts/compute_effect_sizes.py
git commit -m "feat(eval): CLI to emit per-run length effect sizes from pairs.jsonl"
```

---

## Task 3: Render existing per-run charts for Arm A (+ seed1)

**Files:** none (re-run the existing, unchanged `scripts/plot_comparison.py`).

- [ ] **Step 1: Render Arm A and seed1 charts**

Run: `uv run python scripts/plot_comparison.py --name armA && uv run python scripts/plot_comparison.py --name seed1`
Expected: `reports/armA/` and `reports/seed1/` each gain `headline_distances.png`, `voice_tics.png`, `shape_distribution.png`, `length_distribution.png`, `latency_cost.png`, `summary.md`. `reports/armA/headline_distances.png` is **F3**.

- [ ] **Step 2: Copy the report-used charts into `report/fig/` (aggregate-only, committable)**

Run:
```bash
cp reports/main/headline_distances.png report/fig/f2_armB_headline.png
cp reports/armA/headline_distances.png report/fig/f3_armA_headline.png
cp reports/main/voice_tics.png report/fig/voice_tics_armB.png
cp reports/main/length_distribution.png report/fig/length_dist_armB.png
cp reports/main/shape_distribution.png report/fig/shape_dist_armB.png
cp reports/armA/voice_tics.png report/fig/voice_tics_armA.png
```
Expected: the six PNGs exist under `report/fig/`.

- [ ] **Step 3: Commit the figures**

```bash
git add report/fig/*.png
git commit -m "feat(report): render Arm A per-run charts; stage F2/F3 + supporting figs"
```

---

## Task 4: `plot_report.py` — data-shaping helpers (TDD)

**Files:**
- Create: `scripts/plot_report.py` (helpers first; figures added in Task 5)
- Test: `tests/test_plot_report_helpers.py`

- [ ] **Step 1: Write the failing tests for the pure helpers**

```python
# tests/test_plot_report_helpers.py
from __future__ import annotations

from scripts.plot_report import collect_deltas, leak_rate, machinery_pairs


def _delta(d: float, lo: float, hi: float, ez: bool) -> dict:
    return {"delta": d, "ci_lo": lo, "ci_hi": hi, "excludes_zero": ez, "favored": "b"}


def test_collect_deltas_orders_and_extracts() -> None:
    runs = {
        "main": {"scorecard": {"deltas_api_minus_lora": {"len_wasserstein": _delta(125.9, 107.6, 142.4, True)}}},
        "armA": {"scorecard": {"deltas_api_minus_lora": {"len_wasserstein": _delta(3.57, 1.53, 4.66, True)}}},
    }
    rows = collect_deltas(runs, "len_wasserstein", order=["main", "armA"])
    assert [r["name"] for r in rows] == ["main", "armA"]
    assert rows[0]["delta"] == 125.9 and rows[0]["excludes_zero"] is True


def test_leak_rate_from_guard() -> None:
    on = {"retrieval_leak_guard": {"id_leaks": 17}, "scorecard": {"n_items": 60}}
    off = {"retrieval_leak_guard": {"id_leaks": 0}, "scorecard": {"n_items": 60}}
    assert leak_rate(on) == (17, 60)
    assert leak_rate(off) == (0, 60)


def test_machinery_pairs_api_side() -> None:
    main = {"scorecard": {"arms": {"api": {"len_wasserstein_vs_real": 128.8, "exclaim_rate": 0.651},
                                    "lora": {"len_wasserstein_vs_real": 2.9, "exclaim_rate": 0.0}}}}
    arma = {"scorecard": {"arms": {"api": {"len_wasserstein_vs_real": 6.97, "exclaim_rate": 0.0},
                                    "lora": {"len_wasserstein_vs_real": 3.41, "exclaim_rate": 0.0}}}}
    out = machinery_pairs(main, arma, "len_wasserstein_vs_real")
    assert out["api"] == (128.8, 6.97) and out["lora"] == (2.9, 3.41)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_plot_report_helpers.py -q`
Expected: FAIL — module/functions absent.

- [ ] **Step 3: Implement the helpers + loader at the top of `scripts/plot_report.py`**

```python
# Reason: reads run JSONs / pairs.jsonl whose replies contain Cyrillic.
"""Render the cross-arm / report-only figures for the replication report.

Reads the run JSONs under data/eval/compare/ (+ by_language.json, the human and
turing scorecards when present) and writes publication figures into report/fig/.
Pure data-shaping helpers are unit-tested; the matplotlib drawing mirrors the
untested render-script pattern in plot_comparison.py. Gated figures (F10/F11)
are skipped with a printed notice when their scorecard JSON is absent.

    uv run python scripts/plot_report.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

REAL_C, API_C, LORA_C = "#64748b", "#2563eb", "#16a34a"
COMPARE = Path("data/eval/compare")
FIG = Path("report/fig")


def load_run(name: str) -> dict[str, Any]:
    return json.loads((COMPARE / name / "results.json").read_text(encoding="utf-8"))


def collect_deltas(runs: dict[str, dict], metric: str, order: list[str]) -> list[dict[str, Any]]:
    rows = []
    for name in order:
        d = runs[name]["scorecard"]["deltas_api_minus_lora"][metric]
        rows.append({"name": name, **d})
    return rows


def leak_rate(run: dict[str, Any]) -> tuple[int, int]:
    return run["retrieval_leak_guard"]["id_leaks"], run["scorecard"]["n_items"]


def machinery_pairs(main: dict, arma: dict, field: str) -> dict[str, tuple[float, float]]:
    a, b = main["scorecard"]["arms"], arma["scorecard"]["arms"]
    return {
        "api": (a["api"][field], b["api"][field]),
        "lora": (a["lora"][field], b["lora"][field]),
    }
```

- [ ] **Step 4: Run to verify the helpers pass**

Run: `uv run pytest tests/test_plot_report_helpers.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint, type (scripts not type-gated, but format), commit**

Run: `uv run ruff format scripts/plot_report.py tests/test_plot_report_helpers.py && uv run ruff check scripts tests && uv run pre-commit run ruff-format --files scripts/plot_report.py tests/test_plot_report_helpers.py`
```bash
git add scripts/plot_report.py tests/test_plot_report_helpers.py
git commit -m "feat(report): plot_report data helpers (deltas/leak/machinery) + tests"
```

---

## Task 5: `plot_report.py` — render F1, F4, F5, F6, F7, F9 (+ gated F10/F11)

**Files:**
- Modify: `scripts/plot_report.py` (append figure functions + `main()`)

Each figure is a function `def fig_xN(...) -> None` that saves `FIG / "fN_*.png"` at `dpi=130`, zero-based bars, top/right spines hidden (match `plot_comparison.py`). Use the **finalized captions from the spec/figure-plan** as the Typst caption later — the chart titles here stay short. Data sources and the verdict each must show:

- [ ] **Step 1: F1 — leak guard 28%→0.** Two bars from `leak_rate(load_run("armA_leakon"))` = (17,60) and `leak_rate(load_run("armA_leakoff"))` = (0,60); y-axis percent, annotate `id_leaks/n`. Subtitle notes top_sim≈unchanged (0.386 vs 0.389). Save `f1_leak_guard.png`.

- [ ] **Step 2: F4 — what the machinery buys (B→A slope).** `machinery_pairs(main, armA, "len_wasserstein_vs_real")` and the same for `"exclaim_rate"`. Two small panels (length, exclaim); plot API as a line from "Arm B (bare)"→"Arm A (shipped)" (128.8→7.0; 0.651→0.000) and LoRA as a flat dashed reference (2.9→3.4; 0→0). Save `f4_machinery.png`.

- [ ] **Step 3: F5 — steered vs learned.** Grouped bars of API `exclaim_rate` shipped (`armA` = 0.000) vs learned (`armA_learned` = 0.033) — plus a small inset of the `len_wasserstein` delta + CI for both (`armA` 3.57 [1.53,4.66]; `armA_learned` 4.27 [2.51,5.07]) showing both still favor LoRA. Save `f5_steered_vs_learned.png`.

- [ ] **Step 4: F6 — per-language.** From `COMPARE/armA/by_language.json`: cyrillic (n=261) and latin (n=27) — grouped bars of `shape_js` and `len_wasserstein` API/LoRA, with the delta CI as whiskers; omit buckets < n=20. Save `f6_by_language.png`.

- [ ] **Step 5: F7 — forest plot (statistical centerpiece).** `collect_deltas(runs, metric, order=["main","seed1","armA","armA_learned","armA_leakoff","armA_leakon"])` for both `shape_js` and `len_wasserstein`. Two stacked facets; each row = point (delta) + horizontal 95% CI whiskers; dashed line at 0. Color rows whose `excludes_zero` is True. Save `f7_forest.png`. (Honesty: the n=60 leak arms will be wide ties — that is the point.)

- [ ] **Step 6: F9 — copy/near-copy vs natural floor.** From `armA` `scorecard.copy_leak`: grouped bars API (≈0/0) and LoRA (0.103/0.103) for {exact, near}, with `baseline_real_vs_train` (0.057/0.070) drawn as a horizontal gray band. Save `f9_copy_floor.png`.

- [ ] **Step 6b: F8 — operational profile.** From `load_run("main")` and `load_run("armA")` `operational`: three small panels — (a) p50/p95 latency API vs LoRA (armA 0.96/2.67 vs 1.01/4.23); (b) mean input tokens (armA API 2399.6 vs LoRA 211.6 — the context tax); (c) $/1k replies (armA $0.37 vs $0). Save `f8_operational.png`.

- [ ] **Step 7: F10/F11 — gated.** `def fig_human()` / `def fig_turing()`: if `reports/main/human_eval/human_scorecard.json` (resp. `.../turing/turing_scorecard.json`) is absent, `print("[skip] F10 — panel unrated")` and return. When present, render a single proportion bar with Wilson 95% CI whiskers (the scorecard already carries `wilson_95ci`) and a 0.5 chance line. Save `f10_human_winrate.png` / `f11_turing_detect.png`.

- [ ] **Step 8: `main()`** — `matplotlib.use("Agg")`, `FIG.mkdir(parents=True, exist_ok=True)`, call each `fig_*`, wrapping the gated two so a missing scorecard never aborts the rest.

- [ ] **Step 9: Render + eyeball**

Run: `uv run python scripts/plot_report.py`
Expected: prints saved paths for F1,F4,F5,F6,F7,F8,F9 and two `[skip]` notices for F10/F11. Open `report/fig/f7_forest.png` and confirm the length facet shows `main`/`seed1`/`armA`/`armA_learned` excluding 0 and the two n=60 arms straddling 0.

- [ ] **Step 10: Lint + commit**

Run: `uv run ruff format scripts/plot_report.py && uv run ruff check scripts && uv run pre-commit run ruff-format --files scripts/plot_report.py`
```bash
git add scripts/plot_report.py report/fig/f1_leak_guard.png report/fig/f4_machinery.png report/fig/f5_steered_vs_learned.png report/fig/f6_by_language.png report/fig/f7_forest.png report/fig/f8_operational.png report/fig/f9_copy_floor.png
git commit -m "feat(report): render report figures F1,F4-F9 (F10/F11 gated on ratings)"
```

---

## Task 6: Diagrams D1–D4 (cetz)

**Files:**
- Create: `report/diagrams/d1_architecture.typ`, `d2_split_leak.typ`, `d3_two_arm.typ`, `d4_train_serve.typ`

Each is a standalone `cetz` canvas `#import "@preview/cetz:0.3.1": canvas, draw` that the main report includes via `#image`-equivalent or direct `#include`. Author boxes/arrows from the spec's `diagram_candidates`. Diagrams are visually iterated — compile and eyeball each.

- [ ] **Step 1: D3 first (simplest — two-arm design), as the cetz pattern**

```typst
// report/diagrams/d3_two_arm.typ
#import "@preview/cetz:0.3.1": canvas, draw
#canvas({
  import draw: *
  let box(p, w, h, label, fill) = {
    rect((p.at(0) - w/2, p.at(1) - h/2), (p.at(0) + w/2, p.at(1) + h/2),
      fill: fill, stroke: 0.6pt + gray, radius: 2pt)
    content(p, text(8pt)[#label])
  }
  // Arm B: identical thin prompt -> both backends (isolates weights)
  box((0, 3), 3.4, 0.8, [Arm B — identical *thin* prompt], rgb("#eef2ff"))
  box((-2.2, 1.6), 2.2, 0.7, [gpt-4o-mini], rgb("#dbeafe"))
  box((2.2, 1.6), 2.2, 0.7, [LoRA (thin)], rgb("#dcfce7"))
  line((0, 2.6), (-2.2, 1.95), mark: (end: ">"))
  line((0, 2.6), (2.2, 1.95), mark: (end: ">"))
  // Arm A: shipped product vs thin LoRA (isolates the product)
  box((0, 0), 3.8, 0.8, [Arm A — shipped API stack vs thin LoRA], rgb("#fef9c3"))
  content((0, -1.1), text(7.5pt, fill: gray)[shared scorer · paired bootstrap CIs · leak + copy guards])
})
```

- [ ] **Step 2: Compile-check D3 in isolation**

Create a throwaway `report/diagrams/_check.typ` that imports + shows D3, run `typst compile report/diagrams/_check.typ /tmp/d3.pdf`, confirm it builds, then delete `_check.typ`.
Expected: PDF builds with no cetz errors.

- [ ] **Step 3: Author D1 (system/dual-backend), D2 (data split + leak), D4 (train==serve)** following the same `box`/`line` pattern, content per the spec's `diagram_candidates`:
  - **D1:** Telegram → aiogram → LangGraph chain → `openai_chat` fanning to **gpt-4o-mini** vs **llama.cpp llama-server (LoRA GGUF)**; side stores Qdrant / BM25 / SQLite; show the `GENERATION_BACKEND` branch skipping retrieval on the local path.
  - **D2:** export → PII → bursts → sessions → turns → **fork A** temporal `eval_split` (label *register-skewed, one EN contact = 62% of Latin → 0.47 artifact*, drives indexing) and **fork B** recipient-stratified `eval_split_for` (label *train≈eval≈0.18*, drives train/eval JSONL). Annotate the retrieval-leak point that F1 fixes.
  - **D4:** Colab T4 QLoRA (Qwen2.5-3B, `train_on_responses_only`) → merge → local convert+quantize → **GGUF Q5_K_M** → llama-server; highlight the single `THIN_SYSTEM` string flowing into BOTH train and serve.

- [ ] **Step 4: Commit**

```bash
git add report/diagrams/*.typ
git commit -m "feat(report): cetz diagrams D1-D4 (architecture, split/leak, two-arm, train==serve)"
```

---

## Task 7: Typst report scaffold + `make report`

**Files:**
- Create: `report/report.typ`, `report/refs.bib`, `report/parts/{part1,part2,part3,part4,part5,appendix}.typ` (empty stubs for now)
- Modify: `Makefile` (add `report` target + `.PHONY`)

- [ ] **Step 1: Write `report/report.typ` (preamble + title block + abstract + includes)**

```typst
#set document(title: "Replicating a Texting Voice")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { v(0.6em); text(13pt, weight: "bold")[#it]; v(0.2em) }
#set figure(numbering: "1")
#show figure.caption: set text(9pt)

#align(center)[
  #text(18pt, weight: "bold")[Replicating a Texting Voice]
  #v(0.3em)
  #text(11pt)[Building and honestly evaluating a fine-tuned persona model \
  of one person, against a production RAG + GPT-4o-mini baseline]
  #v(0.5em)
  #text(10pt)[Bohdan Chuprynka · #datetime.today().display()]
]

#align(center)[
  #block(width: 88%, inset: 8pt)[
    #set par(justify: true)
    #text(9.5pt)[*Abstract.* // ~180 words — written in Task 12, after results are final.
    ]
  ]
]

#v(0.5em)
#include "parts/part1.typ"
#include "parts/part2.typ"
#include "parts/part3.typ"
#include "parts/part4.typ"
#include "parts/part5.typ"
#include "parts/appendix.typ"

#bibliography("refs.bib", style: "ieee", title: "References")
```

- [ ] **Step 2: Seed `refs.bib` with the core citations**

Entries (BibTeX): Qwen2.5 (Qwen team 2024/2025), LoRA (Hu et al. 2021), QLoRA (Dettmers et al. 2023), Unsloth (project), Okapi BM25 (Robertson & Zaragoza 2009), MMR (Carbonell & Goldstein 1998), Wasserstein/earth-mover (Rubner et al. 2000), Wilson interval (Wilson 1927), Cliff's delta (Cliff 1993 / Romano 2006), llama.cpp (project), text-embedding-3 / GPT-4o-mini (OpenAI). Minimal but real keys: `@qwen2_5`, `@hu2021lora`, `@dettmers2023qlora`, `@robertson2009bm25`, `@carbonell1998mmr`, `@rubner2000emd`, `@wilson1927`, `@cliff1993`, `@llamacpp`.

- [ ] **Step 3: Create empty part stubs** so the include compiles:

```typst
// report/parts/part1.typ  (and part2..part5, appendix likewise — one heading each)
= Building the replica
// content added in Task 8
```

- [ ] **Step 4: Add the Makefile target**

In `Makefile`, add `report` to the first `.PHONY` line and append:
```makefile
# Build the replication report: render report figures, then compile the PDF.
# Needs `brew install typst`; F10/F11 render only once the human panels are rated.
report:
	uv run python scripts/plot_report.py
	typst compile report/report.typ report/persona-rag-report.pdf
	@echo "wrote report/persona-rag-report.pdf"
```

- [ ] **Step 5: Compile the skeleton**

Run: `typst compile report/report.typ report/persona-rag-report.pdf`
Expected: a PDF builds — title block, abstract placeholder, six empty numbered sections, an (empty) references heading. cetz/ieee packages fetch on first run (needs network once).

- [ ] **Step 6: Commit**

```bash
git add report/report.typ report/refs.bib report/parts/*.typ Makefile
git commit -m "feat(report): Typst scaffold (title/abstract/includes) + make report target"
```

---

## Task 8: Draft Part I — Building the replica

**Files:** Modify `report/parts/part1.typ`

Prose drawn from **spec §3 Part I** + system facts in the workflow inventory (architecture, data_pipeline, rag_retrieval, lora_training, serving, decode_levers). Tell it as an implementation journey. Subsections + must-include content:

- [ ] **Step 1: Write the five subsections**
  - **1.1 The target.** What "sounds like Bohdan" is: uk/ru/en code-switch (~0.18 aggregate Latin), terse multi-bubble bursts, the `)` smiley tic, *no* `!`, varied openers, lowercase. State the project's one subjective metric.
  - **1.2 System architecture.** aiogram → LangGraph node chain → `openai_chat` dual backend. Reference `@fig-d1`.
  - **1.3 Data → turns → the split problem.** export → PII redact → burst-collapse (300s) → session-split (6h, ≥4 turns) → `(context→reply)` turns; the **two splits** and *why split #2 exists* (temporal tail register-skew: one EN contact = 62% of all Latin → 0.47 artifact; recipient-stratified hash → train≈eval≈0.18). Reference `@fig-d2`.
  - **1.4 Retrieval.** hybrid dense (`text-embedding-3-small`, Qdrant) + BM25, fuse α=0.7, recency 180d half-life, floor 0.15, MMR λ=0.6 → top-4 reversed; insights distillation; the ~1600-token `SYSTEM_TEMPLATE`.
  - **1.5 The fine-tune + serving.** Qwen2.5-3B QLoRA r=32/α=64, `train_on_responses_only`, the byte-identical `THIN_SYSTEM` **train==serve** invariant; Colab T4 → merge → local `convert_hf_to_gguf` + `llama-quantize` → **Q5_K_M** → `llama-server` (and *why* not Ollama: brew formula ships no runtime). Decode levers `PAREN=+2/EXCLAIM=−5` (OpenAI-only). Reference `@fig-d4`. Close on the finding that motivates the whole effort: shape obeys prompt-conditioning, **lexical tics do not → fine-tuning was necessary.**
  - Embed D1/D2/D4 as `#figure([#include "../diagrams/dN_*.typ"], caption: [...]) <fig-dN>`.

- [ ] **Step 2: Compile + commit**

Run: `typst compile report/report.typ report/persona-rag-report.pdf`
Expected: Part I renders with the three diagrams and cross-refs resolve.
```bash
git add report/parts/part1.typ && git commit -m "docs(report): Part I — building the replica (system, data, fine-tune)"
```

---

## Task 9: Draft Part II — Can we trust the measurement?

**Files:** Modify `report/parts/part2.typ`

Prose from **spec §3 Part II** + `eval-architecture-audit.md` + `compare.py` metric definitions.

- [ ] **Step 1: Write the subsections**
  - **2.1 Metrics.** Formal definitions *with math* (`$...$`): `shape_js` (JS-divergence of bubble-count histograms), `len_wasserstein` (+norm), `exclaim_rate`, `paren_smiley_rate`, `opener_entropy`, `distinct_reply_rate`, `copy/near-copy` vs the measured natural floor, language buckets, leak telemetry. Use the spec's `metric_definitions`.
  - **2.2 The audit.** The ~90% asymmetric train/test leak (scored the temporal `eval_split`; the LoRA held out the recipient-stratified `eval_split_for` → ~90% of scored turns were in the LoRA's train pool, API saw none); plus the prompt/retrieval/lever confounds and absence of CIs. The honest "I found my own eval was broken" beat. Reference `@fig-d2`.
  - **2.3 The fair harness.** paired bootstrap CIs (2000 resamples), the LoRA-disjoint split for *both* arms, NaN-not-0.0, n=300 (drops the shape noise floor ~0.06→0.02).
  - **2.4 Two-arm design.** Arm B isolates **weights**; Arm A isolates the **product**. Reference `@fig-d3`.
  - **2.5 The retrieval leak guard.** per-item `exclude_ids`, the two-leg assertion; **28%→0** proven. Reference `@fig-leak` (F1).
  - **2.6 Pre-registered "better" + ship rule.** State the rule *before* results (human win-rate Wilson-CI-excludes-0.5; guardrails override; ties break to cheaper/local). Note it was fixed in advance — the structural defense against metric-shopping.

- [ ] **Step 2: Compile + commit**
```bash
git add report/parts/part2.typ && git commit -m "docs(report): Part II — trustworthy measurement (audit, harness, guard)"
```

---

## Task 10: Draft Part III — Relative fidelity

**Files:** Modify `report/parts/part3.typ`

Prose from **spec §3 Part III** + `comparison-findings.md` + `arm-a-findings.md` + the computed `effect_sizes.json`. Use the spec's `reusable_tables` verbatim as Typst tables.

- [ ] **Step 1: Write the subsections, each anchored to its figure/table**
  - **3.1 Arm B (controlled).** Table: arm-B two-seed. LoRA length EMD 2.9 vs 128.8 (CI [107.6,142.4]), `!` 0.00 vs 0.65, shape tie. `@fig-armB` (F2), supporting voice_tics/length/shape in Appendix D.
  - **3.2 Arm A (production).** Table: arm-A headline. machinery closes the gap; LoRA still wins length 3.4 vs 7.0 (CI [1.5,4.7]) + openers 5.76 vs 3.70; shape/`!` tie. `@fig-armA` (F3).
  - **3.3 What the machinery buys.** B→A API side: len 128.8→7.0, `!` 0.65→0.00, openers **regress** 5.02→3.70. `@fig-machinery` (F4) + the machinery table.
  - **3.4 Steered vs learned.** `!` 0.65→0.033 (prompt) →0.000 (bias); verdict unchanged. `@fig-steered` (F5) + steered/learned table.
  - **3.5 Per-language.** cyrillic (87%, n=261) wins length; English (n=27) tie + shared weakness (low-confidence). `@fig-lang` (F6) + per-language table.
  - **3.6 Effect size & robustness.** From `effect_sizes.json`: **Cliff's δ ≈ 0.95 (large), LoRA closer ≈ 292/300, sign-test p ≪ 1e-50**, rank-biserial; the forest plot across all six runs (incl. seed1 replication, leak arms underpowered). `@fig-forest` (F7).
  - **3.7 Anti-memorization.** LoRA copy 10.3% vs natural floor 7.0% (real-vs-train); not parroting. `@fig-copy` (F9).
  - **3.8 Operational.** $0 vs $0.37/1k; ~11× context tax (2400 vs 210 tok); p50 near-tie. `@fig-ops` (F8).

- [ ] **Step 2: Compile + commit**
```bash
git add report/parts/part3.typ && git commit -m "docs(report): Part III — relative fidelity (arms, ablations, effect sizes)"
```

---

## Task 11: Draft Part IV — Absolute fidelity

**Files:** Modify `report/parts/part4.typ`

Prose from **spec §3 Part IV** + `turing-test-design.md`. **Gated content uses clearly-marked placeholders.**

- [ ] **Step 1: Write the subsections**
  - **4.1 The blind human panel.** Protocol (forced choice, randomized, key/blind split). *Current* qualitative verdict: the API is **trivially discriminable** by eye → LoRA wins voice. `@fig-human` (F10) rendered as a placeholder note: *"Awaiting ratings — win-rate + Wilson CI populate `score_human_eval.py` output once `choices.json` exists."*
  - **4.2 The Turing test (the climax).** The absolute question: can Bohdan tell the LoRA from his *own* reply? The pass condition **flips** — detection Wilson CI *includes* 0.5 = indistinguishable. `@fig-turing` (F11) placeholder.
  - **4.3 The tell taxonomy.** voice tells (decode/training, fixable by tuning) vs knowledge=missing-facts (fixable by RAG/grounding); the split **sizes the RAG business case**. Prior (to be replaced by data): detection ~65–75%, ~40–55% missing-facts.

- [ ] **Step 2: Compile + commit**
```bash
git add report/parts/part4.typ && git commit -m "docs(report): Part IV — absolute fidelity (human panel, Turing climax, tells)"
```

---

## Task 12: Draft Part V + appendices + abstract + references

**Files:** Modify `report/parts/part5.typ`, `report/parts/appendix.typ`, `report/report.typ` (abstract), `report/refs.bib`

- [ ] **Step 1: Part V — What we learned.**
  - **5.1 Verdict & triangulation.** Under the pre-registered rule the **LoRA ships**; three independent methods concur (arm B, arm A, human eye).
  - **5.2 Threats to validity.** The full list from the spec/rigor inventory: construct validity (surface proxies), single rater (rationale + κ caveat), single decode at temp 0.8, leakage residual (paper-grade needs a leak-free re-train), external validity (87% cyrillic, one person), prompt/retrieval confound, decode-lever asymmetry + tokenizer-budget caveat, corpus-level cross-arm, replay fidelity, multiple comparisons (one pre-registered primary), provenance gap (adapter/quant hash unpinned, MLflow uncalled).
  - **5.3 Conclusion & future work.** rate the panels → win-rate + κ; the RAG/grounding layer the tells justify; leak-free re-train for a paper-grade claim.

- [ ] **Step 2: Appendices.** A: reproduce commands (`make compare`, `make compare-arma`, the `--learned`/`--leak-on` runs, `compute_effect_sizes.py`, `make report`) + pinned params (seed, temp 0.8, n_boot 2000, model ids, split). B: metric formulas (expanded). C: full per-run tables (all six runs). D: supporting charts (voice_tics/length/shape for both arms via the `report/fig/*_armA.png` / `*_armB.png` assets).

- [ ] **Step 3: Abstract.** Write the ~180-word abstract in `report.typ` now that every number is fixed (use the spec's `headline_verdict` as the basis, trimmed).

- [ ] **Step 4: Finalize `refs.bib`** — ensure every `@cite` used in the prose has an entry; remove unused.

- [ ] **Step 5: Compile + commit**
```bash
git add report/parts/part5.typ report/parts/appendix.typ report/report.typ report/refs.bib
git commit -m "docs(report): Part V, appendices, abstract, references"
```

---

## Task 13: Final polish + full build

**Files:** any of the above as needed.

- [ ] **Step 1: Full clean build via the Makefile target**

Run: `make report`
Expected: figures render (with two `[skip]` notices for F10/F11), `report/persona-rag-report.pdf` compiles with **zero** Typst warnings about unresolved references or missing figures.

- [ ] **Step 2: Cross-reference + caption audit**

Open the PDF. Verify: every `@fig-*` / `@cite` resolves (no `(??)`); every figure has a numbered, publication-quality caption (use the captions written in the figure-plan/spec verbatim); figures sit near their referencing text; no raw message text anywhere (privacy).

- [ ] **Step 3: Self-review against the spec**

Walk spec §3 (the 5-Part table) — every section + figure present? Walk §5 — all three ADD metrics in, every AVOID item absent? Walk §6 — F10/F11 marked pending, not faked? Fix any gaps inline.

- [ ] **Step 4: Run the whole test + lint suite once**

Run: `uv run pytest -q && uv run ruff check persona_rag tests scripts && uv run mypy persona_rag`
Expected: green (the report adds `effect_size.py` + two test files; everything else untouched).

- [ ] **Step 5: Final commit**
```bash
git add -A report/ && git commit -m "docs(report): final polish — clean compile, cross-refs, captions, self-review"
```

---

## Completion

After all tasks: announce using superpowers:finishing-a-development-branch. The deliverable is `report/persona-rag-report.pdf` (committable, aggregate-only) + the reusable `effect_size.py`, `compute_effect_sizes.py`, `plot_report.py`. F10/F11 + the construct-validity analysis fill in later by rating `reports/main/{human_eval,turing}/rater.html` → `make report` re-renders. The title remains open for a later tweak.
