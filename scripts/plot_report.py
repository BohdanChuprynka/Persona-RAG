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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REAL_C, API_C, LORA_C, MUTE_C = "#64748b", "#2563eb", "#16a34a", "#94a3b8"
COMPARE = Path("data/eval/compare")
FIG = Path("report/fig")


# --------------------------------------------------------------------------- #
# pure data-shaping helpers (unit-tested)
# --------------------------------------------------------------------------- #
def load_run(name: str) -> dict[str, Any]:
    return json.loads((COMPARE / name / "results.json").read_text(encoding="utf-8"))


def collect_deltas(runs: dict[str, dict], metric: str, order: list[str]) -> list[dict[str, Any]]:
    """Pull the api-minus-lora delta + CI for one metric across runs, in order."""
    rows = []
    for name in order:
        d = runs[name]["scorecard"]["deltas_api_minus_lora"][metric]
        rows.append({"name": name, **d})
    return rows


def leak_rate(run: dict[str, Any]) -> tuple[int, int]:
    return run["retrieval_leak_guard"]["id_leaks"], run["scorecard"]["n_items"]


def machinery_pairs(main: dict, arma: dict, field: str) -> dict[str, tuple[float, float]]:
    a, b = main["scorecard"]["arms"], arma["scorecard"]["arms"]
    return {"api": (a["api"][field], b["api"][field]), "lora": (a["lora"][field], b["lora"][field])}


# --------------------------------------------------------------------------- #
# drawing (mirrors plot_comparison.py; not unit-tested)
# --------------------------------------------------------------------------- #
def _despine(ax: Any) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig: Any, name: str) -> None:
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG / name
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def fig_leak_guard() -> None:
    on, off = load_run("armA_leakon"), load_run("armA_leakoff")
    on_leaks, on_n = leak_rate(on)
    off_leaks, off_n = leak_rate(off)
    vals = [100 * on_leaks / on_n, 100 * off_leaks / off_n]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    bars = ax.bar(
        ["leak-on\n(exclusion off)", "leak-off\n(shipped guard)"], vals, color=[API_C, LORA_C]
    )
    ax.bar_label(bars, labels=[f"{on_leaks}/{on_n}", f"{off_leaks}/{off_n}"], padding=3)
    ax.set_ylim(0, max(vals) * 1.3 or 1)
    ax.set_ylabel("gold answer-key retrieved into few-shot (%)")
    ax.set_title("Retrieval leak guard: the gold answer-key, before vs after")
    on_ts = on["retrieval_leak_guard"]["top_sim_mean"]
    off_ts = off["retrieval_leak_guard"]["top_sim_mean"]
    ax.set_xlabel(
        f"mean top-1 similarity ~unchanged ({on_ts:.3f} vs {off_ts:.3f}): "
        "the guard removes contamination, not signal"
    )
    _despine(ax)
    _save(fig, "f1_leak_guard.png")


def fig_machinery() -> None:
    main, arma = load_run("main"), load_run("armA")
    specs = [
        ("len_wasserstein_vs_real", "reply length (Wasserstein vs Bohdan)"),
        ("exclaim_rate", "exclamation rate"),
    ]
    labels = ["Arm B\n(bare prompt)", "Arm A\n(shipped stack)"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    for ax, (field, title) in zip(axes, specs, strict=True):
        mp = machinery_pairs(main, arma, field)
        ax.plot([0, 1], mp["api"], "-o", color=API_C, label="API")
        ax.plot([0, 1], mp["lora"], "--o", color=LORA_C, label="LoRA (reference)")
        for xi, yi in zip((0, 1), mp["api"], strict=True):
            ax.annotate(
                f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 8), fontsize=8
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(bottom=0)
        ax.set_title(title)
        _despine(ax)
    axes[0].legend(loc="upper right")
    fig.suptitle("What the production machinery buys (the API moves; the LoRA barely does)")
    _save(fig, "f4_machinery.png")


def fig_steered_vs_learned() -> None:
    ship, learn = load_run("armA"), load_run("armA_learned")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.3))
    ex = [
        ship["scorecard"]["arms"]["api"]["exclaim_rate"],
        learn["scorecard"]["arms"]["api"]["exclaim_rate"],
    ]
    bars = ax1.bar(["shipped\n(EXCLAIM=-5)", "learned\n(no bias)"], ex, color=[API_C, "#93c5fd"])
    ax1.bar_label(bars, fmt="%.3f", padding=3)
    ax1.set_ylim(0, max(ex) * 1.4 or 0.05)
    ax1.set_title("API exclamation rate")
    _despine(ax1)
    for i, run in enumerate((ship, learn)):
        d = run["scorecard"]["deltas_api_minus_lora"]["len_wasserstein"]
        lo, hi = d["delta"] - d["ci_lo"], d["ci_hi"] - d["delta"]
        ax2.errorbar(i, d["delta"], yerr=[[lo], [hi]], fmt="o", color=LORA_C, capsize=5)
    ax2.axhline(0, ls="--", color="gray")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["shipped", "learned"])
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_title("length Δ (API-LoRA), 95% CI\n(>0 favors LoRA)")
    _despine(ax2)
    fig.suptitle("Steered vs learned: the levers move the tic, not the verdict")
    _save(fig, "f5_steered_vs_learned.png")


def fig_by_language() -> None:
    bl = json.loads((COMPARE / "armA" / "by_language.json").read_text(encoding="utf-8"))
    langs = [lang for lang in ("cyrillic", "latin") if lang in bl]
    specs = [
        ("shape_js_vs_real", "message shape (JS divergence)"),
        ("len_wasserstein_vs_real", "reply length (Wasserstein)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
    for ax, (field, title) in zip(axes, specs, strict=True):
        x = range(len(langs))
        w = 0.35
        api = [bl[lang]["arms"]["api"][field] for lang in langs]
        lora = [bl[lang]["arms"]["lora"][field] for lang in langs]
        ax.bar([i - w / 2 for i in x], api, w, color=API_C, label="API")
        ax.bar([i + w / 2 for i in x], lora, w, color=LORA_C, label="LoRA")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{lang}\n(n={bl[lang]['n_items']})" for lang in langs])
        ax.set_ylim(bottom=0)
        ax.set_title(title)
        _despine(ax)
    axes[0].legend(loc="upper left")
    fig.suptitle(
        "Per-language fidelity (Arm A): cyrillic drives the verdict; English a shared weakness"
    )
    _save(fig, "f6_by_language.png")


def fig_forest() -> None:
    order = ["main", "seed1", "armA", "armA_learned", "armA_leakoff", "armA_leakon"]
    runs = {n: load_run(n) for n in order}
    specs = [
        ("shape_js", "message shape Δ (JS)", False),
        ("len_wasserstein", "reply length Δ (Wasserstein)", True),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (metric, title, symlog) in zip(axes, specs, strict=True):
        rows = collect_deltas(runs, metric, order)
        for y, r in enumerate(rows):
            color = LORA_C if r["excludes_zero"] else MUTE_C
            ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=color, lw=2.2)
            ax.plot(r["delta"], y, "o", color=color, zorder=3)
            ax.annotate(
                f"{r['delta']:.3g}",
                (r["delta"], y),
                textcoords="offset points",
                xytext=(0, 7),
                fontsize=7,
                ha="center",
                color=color,
            )
        ax.axvline(0, ls="--", color="gray", lw=1)
        if symlog:
            ax.set_xscale("symlog", linthresh=1)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["name"] for r in rows])
        ax.invert_yaxis()
        ax.set_xlabel("API - LoRA   (>0 favors LoRA)")
        ax.set_title(title)
        _despine(ax)
    fig.suptitle("Effect sizes across all runs (color = 95% CI excludes 0; gray = tie)")
    _save(fig, "f7_forest.png")


def fig_operational() -> None:
    op = load_run("armA")["operational"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4))
    lat = [
        op["api"]["p50_latency_s"],
        op["api"]["p95_latency_s"],
        op["lora"]["p50_latency_s"],
        op["lora"]["p95_latency_s"],
    ]
    bars = axes[0].bar(
        ["API\np50", "API\np95", "LoRA\np50", "LoRA\np95"],
        lat,
        color=[API_C, API_C, LORA_C, LORA_C],
    )
    axes[0].bar_label(bars, fmt="%.2fs", padding=3)
    axes[0].set_ylim(0, max(lat) * 1.25)
    axes[0].set_title("latency (s / reply)")
    _despine(axes[0])
    tok = [op["api"]["mean_in_tokens"], op["lora"]["mean_in_tokens"]]
    bars = axes[1].bar(["API", "LoRA"], tok, color=[API_C, LORA_C])
    axes[1].bar_label(bars, fmt="%.0f", padding=3)
    axes[1].set_ylim(0, max(tok) * 1.25)
    axes[1].set_title("mean input tokens (the context tax)")
    _despine(axes[1])
    cost = [op["api"].get("usd_per_1k_replies", 0.0), 0.0]
    bars = axes[2].bar(["API", "LoRA"], cost, color=[API_C, LORA_C])
    axes[2].bar_label(bars, labels=[f"${cost[0]:.2f}", "$0"], padding=3)
    axes[2].set_ylim(0, max(cost) * 1.3 or 1)
    axes[2].set_title("USD / 1,000 replies")
    _despine(axes[2])
    fig.suptitle(
        "Operational profile (Arm A): the LoRA is free and lean; the API pays a ~11x context tax"
    )
    _save(fig, "f8_operational.png")


def fig_copy_floor() -> None:
    cl = load_run("armA")["scorecard"]["copy_leak"]
    base = cl["baseline_real_vs_train"]
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    x = range(2)
    w = 0.35
    api = [cl["api"]["exact"], cl["api"]["near"]]
    lora = [cl["lora"]["exact"], cl["lora"]["near"]]
    ax.bar([i - w / 2 for i in x], api, w, color=API_C, label="API")
    ax.bar([i + w / 2 for i in x], lora, w, color=LORA_C, label="LoRA")
    ax.axhline(
        base["near"],
        ls="--",
        color=REAL_C,
        label=f"natural floor (real vs train, near {base['near']:.3f})",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(["exact copy", "near copy"])
    ax.set_ylim(0, max(api + lora + [base["near"]]) * 1.35)
    ax.set_ylabel("share of replies reproducing a training reply")
    ax.set_title("Anti-memorization: the LoRA sits just above the human reuse floor")
    ax.legend(loc="upper left", fontsize=8)
    _despine(ax)
    _save(fig, "f9_copy_floor.png")


def _winrate_fig(scorecard: dict[str, Any], title: str, rate_key: str, name: str) -> None:
    rate = scorecard.get(rate_key, scorecard.get("rate"))
    ci = scorecard.get("wilson_95ci", scorecard.get("wilson_ci", [None, None]))
    fig, ax = plt.subplots(figsize=(5.4, 4))
    ax.bar(["panel"], [rate], width=0.5, color=LORA_C)
    if ci[0] is not None:
        ax.errorbar(
            0, rate, yerr=[[rate - ci[0]], [ci[1] - rate]], fmt="none", ecolor="black", capsize=6
        )
    ax.axhline(0.5, ls="--", color="gray", label="chance (0.5)")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    _despine(ax)
    _save(fig, name)


def fig_human() -> None:
    p = Path("reports/main/human_eval/human_scorecard.json")
    if not p.exists():
        print("[skip] F10 (human win-rate) — panel unrated (no human_scorecard.json)")
        return
    _winrate_fig(
        json.loads(p.read_text(encoding="utf-8")),
        "Blind human panel: LoRA win-rate (Wilson 95% CI)",
        "lora_win_rate",
        "f10_human_winrate.png",
    )


def fig_turing() -> None:
    p = Path("reports/main/turing/turing_scorecard.json")
    if not p.exists():
        print("[skip] F11 (Turing detect-rate) — panel unrated (no turing_scorecard.json)")
        return
    _winrate_fig(
        json.loads(p.read_text(encoding="utf-8")),
        "Turing slice: detection rate (Wilson 95% CI)",
        "detection_rate",
        "f11_turing_detect.png",
    )


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in (
        fig_leak_guard,
        fig_machinery,
        fig_steered_vs_learned,
        fig_by_language,
        fig_forest,
        fig_operational,
        fig_copy_floor,
        fig_human,
        fig_turing,
    ):
        try:
            fn()
        except Exception as e:
            print(f"[error] {fn.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
