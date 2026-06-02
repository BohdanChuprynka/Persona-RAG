"""Render charts + an auto summary from a compare_persona.py run.

Reads ``data/eval/compare/<name>/{results.json, pairs.jsonl}`` and writes PNG
charts + ``summary.md`` under ``reports/<name>/``. All voice numbers (incl. the
real reference) are recomputed from pairs.jsonl with the canonical metric
functions, so the charts can't drift from the audited primitives.

    uv run python scripts/plot_comparison.py --name main
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

from persona_rag.eval.compare import (
    distinct_reply_rate,
    empty_rate,
    exclaim_rate,
    opener_entropy,
)
from persona_rag.eval.distribution import (
    latin_script_rate,
    opener_top_share,
    paren_smiley_rate,
    per_bubble_lengths,
    shape_histogram,
)

REAL_C, API_C, LORA_C = "#64748b", "#2563eb", "#16a34a"


def _load(name: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    base = Path("data/eval/compare") / name
    results = json.loads((base / "results.json").read_text(encoding="utf-8"))
    pairs = [
        json.loads(line)
        for line in (base / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return results, pairs


def _cols(pairs: list[dict[str, str]]) -> dict[str, list[str]]:
    return {
        "real": [p["real"] for p in pairs],
        "api": [p["gen_api"] for p in pairs],
        "lora": [p["gen_lora"] for p in pairs],
    }


def _tics(texts: list[str]) -> dict[str, float]:
    return {
        "latin_script_rate": latin_script_rate(texts),
        "paren_smiley_rate": paren_smiley_rate(texts),
        "exclaim_rate": exclaim_rate(texts),
        "opener_top_share": opener_top_share(texts),
        "opener_entropy": opener_entropy(texts),
        "distinct_reply_rate": distinct_reply_rate(texts),
        "empty_rate": empty_rate(texts),
    }


def _fig_headline(results: dict[str, Any], out: Path) -> None:
    arms = results["scorecard"]["arms"]
    d = results["scorecard"]["deltas_api_minus_lora"]
    metrics = [("shape_js_vs_real", "shape_js"), ("len_wasserstein_vs_real", "len_wasserstein")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (key, short) in zip(axes, metrics, strict=True):
        vals = [arms["api"][key], arms["lora"][key]]
        bars = ax.bar(["API\n(gpt-4o-mini)", "LoRA\n(Qwen2.5-3B)"], vals, color=[API_C, LORA_C])
        ax.set_title(f"{short} - distance to Bohdan (lower = closer)")
        ax.bar_label(bars, fmt="%.3f", padding=3)
        dd = d[short]
        verdict = dd["favored"] if dd["excludes_zero"] else "tie (CI spans 0)"
        lbl = f"d(api-lora)={dd['delta']:.3f}  CI[{dd['ci_lo']:.3f},{dd['ci_hi']:.3f}] -> {verdict}"
        ax.set_xlabel(lbl)
    fig.suptitle(f"Controlled A/B (identical thin prompt) - n={results['scorecard']['n_items']}")
    fig.tight_layout()
    fig.savefig(out / "headline_distances.png", dpi=130)
    plt.close(fig)


def _fig_tics(cols: dict[str, list[str]], out: Path) -> None:
    t = {k: _tics(v) for k, v in cols.items()}
    keys = ["latin_script_rate", "paren_smiley_rate", "exclaim_rate", "opener_top_share"]
    x = range(len(keys))
    w = 0.27
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar([i - w for i in x], [t["real"][k] for k in keys], w, label="real (Bohdan)", color=REAL_C)
    ax.bar(list(x), [t["api"][k] for k in keys], w, label="API", color=API_C)
    ax.bar([i + w for i in x], [t["lora"][k] for k in keys], w, label="LoRA", color=LORA_C)
    ax.set_xticks(list(x))
    ax.set_xticklabels(keys, rotation=15)
    ax.set_title("Voice tics vs the real reference (closer to gray = better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "voice_tics.png", dpi=130)
    plt.close(fig)


def _fig_shape(cols: dict[str, list[str]], out: Path) -> None:
    hists = {k: shape_histogram(v) for k, v in cols.items()}
    buckets = sorted(hists["real"].keys())
    x = range(len(buckets))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar([i - w for i in x], [hists["real"][b] for b in buckets], w, label="real", color=REAL_C)
    ax.bar(list(x), [hists["api"][b] for b in buckets], w, label="API", color=API_C)
    ax.bar([i + w for i in x], [hists["lora"][b] for b in buckets], w, label="LoRA", color=LORA_C)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(b) for b in buckets])
    ax.set_xlabel("bubbles per reply")
    ax.set_ylabel("share of replies")
    ax.set_title("Message-shape distribution (Telegram bubbles per reply)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "shape_distribution.png", dpi=130)
    plt.close(fig)


def _fig_lengths(cols: dict[str, list[str]], out: Path) -> None:
    clip = 160
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for k, c in (("real", REAL_C), ("api", API_C), ("lora", LORA_C)):
        lens = [min(x, clip) for x in per_bubble_lengths(cols[k])]
        ax.hist(lens, bins=24, density=True, histtype="step", linewidth=2, label=k, color=c)
    ax.set_xlabel(f"per-bubble length (chars, clipped at {clip})")
    ax.set_ylabel("density")
    ax.set_title("Per-bubble length distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "length_distribution.png", dpi=130)
    plt.close(fig)


def _fig_ops(results: dict[str, Any], out: Path) -> None:
    op = results["operational"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = ["API p50", "API p95", "LoRA p50", "LoRA p95"]
    vals = [
        op["api"]["p50_latency_s"],
        op["api"]["p95_latency_s"],
        op["lora"]["p50_latency_s"],
        op["lora"]["p95_latency_s"],
    ]
    bars = ax.bar(labels, vals, color=[API_C, API_C, LORA_C, LORA_C])
    ax.bar_label(bars, fmt="%.2fs", padding=3)
    ax.set_ylabel("latency (s)")
    cost = op["api"].get("usd_per_1k_replies", "?")
    ax.set_title(f"Latency - API ${cost}/1k replies vs LoRA $0 (local)")
    fig.tight_layout()
    fig.savefig(out / "latency_cost.png", dpi=130)
    plt.close(fig)


def _summary_md(results: dict[str, Any], cols: dict[str, list[str]], out: Path) -> None:
    arms = results["scorecard"]["arms"]
    d = results["scorecard"]["deltas_api_minus_lora"]
    cl = results["scorecard"].get("copy_leak", {})
    op = results["operational"]
    p = results["params"]
    t = {k: _tics(v) for k, v in cols.items()}
    lines = [
        f"# Comparison summary - `{results['name']}`",
        "",
        f"- **Arm:** {results['arm']}",
        f"- **n:** {p['n']}  seed {p['seed']}  temp {p['temperature']}",
        f"- **backends:** API `{p['api_model']}` vs LoRA `{p['lora_model']}`",
        f"- **hold-out:** {p['eval_split']}",
        "",
        "## Headline distances (lower = closer to Bohdan)",
        "",
        "| metric | API | LoRA | delta(api-lora) | 95% CI | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for key, short in (
        ("shape_js_vs_real", "shape_js"),
        ("len_wasserstein_vs_real", "len_wasserstein"),
    ):
        dd = d[short]
        verdict = dd["favored"] if dd["excludes_zero"] else "tie"
        ci = f"[{dd['ci_lo']:.3f}, {dd['ci_hi']:.3f}]"
        lines.append(
            f"| {short} | {arms['api'][key]:.3f} | {arms['lora'][key]:.3f} "
            f"| {dd['delta']:.3f} | {ci} | **{verdict}** |"
        )
    lines += [
        "",
        "## Voice tics vs real (closer to the real column = better)",
        "",
        "| tic | real | API | LoRA |",
        "|---|---|---|---|",
    ]
    for k in (
        "latin_script_rate",
        "paren_smiley_rate",
        "exclaim_rate",
        "opener_top_share",
        "opener_entropy",
    ):
        lines.append(f"| {k} | {t['real'][k]:.3f} | {t['api'][k]:.3f} | {t['lora'][k]:.3f} |")
    if cl:
        api_cl = f"{cl['api']['exact']:.3f}|{cl['api']['near']:.3f}"
        lora_cl = f"{cl['lora']['exact']:.3f}|{cl['lora']['near']:.3f}"
        base = cl.get("baseline_real_vs_train")
        base_cl = f"{base['exact']:.3f}|{base['near']:.3f}" if base else "n/a"
        lines += [
            "",
            "## Anti-gaming guards",
            "",
            f"- copy/leak (exact|near): API {api_cl}  LoRA {lora_cl}  "
            f"(natural floor, real-vs-train: {base_cl})",
            f"- distinct-reply: API {arms['api']['distinct_reply_rate']:.3f}  "
            f"LoRA {arms['lora']['distinct_reply_rate']:.3f}",
            f"- empty rate: API {arms['api']['empty_rate']:.3f}  "
            f"LoRA {arms['lora']['empty_rate']:.3f}",
        ]
    lines += [
        "",
        "## Operational",
        "",
        f"- API: p50 {op['api']['p50_latency_s']}s / p95 {op['api']['p95_latency_s']}s, "
        f"${op['api'].get('usd_per_1k_replies', '?')}/1k replies",
        f"- LoRA: p50 {op['lora']['p50_latency_s']}s / p95 {op['lora']['p95_latency_s']}s, "
        f"$0/1k (local)",
        "",
        "*Charts: headline_distances.png, voice_tics.png, shape_distribution.png, "
        "length_distribution.png, latency_cost.png*",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render charts + summary from a comparison run.")
    ap.add_argument("--name", default="main")
    a = ap.parse_args()
    matplotlib.use("Agg")
    results, pairs = _load(a.name)
    cols = _cols(pairs)
    out = Path("reports") / a.name
    out.mkdir(parents=True, exist_ok=True)
    _fig_headline(results, out)
    _fig_tics(cols, out)
    _fig_shape(cols, out)
    _fig_lengths(cols, out)
    _fig_ops(results, out)
    _summary_md(results, cols, out)
    print(f"wrote charts + summary.md -> {out}")


if __name__ == "__main__":
    main()
