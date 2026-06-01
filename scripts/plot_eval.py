"""Plot the persona scorecards into one comparison figure.

    uv run python scripts/plot_eval.py

Reads data/eval/<run>/scorecard.json for the named runs and writes
data/eval/metrics_comparison.png — generated vs Bohdan's real distribution.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [
    ("baseline", "baseline-consolidated"),
    ("paren-bias=2", "paren-bias-2"),
    ("best-of-4", "best-of-4"),
]


def _load(slug: str) -> dict | None:
    p = Path("data/eval") / slug / "scorecard.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> None:
    cards = [(label, _load(slug)) for label, slug in RUNS]
    cards = [(lbl, c) for lbl, c in cards if c]
    if not cards:
        print("no scorecards found")
        return

    # Real target from the baseline card (same held-out set).
    real = cards[0][1]["distance"]["real"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle(
        "Persona fidelity — generated vs Bohdan's real distribution "
        "(closer to the dashed line = more like him)",
        fontsize=12,
        fontweight="bold",
    )

    # Panel 1: lexical voice (gen rates) vs real target lines.
    lex = ["paren_smiley_rate", "latin_script_rate", "opener_top_share"]
    lex_names = ["paren ) tic", "code-switch", "opener monotony"]
    x = range(len(lex))
    width = 0.8 / len(cards)
    ax = axes[0]
    for i, (lbl, c) in enumerate(cards):
        g = c["distance"]["gen"]
        ax.bar([xi + i * width for xi in x], [g[k] for k in lex], width, label=lbl)
    for xi, k in zip(x, lex, strict=True):
        ax.hlines(real[k], xi - 0.1, xi + 0.8, colors="black", linestyles="dashed", linewidth=1.4)
    ax.set_xticks([xi + 0.3 for xi in x])
    ax.set_xticklabels(lex_names, fontsize=9)
    ax.set_title("Lexical voice (dashed = real)")
    ax.legend(fontsize=8)

    # Panel 2: headline distances (lower = better).
    ax = axes[1]
    dists = ["shape_js", "len_ks"]
    for i, (lbl, c) in enumerate(cards):
        d = c["distance"]
        ax.bar([j + i * width for j in range(len(dists))], [d[k] for k in dists], width, label=lbl)
    ax.set_xticks([j + 0.3 for j in range(len(dists))])
    ax.set_xticklabels(["shape_js", "len_ks"], fontsize=9)
    ax.set_title("Headline distances (lower = better)")
    ax.legend(fontsize=8)

    # Panel 3: style self-similarity (higher = better).
    ax = axes[2]
    ss = [(lbl, c.get("style_self_sim")) for lbl, c in cards]
    ss = [(lbl, v) for lbl, v in ss if v is not None]
    ax.bar([lbl for lbl, _ in ss], [v for _, v in ss], color="#4c72b0")
    ax.set_ylim(0.9, 0.95)
    ax.set_title("style_self_sim (higher = more like me)")
    ax.tick_params(axis="x", labelrotation=15, labelsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path("data/eval/metrics_comparison.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
