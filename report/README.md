# Replicating a Texting Voice — report

Self-contained Typst sources for the portfolio-grade persona-replication report
(the build + honest evaluation of a fine-tuned Qwen2.5-3B LoRA against the shipped
gpt-4o-mini RAG product). **Aggregate-only** — no raw messages anywhere in here.

## Build

From the **repo root**:

```bash
bash report/build.sh
```

That renders the report figures (`scripts/plot_report.py`) and compiles the PDF
to `report/persona-rag-report.pdf`. Equivalent manual steps:

```bash
uv run python scripts/plot_report.py
typst compile report/report.typ report/persona-rag-report.pdf
```

Needs `typst` (`brew install typst`) and the eval runs under `data/eval/compare/`.
`cetz` (diagrams) and the `ieee` bib style fetch from the Typst package registry
on first compile.

## Layout

- `report.typ` — main document (preamble, title, abstract, includes, bibliography)
- `parts/` — the five narrative Parts + the appendix
- `diagrams/` — `cetz` diagrams D1–D4 (architecture, data split/leak, two-arm design, train==serve)
- `fig/` — rendered figures (aggregate-only PNGs)
- `refs.bib` — references

## Feeding code (lives in its natural home, by repo convention — not duplicated here)

- `persona_rag/eval/effect_size.py` — per-item length effect sizes (Cliff's δ, sign, Wilcoxon)
- `scripts/compute_effect_sizes.py` — writes `effect_sizes.json` per run
- `scripts/plot_report.py` — cross-arm figures F1, F4–F11
- `scripts/plot_comparison.py` — per-run charts (run with `--name armA` / `--name main`)

## Status

F10/F11 (the blind human panel + the Turing slice) render only once those panels
are rated (`choices.json` → the scorers in `scripts/score_{human,turing}_eval.py`).
Until then they appear as clearly-marked "awaiting ratings" placeholders.
