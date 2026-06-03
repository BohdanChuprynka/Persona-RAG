# Persona-Replication Report — Design Spec

- **Date:** 2026-06-03 · **Branch:** `feat/eval-ab-comparison` (local, not pushed) · **Status:** frame APPROVED, build pending
- **Source material (all aggregate, privacy-safe):** `2026-06-02-eval-architecture-audit.md`, `2026-06-02-comparison-findings.md` (arm B), `2026-06-02-arm-a-findings.md` (arm A), `2026-06-02-turing-test-design.md`, the two comparison specs, and this session's research-workflow inventory.
- **Companion plan:** `docs/superpowers/plans/2026-06-03-persona-replication-report.md` (the executable build).

## 1. Goal & thesis

A comprehensive, paper-formal **replication story**: *how we built and honestly evaluated a fine-tuned model that texts like one specific person*, using the concrete **Qwen2.5-3B LoRA vs. the shipped gpt-4o-mini RAG product** as the running thread. Narrative spine, real-world research rigor pervasive, nothing trimmed.

> **Thesis.** Can you faithfully replicate one person's texting voice from their own chat history — and *prove* how close you got?

The frame is **"build it (the replication story) + prove it (paper rigor)"**, organized around **two fidelity axes**:
- **Relative fidelity** — does the model beat the strong baseline (the full gpt-4o-mini product) at sounding like the person? (Arms B/A)
- **Absolute fidelity** — can the person *themselves* tell the model's reply from their own? (the Turing test — the climax)

The comparison is the *measurement instrument*; replication is the *mission*. Scope the claim precisely — **one person's texting voice, learned from their own chat logs** — so "human replication" stays honest, not grandiose.

## 2. Audience, format, privacy

- **Audience:** portfolio / hiring (ML engineers, hiring managers). Rigor-forward, decision-driven, honest about limits.
- **Format:** **Typst → PDF**, single-column, arXiv-preprint aesthetic (`arkheion`-style template or a custom preamble). Serif paper font (New Computer Modern / Libertinus). Numbered figures with captions + cross-references; a references section.
- **Length target:** ~16–22 pp + appendices (comprehensive — capture the full body of work).
- **Privacy (hard rule):** **aggregate-only**, no raw messages — same rule the findings docs already follow. The Typst sources, the aggregate figures, and the compiled PDF live under a committable `report/`; `data/` and `reports/` (raw kits, `pairs.jsonl`) stay git-ignored. Every figure is distributional/aggregate → safe to commit.

## 3. Document structure — 5 Parts (narrative) × formal sections

| Part | Story beat | Formal sections (content → source) | Visuals |
|---|---|---|---|
| **I — Building the replica** | *How you actually clone a texting voice* | system architecture (dual-backend LangGraph chain); data pipeline (export → PII → bursts → sessions → turns); **the split problem** (temporal tail register-skew: one EN contact = 62% of Latin → 0.47 artifact → recipient-stratified `eval_split_for`, train≈eval≈0.18); hybrid retrieval (dense+BM25, fuse α=0.7, recency 180d, floor 0.15, MMR λ=0.6, insights, the ~1600-tok `SYSTEM_TEMPLATE`); the LoRA (Qwen2.5-3B QLoRA r=32/α=64, `train_on_responses_only`, **train==serve `THIN_SYSTEM`**, Colab T4 → merge → local GGUF Q5_K_M → `llama-server`); decode levers (`PAREN=+2/EXCLAIM=−5`, OpenAI-only; register/shape directives) | D1, D2, D4 |
| **II — Can we trust the measurement?** | *The hard part nobody shows* | the voice target + **metric definitions** (math); **the audit** — the ~90% asymmetric train/test leak (scored temporal split; LoRA held out recipient split) + confounds + no-CIs; the fair harness (paired bootstrap CIs, NaN-not-0.0, n=300); **two-arm design**; the **retrieval leak guard** (per-item `exclude_ids`, two-leg assertion, **28%→0** proven); **pre-registered "better" + ship rule** (stated before results) | D3, F1 |
| **III — Relative fidelity** | *Does it beat the strong baseline?* | Arm B controlled (LoRA length EMD 2.9 vs 128.8 CI-excl-0, `!` 0.00 vs 0.65, shape tie; 2-seed replication); Arm A production (machinery closes the gap; LoRA still wins length 3.4 vs 7.0 CI [1.5,4.7] + openers 5.76 vs 3.70); what the machinery buys (len 128.8→7.0, `!` 0.65→0.00, openers regress 5.02→3.70); steered-vs-learned (no-`!` mostly the prompt 0.65→0.033, bias only the last 0.033→0); per-language (cyrillic 87% drives it; English tie + shared weakness); **effect sizes** (Arm B Cliff's δ 0.95 / 292-of-299 vs Arm A δ 0.04 / per-item wash — the machinery closes the per-message gap; forest across all 6 runs); anti-memorization (LoRA 10.3% vs floor 7%); operational ($0 vs $0.37/1k, ~11× context tax) | F2, F3, F4, F5, F6, F7, F9, F8 (+ voice_tics / length / shape supporting) |
| **IV — Absolute fidelity** | *Can the person tell it from themselves?* | the blind human panel (qualitative now: API trivially discriminable → LoRA wins voice; win-rate + Wilson CI when rated); **the Turing test as climax** (pass = detection CI *includes* 0.5 = indistinguishable); the **voice-vs-knowledge tell taxonomy** → sizes the RAG/grounding investment | F10, F11 (gated) |
| **V — What we learned** | *The verdict + the road ahead* | triangulation (three methods concur); **threats to validity** (construct, single-rater, single-decode, leakage residual, confounds, corpus-level cross-arm, replay fidelity, multiple comparisons, provenance); conclusion; future work (rate the panels, multi-rater κ, leak-free re-train, the RAG layer the tells justify) | — |
| **Appendices** | | A: reproduce commands + pinned params; B: metric formulas; C: full per-run tables; D: supporting charts (both arms); references | — |

## 4. Figures & diagrams

**Diagrams (non-data, authored in `cetz` — Typst-native, version-controlled, no external toolchain; fall back to a committed SVG if one gets unwieldy):**
- **D1** system / dual-backend architecture · **D2** data split + the leak · **D3** two-arm experimental design · **D4** train==serve.

**Figures (data, aggregate):**

| id | shows | status | source |
|---|---|---|---|
| F1 | leak guard 28%→0 (+ top_sim unchanged) | new (`plot_report.py`) | `armA_leakon` / `armA_leakoff` results.json |
| F2 | Arm B headline (shape tie, length LoRA) | exists — relabel | `reports/main/headline_distances.png` |
| F3 | Arm A headline | run existing `plot_comparison.py --name armA` | `armA` |
| F4 | what the machinery buys (B→A slope) | new | `main` + `armA` |
| F5 | steered vs learned levers | new | `armA` + `armA_learned` |
| F6 | per-language (cyrillic vs latin) | new | `armA/by_language.json` |
| F7 | **forest plot** — deltas + 95% CIs across all 6 runs | new | all `results.json` |
| F8 | operational (latency + token tax + cost) | new (richer than `_fig_ops`) | `main` + `armA` |
| F9 | copy/near-copy vs natural floor | new | `armA` copy_leak |
| F10 | human win-rate + Wilson CI | **gated** (render on rating) | `human_scorecard.json` |
| F11 | Turing detection-rate + Wilson CI | **gated** | `turing_scorecard.json` |

Supporting per-run charts (`voice_tics`, `shape_distribution`, `length_distribution`) come free from `plot_comparison.py --name {main,armA}` and go in Appendix D / behind F2–F3.

## 5. Metrics & rigor (honoring "push rigor, no contrived metrics")

**ADD** (standard, computable from `pairs.jsonl` already on disk — no new generation):
1. **Cliff's δ + matched-pairs rank-biserial on per-item length error** — converts a corpus EMD into a standardized magnitude + consistency claim. *Verified, and it splits the two arms sharply:* **Arm B δ = 0.949 (large), LoRA closer 292/299, sign-test p ≈ 8e-77** (overwhelming, per-item consistent); **Arm A δ = 0.043 (negligible), 147/293 — a per-item coin-flip.** The production machinery closes the per-*message* length gap to a wash; the LoRA's Arm A length edge is purely *distributional* (corpus EMD 3.4 vs 7.0, CI excludes 0), not per-item. This honest contrast is itself a "what the machinery buys" result, and the headline rigor upgrade.
2. **Sign / Wilcoxon signed-rank test** on per-item length deltas — the per-item complement to the corpus bootstrap (the audit prescribed it; only the bootstrap was implemented).
3. **Metric↔human agreement** (construct validity) — *gated on the panel being rated*; descriptive concordance, not a fitted model.

**PRESENT WELL** (already measured — frame to paper standard): Wilson CIs (human + Turing); the **ablation table** (arms already run); empirical noise-floor as the MDE statement; power note for the human panel; decode-variance caveat; **leakage as a first-class methods section**; copy-vs-natural-floor; exact CIs everywhere; per-language; single-rater rationale + caveat; the pre-registered "better" rule; triangulation; provenance stamping.

**AVOID** (explicit non-goals — the "don't hallucinate metrics" constraint, made concrete): composite "persona-fidelity score"; BLEU/ROUGE/METEOR/chrF vs the gold reply; cross-tokenizer perplexity; generic LLM-judge 1–10; Cohen's d on skewed lengths; a fake per-item `shape_js`; parametric power calc for the distances; exotic stylometry (Yule's K, Burrows's Δ); a fake multi-model leaderboard.

## 6. Human-panel handling

**Placeholders now, fill later.** F10, F11, and the §III/§IV metric↔human agreement render/compute once `choices.json` exists; the qualitative "API trivially discriminable → LoRA wins voice" carries the human beat today. The rater kits + scorers already exist (`reports/main/{human_eval,turing}/`, `score_human_eval.py`, `score_turing_eval.py`); only ratings are pending.

## 7. Build approach (artifacts)

- `persona_rag/eval/effect_size.py` *(new, unit-tested)* — `per_item_length_error`, `cliffs_delta`, `matched_pairs_rank_biserial`, `sign_test`, `wilcoxon_signed_rank`. Hand-rolled to match the no-heavy-dep house style unless scipy is already a dependency.
- `scripts/compute_effect_sizes.py` *(new)* — reads `pairs.jsonl`, writes `effect_sizes.json` per run.
- `scripts/plot_comparison.py` — **run as-is** with `--name armA` (and `seed1`) to render the per-run charts incl. the Arm A headline (F3). No code change.
- `scripts/plot_report.py` *(new)* — the cross-arm / new figures F1, F4, F5, F6, F7, F9 + gated F10/F11. Pure data-shaping helpers are unit-tested; matplotlib drawing mirrors the existing (untested) render-script pattern. Shared palette (real `#64748b`, API `#2563eb`, LoRA `#16a34a`).
- Diagrams D1–D4 — `cetz`.
- `report/` — `report.typ` (main) + `parts/*.typ` + `refs.bib` + `fig/`. Template: `arkheion`-style.
- `make report` — render figures + compile the PDF.

## 8. Decisions

frame = **replication story + paper rigor, two fidelity axes**; format = **Typst→PDF**; scope = **comprehensive full-project**; rigor = **push, sensible-metrics-only**; human panel = **placeholders, fill later**; diagrams = **cetz**; **one** document; title = **TBD (tweak later)**.

## 9. Non-goals

No contrived/expanded metrics (see §5 AVOID). No LoRA re-train. No `git push`, no PR, no live Telegram bot. No raw messages in the document. Not a multi-model benchmark — the question is *this* LoRA vs *this* product.

## 10. Open items

- **Title** — working title only; tweak later.
- **Human + Turing panels unrated** → fills F10/F11 and the §III/§IV construct-validity once `choices.json` is produced.
