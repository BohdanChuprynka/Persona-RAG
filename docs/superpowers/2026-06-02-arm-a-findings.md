# Arm A (production-realism) — findings

- **Date:** 2026-06-02 · **Branch:** `feat/eval-ab-comparison` (local, not pushed)
- **Spec:** `specs/2026-06-02-arm-a-production-realism-design.md` · **Plan:** `plans/2026-06-02-arm-a-production-realism.md`
- **Companion:** arm B (controlled) findings — `2026-06-02-comparison-findings.md`
- **Runs:** `data/eval/compare/{armA, armA_learned, armA_leakoff, armA_leakon}/` (git-ignored — real chat content)

## TL;DR

Arm A pits the **shipped OpenAI product** (rich ~1600-token RAG prompt + hybrid retrieval + register/shape directives + the live `.env` decode levers `PAREN=+2 / EXCLAIM=-5`) against the **LoRA** in its real thin serving config, on the recipient-stratified hold-out, with the gold turn excluded per-item from retrieval (a hard guard, proven below).

**Result (n=300):** the API's machinery closes almost the entire voice gap arm B exposed — but the fine-tune is still measurably **closer on reply length** and has **more varied openers**, at **$0** local and no per-message phone-home. Shape and `!`-suppression are ties. The fine-tune's edge **shrinks under production machinery but does not vanish.**

## 1. The leak guard is real and proven

The hold-out gold turns sit in the same Qdrant + BM25 corpus the API retrieves from; the pre-existing `exclude_eval` filter targets the *temporal* split, not the recipient-stratified one we score. Measured with the new `--leak-on` switch (exclusion disabled, n=60):

| | exclusion OFF (`--leak-on`) | exclusion ON (headline) |
|---|---|---|
| gold turn retrieved (`id_leaks`) | **17 / 60 (28%)** | **0** |

So ~28% of items would have had the **exact answer key** in the API's few-shot pool. The per-item `exclude_ids` guard removes all of it (`id_leaks=0` on every headline run), and the two-leg assertion does **not** false-fire on the ~13% of items whose gold reply is a ubiquitous short token (`да`/`ок`) recurring under another turn-id (those are fair neighbours, only counted).

**Honest nuance:** even when handed the gold, `gpt-4o-mini` *paraphrased* rather than copied (copy-rate stayed 0), so the leak's scorecard impact at n=60 was modest — but it is real contamination (the model saw the answer 28% of the time) and we eliminate it regardless.

## 2. Headline — production vs production (n=300, shipped levers, `id_leaks=0`)

| metric (lower = closer to Bohdan) | API (shipped) | LoRA (thin) | Δ API−LoRA (95% CI) | verdict |
|---|---|---|---|---|
| `shape_js` (bubble-count dist.) | 0.0353 | 0.0339 | +0.001 [−0.040, +0.040] | **tie** |
| `len_wasserstein` (per-bubble length EMD) | 6.97 | **3.41** | +3.57 [+1.53, +4.66] | **LoRA** |
| `exclaim_rate` | 0.000 | 0.000 | — | **tie** (both perfect) |
| `opener_entropy` (higher = more varied) | 3.70 | **5.76** | — | LoRA |
| `distinct_reply_rate` | 0.983 | 0.980 | — | tie (no mode-collapse) |
| copy / near-copy vs train | 0.000 / 0.000 | 0.103 / 0.103 | — | LoRA at its ~10% short-text floor |
| cost / 1k replies | $0.37 | **$0** (local) | — | LoRA |
| p50 latency | 0.96s | 1.01s | — | ~tie |

Retrieval guard telemetry: `top_sim_mean=0.39`, `top_sim≥0.9 = 0` — neighbours are genuinely different turns, not near-duplicates, so the API's numbers reflect generalization, not retrieved-twin echo. API error rate 0.3% (1/300, a residual 429 after 4 retries).

## 3. What the API's machinery actually buys (arm B → arm A, API side)

Both arms score the same recipient-stratified hold-out; arm B gave the API the *thin* prompt, arm A gives it the *shipped* prompt + retrieval + levers.

| API metric | arm B (bare) | arm A (shipped) | effect of the machinery |
|---|---|---|---|
| `len_wasserstein` | 128.8 | 6.97 | **massive fix** (rich prompt + few-shot rein in the length blowup) |
| `exclaim_rate` | 0.651 | 0.000 | **fully suppressed** (the shipped `EXCLAIM_LOGIT_BIAS=-5`) |
| `shape_js` | 0.052 | 0.035 | improved |
| `opener_entropy` | 5.02 | 3.70 | **regressed** — the few-shot/directives *homogenize* openers below even bare gpt-4o-mini |

The machinery is what makes the API competitive at all — it converts a backend that was wildly off-register (arm B) into a near-match. The one counter-intuitive cost: it makes the API's openers *less* varied than the bare model, and well below the LoRA.

**LoRA consistency check:** the LoRA arm is near-identical across arms B↔A (opener_entropy 5.77→5.76, len_wasserstein 2.88→3.41, exclaim 0.00→0.00) — same thin input, same model, as expected. This validates the harness and isolates every arm-A change to the API side.

## 4. Interpretation — which ships better?

Even as a **fully-equipped product**, the API only *ties* the LoRA on bubble-shape and `!`-suppression, while the **LoRA stays measurably ahead on reply length** (EMD 3.4 vs 7.0, CI excludes 0) and **opener variety**, runs at **$0** with no per-message OpenAI embedding phone-home, and never leaks. Against that, the API's no-`!` is *bought* by a hard-coded logit bias (see §5), not learned, and its retrieval adds the leak surface we had to guard.

Per the decide rule (spec §2: a voice-tie + cheaper/local ships the LoRA), the automatic-metric verdict is **the LoRA ships** — its edge narrows under the API's machinery but does not disappear, and cost/privacy break the ties. This **agrees with the rated blind human panel** (Bohdan found the API *trivially discriminable* — he picks it out every time — so the LoRA wins voice decisively) and with arm B. **Three independent methods now concur: the LoRA ships.** The open frontier is no longer API-vs-LoRA but the *Turing* test — is the LoRA distinguishable from Bohdan's **real** replies? (see `2026-06-02-turing-test-design.md`).

## 5. Steered vs learned (`--learned` diagnostic, n=300)

Re-running the API arm with levers forced to 0/0 isolates what the rich prompt produces on its own:

| API `exclaim_rate` | value | source |
|---|---|---|
| arm B (bare thin prompt) | 0.651 | baseline gpt-4o-mini |
| arm A `--learned` (rich prompt, no bias) | 0.033 | the rich prompt + few-shot + directives |
| arm A shipped (rich prompt + `EXCLAIM=-5`) | 0.000 | + the logit-bias finishes it |

So the API's no-`!` is **mostly the rich prompt** (0.65 → 0.033 — the few-shot examples and "match the examples" directives genuinely teach `!`-restraint); the shipped logit-bias supplies only the **final 0.03** nudge to zero. The tic-suppression is *largely earned by prompting*, not a pure hard-coded hack — a fairer reading than "the API only looks good because of the bias." Everything else is lever-insensitive (shape 0.034, `len_wasserstein` 7.24, opener_entropy 3.83), and the verdict is unchanged: shape a tie, **LoRA still wins length** (Δ 4.27, CI [2.5, 5.1]).

## 6. Caveats

- **Corpus-level, not within-item.** Arm A and arm B score the same hold-out *distribution* but not byte-identical item sets (arm B shuffles a no-id ShareGPT file; arm A loads id-bearing DB rows). Cross-arm deltas are aggregate, not paired. (Within-item alignment was deferred as a fragile nice-to-have.)
- **Runtime-faithful query.** Retrieval + register/shape use `ctx[-1]` only (what the live bot sends, `chat.py:51`), with `ctx[:-1]` as session — not the joined context. Both arms replay the same `(incoming, session)`; each prompt builder consumes it as it really does.
- **Replay gaps (logged in `results.params`):** `user_memory=""` (first-contact), `session` reconstructed from the item's own context, insights from time-of-run tables, `style_anchors.json` shared with prod.
- **The human panel is the real verdict** for both arms.

## 7. Reproduce

```bash
# Qdrant up (make up) + index built (make ingest) + llama-server serving the LoRA.
make compare-arma                                                        # shipped headline
uv run python scripts/compare_persona_armA.py --n 300 --learned --name armA_learned
uv run python scripts/compare_persona_armA.py --n 60 --leak-on --name armA_leakon   # leak proof
uv run pytest tests/test_eval_armA.py -q
```
