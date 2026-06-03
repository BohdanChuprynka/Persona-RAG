# Persona-RAG Eval-Architecture Audit — API vs LoRA A/B

**Date:** 2026-06-02
**Repo SHA:** `6e79973`
**Author:** evaluation architect (audit synthesised from per-component analyses + adversarial critiques, all load-bearing claims re-verified against source)

**Scope.** Prepare a *trustworthy*, staged comparison of two generation backends for the Persona-RAG Telegram bot:

- **API arm** — `gpt-4o-mini` served the **rich ~1600-token RAG prompt** (`SYSTEM_TEMPLATE` + retrieved few-shot real replies + register/shape directives + voice logit-bias).
- **LoRA arm** — local fine-tuned **Qwen2.5-3B** (llama.cpp/ollama) served the **THIN prompt** (`THIN_SYSTEM` anchor + joined context only). Train == serve by design.

**The single target metric** is subjective: *"does this read like something Bohdan would actually text?"* — uk/en/ru code-switch, per-context mirroring, opener variety, the `)` smiley tic, no `!`, casual casing, multi-bubble burst shape.

**Staging.** decide (ship which backend?) → portfolio (blog/case study) → maybe paper. Each stage raises the evidentiary bar; this audit defines a protocol that is honest enough for all three.

> **Bottom line up front.** The metric *primitives* in `persona_rag/eval/distribution.py` and `authorship.py` are well-built and well-tested. But the **harness that would compare two backends does not exist yet**, and the path that does exist (`scripts/eval_persona.py`) is **structurally unfair** for an API-vs-LoRA comparison on at least three counts (eval-split mismatch + train/test leak, prompt+retrieval confound, decode-lever asymmetry) and reports **point estimates with no uncertainty**. No "fine-tune beats RAG" (or vice-versa) claim is defensible until the must-fix items below are closed.

---

## 1. Inventory — what is currently measured and how

| Component | File(s) | What it computes | Consumed by | Status for the A/B |
|---|---|---|---|---|
| **Distributional distances** | `persona_rag/eval/distribution.py` | `persona_distance(real, gen)` → `shape_js` (JS-div of 1..6 bubble-count histogram, bounded [0,1]), `len_wasserstein` + `len_ks` (per-bubble char-length EMD/KS), `pct_single_real/gen`. `summarize()` fingerprint: `latin_script_rate` (code-switch), `opener_top_share` (monotony), `paren_smiley_rate` (the `)` tic, unbalanced-close heuristic), `caps_ratio_mean`, `emoji_rate_mean`, `punct_density_mean`, bubble-length median/mean. | `eval_persona.py:151` | **Right primitive, wrong altitude.** Pooled corpus-vs-corpus, point estimates, no CIs. Headline is a *vector* of incommensurable units (`shape_js∈[0,1]` vs `len_wasserstein` in chars) with **no composite and no weighting**. The named voice tics are printed as diagnostics but are **NOT distances** and cannot move the headline. |
| **Authorship "is-this-me" cosine** | `persona_rag/eval/authorship.py` | `self_similarity(gen, ref)` = mean StyleDistance-embedding cosine of gen replies to a centroid of real replies. Frozen off-the-shelf encoder; one-class, no negatives. | `eval_persona.py:96-109` (optional, `None` when torch absent); **same `cached_reference_vector` is the live best-of-N reward** in `generate/select.py:42-43`. | **Complementary axis, but leaky + circular.** Eval centroid built from the *same* held-out reals it scores (`reference_vector(real)`, `eval_persona.py:105`). Unvalidated on uk/ru Cyrillic. Doubles as the API-side selector reward → contamination if `BEST_OF_N>1`. |
| **Legacy stylometry** | `persona_rag/eval/stylometry.py` | `compute_features` (7 scalars: len, emoji_rate, caps_ratio, punct_density, avg_word_len, lexical_diversity); `mean_abs_deviation` (corpus-mean-vs-corpus-mean gap). | `distribution.summarize` reuses `compute_features` for 3 keys. **`mean_abs_deviation` has zero callers.** | **`compute_features` = keep as primitive. `mean_abs_deviation` = dead, gameable, mode-collapse-blind. Do not wire into any scorecard.** |
| **Eval runner** | `scripts/eval_persona.py` | Samples N held-out turns (`eval_split==True`, default n=80, seed=0), replays each through the full compiled LangGraph in `SHADOW_MODE`, scores gen-vs-real via `persona_distance` + optional `style_self_sim`. Writes `data/eval/<name>/{pairs.csv, scorecard.json}`; prints scorecard. | `make eval` | **Single-arm descriptive scorecard, not a comparator.** Backend chosen by env (no `--backend` flag). Comparison = "run twice, eyeball two JSON files" (EVAL.md:99). One decode per turn at temp 0.8. Silent drop of failed/empty gens → effective-n drift between arms. |
| **Shadow logger** | `persona_rag/shadow/logger.py`, `graph/nodes/shadow_log.py` | Capture-only JSONL sink (incoming, generated, retrieved_ids, memory, decode params, `your_actual_reply=null`). | nothing (no reader exists) | **Backend MISLABELLED:** hardcodes `OPENAI_CHAT_MODEL` even on the LoRA path (ignores `active_model()`). No backend/prompt-variant field, no join key to ground truth, dead `session_id`, unguarded concurrent append, mislogged temperature under best-of-N. Not usable for A/B as-is. |
| **MLflow wrapper** | `persona_rag/eval/mlflow_wrap.py` | `log_eval_run(params, metrics, tags, artifacts)` thin pass-through. | **zero callers** outside its unit test (EVAL.md:129 concedes this). | **Dead in the loop.** Type contract (`metrics: dict[str,float]`) clashes with the nested `persona_distance` payload and crashes on `style_self_sim=None`. No run store exists today. |
| **Stress test** | `scripts/stress_test_persona.py` | 20 hardcoded uk-only prompts through the real graph, dumps markdown "evidence cards" (reply + retrieval evidence + latency). Three terminal counters (error / zero-insight / zero-turn). | manual eyeballing | **Qualitative grounding/wiring smoke test, NOT an evaluator.** No asserts, always exits 0. On the LoRA path `retrieve_hybrid` is skipped → 20/20 zero-turns → LoRA report *looks broken* vs API by construction. |
| **Dataset / split** | `persona_rag/finetune/dataset.py`, `ingest/turns.py` | TWO splits: (a) `mark_eval_split` (`turns.py:52-58`) = **temporal tail** (last 10% by timestamp), written to the DB `eval_split` column. (b) `eval_split_for` (`dataset.py:62-69`) = **recipient-stratified SHA-1 hash**, used by the LoRA export (`iter_records:117` ignores the DB column). | (a) → `eval_persona.py:65`, retrieval exclusion, `cached_reference_vector`. (b) → LoRA train/eval. | **THE central tension.** The two are disjoint partitions. The codebase's own docs (`dataset.py:67`, EVAL.md:35) say the temporal tail is English-heavy and "made `latin_script_rate` unreachable" — yet the runner scores BOTH arms on it. |
| **Insights gate** | `persona_rag/insights/verifier.py`, `verification.py` | LLM-judge provenance check (YES/NO/AMBIGUOUS) + human review FSM. Drops only NO; keeps YES/AMBIGUOUS/None(fail-open). | distill-time gate on the shared facts corpus | **Not an A/B metric — a CONTROLLED CONSTANT.** Gates the facts the *rich API prompt* consumes. Run once at distill; must be frozen identically for both arms or it becomes a content confound (heavier on the API arm). |

**What is NOT measured anywhere today:** any paired backend-vs-backend delta; any statistical uncertainty (no bootstrap/CI/p-value — confirmed zero across `persona_rag/` + `scripts/`); decode-noise variance; per-register / per-language / per-recipient breakdown; **latency**; **cost / tokens** (`chat_complete` discards `resp.usage`, `llm_client.py:148`); a **blind human win-rate**; any **content/semantic-appropriateness** signal (every metric is pure surface/form).

---

## 2. Strengths to keep

1. **Distributional-by-construction is the correct instrument.** `shape_js` + `len_wasserstein`/`len_ks` genuinely catch the project's stated failure #1 (the constant-3-bubble/45-char mode collapse that a mean-based metric scores ~0 on). `js_divergence` is log2-bounded to [0,1] → comparable across runs. Unit-tested with known values (`tests/test_eval_distribution.py`).
2. **Single source of truth for "a bubble".** `distribution.py` imports `split_bubbles`/`count_bubbles` from the canonical `generate/bubbles.py` shared by delivery/generation/eval, so measurement cannot drift from production.
3. **The `)` tic is measured correctly.** `_has_paren_smiley` uses the *unbalanced* close-paren heuristic (`count(')')>count('(')`), so real parentheticals don't false-positive — a subtlety the emoji-codepoint metric is blind to.
4. **Pure, deterministic, I/O-free metric functions.** Reusable in the Colab LoRA loop and `export_finetune_data` with no DB; the same `compute_features` primitive is shared by both eval sides.
5. **`pairs.csv` is a sound blind-A/B substrate.** Written through the `csv` module (quotes/commas/newlines round-trip); `(incoming, real, generated)` is exactly the triple a forced-choice rater and a future DPO set need.
6. **Cost-safe replay.** The runner forces `SHADOW_MODE=true` and `MEMORY_UPDATE_INTERVAL_TURNS=0` *before* settings cache, so eval never sends Telegram messages or fires paid memory-update calls.
7. **Deterministic, recipient-stratified split exists and is the right design.** `eval_split_for` (used by the LoRA) preserves the recipient mix and therefore the code-switch register — it is the split the comparison *should* standardise on.
8. **`recipient_id_hash` is on the row** (`models.py:35`, `db/models.py:34`) — so recipient-stratified sampling AND per-register breakdowns are *feasible without new plumbing*; the runner simply doesn't use it yet.
9. **Run params are already captured** in `scorecard.json` (backend, model, temperature, top_k, mmr, register_aware, shape_hint, best_of_n, paren_logit_bias, score_floor) — provenance for each arm exists even though nothing compares it.

---

## 3. Top trust-risks for the A/B, RANKED

> Ranked by how badly each one can flip or fabricate a "winner." R1–R3 are **disqualifying** for any cross-backend claim until fixed.

### R1 — Eval-split mismatch → asymmetric ~90% train/test leak that favours the LoRA *(must-fix #1)*
- **Mechanism (verified in source):** `eval_persona.py:65` samples `PersonaTurnRow.eval_split == True` = the **temporal tail** (`turns.py:57-58`, last 10% by timestamp). The LoRA's held-out set is the **recipient-stratified hash** `eval_split_for` (`dataset.py:62-69`); `iter_records:117` trains/holds-out by that hash and **explicitly ignores the DB column**.
- **Consequence:** the two hold-out sets are disjoint partitions that overlap only ~`frac` (≈10%) by chance. So **~90% of the turns the runner scores were in the LoRA's *training* pool**, while `gpt-4o-mini` saw none of them. The LoRA is graded largely on memorised data; the API on truly held-out data. **Direction of bias is unambiguous: toward the LoRA.**
- **Compounding asymmetry:** the rich-API retrieval path *excludes* the temporal-eval rows from few-shot (`qdrant_store` `eval_split==False` filter; BM25 corpus built `not eval_split`) — so the API can retrieve **zero** scored turns by construction (clean), while the LoRA trained on most of them (dirty). The leak is one-sided.
- **Register skew on top:** the codebase itself documents the temporal tail as English-heavy — `dataset.py:67` / EVAL.md:35: it "made the `latin_script_rate` target unreachable." So both arms are *also* judged on a non-representative, code-switch-poor slice the LoRA was deliberately steered away from.
- **Fix:** standardise BOTH arms on ONE shared hold-out drawn by `eval_split_for` (recipient-stratified), exclude those ids from the LoRA training export, and assert no scored `turn_id` appears in the LoRA train manifest.

### R2 — Prompt + retrieval confound (the "thin vs rich" fairness problem) → backend effect is unattributable *(must-fix #2)*
- **Mechanism (verified):** `build_messages` branches on `GENERATION_BACKEND` (`prompt.py:248`). The LoRA gets `build_thin_messages` (one-line `THIN_SYSTEM` + a single joined-context user turn). The API gets `SYSTEM_TEMPLATE` (~1600 tok) + retrieved few-shot assistant turns + register/shape/heated directives. Additionally `_route_after_auth` (`compile.py:32-34`) sends the ollama path straight to `load_memory`, **skipping `retrieve_hybrid` entirely** — the LoRA has *no* few-shot at all.
- **Consequence:** three independent variables move together with "backend": **prompt richness**, **retrieval present/absent**, and the **system anchor**. Any `shape_js`/voice-distance gap conflates "gpt-4o-mini vs Qwen2.5-3B" with "rich RAG prompt vs thin prompt" and "with retrieval vs without." You cannot attribute a result to the thing you are choosing.
- **This is intentional (train==serve realism), so it cannot be "fixed" inside a metric** — it must be controlled in the harness via *conditions* (see Protocol §4). The honest design measures **both** a *production-realism* arm (each backend in its native prompt) and a *controlled* arm (a shared prompt / matched retrieval) and reports them separately.

### R3 — Decode-lever asymmetry → the API has a thumb on the scale for the exact tics being scored *(must-fix #3)*
- **Mechanism (verified):** `voice_logit_bias()` returns `None` unless `GENERATION_BACKEND=='openai'` (`llm_client.py:103`) and ollama ignores `logit_bias` anyway. So the `)` tic is *decode-injected* on the API and *learned-only* on the LoRA. `REGISTER_AWARE_ENABLED` and `SHAPE_HINT_ENABLED` default **on** and inject per-reply directives **only on the API path** (`prompt.py:283-295`). `MAX_REPLY_TOKENS=500` is a *token* budget — gpt-4o-mini (o200k) and Qwen2.5 BPE tokenise Cyrillic at different bytes/token, so equal max_tokens ≠ equal allowed character length, biasing `len_wasserstein`.
- **Consequence:** comparing the two arms on `paren_smiley_rate` (a headline tic) while only one arm is biased toward `)` is apples-to-oranges; same for any length-distribution metric under mismatched char budgets.
- **Note:** the levers default OFF in config (`PAREN_LOGIT_BIAS=0`, `EXCLAIM_LOGIT_BIAS=0`, `BEST_OF_N=1`) — but `REGISTER_AWARE`/`SHAPE_HINT`/`MMR` default ON. **For the official run, pin every lever to a logged, matched state** and report `paren_smiley_rate` as a *steered* (not learned) signal for the API if any bias is on.

### R4 — No statistical uncertainty; n=80, single seed, single decode → "winners" are within noise
- **Mechanism (verified):** `persona_distance` returns bare floats; grep finds zero bootstrap/CI/p-value/stderr anywhere. The runner does **one** decode per turn at `TEMPERATURE=0.8`; `--seed` fixes only *which* turns are sampled, not the decode. `shape_js` is a JS-div over a 6-bucket histogram from ≤80 replies.
- **Empirical noise floor (from the supplied analyses, reproduced):** drawing real and gen from the *identical* distribution, `shape_js` across seeds — n=80 → mean 0.014, **max 0.064**; n=160 → mean 0.007, max 0.021. **So at n=80 any `shape_js` gap below ~0.06 is pure sampling noise** and a "winner" can flip on reseed. Add unmeasured decode variance on top (both arms sample).
- **Fix:** paired design + bootstrap 95% CIs + ≥3 seeds + report *surviving* n per arm. A delta must clear its CI to count.

### R5 — `style_self_sim` is self-referential, one-way-leaky, circular, and unvalidated on the target languages
- Eval centroid built from the **same** held-out reals it scores (`eval_persona.py:105`); under the hash split those reals sit in the LoRA's *training* distribution → flatters the LoRA. The identical centroid is the **live best-of-N reward** (`select.py:42`) → if `BEST_OF_N>1` for either arm, that arm argmaxes the very metric reported as a win. StyleDistance is English-trained; behaviour on uk/ru Cyrillic and code-switch is **unvalidated** (no test). One-class (no negatives) → measures "near the casual-register centroid," not "distinctively Bohdan."
- **Fix:** build the centroid from a **frozen validator split in neither arm's training data**; report a contrastive margin (Bohdan-centroid minus other-sender-centroid) if other-author data is available; force `BEST_OF_N=1` for both arms; sanity-check the encoder clusters uk/en/ru paraphrases before trusting it.

### R6 — Degenerate-generation masking + effective-n drift → failures *flatter* the worse backend
- `wasserstein_1d`/`ks_statistic` return **0.0** on empty input (`distribution.py:137,153`); `shape_histogram` drops zero-bubble texts (L106). An all-blank gen corpus scores `len_wasserstein=0.0` and `len_ks=0.0` — **artificially perfect** on 2 of 3 headline distances. The runner silently `continue`s on failed/empty gens (`eval_persona.py:136-140`), recording only final `n_generated`. A flakier backend (the local LoRA is the likelier culprit) has its failures *deleted* from the denominator → smaller, cleaner surviving sample → every distance biased lower.
- **Fix:** return `NaN`/sentinel on empty (not 0.0); record `n_requested / n_failed / n_empty` per arm; treat an empty reply as a maximally-wrong sample or report the drop rate; score both arms on the *same* surviving turn set.

### R7 — Headline is all-surface and gameable; no composite, no semantic axis, no anti-gaming guard
- Every headline distance matches a **marginal** distribution. A model that ignores the incoming and samples canned Bohdan-shaped bursts at the right length/script/tic rates scores near-0 while being content-garbage — `persona_distance` never pairs gen-vs-real for the *same* prompt. `opener_top_share` only penalises the *single* most-common opener → a 2–4-opener rotation games it. `paren_smiley_rate` is gamed by appending `)`. Because there is no composite and the tics aren't distances, a model only has to match the few shape/length numbers to "win," and ~11 numbers printed with no correction invites metric-shopping (EVAL.md:99 even says "look at the specific tic").
- **Fix:** add a leak/copy-rate detector (fraction of gens that near-exactly copy any training reply — critical given R1), an opener-entropy metric, a content/semantic-relevance signal, and **one pre-registered composite acceptance rule**. The blind human A/B (R8) is the only non-gameable backstop.

### R8 — The "blind A/B" is aspirational, not implemented
- EVAL.md:103 calls `pairs.csv` "the blind A/B substrate," but **nothing** randomises real/gen order, presents pairs to a rater, collects choices, or computes a win-rate. The project's *true* target ("feels like Bohdan") has **no measurement today**. The distributional numbers are unvalidated proxies for it.

### R9 — No reproducibility / lineage; MLflow uncalled; shadow log backend-mislabelled
- MLflow `log_eval_run` has zero callers; runs live as loose JSON in `data/eval/<name>/` with a default name (`...-baseline`) that **silently overwrites** on reuse and carries no backend tag. The shadow logger hardcodes `OPENAI_CHAT_MODEL` even on the LoRA path (ignores `active_model()`), so any live A/B built on it silently compares gpt-4o-mini to gpt-4o-mini. `active_model()` on ollama returns only the model name — the quant / llama.cpp build / adapter hash is never pinned.

### R10 — Per-register / per-language aggregation hides the thing that matters
- `persona_distance` pools ALL turns; there is no breakdown by recipient/register/language even though `recipient_id_hash` and `detect_language` are on the row. A backend that nails casual-uk but fails serious-en shows a deceptively middling single number — and per-context mirroring is half the stated voice goal, erased because `_seed_context` injects context under the admin id and drops the real recipient.

---

## 4. Recommended A/B evaluation PROTOCOL

The protocol is **staged**: the *decide* gate runs the cheap automatic screen + a small blind human panel; *portfolio* adds the full human panel + ablation conditions; *paper* adds larger n, inter-rater reliability, and pre-registration.

### 4.0 Build the harness that doesn't exist
Create a single `scripts/compare_persona.py` that loops **once over a shared held-out set** and, per turn, generates from **both** arms (and any ablation arms), producing rows of `(turn_id, recipient_hash, register, lang, incoming, real, gen_api, gen_lora, ...)`. This makes the comparison **paired**, fixes the surviving-n to be identical across arms, and feeds both the automatic metrics and the blind kit. `eval_persona.py` stays as the single-arm scorecard.

### 4.1 The shared held-out set (kills R1)
- Sample by `eval_split_for(turn_id)` (recipient-stratified hash), **not** the DB temporal column. Exclude those exact ids from the LoRA training export. Assert in the harness that no scored `turn_id` is in the LoRA train manifest, and stamp the split id + frac + corpus hash into every output.
- Recommended size: **n = 300** turns for the *decide* automatic screen (drops the n=80 shape_js noise floor from ~0.06 to ~0.02 and gives the rare tics — emoji, paren — enough events). Use the same 300 for every arm.

### 4.2 Automatic metrics to use (and to drop)
- **Headline (report with CIs):** `shape_js`, `len_wasserstein` (also report a **normalised** length distance — EMD over log-lengths or `len_wasserstein / median_real_len` — so a few long bubbles don't dominate), and `style_self_sim` *only* with a frozen, model-disjoint centroid (R5).
- **Voice-tic panel (report with CIs, as diagnostics, not the verdict):** `latin_script_rate`, `opener_top_share` **plus an opener-entropy** companion, `paren_smiley_rate` (tagged *steered* for the API if logit-bias on), `caps_ratio_mean`, an explicit `exclaim_rate` for the "no `!`" rule.
- **Anti-gaming guards (must report):** **copy/leak rate** (fraction of gens that near-exactly match any training reply — non-negotiable given R1), **distinct-reply rate / dup-rate** (mode-collapse detector), and **empty/failed rate per arm** with `NaN` (not 0.0) on degenerate input (R6).
- **Drop:** `eval/stylometry.mean_abs_deviation` (dead, gameable, mode-collapse-blind). Add a deprecation docstring so no one re-wires it.
- **Fix before use:** `latin_script_rate` double-counts mixed-script tokens — classify each token once (Latin XOR Cyrillic) before reporting.

### 4.3 Blind human-preference protocol (the verdict for *decide* and *portfolio*)
This is the only measurement of the actual target. Design:
1. **Source.** From the shared held-out set, draw the prompts. For each, present the **incoming context** + two candidates: `gen_api` and `gen_lora`, in **randomised left/right order** (store the unshuffle key + a rater seed separately; strip all backend/model/param labels — never render the shadow-log model field).
2. **Question.** Forced choice: *"Which reply is more like something Bohdan would actually send here?"* + a "can't tell / both equally" escape. Optionally a second item: show the **real** reply alongside the *better* candidate for a real-vs-best **lower bound** ("how close to the ceiling").
3. **Rater(s).** Bohdan is the gold rater (it's his voice). For *portfolio*/*paper*, add 1–2 people who know his texting style; compute inter-rater agreement (Cohen's/Fleiss' κ) to show the judgment is reproducible.
4. **Size.** *decide*: **~100 paired items** from Bohdan alone (a 60/40 split is detectable; see §4.4). *portfolio*: ~150–200 items, ≥2 raters. *paper*: ≥300 items, ≥3 raters, pre-registered.
5. **Hygiene.** Include a few **attention checks** (real-vs-obvious-garbage pairs) to catch click-through; randomise item order per rater; one sitting ≤ ~80 items to limit fatigue.
6. **Substrate.** A tiny CLI/HTML kit reading the harness output; persists `(item_id, shown_order, choice, rater_id, ts)` and the separate un-blinding key.

### 4.4 Statistical tests + rough sample sizes
- **Human win-rate (primary verdict):** two-sided **binomial / exact sign test** on the per-item winner (drop ties or treat as 0.5). Report win-rate + Wilson 95% CI. Rough power: to call a **60/40** preference at α=0.05, power 0.8 ≈ **~100 decisive items**; a **55/45** needs ~**400**. So *decide* (n≈100) can only resolve a *clear* preference — which is the right bar for a ship decision. For a **Bradley-Terry** strength estimate (portfolio/paper), reuse the same paired data.
- **Automatic distances (screen + supporting evidence):** **paired bootstrap** (≥10k resamples) over the shared turns → 95% CI on `Δshape_js`, `Δlen_wasserstein`, `Δstyle_self_sim`; a **Wilcoxon signed-rank / per-item sign test** on per-turn deltas. A delta counts only if its CI excludes 0. Run **≥3 seeds**; treat across-seed spread as the noise floor a delta must clear (n=300 puts the `shape_js` floor ≈0.02).
- **Multiple comparisons:** the headline verdict is the **human win-rate** + **one pre-registered composite** (§4.6). The tic panel is descriptive; if any tic is used to support a claim, apply Holm-Bonferroni across the panel.

### 4.5 Fairness controls (run these as explicit *conditions*, all logged)
| Confound | Control |
|---|---|
| Split / leak (R1) | One shared `eval_split_for` hold-out, LoRA-train-disjoint, asserted; corpus hash stamped. |
| Prompt+retrieval (R2) | Report **two arms**: **(A) production-realism** — each backend native (API rich+retrieval, LoRA thin); **(B) controlled** — gpt-4o-mini *also* on the THIN prompt with retrieval off, both at matched temp & N=1. (A) answers "which product ships better," (B) isolates "weights vs scaffold." |
| Decode levers (R3) | Pin & log every lever for the official run: `BEST_OF_N=1`, `PAREN/EXCLAIM_LOGIT_BIAS=0` (or report API with/without bias), and either match `REGISTER_AWARE`/`SHAPE_HINT` across arms or declare them part of the "API scaffold" in arm (A). Set per-backend token caps that yield comparable **character** budgets, or assert truncation rarely binds. |
| Decode variance (R4) | Report at serving temp (realistic) *and* at temp≈0 / greedy (stable anchor); ≥3 seeds; CIs. |
| Centroid leak / circularity (R5) | Frozen model-disjoint validator centroid; `BEST_OF_N=1`. |
| Effective-n (R6) | Same surviving turn set for both arms; `NaN` on empty; per-arm failure rates reported. |
| Facts corpus (insights) | **Freeze** the verified `InsightRow` table + Qdrant insights collection to a versioned snapshot; both arms load the identical bytes; no re-distill between arms. Report the corpus composition (YES / AMBIGUOUS-kept / fail-open-None counts). |
| Cost/latency (decision-relevant) | Instrument per-call wall-time and read `resp.usage` (currently discarded, `llm_client.py:148`); report p50/p95 latency, mean tokens, $/1k-replies (API) and local ms + tok/s (LoRA). |

### 4.6 A FORMAL definition of "better"
Pre-register this **before looking at results** (prevents the metric-shopping in R7):

> **Primary (the verdict).** Backend X is *better on voice* iff, on the shared LoRA-disjoint hold-out under the **production-realism arm (A)**, the **blind human win-rate** for X over Y has a Wilson 95% CI that **excludes 0.5**. If the CI straddles 0.5 → declared **a voice tie**.
>
> **Guardrails (any failure overrides a human "win").** The winner must not be won by degeneracy: (i) **copy/leak rate ≤ ε** (e.g. ≤5% near-exact training-reply copies); (ii) **distinct-reply rate ≥ τ** (no mode collapse); (iii) the bootstrap CI on `Δshape_js` must **not** show X *worse* by more than the noise floor. A backend that "wins" preference via memorisation or collapse does **not** qualify.
>
> **Ship rule (decide stage).** Ship the **LoRA** if it is a voice-win OR a voice-tie *and* materially cheaper/faster (local, $0/reply, lower p95) — i.e. ties break on cost/latency/offline-capability, which is the LoRA's value proposition. Ship the **API** only if it is a *clear* voice-win that survives the guardrails. Record the decision + the numbers that drove it.
>
> **Construct-validity check (portfolio/paper).** Once human labels exist, report the **rank correlation** of each automatic metric with the human win-rate, so the writeup can state which proxy (`shape_js` / `len_wasserstein` / `style_self_sim`) actually predicts "feels like Bohdan" — and weight any composite accordingly.

---

## 5. Prioritized improvements

### Must-fix — *before any cross-backend number is trustworthy*
1. **Unify the eval split + assert no leak (R1).** Sample by `eval_split_for`; exclude those ids from the LoRA export; assert no scored `turn_id` in the LoRA train manifest; stamp split id + corpus hash. *Single highest-value fix.*
2. **Build the paired comparator (`compare_persona.py`) (R4/R6/R10).** One loop over the shared set, both arms per turn, identical surviving-n, per-turn deltas; emits the blind-kit rows and the metric rows.
3. **Control the prompt+retrieval+lever confounds via explicit conditions (R2/R3).** Production-realism arm (A) + controlled arm (B); pin & log every lever; match character budgets.
4. **Add uncertainty (R4).** Paired bootstrap 95% CIs + sign/Wilcoxon on `shape_js`, `len_wasserstein`, `style_self_sim`; ≥3 seeds; n≈300; report surviving-n per arm.
5. **Stop the degenerate-input masking (R6).** `NaN` (not 0.0) on empty in `wasserstein_1d`/`ks_statistic`; record `n_failed`/`n_empty`; count empties as wrong or report the drop rate.
6. **Add anti-gaming guards (R7).** Copy/leak rate (critical under R1), distinct-reply/dup rate, opener-entropy; treat the human win-rate + one pre-registered composite as the verdict, the tic panel as descriptive.
7. **De-leak `style_self_sim` (R5).** Frozen model-disjoint validator centroid; `BEST_OF_N=1` both arms; or demote it to diagnostic-only.
8. **Build the blind human-preference kit (R8).** Randomised-order forced choice over the shared set; stored un-blinding key; the only verdict on the real target. (Strip the shadow-log model field if reused — it's mislabelled.)
9. **Freeze the facts corpus (insights).** Versioned snapshot loaded identically by both arms; no re-distill between arms; report corpus composition. Fix the reject-Qdrant-id bug (`verification.py:153` passes the raw id, not `to_qdrant_point_id(...)`) so human-rejected facts don't linger in the API's retrieval pool.

### Nice-to-have — *raises portfolio/paper quality, not a blocker for decide*
- **Instrument latency + cost** (decision-relevant; borderline must-fix for the ship decision): per-call wall-time + `resp.usage`; p50/p95, $/1k, tok/s. The LoRA's entire value prop is currently unscorable.
- **Per-register / per-language / per-recipient breakdown** (R10): small table of `shape_js`/win-rate per bucket using `recipient_id_hash` + `detect_language`.
- **Wire MLflow** (R9): call `log_eval_run` per arm with flattened scalar metrics + a sanitize step (drop nested `real`/`gen` dicts, strip `None`) + git SHA + split id + adapter/quant/llama.cpp build hash; self-labelling run names that refuse to overwrite. Guard MLflow I/O in try/except with file-only fallback.
- **Fix `latin_script_rate` double-count**; **widen/parameterise the shape overflow bucket** past 6; **enrich opener metric** (top-3 / full-distribution JS).
- **Add a content/semantic-relevance axis** (cheap embedding sim of gen-vs-real for the same incoming) so a shape-right-but-off-topic reply can't score ~0.
- **Construct-validity calibration**: rank-correlate each automatic metric with the human win-rate once labels exist.
- **Re-scope the stress test** as a per-backend grounding smoke harness with a non-zero exit gate; stamp backend provenance; render the captured-but-dropped `system_prompt_excerpt`; label the LoRA's empty-retrieval section as *by design*, not a defect.
- **Calibration anchors** for `style_self_sim`: real-vs-real self-cosine (ceiling) and different-author cosine (floor) in the scorecard so an absolute 0.83 is interpretable.

---

## 6. Open questions for Bohdan (decisions only he can make)

1. **What does "ship the LoRA" actually require?** Is a **voice-tie + cheaper/local/offline** enough to ship the LoRA (cost/latency break ties), or must the LoRA *clearly win* on voice to justify replacing the API? This single threshold drives the formal "better" definition in §4.6.
2. **Production-realism vs controlled — which arm is the headline?** The honest comparison reports both: (A) each backend in its native prompt ("which product ships better") and (B) a shared prompt isolating weights-vs-scaffold ("which *model* is more Bohdan"). For the *decide* and *portfolio* narratives, which is the headline and which is supporting?
3. **How much of your time for the blind panel?** The verdict needs ~100 paired judgments from you for *decide* (≈1–1.5 h), more for portfolio/paper. Are you the sole rater, or do we recruit 1–2 people who know your texting style for inter-rater reliability (needed for a credible paper)?
4. **Re-train the LoRA on the unified split, or accept the current adapter and only re-pick the eval set?** Fixing R1 cleanly means excluding the shared hold-out from training and **re-running the Colab LoRA**. Are you willing to re-train, or do we accept a known (and disclosed) small residual overlap and proceed? *(Re-training is the only way to make a paper-grade leak-free claim.)*
5. **Acceptable degeneracy thresholds.** What copy/leak-rate (ε) and distinct-reply-rate (τ) do you consider acceptable? A LoRA that wins preference partly by reproducing your real lines may *feel* perfect but is overfitting — where's your line?
6. **Cost/latency weighting in the decision.** How much does local/offline/$0-per-reply matter to you relative to a small voice edge? This sets how aggressively ties (and near-ties) break toward the LoRA.
7. **Multilingual `style_self_sim` encoder.** StyleDistance is English-trained and unvalidated on your uk/ru code-switch. Do you want to validate/swap it for a multilingual style encoder (extra setup), or keep `style_self_sim` as a *diagnostic-only* signal and lean on `shape_js` + the human panel for the verdict?
8. **Stage gate after *decide*.** Is the portfolio/paper contingent on a clear, surprising result (e.g. "a 3B local LoRA matches gpt-4o-mini on persona voice"), or do you want to write it up regardless of which way it lands?
