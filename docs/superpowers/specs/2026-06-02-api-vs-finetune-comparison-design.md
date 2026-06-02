# Persona-RAG — API vs Fine-tuned LoRA: Staged Comparison Design

**Date:** 2026-06-02
**Branch:** `feat/eval-ab-comparison`
**Builds on:** [`docs/superpowers/2026-06-02-eval-architecture-audit.md`](../2026-06-02-eval-architecture-audit.md)
**Status:** design + decisions. Implemented in this branch (see §8). *Decisions marked "(default — confirm)" are mine pending Bohdan's review.*

---

## 1. Goal & staging

Determine, **trustworthily**, whether the locally fine-tuned **Qwen2.5-3B LoRA** matches or beats **`gpt-4o-mini`** at speaking in Bohdan's texting voice — and decide which to run. Staged so each stage raises the evidentiary bar:

1. **decide** — cheap automatic screen + a small blind human panel → ship which backend?
2. **portfolio** — full human panel + ablation arms + writeup.
3. **paper** — larger n, inter-rater reliability, pre-registration, leak-free re-train.

The single target is subjective: *"does this read like something Bohdan would actually text?"* (uk/en/ru code-switch + per-context mirroring, opener variety, the `)` tic, no `!`, casual casing, multi-bubble shape).

---

## 2. Decisions on the audit's 8 open questions *(defaults — confirm)*

1. **Ship threshold.** Primary verdict = **blind human win-rate**. A **voice-tie + cheaper/local/offline ⇒ ship the LoRA**; ship the API only on a *clear* human voice-win that survives guardrails. (Cost/latency breaks ties toward the LoRA — that's its value prop.)
2. **Headline arm.** The **controlled arm (B)** is the scientific headline — *"which model is more Bohdan under identical minimal prompting"* — because it is the only attributable, leak-free comparison we can run today. The **production-realism arm (A)** (*"which shipped product wins"*) is reported once built (deferred, §7).
3. **Raters.** Bohdan is the sole gold rater for *decide* (~100 paired items). Kit is built so 1–2 extra raters can be added later for inter-rater reliability (κ).
4. **Re-train?** **No re-train now** (no local GPU; training is the free-Colab kit). We fix the audit's #1 leak by scoring on the **existing LoRA-disjoint `eval.jsonl`** hold-out — leak-free *without* retraining. A unified-split re-train is the paper-grade follow-up.
5. **Degeneracy thresholds.** Provisional: **copy/leak rate ε ≤ 5%** (near-exact training-reply reproduction), **distinct-reply rate τ ≥ 0.90** (mode-collapse guard). Confirm/calibrate.
6. **Cost/latency weight.** **High.** Local, offline, $0/reply is a primary decision factor; near-ties break to the LoRA.
7. **`style_self_sim`.** **Diagnostic-only.** The StyleDistance encoder is English-trained and unvalidated on uk/ru code-switch; the verdict leans on `shape_js` + the human panel, not this cosine.
8. **Stage gate.** **Write up regardless of which way it lands** — the methodology (a fair RAG-vs-fine-tuning persona comparison) is the portfolio value. Paper only if the result is clean *and* interesting (e.g. a 3B local model matching `gpt-4o-mini` on voice).

---

## 3. The fair comparison — how it fixes the audit

The audit's three disqualifying problems and how this design closes them:

| Audit risk | Fix in this design |
|---|---|
| **R1 — ~90% train/test leak** (runner scored the temporal split; LoRA held out the hash split) | Score **both** backends on `data/finetune/eval.jsonl` — the recipient-stratified `eval_split_for` hold-out the LoRA *trained disjoint from* (`train.jsonl`). Leak-free by construction; the API trained on nothing. |
| **R2 — thin-vs-rich prompt confound** | **Arm B (controlled):** the *identical* thin prompt (`THIN_SYSTEM` + joined context) goes to **both** backends; no retrieval, no directives. Isolates *weights*, not scaffolding. (Arm A, which keeps each backend's native prompt, is deferred — §7.) |
| **R3 — decode-lever asymmetry** | No `logit_bias`, no register/shape directives on either side; matched `temperature` + `n=1` + matched `max_tokens`; truncation rate reported so we can confirm the char-budget confound doesn't bind. |
| **R4 — no uncertainty** | Paired bootstrap 95% CIs on the api−lora deltas; cross-seed replication; surviving-n reported. |
| **R6 — degenerate masking** | Empty generations counted (not silently dropped); metrics return `NaN` (not `0.0`) on empty input; per-arm empty/failed rates reported. |
| **R7 — gameable, no guard** | Copy/leak rate + distinct-reply rate reported; the **human panel** is the non-gameable verdict. |

**Shared hold-out:** `data/finetune/eval.jsonl` — 1,104 records, ShareGPT `{system: THIN_SYSTEM, human: joined-context, gpt: real reply}`. Sample **n = 300** for the screen (drops the `shape_js` noise floor from ~0.06 at n=80 to ~0.02).

---

## 4. Metrics (per audit §4.2)

- **Headline (with CIs):** `shape_js`, `len_wasserstein` (+ a length distance normalized by median real length). `style_self_sim` *diagnostic-only* and only with a frozen, model-disjoint centroid.
- **Voice-tic panel (diagnostic, with CIs):** `latin_script_rate`, `opener_top_share` **+ opener-entropy**, `paren_smiley_rate`, `caps_ratio_mean`, explicit `exclaim_rate` (the "no `!`" rule). Reported for each backend **and the real reference**, so "closer to real" is visible.
- **Anti-gaming guards (always reported):** copy/leak rate vs `train.jsonl`; distinct-reply rate; empty/failed rate per arm.
- **Reference** = the real `gpt` replies from the same 300 items (paired).

---

## 5. Statistics

- **Automatic screen:** paired bootstrap (10k resamples, same resampled indices for both arms) → 95% CI on `Δshape_js`, `Δlen_wasserstein`. A delta counts only if its CI excludes 0. Cross-seed replication (seed 0 @ n=300, seed 1 @ n=150) to expose the noise floor.
- **Human verdict:** forced-choice win-rate with **Wilson 95% CI**; winner iff the CI excludes 0.5, else **declared a tie**. ~100 items resolves a clear (≈60/40) preference, which is the right bar for a ship decision.

---

## 6. Formal definition of "better" *(pre-registered, per audit §4.6)*

> **Primary (verdict):** backend X is *better on voice* iff its **blind human win-rate** over Y has a Wilson 95% CI excluding 0.5. CI straddles 0.5 ⇒ **voice tie**.
> **Guardrails (override a human win):** (i) copy/leak rate ≤ ε; (ii) distinct-reply rate ≥ τ; (iii) bootstrap CI on `Δshape_js` must not show X worse beyond the noise floor. A win by memorization or mode-collapse does not qualify.
> **Ship rule (decide):** ship the **LoRA** on a voice-win **or** a voice-tie + materially cheaper/faster/offline; ship the **API** only on a clear, guardrail-clean voice-win. Record the numbers that drove it.
> **Construct validity (portfolio/paper):** once human labels exist, rank-correlate each automatic metric with the human win-rate to learn which proxy actually predicts "feels like Bohdan."

---

## 7. Deferred (next builds, documented so they're not forgotten)

- **Arm A — production-realism.** API in its native rich-prompt+retrieval condition vs the LoRA thin. Requires a **turn_id-tagged** hold-out export and **excluding those ids from the API's retrieval** (else the API can retrieve the gold reply — a fresh leak). Build a `turn_id`-carrying variant of the export, then extend the runner with an arm-A path through the graph with retrieval exclusion.
- **Leak-free re-train** on a unified split (paper-grade clean claim).
- **Multilingual style encoder** validation/swap (R5) before `style_self_sim` is trusted beyond diagnostic.
- **MLflow wiring** + per-recipient/per-language breakdown tables (R9/R10).
- **Decode-variance**: re-decode at ≥3 seeds and a greedy (temp≈0) stable anchor (cost permitting).

---

## 8. What's implemented in this branch

- `persona_rag/eval/compare.py` — pure, tested comparison logic: paired bootstrap CIs, copy/leak rate, distinct-reply rate, empty/failed accounting, opener-entropy, exclaim-rate, language bucketing, NaN-safe metric wrappers, scorecard assembly.
- `scripts/compare_persona.py` — the arm-B runner: loads `eval.jsonl`, generates from both backends under the identical thin prompt, captures latency + token usage, writes `data/eval/compare/<ts>/{results.json, pairs.jsonl}`.
- `scripts/build_human_eval.py` — blind randomized rating kit (`pairs_blind.csv` + `key.json` un-blinding key) + a minimal HTML rater.
- `scripts/plot_comparison.py` — charts (per-metric api vs lora vs real, distribution overlays, scorecard) → `reports/`.
- `reports/` — generated findings + charts.
- `tests/test_eval_compare.py` — unit tests for the pure logic.

All on `feat/eval-ab-comparison`, committed incrementally, **not pushed**. Bohdan reviews and polishes on return.
