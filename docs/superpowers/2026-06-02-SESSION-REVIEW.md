# Autonomous session review — 2026-06-02

Read this first. Everything is on branch **`feat/eval-ab-comparison`** (local commits, **not pushed**). Nothing went to GitHub or live Telegram.

## TL;DR

1. **Your fine-tuned model is live and served.** GGUF built locally (`models/bohdan-q5_k_m.gguf`), served by `llama-server` on `:11434` as `bohdan`; the bot's `--local` preflight passes. (Homebrew's `ollama` is broken — ships no runtime — so we serve via llama.cpp; details in [[persona-rag-state-and-voice-gap]] / the earlier session.)
2. **The eval stack got a hard audit** → found a disqualifying ~90% train/test leak in the old runner. We built a *fair* comparison from scratch.
3. **First trustworthy result (controlled arm, n=300, replicated):** the 3B fine-tune matches your terse, no-`!` register far better than raw `gpt-4o-mini`; bubble-shape is a tie; runs at $0 locally. **Not the final verdict** — needs the human panel + the production-RAG arm + a memorization check (below).

## UPDATE (2026-06-02, continued) — LoRA-vs-you (Turing) test added

While grading the blind API-vs-LoRA panel you found it **trivially discriminable** — you can ID the API every time. That closes the *relative* question (LoRA ≫ API on voice) but proves only that the API is obviously-not-you, not that the LoRA is indistinguishable from *you*. So I built the test that's still hard — the **absolute** one:

- **`make turing-build`** → `reports/main/turing/rater.html`: blind, randomized **your real reply vs the LoRA** for the same context, forced choice **"which is the machine?"**, with a one-tap **tell** (why) per catch. Reuses the already-generated pairs — no regeneration, $0.
- Rate, download `choices.json` into that folder, then **`make turing-score`** → detection rate + Wilson CI + the **voice-vs-knowledge tell split** + per-language.
- **Pass = indistinguishable:** a detection CI that includes 0.5 means the LoRA passes as you. The tell split localizes what's left — **knowledge** (missing-facts) ⇒ RAG; **voice** ⇒ decode/training — i.e. it sizes the RAG business case empirically.
- Design + honest caveat (one ground-truth reply ≠ the only valid reply; attribute the tells): [`2026-06-02-turing-test-design.md`](2026-06-02-turing-test-design.md). Pure logic + 7 tests in `compare.py` / `test_eval_compare.py`. The API-vs-LoRA kit is untouched.

## What I built (commit order on the branch)

1. `chore:` serve LoRA from in-repo `models/` via llama.cpp.
2. `docs:` **eval-architecture audit** (`docs/superpowers/2026-06-02-eval-architecture-audit.md`) + **comparison design spec** (`docs/superpowers/specs/2026-06-02-api-vs-finetune-comparison-design.md`).
3. `feat(eval):` the fair **comparison harness** — `persona_rag/eval/compare.py` (pure, 14 unit tests) + `scripts/compare_persona.py` (runner).
4. `feat(eval):` **charts + human-kit** scripts (`scripts/plot_comparison.py`, `scripts/build_human_eval.py`) + config.
5. `docs:` **findings** (`2026-06-02-comparison-findings.md`) + **portfolio draft** (`docs/portfolio/...`) + this review.

`git log --oneline feat/eval-ab-comparison` shows them. All lint/type/test gates green.

## Decisions I made for you (defaults — override freely)

Recorded in the design spec §2. The load-bearing ones:
- **"Better" = blind human win-rate**; a voice-tie + cheaper/local **ships the LoRA** (cost breaks ties).
- **Controlled arm is the scientific headline**; production-realism arm is the deployment view (not built yet).
- **No re-train** — fixed the leak by scoring on the existing LoRA-disjoint `eval.jsonl` instead.
- **`style_self_sim` demoted to diagnostic** (English encoder, unvalidated on uk/ru).
- **Quant = q5_k_m** (the doc default).

## Open questions only you can answer (prioritized)

1. **Do you want the production-realism arm (A)?** It compares the *shipped* API (rich RAG prompt + retrieval) vs the LoRA — the honest "which ships better." Ready-to-execute plan + leak-safety design in [`2026-06-02-arm-a-plan.md`](2026-06-02-arm-a-plan.md) (≈ half a day). The controlled arm we have isolates the *model*; arm A judges the *product*.
2. **The blind human panel** (~10–15 min): open `reports/main/human_eval/rater.html`, rate 100 pairs, download `choices.json` into that folder, then `make compare-score`. The scorer (win-rate + Wilson CI + per-language) is built and tested. *You've already called this one by eye (LoRA wins) — so the more meaningful verdict to rate now is the **LoRA-vs-you Turing panel** in the UPDATE above (`make turing-build` → rate → `make turing-score`).*
3. ~~**Copy-rate threshold**~~ — **resolved while you were out:** the LoRA's ~15% near-copy ≈ the **11% natural short-text floor** (your own held-out replies near-match train 11.1%), and it near-matches *unseen* reals only 2.7% → not overfitting. No threshold decision needed.
4. The remaining audit questions (raters/IRR, multilingual style encoder, paper go/no-go) — spec §2 / audit §6.

## How to review (suggested order)

1. This file → `docs/superpowers/2026-06-02-comparison-findings.md` (the result + caveats).
2. `reports/main/` — open the 5 PNG charts + `summary.md`.
3. The spec (`docs/superpowers/specs/2026-06-02-...-design.md`) and audit if you want the why.
4. `reports/main/human_eval/rater.html` — do the blind panel.
5. Skim the code: `persona_rag/eval/compare.py` + `scripts/compare_persona.py`.

## Not done / next builds

- **Arm A** (production-realism) — ready-to-execute plan in [`2026-06-02-arm-a-plan.md`](2026-06-02-arm-a-plan.md).
- **Run the human panel** — kit + scorer built (`make compare-score`); just needs your ratings.
- **MLflow wiring + per-recipient breakdown**, multilingual style encoder, greedy/multi-seed decode — all in the audit's "nice-to-have."

*(Done while you were out: copy-rate baseline → caveat resolved; human-panel scorer + tests; arm-A plan; `make compare*` targets.)*

## Privacy

`reports/` is git-ignored — the human-eval kit embeds real chat content. Charts/summary are aggregate but kept local too; publish selectively after you review. `data/` stays ignored as before.

## Reproduce

```bash
# llama-server must be up: llama-server -m models/bohdan-q5_k_m.gguf -a bohdan --host 127.0.0.1 --port 11434 -c 8192 --parallel 4 --jinja
make ... # or:
uv run python scripts/compare_persona.py --n 300 --seed 0 --name main
uv run python scripts/plot_comparison.py --name main
uv run python scripts/build_human_eval.py --name main --n 100
uv run pytest tests/test_eval_compare.py -q
```
