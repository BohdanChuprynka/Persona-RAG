# LoRA-vs-real (Turing) test — design

## Why

The blind **API-vs-LoRA** panel turned out *trivially discriminable* — Bohdan can
identify the API reply every time. That **closes the relative question** (the LoRA
beats `gpt-4o-mini` on voice, decisively) but it proves only that the API is
obviously-not-him. It does **not** prove the LoRA is indistinguishable from *him*.
The still-hard, still-interesting question is the **absolute** one:

> Can the persona-target tell the LoRA's reply apart from his own real reply?

## Method

For each held-out context, pair the LoRA generation against Bohdan's **real**
reply (both already in `data/eval/compare/<name>/pairs.jsonl` — **no regeneration,
$0**). Blind, randomized A/B, one forced question: **"which reply is the machine?"**
Every catch also records a one-tap **tell** (why he thinks so).

- **The pass condition flips.** Detection rate ≈ 0.5 — a Wilson 95% CI that
  *includes* 0.5 — means he can't beat chance ⇒ the LoRA is statistically
  **indistinguishable** from him (it passes). A CI strictly above 0.5 means he
  can tell.
- This is the harshest possible test: the persona-target is the strongest
  discriminator of his own voice. Passing here is a real claim.

## The tell taxonomy (the diagnostic payoff)

A bare detection rate says *whether* he can tell, not *where it messes up*. So
each catch is tagged and bucketed:

- **voice** — `wording`, `length`, `punct/caps`, `too-generic`, `topic`.
  Fixable by decode params / more training.
- **knowledge** — `missing-facts`: the real reply carried specifics the model
  can't know. Fixable by **retrieval / RAG**.

The **voice-vs-knowledge split is the RAG business case.** If catches are mostly
`missing-facts`, voice is solved and the remaining gap is *grounding* (→ Obsidian
+ chat-history RAG, multilingual retrieval). If mostly voice, it's a
decode/training problem, and RAG won't help much.

## Honest caveat (read the verdict with the split, never the rate alone)

There is exactly **one** ground-truth reply per context, but many replies could
be equally "him." A catch may reflect *missing private knowledge*, not a voice
defect — which is exactly why the tell attribution matters: it separates "wrong
voice" from "doesn't know my life." Backchannel turns ("ok", "ya", "haha") are
expected to be indistinguishable; fact-dependent turns are where catches cluster.

## How to run

```bash
make turing-build      # -> reports/main/turing/rater.html (+ key.json, pairs_blind.csv)
# open rater.html, rate, download choices.json INTO reports/main/turing/
make turing-score      # detection rate + Wilson CI + voice/knowledge tells + per-language
```

`reports/` stays git-ignored (the kit embeds real chat). Pure logic + 7 unit
tests: `persona_rag/eval/compare.py` (`build_turing_kit`, `score_detection`,
`bucket_tells`) and `tests/test_eval_compare.py`. The API-vs-LoRA kit
(`build_human_eval.py` / `score_human_eval.py`) is **unchanged** — this is purely
additive.

## Prior (to be replaced by the run)

Guess before rating: detection ~**65-75%** (LoRA alone), with ~**40-55%** of
catches being `missing-facts`. Non-uniform — strong on fact-dependent replies,
≈chance on backchannel. The run replaces this guess with data, and the
`missing-facts` rate is the direct estimate of how much an Obsidian/RAG layer
would buy.
