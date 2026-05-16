# Evaluation

How to measure whether the bot actually sounds like the persona.

## Why this matters

The predecessor project used BLEU/ROUGE — n-gram overlap with a single reference reply. Persona is a *distribution*. Two replies with zero word overlap can both be perfectly in-character. BLEU/ROUGE optimize for the wrong thing.

This project measures three dimensions independently:

1. **Stylometric match** — does the bot's output have the same statistical fingerprint as the persona's real corpus?
2. **Semantic plausibility** — does the bot's reply continue the conversation in a way the persona could have?
3. **Human A/B** — can a friend tell the bot apart from the real persona?

## Held-out split

At ingest time, the last 10% of `PersonaTurn` rows by `timestamp` are marked `eval=true` and **excluded from retrieval**. They never appear in few-shot examples. They serve as the ground truth for everything below.

Why time-based split, not random? Persona drifts. Random splits leak today's style into the past corpus. Time-split eval is the only honest test.

## Metric 1 — Stylometric match

For each held-out turn `t`:

- Feed the bot `t.incoming_context` as if it were a real incoming message
- Generate a reply `r`
- Compute features for both `r` and `t.your_reply`:

| Feature | What |
|---|---|
| `len_chars` | Character length |
| `len_words` | Word count |
| `emoji_rate` | Emojis / total characters |
| `caps_ratio` | Uppercase chars / total alpha |
| `punct_density` | Punctuation per word |
| `lang_mix` | Fraction of tokens in each language (multilingual) |
| `avg_word_len` | Mean word length |
| `lexical_diversity` | type/token ratio over the reply |
| `top_bigram_jaccard` | Jaccard of top-20 bigrams vs persona corpus |

Report per-feature mean absolute deviation between bot and real, plus a single composite score (z-normalized sum, lower = better).

Run with `make eval` → `scripts/eval_persona.py --metric stylometry`.

## Metric 2 — Semantic plausibility (perplexity proxy)

For each held-out turn `t`:

- Build the bot's prompt as in production
- Ask the LLM to *score* the real reply with logprobs instead of generating
- Compute mean per-token logprob of `t.your_reply` under the bot's prompt distribution

Higher mean logprob = the bot's prompt setup "expects" replies that match the persona's actual reply. This is a cheap proxy for true perplexity that works with closed-source models via OpenAI's `logprobs` API.

Comparison baselines:

- Same metric with TOP_K=0 (no retrieval) — measures the lift retrieval provides
- Same metric with random retrievals — measures whether *relevant* retrievals matter
- Same metric with shuffled few-shot order — measures whether ordering matters

## Metric 3 — Human A/B

The gold standard. Two protocols:

### Shadow mode (no friends required)

Bot runs in production but **doesn't send replies** — it logs `(incoming_msg, generated_reply, your_actual_reply)` triples to a CSV. After a week of shadow, you self-rate the bot's replies on a 1–5 scale:

1 = clearly not me
3 = could go either way
5 = I would have said this

Aim for ≥ 4.0 mean before flipping to live.

### Friend A/B (after launch)

Recruit 3–5 trusted friends. Each session, randomly half of the bot's replies are real-you (admin types) and half are bot. At session end, friends label which were bot. Track:

- Detection accuracy (50% = perfect bot)
- Detection patterns (which features tip them off?)

This is the only test that matters in the end. The two automated metrics are early-warning systems.

## Reporting

`scripts/eval_persona.py` writes a markdown report:

```
data/eval/2026-XX-XX-eval-report.md
```

Contains:

- Metric 1: per-feature deviation table + composite
- Metric 2: mean logprob with baselines
- Metric 3: per-session A/B accuracy if data exists
- Trend chart (markdown table) vs prior runs

CI does NOT run eval — it touches OpenAI, costs money, and is slow. Run manually after schema or prompt changes.

## When to act on the numbers

| Signal | Likely cause | Fix |
|---|---|---|
| Stylometric `emoji_rate` way off | Bot strips emojis | Check tokenizer / prompt instructions |
| `len_chars` too high | LLM verbosity bias | Lower `MAX_REPLY_TOKENS`, add "short replies" instruction |
| Mean logprob low + retrieval baseline high | Retrieval is hurting | Check embedding model, recency weight |
| Mean logprob low across the board | Wrong base model | Try `gpt-4o` instead of `gpt-4o-mini` |
| Friend A/B accuracy > 70% | Bot is identifiable | Look at friend's notes — what gave it away? |
| Friend A/B accuracy ~ 50% | You're done. Ship it. | — |
