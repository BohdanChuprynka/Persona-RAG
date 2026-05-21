# Evaluation

How to measure whether the bot actually sounds like the persona. Every run tracked in MLflow.

## Why this matters

The predecessor project used BLEU/ROUGE — n-gram overlap with a single reference reply. Persona is a *distribution*. Two replies with zero word overlap can both be perfectly in-character. BLEU/ROUGE optimize for the wrong thing.

This project measures three dimensions independently and logs them per run to MLflow:

1. **Stylometric match** — statistical fingerprint similarity
2. **Semantic plausibility** — perplexity-proxy against persona's real replies
3. **Human A/B** — can a friend tell bot apart from real persona

## Held-out split

Last 10% of `PersonaTurn` rows by `timestamp` are tagged `eval_split=true` at ingest time and **excluded from retrieval** (Qdrant filter on `eval_split=false`). They never appear in few-shot examples. They serve as the ground truth for everything below.

Time-based, not random — random splits leak today's style into the past corpus.

## Shadow mode (data collection runtime)

`SHADOW_MODE=true` turns the bot into a logger. Every incoming message:

1. Walks the full LangGraph (auth → retrieve → prompt → generate → guardrails)
2. **Does not send** the reply to Telegram
3. Appends a JSONL row to `data/shadow_log.jsonl`:

```json
{
  "ts": "2026-05-17T14:23:11Z",
  "session_id": "uuid",
  "user_id_hash": "blake2b(...)",
  "incoming": "actual friend message",
  "context": ["..."],
  "retrieved_ids": ["uuid1", "uuid2"],
  "memory_summary": "...",
  "generated_reply": "what the bot would have said",
  "your_actual_reply": null,
  "model": "gpt-4o-mini",
  "params": {"top_k": 8, "alpha": 0.7, "temperature": 0.8, "prompt_version": "v1"}
}
```

`your_actual_reply` is backfilled later from your real Telegram export when you re-ingest. Once enough triples accumulate, this becomes:

- The dataset for **Metric 3** (human A/B against real ground truth)
- The DPO training set for Phase 2 (chosen=your_actual, rejected=generated)

## Metric 1 — Stylometric match

`scripts/eval_persona.py --metric stylometry`

For each held-out turn `t`:

- Feed bot `t.incoming_context` as if real incoming
- Generate a reply `r`
- Compute features for both `r` and `t.your_reply`

| Feature | What |
|---|---|
| `len_chars` | Character length |
| `len_words` | Word count |
| `emoji_rate` | Emojis / total characters |
| `caps_ratio` | Uppercase chars / total alpha |
| `punct_density` | Punctuation per word |
| `lang_mix` | Fraction in each language |
| `avg_word_len` | Mean word length |
| `lexical_diversity` | type/token ratio |
| `top_bigram_jaccard` | Jaccard of top-20 bigrams vs persona corpus |

Report: per-feature mean absolute deviation between bot and real + composite score (z-normalized sum, lower=better).

**MLflow logging:**
- Params: model, top_k, alpha, recency_half_life, prompt_version, temperature
- Metrics: each feature MAD + composite
- Artifacts: per-turn diff CSV

## Metric 2 — Semantic plausibility

`scripts/eval_persona.py --metric perplexity-proxy`

For each held-out turn `t`:

- Build the bot's prompt as in production
- Ask the LLM to *score* the real reply with logprobs (no generation)
- Compute mean per-token logprob of `t.your_reply` under the bot's prompt distribution

Higher mean logprob = the bot's prompt setup "expects" replies that match the persona's actual reply. Proxy for true perplexity that works with OpenAI's `logprobs` API.

Comparison baselines (each its own MLflow run):

- `TOP_K=0` (no retrieval) — measures the lift retrieval provides
- Random retrievals (same `TOP_K`) — measures whether *relevant* retrievals matter
- Shuffled few-shot order — measures whether ordering matters
- `gpt-4o-mini` vs `gpt-4o` — model lift

## Metric 3 — Human A/B

The gold standard. Two protocols, both logged to MLflow.

### Shadow-vs-real A/B (no friends required)

Once shadow mode has logged `your_actual_reply` for ≥100 triples:

`scripts/eval_persona.py --metric ab-self`

For each triple:
- Present `(generated_reply, your_actual_reply)` in random order to a rater (you, blind)
- Rater picks "more like me"

Track:
- Bot-pick rate (target ≈ 50% = indistinguishable)
- Per-feature breakdown of when bot is picked

### Friend A/B (after launch)

Recruit 3–5 trusted friends. Randomly half of the bot's replies in a session are real-you (you type), half are bot. At session end, friends label which were bot.

Track:
- Detection accuracy per friend (50% = perfect bot)
- Patterns from notes — what features tip them off

## MLflow run anatomy

Each `make eval` invocation creates one parent run with N child runs (one per metric):

```
experiment: persona-rag-eval
├── run: 2026-05-17-v1.2-baseline
│   ├── params: {prompt_version: v1.2, top_k: 8, alpha: 0.7, model: gpt-4o-mini, ...}
│   ├── tags: {dataset_version: ingest-2026-05-15, persona_name: <name>}
│   ├── metrics:
│   │   stylometry_composite: 1.23
│   │   stylometry_emoji_rate_mad: 0.005
│   │   stylometry_len_chars_mad: 12.3
│   │   perplexity_proxy_mean_logprob: -1.47
│   │   perplexity_proxy_baseline_no_retrieval: -2.91
│   │   ab_self_bot_pick_rate: 0.42
│   └── artifacts:
│       per_turn_features.csv
│       sample_replies.md
└── ...
```

Compare runs side-by-side in the MLflow UI (`localhost:5001` via docker-compose). Filter by `prompt_version` or `model` to A/B prompt designs.

## Reporting

`scripts/eval_persona.py` also writes a human-readable markdown report:

```
data/eval/2026-MM-DD-eval-report.md
```

Contains:
- Headline composite numbers
- Per-feature deviation table
- 10 sampled (real, bot) pairs side-by-side for qualitative check
- Link to MLflow run
- Trend chart vs last 5 runs

CI does NOT run eval — costs money, slow. Run manually after prompt or schema changes.

## When to act on the numbers

| Signal | Likely cause | Fix |
|---|---|---|
| Stylometric `emoji_rate` way off | Bot strips emojis | Check tokenizer / prompt instructions |
| `len_chars` too high | LLM verbosity bias | Lower `MAX_REPLY_TOKENS`, add "short replies" instruction |
| Mean logprob low + retrieval baseline high | Retrieval is hurting | Check embedding model, recency weight |
| Mean logprob low across the board | Wrong base model | Try `gpt-4o` |
| Bot-pick rate ≪ 50% | Bot is identifiable as bot | Look at qualitative samples |
| Bot-pick rate ≈ 50% | Indistinguishable. Ship it. | — |
| Bot-pick rate > 60% | Suspiciously human — check for leakage of held-out into retrieval | Verify `eval_split` filter is on |
