= Building the replica <sec-build>

This part is the implementation story: what it takes to make a model that texts
like one specific person, and the design choices that the evaluation later has to
account for. The running subject is a fine-tuned #emph[Qwen2.5-3B] LoRA
@qwen2_5 @hu2021lora served locally, set against the shipped OpenAI gpt-4o-mini
product it was built to replace.

== The target: what "sounds like me" means

The system has exactly one thing it must get right — _does this read like something
the owner would actually text?_ That target is subjective but it has measurable
fingerprints: heavy Ukrainian/Russian/English code-switching (about 0.18 of
characters in Latin script, aggregated across contacts); terse, multi-bubble
bursts (several short messages rather than one paragraph); a `)` smiley tic; almost
no `!`; varied openers; lowercase casing. The deployed gpt-4o-mini was capable and
fluent, but on these register markers it did not sound like the person — which is
what motivated a fine-tune.

== System architecture

Each inbound Telegram message runs through a single LangGraph pipeline
(@fig-d1): authentication, hybrid retrieval, contact memory, self-insights,
prompt assembly, generation, guardrails, and bubble-wise delivery. The
generation node fans out to two interchangeable backends — the OpenAI API
(gpt-4o-mini) and a local `llama.cpp` server hosting the fine-tuned LoRA. A single
`GENERATION_BACKEND` switch selects which, and routes the local path to _skip_
retrieval entirely: the fine-tune serves on the same thin prompt it trained on.

#figure(
  include "../diagrams/d1_architecture.typ",
  caption: [Persona-RAG. One LangGraph pipeline; the generation node fans to two
  backends. The API path carries a rich retrieval-augmented prompt and decode
  levers; the local LoRA path skips retrieval and serves a thin prompt.],
) <fig-d1>

== Data, turns, and the split problem

Raw Telegram/Instagram exports are PII-redacted, collapsed into bursts (messages
within 300s of each other), cut into sessions on 6-hour idle gaps, and reduced to
_persona turns_: each turn is one reply the owner sent, paired with the preceding
context. This `(context → reply)` pair is the atomic unit for both retrieval and
fine-tuning.

Two different held-out splits exist, and the distinction is load-bearing for the
evaluation (@fig-d2). The product indexes and serves against a _temporal_ split
(the most recent 10% of turns by timestamp). But the temporal tail is
register-skewed: a single English-speaking contact accounts for 62% of all Latin
script in the corpus, so the recent slice reads about 0.47 Latin against the
person's true \~0.18 — an artifact that makes the voice target unreachable. The
fine-tune therefore trains and evaluates on a _recipient-stratified_ split
(`eval_split_for`), a deterministic hash that holds out \~10% of _every_ contact
independently, so train and eval share the same code-switch register
(train ≈ eval ≈ 0.18). Every fair comparison in this report scores on that
recipient-stratified, LoRA-disjoint hold-out.

#figure(
  include "../diagrams/d2_split_leak.typ",
  caption: [Data pipeline and the two hold-out splits. The temporal split (left)
  drives indexing but is register-skewed; the recipient-stratified split (right)
  gives the fine-tune an honest, reachable voice target. The retrieval leak that
  the temporal index enables is removed per-item (@fig-leak).],
) <fig-d2>

== Retrieval

The API path retrieves the owner's own past replies as few-shot examples. Dense
search (`text-embedding-3-small` over Qdrant) and BM25 @robertson2009bm25 are
min-max-normalised and fused (0.7 weight on dense), decayed by recency (180-day
half-life), floored at a minimum score, and diversified by Maximal Marginal
Relevance @carbonell1998mmr ($lambda = 0.6$) down to the four best, which are then
reversed so the single most relevant example lands last. A parallel store of
distilled "self-insights" adds a bio-facts block. All of it renders into a
\~1600-token system prompt with explicit, enforced voice rules.

== The fine-tune, and the train==serve invariant

The fine-tune is a QLoRA @dettmers2023qlora adapter (rank 32, $alpha = 64$) on
Qwen2.5-3B-Instruct, trained with Unsloth @unsloth on a free Colab T4. Loss is
masked to the assistant turn only, and multi-bubble newlines are preserved so the
model learns the burst shape. Crucially, the system turn used in training is the
byte-identical one-line persona anchor used at serving time (@fig-d4) — any drift
between the two would be train/serve skew. After training, the adapter is merged
and then converted and quantized _locally_ to a GGUF `Q5_K_M` model, served through
`llama.cpp`'s OpenAI-compatible server @llamacpp (the Homebrew Ollama formula on
the target machine shipped no inference runtime).

#figure(
  include "../diagrams/d4_train_serve.typ",
  caption: [Train==serve. The same thin persona-anchor string conditions both
  training and serving; the adapter is merged, quantized to GGUF Q5_K_M, and
  served locally.],
) <fig-d4>

Why fine-tune at all, rather than prompt harder? Message _shape_ was already solved
by prompt shape-conditioning. But lexical voice — the code-switch, opener variety,
the `)` tic, the absent `!` — does not respond to soft prompt instructions: the base
model obeys hard per-reply directives but ignores style requests. Those tics had to
be learned from the data. The production API path compensates differently, with two
decode levers (a positive logit bias on the `)` token, a strong negative bias on `!`)
and per-reply register directives — machinery the next part puts on trial.
