# Arm A (production-realism comparison) — design spec

- **Date:** 2026-06-02
- **Branch:** `feat/eval-ab-comparison` (local commits, not pushed)
- **Status:** approved design; ready for implementation plan (writing-plans)
- **Supersedes:** the sketch in `docs/superpowers/2026-06-02-arm-a-plan.md`
- **Produced via:** the `arm-a-plan-research` workflow (4 parallel code-maps -> synthesize -> adversarial critique); every claim below verified against the live code / DB / `.env`.

## 1. Goal & scope

Arm A answers the **product** question: *"which deployed system replies more like Bohdan on a message it has never seen?"* It compares:

- **API arm** = the SHIPPED OpenAI product: `gpt-4o-mini` + the rich ~1600-token `SYSTEM_TEMPLATE` prompt + hybrid dense+BM25 retrieval injecting few-shot turns + register/shape directives + the shipped decode levers.
- **LoRA arm** = the fine-tuned Qwen2.5-3B in its REAL thin serving config (`THIN_SYSTEM` one-liner + joined context, no retrieval), served via `llama-server` on `OLLAMA_BASE_URL`.

Both are graded on the **recipient-stratified hold-out** (`eval_split_for(turn_id) == True` — the LoRA-disjoint set behind `data/finetune/eval.jsonl`).

This contrasts with **arm B** (`scripts/compare_persona.py`, already built), which gives BOTH backends the identical thin prompt and no retrieval, isolating the *weights*. Arm A deliberately re-introduces all the API-side machinery arm B strips, to measure the deployed product.

**Non-goals:** no re-training; no change to production runtime behaviour (all new retrieval params default to no-op); the blind human panel remains the overriding verdict for both arms.

## 2. Definition of "better" (unchanged from the comparison spec)

Lower distributional distance to Bohdan's real register (`len_wasserstein`, `shape_js`) + on-voice tic rates (`exclaim_rate`, paren rate), with paired bootstrap CIs. A voice-tie that is cheaper/local ships the LoRA. The blind human win-rate overrides automatic metrics. See `docs/superpowers/specs/2026-06-02-api-vs-finetune-comparison-design.md` §2.

## 3. The leak and the guard (the crux)

The hold-out gold turns are **still in the retrieval corpus**. The existing `exclude_eval` guard (`qdrant_store.search_dense`, `eval_split` field) filters only the **temporal** `eval_split` column; arm A scores the **recipient-stratified** `eval_split_for` split — a *different* set — and BM25 applies no eval filter at all. **Empirically: ~89% (2203/2466) of hold-out turns are BM25-retrievable; the temporal filter covers only 263.** So without a new guard, the API can retrieve the exact gold reply as a few-shot example and fake a win.

**The guard (API arm only — the LoRA path skips retrieval entirely, `graph/compile.py:32`, so it cannot leak):**

1. **Per-item retriever-level exclusion** of the scored hold-out ids (the whole eval set), on **both** retrievers.
2. **A hard two-leg leak assertion** run per item, before prompt assembly (belt-and-suspenders over the exclusion).
3. **Top-hit similarity logging** so an API win is auditable as generalization vs. a near-duplicate already in history.
4. **Keep all other history** in retrieval — a near-twin from a *different* turn is fair (it is available in production too). Remove only the exact answer key.

## 4. Decisions

| Decision | Choice | Why |
|---|---|---|
| **Exclusion mechanism** | Thread `exclude_ids: set[str] \| None = None` through the live retrieval stack (default `None` = zero prod change). | Minimal, additive; filters at the retriever (pre-MMR-pool) so it preserves pool fullness + `TOP_K` honesty and removes only the exact answer key (keeps fair near-twins). A train-only index rebuild over-excludes the whole hold-out and diverges from what prod serves. |
| **Runner location** | Sibling `scripts/compare_persona_armA.py` importing arm B's scaffold + scorer. | Keeps arm B's leak-free runner pristine; quarantines the fail-the-run code + the `GENERATION_BACKEND` flip in one auditable file; zero metric duplication. |
| **Eval item set** | Reuse arm B's EXACT hold-out (re-derived from the DB with `iter_records`' filters; deterministic order), recover ids, **assert byte-identical alignment**; regenerate LoRA gens locally ($0). Fall back to corpus-level A-vs-B if alignment can't be proven. | Unlocks the within-item B->A delta (does the machinery close the gap arm B exposed?). Arm B's `_sample(seed)` shuffles + drops ids, so its `pairs.jsonl` gen_lora is NOT alignable — regenerate. |
| **Retrieval-query fidelity** | **`ctx[-1]` — runtime-faithful.** `incoming = ctx[-1]`; `session = ChatMessages(ctx[:-1])`; `retrieve(ctx[-1], exclude_ids=...)`; `detect_register()/target_bubbles()` on `ctx[-1]`. | The live bot queries with the latest message only (`chat.py:51 incoming = message.text`), earlier turns in `session`. Most honest production-realism. |
| **Lever headline** | **Shipped `.env` levers = headline** (`PAREN_LOGIT_BIAS=2`, `EXCLAIM_LOGIT_BIAS=-5`, `BEST_OF_N=1`), read via `get_settings()`. `--learned` (0/0/1) = isolation diagnostic. | `.env` overrides the 0/0 config defaults, so the deployed bot IS steered. The product headline must measure the product. |

**Both arms start from the same reconstructed state** (`incoming = ctx[-1]`, `session = ctx[:-1]`) — exactly the live graph state. Each production prompt builder then consumes it as it really does: the API as rich multi-turn + retrieval few-shot; the LoRA via `build_thin_messages`, which joins `session + incoming` back into one user turn (its trained/served shape, equal to arm B's `eval.jsonl` `human`). Both are faithful; no artificial asymmetry is introduced. A useful consequence: the LoRA's arm-A gens should be ~identical to its arm-B gens (same input, same model) — a built-in sanity check.

## 5. Architecture & components

**Governing principle: reuse production code paths verbatim.** Arm A calls the real `retrieve()`, `build_messages()`, `retrieve_insights()`, `detect_register()`/`target_bubbles()`, and `voice_logit_bias()` — it does NOT hand-reconstruct prompts. The ONLY new code is: `exclude_ids` plumbing, the leak guard, a `logit_bias` forward in `_gen_all`, and the runner.

### 5.1 `exclude_ids` plumbing (4 functions, additive, default `None`)

- `persona_rag/retrieval/__init__.py:14` `retrieve(...)` — add `exclude_ids`; forward to `retrieve_dense` (line 31) and `retrieve_bm25` (line 32). Single chokepoint reaching both retrievers.
- `persona_rag/index/qdrant_store.py:106` `search_dense(...)` — add `exclude_ids`; when truthy attach `must_not=[HasIdCondition(has_id=list(exclude_ids))]` to the `Filter` (currently only `must=` is built; `HasIdCondition` already imported at line 11). Persona-turn Qdrant **point id IS `turn.id`** (raw uuid4, `upsert_turns:81`), so `HasIdCondition` matches the gold directly, server-side, before it can enter the pool. (Payload-field `FieldCondition(key="id", ...)` is the fallback form if the collection is ever re-ingested with an id remap.)
- `persona_rag/retrieval/dense.py:11` `retrieve_dense(...)` — add `exclude_ids`; forward to `search_dense` (line 19).
- `persona_rag/retrieval/bm25.py:14` `retrieve_bm25(...)` — add `exclude_ids`; drop ids in the ranked `pairs` **before** the `[:top_k]` slice (line 20) so `top_k` stays honest: `pairs = [p for p in sorted(...) if not exclude_ids or p[0] not in exclude_ids][:top_k]`. ids are `turn.id` strings. Mandatory and independent of dense: `data/bm25.pkl` is built with only the temporal pre-filter (`scripts/reindex.py:90-91`), so the recipient-stratified golds are in the pickle. `fuse_scores` (`hybrid.py:16`) joins on `turn.id`, so an id removed at both retrievers cannot reappear post-fusion.

### 5.2 Leak guard (in the runner, per item, after `retrieve()` and BEFORE `build_messages`)

Two legs with DIFFERENT actions (the critical fix — a blanket exact-text fail would falsely kill ~13% of items where ubiquitous short replies like `да`/`окей`/`ок` recur under other turn-ids):

- **ID leg (hard-fail):** `gold turn_id in {r.turn.id for r in retrieved}` -> `raise RuntimeError` (fail the whole run). The retriever-level exclusion should already prevent this, so a fire here means a real bug.
- **Exact-text leg (conditional):** if `_norm(gold_reply) in {_norm(r.turn.your_reply) for r in retrieved}` (reuse `persona_rag.eval.compare._norm`, `compare.py:49`):
  - **hard-fail** only when the match also shares the **same incoming context** as the gold (`_norm(joined retrieved.incoming_context) == _norm(joined gold.incoming_context)`) — a genuine duplicate of THIS turn under a re-minted id;
  - otherwise **log a counter** (`n_exact_text_dup_diff_context`) — a fair near-twin per the keep-the-book rule; never raise.
- **Top-hit similarity:** log per-item `top_sim` (the most-relevant retrieved hit's score) into `pairs.jsonl`; aggregate `max/mean/share>=0.9` into `results.json` under `retrieval_leak_guard`.

### 5.3 The runner — `scripts/compare_persona_armA.py`

Imports from `compare_persona.py`: `_latency_cost`, `_pctile`, `_print_summary`, `API_PRICE_IN/OUT`, and a `logit_bias`-forwarding `_gen_all` (see 5.4); imports `compare_scorecard` from `persona_rag.eval.compare`. Flow per run:

1. **Load hold-out from the DB** (not `eval.jsonl`, which carries no turn_id): `select PersonaTurnRow where eval_split_for(id) is True`, applying `iter_records`' exact filters (`clean_reply`, `min_reply_chars`, non-empty ctx, optional `since_months`) in deterministic order -> tuples `(turn_id, recipient_id_hash, incoming_context: list[str], your_reply)`. Collect `eval_ids = {turn_id}`.
2. **Align to arm B (item-set reuse):** recover arm B's sampled subset by replaying `_sample(seed)` over the same ordered source and joining `eval.jsonl` `human` text back to the DB row's joined `incoming_context`; **assert byte-identical alignment**. If it can't be proven, downgrade to a corpus-level A-vs-B comparison and drop the within-item-delta language (log which mode was used).
3. **API arm (shipped product):** pin `os.environ["GENERATION_BACKEND"]="openai"` + the lever envs, `get_settings.cache_clear()`, assert `settings.GENERATION_BACKEND == "openai"`. Per item: `retrieved = await retrieve(ctx[-1], exclude_ids=eval_ids, ...)` -> run the **leak guard** (5.2) -> `insights = retrieve_insights(ctx[-1])` wrapped `try/except -> {"semantic":[],"static":{}}` -> `messages = build_messages(persona_name=s.PERSONA_NAME, persona_description=generate_persona_description(fallback=s.PERSONA_DESCRIPTION) if s.INSIGHTS_USE_GENERATED_PERSONA_DESCRIPTION else s.PERSONA_DESCRIPTION, style_anchors=_load_anchors(), user_memory="", retrieved=retrieved, session=ChatMessages(ctx[:-1]), incoming=ctx[-1], insights=insights)`. **Assert the rich `SYSTEM_TEMPLATE` markers are present** in `messages` before generating (guards against a silent thin-prompt degrade). Generate via `_gen_all(..., logit_bias=voice_logit_bias())`.
4. **LoRA arm (shipped thin):** build messages explicitly via `build_thin_messages(incoming=ctx[-1], session=ChatMessages(ctx[:-1]))` (reproduces the joined-context serving shape) — do NOT route through `build_messages` under a flipped global. Generate against `OLLAMA_BASE_URL` with `logit_bias=None`.
5. **Score:** `compare_scorecard(real, gen_api, gen_lora, train_replies=...)` (reused verbatim). Capture latency + cost.
6. **Outputs:** `data/eval/compare/armA/{results.json, pairs.jsonl}` (pairs carry per-item `top_sim`, `lang`, `turn_id`); `results.params` records levers (scalars + resolved `voice_logit_bias()` dict), the `ctx[-1]` query choice + cross-arm asymmetry note, `style_anchors` mtime/n_turns, insights snapshot, alignment mode, and the `retrieval_leak_guard` block. Render with the existing `scripts/plot_comparison.py --name armA`; report arm A next to arm B.

### 5.4 Levers

- Add `logit_bias: dict[int,int] | None = None` to `_gen_all` (or a thin arm-A wrapper) and pass it into `client.chat.completions.create`. (Arm B's `_gen_all` at `compare_persona.py:80` omits it, so today the bias silently no-ops.)
- Resolve the dict via `voice_logit_bias()` (`llm_client.py:94`) **while** `GENERATION_BACKEND=="openai"` (it is hard-gated at `llm_client.py:103`, returns `None` otherwise).
- **Headline pass** = shipped `.env` levers (2/-5, `BEST_OF_N=1`). **`--learned` pass** = force `PAREN=0/EXCLAIM=0` to isolate learned vs steered tics. Log both the scalars and the resolved dict; label each pass `shipped` vs `learned-isolated`.

### 5.5 Fidelity rules (all from the critique, all verified)

- Source `persona_name`, the `persona_description` fallback, and levers from `get_settings()` (the same `.env` the bot uses) — never hardcode.
- `style_anchors.json` (dated 2026-05-21, n_turns=24551) is the artifact the live bot serves, so using it is faithful; just record its mtime/n_turns for provenance (do NOT treat its staleness as a blocker).
- `retrieve_insights` makes its OWN embedding call -> **2 embeds/item** (dense query + insights); embed both with the same `ctx[-1]` string; bill 2 embeds/item in the cost line.
- Citation fix: the BM25 corpus filter lives in `scripts/reindex.py:90-91` (and `ingest/pipeline.py:186-187`), NOT `persona_rag/index/reindex.py`.

## 6. Data flow (per hold-out item, API arm)

```
item (turn_id, recipient_id_hash, ctx=incoming_context[list], gold_reply)
  -> retrieved = retrieve(ctx[-1], exclude_ids=eval_ids, language=lang)   # dense must_not + bm25 drop
  -> LEAK GUARD: id-leg (hard-fail) + exact-text-leg (fail iff same-context else log) + log top_sim
  -> insights = retrieve_insights(ctx[-1])            # try/except -> empty
  -> messages = build_messages(... retrieved, session=ChatMessages(ctx[:-1]), incoming=ctx[-1], insights)
  -> assert SYSTEM_TEMPLATE present in messages
  -> gen_api = _gen_all(messages, logit_bias=voice_logit_bias())
LoRA arm: messages = build_thin_messages(incoming=ctx[-1], session=ChatMessages(ctx[:-1])); gen_lora = _gen_all(..., logit_bias=None)
score: compare_scorecard(real=gold_replies, gen_a=gen_api, gen_b=gen_lora, train_replies=...)
```

## 7. Validation — leak-ON vs leak-OFF (proves the guard)

Run the API arm twice on the same items: **leak-OFF** (exclusion + assertion on, the real run) and **leak-ON** (exclusion disabled, assertion downgraded to a counter). Expect leak-ON to show a sharply better (fake) API `len_wasserstein`/copy rate driven by retrieved-gold transcription, and leak-OFF to remove it. This quantifies the leak magnitude and demonstrates the guard closes it. Report the delta in the findings.

## 8. Tests (`tests/test_eval_armA.py` + extend `tests/test_eval_compare.py`)

- `search_dense` with `exclude_ids` builds a `must_not` `HasIdCondition` and the excluded id never appears (monkeypatch the qdrant client).
- `retrieve_bm25` with `exclude_ids` drops the id before `[:top_k]` and still returns `top_k` results when enough remain (build the `(bm25, ids)` dict-pickle shape from `bm25_store.load`, `bm25_store.py:31`).
- Leak guard: ID-leg raises on a planted gold turn_id; exact-text-leg raises on same-context dup; exact-text-leg only increments the counter (no raise) on a different-context `ок`.
- `_gen_all` forwards `logit_bias` into `create()` (capture kwargs).
- API arm assembles a prompt containing `SYSTEM_TEMPLATE` markers (rich), not `THIN_SYSTEM`, under `GENERATION_BACKEND=openai` + `cache_clear()`.

## 9. Cost & knobs

- **Cost:** ~$0.11 per 300-item pass (`gpt-4o-mini`, rich ~1.6k in + ~200 out); 2 embeds/item negligible. Under $1 even running shipped + `--learned` + leak-validation passes. Hard cap: stay < $1 total.
- **Run-time knobs (flags, with defaults):** `--n` (default 300; full hold-out ~1104 for paper-grade), `--learned` (add the 0/0 isolation pass), `--leak-on` (the validation pass), `--seed`, `--name armA`.

## 10. Caveats (documented, not hidden)

- Both arms replay the same `(incoming=ctx[-1], session=ctx[:-1])`; each prompt builder uses it faithfully (API rich multi-turn; LoRA joins to one user turn). No artificial asymmetry.
- Genuine replay gaps (logged in `results.params`): `user_memory=""`, `session` reconstructed from the item's own context (not the live session store), and insights from time-of-run tables.
- Within-item B->A delta is reported only if byte-identical alignment is asserted; else corpus-level.

## 11. File-touch summary

- **Edit (additive, default-`None`):** `persona_rag/retrieval/__init__.py`, `persona_rag/retrieval/dense.py`, `persona_rag/retrieval/bm25.py`, `persona_rag/index/qdrant_store.py`, `scripts/compare_persona.py` (`_gen_all` gains `logit_bias`).
- **New:** `scripts/compare_persona_armA.py`, `tests/test_eval_armA.py`.
- **Outputs (git-ignored):** `data/eval/compare/armA/`, `reports/armA/`.
- **Makefile:** add `compare-arma` (+ `compare-arma-plot`).
