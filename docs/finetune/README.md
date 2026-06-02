# Persona LoRA: fine-tune kit (free Google Colab)

The RAG prompt nails **shape** (when to send 1 vs 3 bubbles) but cannot reach
Bohdan's **lexical voice**: uk/en/ru code-switch, opener variety, the `)` smiley
tic, his real casing. Every persona-cloning project surveyed got voice fidelity
only from **fine-tuning on turn pairs**, never from prompt rules. This kit does
that on a free T4.

> You don't pay for this. It runs on Colab's free GPU. ~20–40 min end to end.

---

## The notebook is generated: never hand-edit it

`scripts/build_colab_notebook.py` is the **single source of truth** for the
Colab kit. `notebooks/finetune_persona_colab.ipynb` is the build artifact:

```bash
uv run python scripts/build_colab_notebook.py   # regenerates the .ipynb
```

Edits go into the generator's `CELLS` list, then you regenerate. A test asserts
the committed `.ipynb` equals `build_notebook()`, so a hand-edited notebook fails
CI. The architecture rationale lives in this doc, not in the notebook cells.

`THIN_SYSTEM` is the literal in `persona_rag/generate/persona.py` (serving),
re-exported by `persona_rag/finetune/dataset.py` as `DEFAULT_SYSTEM = THIN_SYSTEM`
(so the export reuses the same object). The generator
`scripts/build_colab_notebook.py` keeps a byte-identical copy of the literal. So
train and serve share the exact system string.

---

## The one architectural rule: **train == serve**

This is the load-bearing fix from the 2026-06-01 architecture audit. The adapter
is trained on a **thin** prompt:

```
[system: "Ти Богдан. Пиши так, як ти зазвичай пишеш у телеграмі."]
[user:   <the joined incoming context>]
[assistant: <your reply>]        # loss flows ONLY here (train_on_responses_only)
```

So it must be **served** under that exact shape. When `GENERATION_BACKEND=ollama`,
`persona_rag/generate/prompt.build_messages` auto-switches to `build_thin_messages`
(persona anchor plus the joined context, nothing else). It does **not** send the
1600-token English RAG template, the retrieved few-shot turns, or the
register/shape directives. Feeding a 3B LoRA that instruction wall drags it back
to its generic assistant register and undoes the fine-tune. That single skew was
the dominant finding of the audit (it appeared independently in 5 of 7 review
dimensions). The notebook's eval and probes use the same thin shape, so the LoRA
is graded in its native condition.

---

## The honest target (read this before you "fix" the numbers)

The original kit chased `latin_script_rate = 0.468`. **That number was an
artifact.** The DB's `eval_split` was a *temporal tail* (last 10% of turns by
date), and Bohdan's recent months are dominated by one English-speaking contact
(a single recipient = **62% of all Latin tokens** in the whole corpus). So the
held-out slice looked far more English than Bohdan actually is.

The export now uses a **recipient-stratified, seeded** split
(`dataset.eval_split_for`), so train and eval share the same code-switch register.
The honest, reachable target the export prints:

| metric | train | eval |
|---|---|---|
| `latin_script_rate` | ~0.18 | ~0.18 |
| `paren_smiley_rate` | ~0.05 | ~0.05 |
| `opener_top_share` | ~0.06 | ~0.06 |

The real skill is **per-context mirroring** (reply in English to the English
contact, Ukrainian to everyone else), which a context→reply LoRA learns for free.
A model that hits ~0.18 aggregate *and* mirrors the incoming language is correct;
one that emits 0.47 Latin everywhere is **wrong**. `latin_script_rate` is also
noisy at small n and recipient-dependent. Treat `shape_js`, per-bubble length,
`opener_top_share`, `paren_smiley_rate`, and `caps_first` as the stabler signals.

---

## TL;DR: morning checklist

```bash
# 1. (local) export real pairs → ShareGPT JSONL (cleans <REDACTED>/URL leaks,
#    recipient-stratified split, prints the honest register-matched target).
#    --since-months trains on CURRENT-you: code-switch climbs over a shorter
#    window, so all-time sounds dated. 12 is a good default; the CLI default is
#    all-time (no --since-months).
uv run python scripts/export_finetune_data.py --since-months 12   # -> data/finetune/{train,eval}.jsonl

# 2. (optional, only if you touched the generator) rebuild the notebook
uv run python scripts/build_colab_notebook.py

# 3. open notebooks/finetune_persona_colab.ipynb in Colab, Runtime -> T4 GPU,
#    upload train.jsonl AND eval.jsonl. Run all cells. It trains, then prints an
#    in-notebook VOICE_DISTANCE before you download anything, then emits a GGUF.

# 4. (local) install into Ollama (start `ollama serve` in another terminal first)
unzip bohdan-lora-gguf.zip -d bohdan-gguf && cd bohdan-gguf
ollama create bohdan -f Modelfile

# 5. (local) run the SAME Telegram pipeline on the local model, no .env edits.
#    --local forces GENERATION_BACKEND=ollama, folds contact facts into the thin
#    system turn, skips the (unused) few-shot retrieval, and preflights the model.
make run-local            # == python -m persona_rag.bot.main --local

# 6. grade it (now graded through the THIN prompt, its native condition)
uv run python scripts/eval_persona.py --n 200 --name lora-v1
```

Keep the run whose `VOICE_DISTANCE` is lowest and whose `latin/paren/opener` sit
closest to the reference, **not** the lowest train loss.

---

## Why these choices (the load-bearing parts)

- **Base = Qwen2.5-3B-Instruct.** Best Cyrillic tokenizer of the small models
  (decisive for uk/ru/en code-switch). Fits a T4 in 4-bit. Llama-3.2-3B /
  Gemma-2-2B are weaker on Ukrainian. (7B is premature: try it only on Colab
  Pro / A100 if the 3B plateaus.)
- **QLoRA, r=32, `alpha=64` (2·r), `dropout=0`.** alpha=2r pushes the update hard
  enough to *override* the base model's polite/monolingual/Title-Case defaults.
  dropout=0 keeps Unsloth's fused fast path.
- **Epochs.** The generated notebook ships `num_train_epochs=1` with a one-line
  bump to 2 if `VOICE_DISTANCE` wants more (resume from the Drive checkpoint is
  cheap). More than 2 starts to parrot. Select by the in-notebook distributional
  eval, not loss.
- **T4-safe batch: `bs=4 × ga=4`** (effective 16), `max_seq_length=1536`. Half
  the worst-case activation memory of bs=8 with no late-epoch OOM; 1536 still
  covers the 2000-char context cap plus reply without truncating the response.
- **`train_on_responses_only`.** Loss flows only through your reply tokens. The
  single most important switch for style mimicry. The notebook includes a mask
  check cell that prints the reply tokens (context masked to -100).
- **Reply newlines preserved.** The export keeps multi-bubble bursts as newlines
  in the assistant turn; the serving-side bubble-splitter re-splits on the same
  newlines.
- **Leak scrub.** `dataset.clean_reply` drops `<REDACTED>` scrubber scars and
  bare-URL replies and strips inline URLs, so the LoRA never learns to emit them.
- **In-notebook voice eval.** You see real distributional numbers (and sample
  generations) *before* downloading a 2 GB GGUF.
- **Export: adapter first, then GGUF with a 16-bit-merge fallback.** The adapter
  saves in seconds and downloads first, so a fragile GGUF convert never costs the
  whole run. Default quant `q5_k_m` (voice is quant-sensitive; q8_0 for max
  fidelity, q4_k_m to save space).
- **Modelfile** sets `num_ctx 4096` (Ollama's default would truncate),
  `num_predict 256`, `temperature 0.8`, `top_p 0.95`, `repeat_penalty 1.1`, and
  `SYSTEM = THIN_SYSTEM`, matching training exactly.

## How it plugs into serving

`make run-local` (or `python -m persona_rag.bot.main --local`) runs the same
Telegram pipeline against the local LoRA via Ollama's OpenAI-compatible API. The
flag forces `GENERATION_BACKEND=ollama` and sets `OLLAMA_FACTS_IN_SYSTEM=true`,
then **preflights** the Ollama server at startup. If it's down or the model isn't
installed, you get the exact `ollama serve` / `ollama create` fix command instead
of an opaque 500 mid-chat (`persona_rag/generate/ollama_health.py`). It's a
preflight, not a process manager: the bot never spawns or owns `ollama serve`.

On the ollama path the graph **skips the few-shot retrieval node entirely**.
`_route_after_auth` in `persona_rag/graph/compile.py` routes `auth → load_memory`
instead of `auth → retrieve_hybrid`, because the LoRA serves the thin prompt and
never injects retrieved turns. Register detection and shape-conditioning are
likewise **bypassed for prompt construction** (the LoRA carries voice and shape
itself), while the **post-generation guardrails and bubble-splitting still run**:
PII redaction and burst delivery are unchanged. `voice_logit_bias` is OpenAI-only
(its tiktoken token ids are wrong for Qwen, and Ollama ignores `logit_bias`
anyway; the `)` tic and the absence of `!` are learned straight from the data).

This "fine-tune-for-voice plus RAG-for-facts" split is the consensus architecture
from the research. The facts layer (`load_memory` plus `retrieve_insights`) is
kept and folded into the thin system turn, so the bot still knows contact-specific
facts, at the cost of a small OpenAI **embedding** call for insight lookup
(`OPENAI_API_KEY` stays required; set `OLLAMA_FACTS_IN_SYSTEM=false` to drop even
that and run the voice clone alone).

## If it still drifts generic (ORPO escalation)

If straight SFT plateaus on voice, add one preference stage. Build pairs where
`chosen` = your real reply and `rejected` = the same reply rewritten by
gpt-4o-mini as a "neutral polite assistant: proper caps, full punctuation, single
language, no emoticons." Then ORPO over the same adapter, which explicitly teaches
the gap between your voice and the model's default politeness. Reuses the same
distributional eval to validate. (Costs a little OpenAI spend to generate the
rejects, so do it only if SFT isn't enough.)

## Reference dataset

`data/finetune/train.jsonl` and `eval.jsonl` come from the recipient-stratified
seeded split (`eval_split_for`), so they are register-matched and the eval target
is honest. This split is intentionally **different** from the DB temporal
`eval_split` the API-path baseline used. That temporal split is what produced the
unreachable 0.468 latin target.
