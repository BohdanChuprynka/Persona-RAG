# Persona LoRA — fine-tune kit (free Google Colab)

The RAG prompt nails **shape** (when to send 1 vs 3 bubbles) but cannot reach
Bohdan's **lexical voice** — uk/en/ru code-switch, opener variety, the `)` smiley
tic, his real casing. Every persona-cloning project surveyed got voice fidelity
only from **fine-tuning on turn pairs**, never from prompt rules. This kit does
that on a free T4.

> You don't pay for this. It runs on Colab's free GPU. ~20–40 min end to end.

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
`persona_rag/generate/prompt.build_messages` auto-switches to
`build_thin_messages` — persona anchor + the joined context, nothing else. It
does **not** send the 1600-token English RAG template, the retrieved few-shot
turns, or the register/shape directives. Feeding a 3B LoRA that instruction wall
drags it back to its generic assistant register and undoes the fine-tune — that
single skew was the dominant finding of the audit (it appeared independently in 5
of 7 review dimensions). The notebook's eval and probes use the same thin shape,
so the LoRA is graded in its native condition.

`THIN_SYSTEM` lives in `persona_rag/generate/persona.py` and is imported by both
the export and the serving path, so train and serve are **byte-identical**.

---

## The honest target (read this before you "fix" the numbers)

The original kit chased `latin_script_rate = 0.468`. **That number was an
artifact.** The DB's `eval_split` was a *temporal tail* (last 10% of turns by
date), and Bohdan's recent months are dominated by one English-speaking contact
(a single recipient = **62% of all Latin tokens** in the whole corpus). So the
held-out slice looked far more English than Bohdan actually is.

The export now uses a **recipient-stratified, seeded** split
(`dataset.eval_split_for`), so train and eval share the same code-switch
register. The honest, reachable target the export prints:

| metric | train | eval |
|---|---|---|
| `latin_script_rate` | ~0.18 | ~0.18 |
| `paren_smiley_rate` | ~0.05 | ~0.05 |
| `opener_top_share` | ~0.06 | ~0.06 |

The real skill is **per-context mirroring** — reply in English to the English
contact, Ukrainian to everyone else — which a context→reply LoRA learns for free.
A model that hits ~0.18 aggregate *and* mirrors the incoming language is correct;
one that emits 0.47 Latin everywhere is **wrong**. `latin_script_rate` is also
noisy at small n and recipient-dependent — treat `shape_js`, per-bubble length,
`opener_top_share`, `paren_smiley_rate` and `caps_first` as the stabler signals.

---

## TL;DR — morning checklist

```bash
# 1. (local) export real pairs → ShareGPT JSONL (cleans <REDACTED>/URL leaks,
#    recipient-stratified split, prints the honest register-matched target)
uv run python scripts/export_finetune_data.py        # -> data/finetune/{train,eval}.jsonl

# 2. open notebooks/finetune_persona_colab.ipynb in Colab, Runtime -> T4 GPU,
#    upload train.jsonl AND eval.jsonl. Run all cells. It trains, then prints an
#    in-notebook VOICE_DISTANCE before you download anything, then emits a GGUF.

# 3. (local) install into Ollama
unzip bohdan-lora-gguf.zip -d bohdan-gguf && cd bohdan-gguf
ollama create bohdan -f Modelfile

# 4. point the bot at it (.env)
GENERATION_BACKEND=ollama
OLLAMA_MODEL=bohdan

# 5. grade it (now graded through the THIN prompt, its native condition)
uv run python scripts/eval_persona.py --n 200 --name lora-v1
```

Keep the run whose `VOICE_DISTANCE` is lowest and whose `latin/paren/opener`
sit closest to the reference — **not** the lowest train loss.

---

## Why these choices (the load-bearing parts)

- **Base = Qwen2.5-3B-Instruct.** Best Cyrillic tokenizer of the small models
  (decisive for uk/ru/en code-switch). Fits a T4 in 4-bit. Llama-3.2-3B /
  Gemma-2-2B are weaker on Ukrainian. (7B is premature — try it only on Colab
  Pro / A100 if the 3B plateaus.)
- **QLoRA, r=32, `alpha=64` (2·r), `dropout=0`, 2 epochs.** alpha=2r pushes the
  update hard enough to *override* the base model's polite/monolingual/Title-Case
  defaults — the entire job. dropout=0 keeps Unsloth's fused fast path (no
  benefit on 20k clean pairs). Two epochs is the sweet spot; more parrots. Select
  by the in-notebook distributional eval, not loss.
- **T4-safe batch: `bs=4 × ga=4`** (effective 16), `max_seq_length=1536`. Half
  the worst-case activation memory of bs=8 with no late-epoch OOM; 1536 still
  covers the 2000-char context cap + reply without truncating the response.
- **`train_on_responses_only`.** Loss flows only through your reply tokens. The
  single most important switch for style mimicry.
- **Reply newlines preserved.** The export keeps multi-bubble bursts as newlines
  in the assistant turn (49% of replies are multi-bubble); the serving-side
  bubble-splitter re-splits on the same newlines.
- **Leak scrub.** The export drops `<REDACTED>` scrubber scars and bare-URL
  replies and strips inline URLs, so the LoRA never learns to emit them.
- **In-notebook voice eval.** You see real distributional numbers (and sample
  generations) *before* downloading a 2 GB GGUF.
- **Export: adapter first, then GGUF with a 16-bit-merge fallback.** The adapter
  saves in seconds and downloads first, so a fragile GGUF convert never costs the
  whole run. Default quant `q5_k_m` (voice is quant-sensitive; q8_0 for max
  fidelity, q4_k_m to save space).
- **Modelfile** sets `num_ctx 4096` (Ollama's 2048 default would truncate),
  `num_predict 256`, `repeat_penalty 1.1`, and `SYSTEM = THIN_SYSTEM` — matching
  training exactly.

## How it plugs in

Flipping `GENERATION_BACKEND=ollama` swaps the generator for the local LoRA via
Ollama's OpenAI-compatible API. Retrieval, the insights layer, register
detection and shape-conditioning are **bypassed for prompt construction** on this
path (the LoRA carries voice + shape itself), but the **post-generation
guardrails and bubble-splitting still run** — PII redaction and burst delivery
are unchanged. `voice_logit_bias` is OpenAI-only (its tiktoken token ids are
wrong for Qwen, and Ollama ignores `logit_bias` anyway; the `)` tic and the
absence of `!` are learned straight from the data). This "fine-tune-for-voice +
RAG-for-facts" split is the consensus architecture from the research; set
`OLLAMA_FACTS_IN_SYSTEM=true` to fold a short facts addendum into the system turn
if you want RAG facts on the LoRA path (mild train/serve trade-off).

## If it still drifts generic (ORPO escalation)

If straight SFT plateaus on voice, add one preference stage. Build pairs where
`chosen` = your real reply and `rejected` = the same reply rewritten by
gpt-4o-mini as a "neutral polite assistant: proper caps, full punctuation, single
language, no emoticons." Then ORPO over the same adapter — this explicitly
teaches the gap between your voice and the model's default politeness. Reuses the
same distributional eval to validate. (Costs a little OpenAI spend to generate
the rejects — do it only if SFT isn't enough.)

## Reference dataset

`data/finetune/train.jsonl` and `eval.jsonl` come from the recipient-stratified
seeded split (`eval_split_for`), so they are register-matched and the eval target
is honest. This split is intentionally **different** from the DB temporal
`eval_split` the API-path baseline used — that temporal split is what produced
the unreachable 0.468 latin target.
