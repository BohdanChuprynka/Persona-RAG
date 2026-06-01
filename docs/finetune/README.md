# Persona LoRA — fine-tune kit (free Google Colab)

The RAG prompt nails **shape** (when to send 1 vs 3 bubbles) but cannot reach
Bohdan's **lexical voice** — uk/en/ru code-switch, opener variety, the `)` smiley
tic, lowercase. The overnight eval proved it: generated `latin_script_rate`
**0.009** vs real **0.468**, `opener_top_share` **0.49** vs real **0.06**. Every
persona-cloning project surveyed got voice fidelity only from **fine-tuning on
turn pairs**, never from prompt rules. This kit does that on a free T4.

> You don't pay for this. It runs on Colab's free GPU. ~20–40 min end to end.

---

## TL;DR — morning checklist

```bash
# 1. (local) export your real pairs to ShareGPT JSONL
uv run python scripts/export_finetune_data.py            # -> data/finetune/{train,eval}.jsonl

# 2. open the notebook in Colab, set Runtime -> T4 GPU, run all cells,
#    upload train.jsonl when asked. It trains, sanity-checks, and emits a GGUF.
#    notebooks/finetune_persona_colab.ipynb

# 3. (local) install the model into Ollama
unzip bohdan-lora-gguf.zip -d bohdan-gguf && cd bohdan-gguf
ollama create bohdan -f Modelfile

# 4. point the bot at it (.env)
GENERATION_BACKEND=ollama
OLLAMA_MODEL=bohdan

# 5. grade it on the SAME held-out turns as the API path
uv run python scripts/eval_persona.py --n 120 --name lora-v1
```

Keep the run if `latin_script_rate`, `paren_smiley_rate`, `opener_top_share`,
and `style_self_sim` move toward the real numbers in the scorecard.

---

## Why these choices (the load-bearing parts)

- **Base = Qwen2.5-3B-Instruct.** Best Cyrillic tokenizer coverage of the small
  models (decisive for uk/ru/en code-switch). Fits a T4 in 4-bit with room to
  train. Llama-3.2-3B / Gemma-2-2B are fine fallbacks but weaker on Ukrainian.
- **QLoRA, r=32, 2 epochs.** ~20.8k pairs sits in the comfortable 1k–10k+ band.
  Two epochs is the sweet spot — more memorises and starts parroting training
  replies verbatim. Select the checkpoint by **distributional eval, not loss.**
- **`train_on_responses_only`.** Loss flows only through Bohdan's reply tokens,
  not the incoming context. This is the single most important switch for style
  mimicry — without it the model spends capacity modelling other people's
  messages.
- **Reply newlines preserved.** `export_finetune_data.py` keeps the multi-bubble
  bursts as newlines in the assistant turn (49% of replies are multi-bubble).
  The serving-side bubble-splitter re-splits on the same newlines, so the model
  learns to *emit* the burst shape, and delivery reproduces it.
- **No `Name:` transcript prefix.** We want raw voice, not a narrated transcript.
- **Minimal system turn.** `Ти Богдан. Пиши так, як ти зазвичай пишеш у телеграмі.`
  anchors register without drowning the style signal. Drop it with
  `--no-system` if you want to A/B.

## How it plugs in

The generate node (`persona_rag/graph/nodes/openai_chat.py`) calls
`chat_complete`, which routes through `GENERATION_BACKEND`. Ollama exposes an
OpenAI-compatible API, so flipping `GENERATION_BACKEND=ollama` swaps the
generator with **zero other changes** — retrieval, the insights layer, register
detection, shape-conditioning, and bubble-splitting all stay exactly as they are.
The fine-tune supplies *voice*; the RAG pipeline still supplies *facts + shape*.
This "fine-tune-for-voice + RAG-for-facts" split is the consensus architecture
from the research.

## If it still drifts generic (ORPO escalation)

If straight SFT plateaus on `style_self_sim`, add one preference stage. Build
pairs where `chosen` = Bohdan's real reply and `rejected` = the same reply
rewritten by gpt-4o-mini as a "neutral polite assistant: proper caps, full
punctuation, single language, no emoticons." Then ORPO over the same adapter.
This explicitly teaches the gap between his voice and the model's default
politeness. Reuses the same distributional eval to validate. (Costs a little
OpenAI spend to generate the rejects — do it only if SFT isn't enough.)

## Reference dataset

`data/finetune/train.jsonl` is the held-OUT-excluded training split;
`data/finetune/eval.jsonl` is the **same** `eval_split` rows the API eval uses,
so a LoRA run is directly comparable to the gpt-4o-mini baseline in
`data/eval/baseline-consolidated/`.
