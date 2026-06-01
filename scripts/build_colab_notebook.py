# ruff: noqa: RUF001
"""Generate notebooks/finetune_persona_colab.ipynb (the morning Colab kit).

Run: uv run python scripts/build_colab_notebook.py
Keeps the notebook reproducible and the cell sources readable as Python here.
"""

from __future__ import annotations

import json
from pathlib import Path


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _src(lines),
    }


def _src(lines: tuple[str, ...]) -> list[str]:
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


CELLS = [
    md(
        "# 🧬 Persona LoRA — fine-tune a small model to text like Bohdan",
        "",
        "Closes the **lexical-voice gap** the RAG prompt can't reach: uk/en/ru code-switch,",
        "opener variety, the `)` smiley tic, lowercase, multi-bubble bursts. Runs on a **free",
        "Colab T4**.",
        "",
        "**Base:** Qwen2.5-3B-Instruct (best Cyrillic tokenizer of the small models).",
        "QLoRA 4-bit, via Unsloth.",
        "",
        "### Before you start",
        "1. `Runtime → Change runtime type → T4 GPU`.",
        "2. Run `uv run python scripts/export_finetune_data.py` locally, then upload",
        "   `data/finetune/train.jsonl` (and `eval.jsonl`) in the **Upload data** cell below.",
        "",
        "End to end: ~20–40 min on a T4.",
    ),
    md("## 1. Install Unsloth"),
    code(
        "%%capture",
        "!pip install -q unsloth",
        "# Pin a known-good trl for train_on_responses_only:",
        '!pip install -q --no-deps "trl<0.12.0" peft accelerate bitsandbytes',
    ),
    md("## 2. Upload your data", "Run, then pick `train.jsonl` (and optionally `eval.jsonl`)."),
    code(
        "from google.colab import files",
        "uploaded = files.upload()  # choose train.jsonl (+ eval.jsonl)",
        "print(list(uploaded.keys()))",
    ),
    md("## 3. Load Qwen2.5-3B-Instruct in 4-bit"),
    code(
        "from unsloth import FastLanguageModel",
        "import torch",
        "",
        "MAX_SEQ_LEN = 2048",
        "model, tokenizer = FastLanguageModel.from_pretrained(",
        '    model_name = "unsloth/Qwen2.5-3B-Instruct",',
        "    max_seq_length = MAX_SEQ_LEN,",
        "    load_in_4bit = True,",
        "    dtype = None,",
        ")",
    ),
    md(
        "## 4. Attach the LoRA adapter",
        "`r=32` is comfortable for ~thousands of pairs; all attn+MLP projections.",
    ),
    code(
        "model = FastLanguageModel.get_peft_model(",
        "    model,",
        "    r = 32,",
        "    lora_alpha = 32,",
        "    lora_dropout = 0.05,",
        "    target_modules = [",
        '        "q_proj", "k_proj", "v_proj", "o_proj",',
        '        "gate_proj", "up_proj", "down_proj",',
        "    ],",
        '    bias = "none",',
        '    use_gradient_checkpointing = "unsloth",',
        "    random_state = 3407,",
        ")",
    ),
    md(
        "## 5. Build the dataset",
        "ShareGPT → Qwen chat template. `train_on_responses_only` makes the loss flow **only",
        "through Bohdan's reply tokens** — the single most important switch for style mimicry.",
    ),
    code(
        "from datasets import load_dataset",
        "from unsloth.chat_templates import standardize_sharegpt, get_chat_template",
        "",
        'tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")',
        "",
        'ds = load_dataset("json", data_files="train.jsonl", split="train")',
        "ds = standardize_sharegpt(ds)",
        "",
        "def _fmt(batch):",
        "    texts = [",
        "        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)",
        '        for c in batch["conversations"]',
        "    ]",
        '    return {"text": texts}',
        "",
        "ds = ds.map(_fmt, batched=True)",
        'print(ds[0]["text"][:400])',
    ),
    md("## 6. Trainer (loss on responses only)"),
    code(
        "from trl import SFTTrainer, SFTConfig",
        "from unsloth.chat_templates import train_on_responses_only",
        "",
        "trainer = SFTTrainer(",
        "    model = model,",
        "    tokenizer = tokenizer,",
        "    train_dataset = ds,",
        '    dataset_text_field = "text",',
        "    max_seq_length = MAX_SEQ_LEN,",
        "    packing = False,",
        "    args = SFTConfig(",
        "        per_device_train_batch_size = 8,",
        "        gradient_accumulation_steps = 2,",
        "        warmup_ratio = 0.05,",
        "        num_train_epochs = 2,          # 2 is enough; more memorises/parrots",
        "        learning_rate = 2e-4,",
        "        logging_steps = 20,",
        '        optim = "adamw_8bit",',
        "        weight_decay = 0.01,",
        '        lr_scheduler_type = "linear",',
        "        seed = 3407,",
        '        output_dir = "outputs",',
        '        report_to = "none",',
        "    ),",
        ")",
        "",
        "trainer = train_on_responses_only(",
        "    trainer,",
        '    instruction_part = "<|im_start|>user\\n",',
        '    response_part = "<|im_start|>assistant\\n",',
        ")",
    ),
    md("## 7. Train"),
    code("trainer_stats = trainer.train()"),
    md(
        "## 8. Sanity-check the voice",
        "Watch for: lowercase, code-switch, the `)` tic, short bursts, and — on the vulnerable",
        "probe — that it engages instead of brushing off.",
    ),
    code(
        "FastLanguageModel.for_inference(model)",
        "",
        "PROBES = [",
        '    "шо там по плану на вечір",',
        '    "сам ти даун шо ти несеш",',
        '    "***REMOVED***, не знаю що робити чесно",',
        "]",
        'SYSTEM = "Ти Богдан. Пиши так, як ти зазвичай пишеш у телеграмі."',
        "",
        "for p in PROBES:",
        "    msgs = [",
        '        {"role": "system", "content": SYSTEM},',
        '        {"role": "user", "content": p},',
        "    ]",
        "    inputs = tokenizer.apply_chat_template(",
        '        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"',
        '    ).to("cuda")',
        "    out = model.generate(input_ids=inputs, max_new_tokens=120, temperature=0.8,",
        "                          do_sample=True, top_p=0.95)",
        '    print("IN :", p)',
        '    print("OUT:", tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))',
        '    print("-" * 60)',
    ),
    md(
        "## 9. Export GGUF + Ollama Modelfile",
        "`q4_k_m` is the cheap, good-enough quant for local serving.",
    ),
    code(
        "# Saves ./model-gguf/ with a *.gguf file. CRITICAL: the Modelfile chat",
        "# template must match training (Qwen) or the tic/code-switch regress.",
        'model.save_pretrained_gguf("model-gguf", tokenizer, quantization_method="q4_k_m")',
        "import os",
        'gguf = [f for f in os.listdir("model-gguf") if f.endswith(".gguf")][0]',
        'print("GGUF:", gguf)',
    ),
    code(
        "modelfile = f'''FROM ./{gguf}",
        'TEMPLATE """{{{{ if .System }}}}<|im_start|>system',
        "{{{{ .System }}}}<|im_end|>",
        "{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user",
        "{{{{ .Prompt }}}}<|im_end|>",
        "{{{{ end }}}}<|im_start|>assistant",
        "{{{{ .Response }}}}<|im_end|>",
        '"""',
        "PARAMETER temperature 0.8",
        "PARAMETER top_p 0.95",
        'PARAMETER stop "<|im_end|>"',
        'SYSTEM """Ти Богдан. Пиши так, як ти зазвичай пишеш у телеграмі."""',
        "'''",
        'open("model-gguf/Modelfile", "w").write(modelfile)',
        "print(modelfile)",
    ),
    code(
        "# Zip + download everything for local Ollama",
        "!cd model-gguf && zip -r ../bohdan-lora-gguf.zip . >/dev/null",
        "from google.colab import files",
        'files.download("bohdan-lora-gguf.zip")',
    ),
    md(
        "## 10. Plug into the bot (local machine)",
        "```bash",
        "unzip bohdan-lora-gguf.zip -d bohdan-gguf && cd bohdan-gguf",
        "ollama create bohdan -f Modelfile",
        "ollama run bohdan 'шо там'        # smoke test",
        "```",
        "Then in the repo `.env`:",
        "```",
        "GENERATION_BACKEND=ollama",
        "OLLAMA_MODEL=bohdan",
        "```",
        "and grade it on the held-out turns (same eval as the API path):",
        "```bash",
        "uv run python scripts/eval_persona.py --n 120 --name lora-v1",
        "```",
        "**Keep the checkpoint that best matches Bohdan's *distribution*** (shape_js,",
        "latin_script_rate, paren_smiley_rate, opener_top_share, style_self_sim) —",
        "**not** the lowest train loss.",
        "If it still drifts generic, run the ORPO stage in `docs/finetune/README.md`.",
    ),
]

NB = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path("notebooks/finetune_persona_colab.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(NB, ensure_ascii=False, indent=1))
print(f"wrote {out} ({len(CELLS)} cells)")
