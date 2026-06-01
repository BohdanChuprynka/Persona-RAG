# ruff: noqa: RUF001, E501
# E501: cell-source string literals are notebook content; wrapping them would
# corrupt the generated cells. RUF001: intentional Cyrillic in the persona anchor.
"""Generate notebooks/finetune_persona_colab.ipynb (the morning Colab kit).

Run: uv run python scripts/build_colab_notebook.py

Minimal, instruction-only notebook. You're needed 3 times (restart, upload,
Drive auth), then training is hands-free (Drive checkpoints + auto-resume).
Architecture rationale lives in docs/finetune/README.md, not in the cells.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Byte-identical to persona_rag.generate.persona.THIN_SYSTEM and the export's
# DEFAULT_SYSTEM. The adapter conditions its whole voice on this string.
THIN_SYSTEM = "Ти Богдан. Пиши так, як ти зазвичай пишеш у телеграмі."


def md(*lines: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines: str) -> dict[str, Any]:
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
        "# Persona LoRA — train a model to text like you",
        "",
        "**You're needed 3 times, then walk away:**",
        "1. **Restart the runtime** after the install cell.",
        "2. **Upload** `train.jsonl` + `eval.jsonl`.",
        "3. **Authorize Google Drive**.",
        "",
        "Training is hands-free after that — checkpoints to Drive, auto-resumes if the T4 drops.",
        "",
        "First: **Runtime → Change runtime type → T4 GPU**.",
    ),
    md("## 1. Install — then restart the runtime"),
    code(
        "# Lift Colab's version cap + force the upgrade. Restart the runtime after this.",
        "!PIP_CONSTRAINT= pip install -q --upgrade --upgrade-strategy eager unsloth unsloth_zoo transformers trl peft protobuf",
        "import importlib.metadata as _m",
        "for _p in ['unsloth', 'transformers', 'trl', 'peft', 'torch', 'protobuf']:",
        "    try:",
        "        print(f'{_p:13s}', _m.version(_p))",
        "    except Exception as _e:",
        "        print(f'{_p:13s} ??', _e)",
    ),
    md(
        "## Restart now: `Runtime → Restart session`",
        "Then run from the next cell. **Do not** re-run the install cell.",
    ),
    code(
        "from unsloth.chat_templates import (  # noqa: F401",
        "    standardize_sharegpt,",
        "    get_chat_template,",
        "    train_on_responses_only,",
        ")",
        'print("OK — unsloth ready")',
    ),
    md("## 2. Upload `train.jsonl` + `eval.jsonl`"),
    code(
        "from google.colab import files",
        "uploaded = files.upload()",
        "assert 'train.jsonl' in uploaded and 'eval.jsonl' in uploaded, 'upload BOTH files'",
        "print(list(uploaded.keys()))",
    ),
    md("## 3. Load model"),
    code(
        "from unsloth import FastLanguageModel",
        "import torch",
        "",
        "MAX_SEQ_LEN = 1536   # covers the 2000-char context cap + reply; leaner than 2048 on a T4",
        "model, tokenizer = FastLanguageModel.from_pretrained(",
        '    model_name = "unsloth/Qwen2.5-3B-Instruct",',
        "    max_seq_length = MAX_SEQ_LEN,",
        "    load_in_4bit = True,",
        "    dtype = None,",
        ")",
    ),
    md("## 4. LoRA adapter"),
    code(
        "model = FastLanguageModel.get_peft_model(",
        "    model,",
        "    r = 32,",
        "    lora_alpha = 64,          # 2*r: override the base register",
        "    lora_dropout = 0,         # keeps Unsloth's fused kernel",
        "    target_modules = [",
        '        "q_proj", "k_proj", "v_proj", "o_proj",',
        '        "gate_proj", "up_proj", "down_proj",',
        "    ],",
        '    bias = "none",',
        '    use_gradient_checkpointing = "unsloth",',
        "    random_state = 3407,",
        ")",
    ),
    md("## 5. Build dataset"),
    code(
        "from datasets import load_dataset",
        "from unsloth.chat_templates import standardize_sharegpt, get_chat_template",
        "",
        'tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")',
        'ds = load_dataset("json", data_files="train.jsonl", split="train")',
        "ds = standardize_sharegpt(ds)",
        "",
        "def _fmt(batch):",
        '    return {"text": [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in batch["conversations"]]}',
        "",
        "ds = ds.map(_fmt, batched=True)",
        'print("examples:", len(ds))',
    ),
    md("## 6. Authorize Google Drive (checkpoints live here)"),
    code(
        "from google.colab import drive",
        "import os",
        "drive.mount('/content/drive')",
        "OUTPUT_DIR = '/content/drive/MyDrive/persona-lora/outputs'",
        "os.makedirs(OUTPUT_DIR, exist_ok=True)",
        "print('checkpoints ->', OUTPUT_DIR)",
    ),
    md("## 7. Trainer"),
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
        "    packing = False,            # required for train_on_responses_only masking",
        "    args = SFTConfig(",
        "        per_device_train_batch_size = 4,",
        "        gradient_accumulation_steps = 4,",
        "        warmup_ratio = 0.05,",
        "        num_train_epochs = 1,              # bump to 2 if VOICE_DISTANCE wants more (resume is cheap)",
        "        learning_rate = 2e-4,",
        "        logging_steps = 20,",
        '        optim = "adamw_8bit",',
        "        weight_decay = 0.01,",
        '        lr_scheduler_type = "linear",',
        "        seed = 3407,",
        '        save_strategy = "steps",',
        "        save_steps = 200,                  # ~25 min between saves; lower = safer",
        "        save_total_limit = 3,",
        "        output_dir = OUTPUT_DIR,           # Drive — survives a disconnect",
        '        report_to = "none",',
        "        padding_free = False,              # new TRL defaults True; breaks with packing=False",
        "    ),",
        ")",
        "trainer = train_on_responses_only(",
        "    trainer,",
        '    instruction_part = "<|im_start|>user\\n",',
        '    response_part = "<|im_start|>assistant\\n",',
        ")",
    ),
    code(
        "# mask check: should print ~your reply only (context masked out)",
        "ex = trainer.train_dataset[0]",
        'if "labels" in ex:',
        '    kept = [t for t in ex["labels"] if t != -100]',
        "else:",
        '    kept = [t for t in trainer.data_collator([ex])["labels"][0].tolist() if t != -100]',
        "print(tokenizer.decode(kept))",
    ),
    md(
        "## 8. Train — walk away now",
        "Resumes from the last Drive checkpoint if the T4 drops (reconnect → Run all → it continues).",
    ),
    code(
        "from transformers.trainer_utils import get_last_checkpoint",
        "last = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None",
        "print('resuming from:', last or '(fresh start)')",
        "trainer_stats = trainer.train(resume_from_checkpoint=last)",
    ),
    code(
        "import gc",
        "gc.collect(); torch.cuda.empty_cache()",
    ),
    md(
        "## 9. Voice eval",
        "`VOICE_DISTANCE` lower = more like you. Keep the run with the lowest one, not the lowest loss.",
    ),
    code(
        "import re, math, bisect, json as _json",
        "from collections import Counter",
        "",
        "_WORD = re.compile(r'[^\\W\\d_]+', re.UNICODE)",
        "_LATIN = re.compile(r'[a-z]')",
        "_CYR = re.compile(r'[Ѐ-ӿ]')",
        "_PAREN_SMILEY = re.compile(r'\\)\\)+|[^\\s(]\\)(?!\\w)')",
        "",
        "def split_bubbles(t):",
        "    if not t: return []",
        "    n = t.replace('\\\\n','\\n').replace('\\r\\n','\\n')",
        "    return [c.strip() for c in n.split('\\n') if c.strip()]",
        "",
        "def _toks(t): return _WORD.findall(t.lower())",
        "",
        "def latin_rate(texts):",
        "    toks = [w for x in texts for w in _toks(x)]",
        "    lat = sum(1 for w in toks if _LATIN.search(w))",
        "    cyr = sum(1 for w in toks if _CYR.search(w))",
        "    return lat/(lat+cyr) if (lat+cyr) else 0.0",
        "",
        "def opener_top(texts):",
        "    op = []",
        "    for x in texts:",
        "        b = split_bubbles(x)",
        "        if b and _toks(b[0]): op.append(_toks(b[0])[0])",
        "    return Counter(op).most_common(1)[0][1]/len(op) if op else 0.0",
        "",
        "def paren_rate(texts):",
        "    bs = [b for t in texts for b in split_bubbles(t)]",
        "    return sum(1 for b in bs if _PAREN_SMILEY.search(b))/len(bs) if bs else 0.0",
        "",
        "def caps_first_rate(texts):",
        "    n=c=0",
        "    for t in texts:",
        "        b = split_bubbles(t)",
        "        if not b: continue",
        "        m = re.search(r'[^\\W\\d_]', b[0])",
        "        if not m: continue",
        "        n += 1; c += 1 if m.group().isupper() else 0",
        "    return c/n if n else 0.0",
        "",
        "def shape_hist(texts, mx=6):",
        "    cnt={}; n=0",
        "    for t in texts:",
        "        c=len(split_bubbles(t))",
        "        if c==0: continue",
        "        b=min(c,mx); cnt[b]=cnt.get(b,0)+1; n+=1",
        "    return {b: cnt.get(b,0)/n for b in range(1,mx+1)} if n else {}",
        "",
        "def js_div(p,q):",
        "    keys=set(p)|set(q); m={k:(p.get(k,0)+q.get(k,0))/2 for k in keys}",
        "    def kl(x): return sum(x[k]*math.log2(x[k]/m[k]) for k in keys if x.get(k,0)>0 and m.get(k,0)>0)",
        "    return 0.5*kl(p)+0.5*kl(q)",
        "",
        "def bubble_lens(texts): return [len(b) for t in texts for b in split_bubbles(t)]",
        "",
        "def wass(a,b):",
        "    if not a or not b: return 0.0",
        "    sa,sb=sorted(a),sorted(b); pts=sorted(set(sa)|set(sb)); tot=0.0",
        "    for i in range(len(pts)-1):",
        "        ca=bisect.bisect_right(sa,pts[i])/len(sa); cb=bisect.bisect_right(sb,pts[i])/len(sb)",
        "        tot+=abs(ca-cb)*(pts[i+1]-pts[i])",
        "    return tot",
        "",
        "def summarize(texts):",
        "    return dict(latin=latin_rate(texts), paren=paren_rate(texts),",
        "                opener_top=opener_top(texts), caps_first_rate=caps_first_rate(texts),",
        "                hist=shape_hist(texts), lens=bubble_lens(texts))",
        "",
        "eval_records = [_json.loads(l) for l in open('eval.jsonl', encoding='utf-8')]",
        "def _human(rec): return next(t['value'] for t in rec['conversations'] if t['from']=='human')",
        "def _gpt(rec):   return next(t['value'] for t in rec['conversations'] if t['from']=='gpt')",
    ),
    code(
        "import random",
        "FastLanguageModel.for_inference(model)",
        f'SYSTEM = "{THIN_SYSTEM}"',
        "",
        "def gen_thin(ctx, max_new_tokens=128):",
        '    msgs = [{"role":"system","content":SYSTEM}, {"role":"user","content":ctx}]',
        '    ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")',
        "    out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=True,",
        "                         temperature=0.8, top_p=0.95, pad_token_id=tokenizer.eos_token_id)",
        "    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()",
        "",
        "def voice_eval(n=64, seed=0):",
        "    rng = random.Random(seed)",
        "    sample = rng.sample(eval_records, min(n, len(eval_records)))",
        "    gen = [gen_thin(_human(r)) for r in sample]; real = [_gpt(r) for r in sample]",
        "    g = summarize(gen); rf = summarize(real)",
        "    shape_js = js_div(rf['hist'], g['hist']); len_w = wass(rf['lens'], g['lens'])",
        "    dist = shape_js + abs(g['latin']-rf['latin']) + abs(g['paren']-rf['paren']) + abs(g['opener_top']-rf['opener_top']) + len_w/50.0",
        "    print(f'  n={len(sample)}  VOICE_DISTANCE={dist:.3f}  (lower = more like you)')",
        "    print(f'  shape_js {shape_js:.3f}   len_wass {len_w:.1f}')",
        "    for k in ('latin','paren','opener_top','caps_first_rate'):",
        "        print(f'  {k:15s} gen={g[k]:.3f}  real={rf[k]:.3f}')",
        "    return dist, gen, real",
        "",
        "dist, gen, real = voice_eval(n=64)",
        "print('\\nsamples:')",
        "for r_, gg in list(zip(real, gen))[:8]:",
        "    print(' real:', repr(r_[:60]), '| gen:', repr(gg[:60]))",
    ),
    md("## 10. Probes"),
    code(
        "for p in [",
        '    "шо там по плану на вечір",',
        '    "сам ти даун шо ти несеш",',
        '    "***REMOVED***, не знаю що робити чесно",',
        '    "yo you coming tonight or nah",',
        "]:",
        '    print("IN :", p)',
        '    print("OUT:", gen_thin(p))',
        '    print("-" * 50)',
    ),
    md("## 11. Export — adapter first, then GGUF"),
    code(
        "# (a) save + download the adapter first (insurance against a GGUF failure)",
        'model.save_pretrained("lora-adapter"); tokenizer.save_pretrained("lora-adapter")',
        "!cd lora-adapter && zip -r ../bohdan-lora-adapter.zip . >/dev/null",
        "from google.colab import files",
        'files.download("bohdan-lora-adapter.zip")',
    ),
    code(
        "# (b) GGUF for Ollama. q5_k_m = good voice + small (~2.4GB). q8_0 = max fidelity.",
        "QUANT = 'q5_k_m'; gguf = None",
        "try:",
        '    model.save_pretrained_gguf("model-gguf", tokenizer, quantization_method=QUANT)',
        "    import os; gguf = [f for f in os.listdir('model-gguf') if f.endswith('.gguf')][0]",
        "    print('GGUF:', gguf)",
        "except Exception as e:",
        "    print('GGUF failed:', repr(e)[:200], '\\nFalling back to merged-16bit')",
        '    model.save_pretrained_merged("model-merged-16bit", tokenizer, save_method="merged_16bit")',
        "    !cd model-merged-16bit && zip -r ../bohdan-merged-16bit.zip . >/dev/null",
        '    files.download("bohdan-merged-16bit.zip")',
    ),
    code(
        "assert gguf, 'no GGUF — use the merged-16bit zip + llama.cpp locally'",
        "modelfile = f'''FROM ./{gguf}",
        'TEMPLATE """{{{{ if .System }}}}<|im_start|>system',
        "{{{{ .System }}}}<|im_end|>",
        "{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user",
        "{{{{ .Prompt }}}}<|im_end|>",
        "{{{{ end }}}}<|im_start|>assistant",
        "{{{{ .Response }}}}<|im_end|>",
        '"""',
        "PARAMETER num_ctx 4096",
        "PARAMETER num_predict 256",
        "PARAMETER temperature 0.8",
        "PARAMETER top_p 0.95",
        "PARAMETER repeat_penalty 1.1",
        'PARAMETER stop "<|im_end|>"',
        f'SYSTEM """{THIN_SYSTEM}"""',
        "'''",
        'open("model-gguf/Modelfile", "w").write(modelfile)',
        "print(modelfile)",
    ),
    code(
        "if gguf:",
        "    !cd model-gguf && zip -r ../bohdan-lora-gguf.zip . >/dev/null",
        '    files.download("bohdan-lora-gguf.zip")',
    ),
    md(
        "## 12. Use it locally (Mac)",
        "```bash",
        "unzip bohdan-lora-gguf.zip -d bohdan-gguf && cd bohdan-gguf",
        "ollama create bohdan -f Modelfile",
        "ollama run bohdan 'шо там'",
        "```",
        "Then in the repo `.env`: `GENERATION_BACKEND=ollama`, `OLLAMA_MODEL=bohdan`.",
    ),
]

NB_PATH = Path("notebooks/finetune_persona_colab.ipynb")


def build_notebook() -> dict[str, Any]:
    """The notebook as a dict — the single source of truth. A test asserts the
    committed .ipynb equals this, so the generated artifact can never silently
    drift from the generator again."""
    return {
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


def main() -> None:
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1))
    print(f"wrote {NB_PATH} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
