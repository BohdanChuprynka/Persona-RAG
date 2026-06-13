"""Capture REAL model outputs for the side-by-side demo (demo/index.html).

The demo must never show invented replies. This runs a handful of synthetic,
privacy-safe prompts through BOTH backends under the *identical thin controlled
prompt* (``THIN_SYSTEM``) — exactly the setup of the report's Arm B — and prints
a ready-to-paste ``DATA_REAL`` block. So the only thing the viewer sees is what
each model genuinely said; the prompts are authored here (no private messages).

  • LoRA  → the local OpenAI-compatible endpoint (OLLAMA_BASE_URL / OLLAMA_MODEL).
            Point OLLAMA_BASE_URL at llama-server locally, or at a tunnelled Colab.
  • API   → gpt-4o-mini (OPENAI_API_KEY / OPENAI_CHAT_MODEL).

Decode params match ``make compare`` (temperature 0.8, max_tokens 200). The LoRA
reply is split into its natural Telegram bubbles (canonical ``split_bubbles``);
the API reply is shown as one block so the burst-vs-wall contrast is visible.
Glosses are left blank on purpose — hand them back to fill accurately.

    uv run python scripts/capture_demo_pairs.py            # all candidates
    uv run python scripts/capture_demo_pairs.py --temp 0.7 # tweak decode

Output (also written to data/eval/demo/data_real.js, gitignored): paste the
entries you like into ``DATA_REAL`` in demo/index.html, then set USE_REAL = true.
Pick the prompts whose REAL replies land best — curating prompts is fair; editing
the replies is not.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from persona_rag._logging import configure_logging, get_logger
from persona_rag.config import get_settings
from persona_rag.generate.bubbles import split_bubbles
from persona_rag.generate.llm_client import _ollama_client, _openai_client
from persona_rag.generate.persona import THIN_SYSTEM

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from compare_persona import _gen_all

log = get_logger()
OUT_PATH = Path("data/eval/demo/data_real.js")

# Authored, privacy-safe prompts — a mix of Ukrainian (shows the real Cyrillic
# voice + the ")" tic) and English (shows the model mirroring the user's
# language). Generate all, keep the 5-6 whose replies land best; gym / guitar /
# coffee hit real interests and draw the longer "gets going on a passion" bursts.
CANDIDATES = [
    # Ukrainian — the real voice (Cyrillic, terse bursts, the paren tic)
    "як справи?",
    "шо робиш?",
    "ходив сьогодні в зал?",
    "як тобі новий айфон?",
    "шо плануєш на вихідні?",
    "порадь хорошу каву",
    # English — shows the model mirroring the user's language (+ the run-1 gems)
    "you still playing guitar?",
    "how are you even awake right now lol",
]


def _js(x: str) -> str:
    """A JS/JSON string literal with non-ASCII (Cyrillic) preserved."""
    return json.dumps(x, ensure_ascii=False)


def _entry(prompt: str, api_text: str, lora_text: str) -> str:
    api_block = " ".join(split_bubbles(api_text)).strip() or api_text.strip()
    lora_bubbles = split_bubbles(lora_text) or ([lora_text.strip()] if lora_text.strip() else [])
    lora_block = ", ".join(f'{{t:{_js(b)},g:""}}' for b in lora_bubbles) or '{t:"",g:""}'
    return (
        f"  {{ prompt:{_js(prompt)},\n"
        f"    api:[{{t:{_js(api_block)}}}],\n"
        f"    lora:[{lora_block}] }},"
    )


async def run(*, temperature: float, max_tokens: int) -> None:
    s = get_settings()
    if not s.OPENAI_API_KEY:
        log.error("no_openai_key", hint="set OPENAI_API_KEY in .env")
        return
    api_client = _openai_client(s.OPENAI_API_KEY)
    lora_client = _ollama_client(s.OLLAMA_BASE_URL)
    messages = [
        [{"role": "system", "content": THIN_SYSTEM}, {"role": "user", "content": p}]
        for p in CANDIDATES
    ]

    log.info("generating_api", model=s.OPENAI_CHAT_MODEL, n=len(messages))
    api_res = await _gen_all(
        api_client,
        s.OPENAI_CHAT_MODEL,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        concurrency=4,
    )
    log.info("generating_lora", model=s.OLLAMA_MODEL, base_url=s.OLLAMA_BASE_URL, n=len(messages))
    lora_res = await _gen_all(
        lora_client,
        s.OLLAMA_MODEL,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        concurrency=1,  # local serve: serial
    )

    errs = [r["err"] for r in (*api_res, *lora_res) if r["err"]]
    if errs:
        log.warning("generation_errors", n=len(errs), first=str(errs[0])[:160])

    entries = [
        _entry(CANDIDATES[i], api_res[i]["text"], lora_res[i]["text"])
        for i in range(len(CANDIDATES))
    ]
    block = "const DATA_REAL = [\n" + "\n".join(entries) + "\n];"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(block + "\n", encoding="utf-8")

    # eyeball table on stderr: prompt, #lora bubbles, the raw lora reply
    print("\n=== captured (review, keep the best 5-6) ===", file=sys.stderr)
    for i, p in enumerate(CANDIDATES):
        bubbles = split_bubbles(lora_res[i]["text"])
        preview = " | ".join(bubbles) if bubbles else "(empty)"
        print(f"  [{len(bubbles)}b] {p!r}\n        lora: {preview}", file=sys.stderr)
    print(
        f"\nwrote {OUT_PATH} — paste the keepers into DATA_REAL, hand back for glosses.\n",
        file=sys.stderr,
    )
    print(block)


def main() -> None:
    configure_logging()
    p = argparse.ArgumentParser(description="Capture real demo pairs (Arm-B thin prompt).")
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=200)
    a = p.parse_args()
    asyncio.run(run(temperature=a.temp, max_tokens=a.max_tokens))


if __name__ == "__main__":
    main()
