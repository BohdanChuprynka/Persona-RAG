"""Export Bohdan's Telegram turn-pairs to ShareGPT JSONL for the Colab LoRA.

    uv run python scripts/export_finetune_data.py --min-reply-chars 2

Writes data/finetune/train.jsonl + eval.jsonl (held-out = the eval_split rows,
so the fine-tune is graded on the SAME held-out turns as scripts/eval_persona.py).
Upload train.jsonl (and eval.jsonl) to the Colab notebook.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from persona_rag.finetune.dataset import DEFAULT_SYSTEM, iter_records, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/finetune", help="output dir")
    p.add_argument(
        "--min-reply-chars",
        type=int,
        default=2,
        help="drop replies shorter than this (filters bare acks if desired)",
    )
    p.add_argument("--max-ctx-chars", type=int, default=2000)
    p.add_argument(
        "--no-system",
        action="store_true",
        help="omit the persona system turn (train on raw pairs only)",
    )
    args = p.parse_args()

    system = None if args.no_system else DEFAULT_SYSTEM
    out_dir = Path(args.out)

    train = list(
        iter_records(
            eval_split=False,
            system=system,
            min_reply_chars=args.min_reply_chars,
            max_ctx_chars=args.max_ctx_chars,
        )
    )
    held = list(
        iter_records(
            eval_split=True,
            system=system,
            min_reply_chars=args.min_reply_chars,
            max_ctx_chars=args.max_ctx_chars,
        )
    )
    n_train = write_jsonl(out_dir / "train.jsonl", train)
    n_eval = write_jsonl(out_dir / "eval.jsonl", held)
    print(f"train.jsonl: {n_train} pairs")
    print(f"eval.jsonl:  {n_eval} pairs")
    print(f"system turn: {'(none)' if system is None else system!r}")
    print(f"-> {out_dir}/")


if __name__ == "__main__":
    main()
