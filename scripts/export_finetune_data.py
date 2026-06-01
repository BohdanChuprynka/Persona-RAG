"""Export Bohdan's Telegram turn-pairs to ShareGPT JSONL for the Colab LoRA.

    uv run python scripts/export_finetune_data.py --since-months 12

Writes data/finetune/train.jsonl + eval.jsonl using a recipient-stratified seeded
split (dataset.eval_split_for), so train and eval share the same code-switch
register and the printed target is honest + reachable. --since-months trains on
current-you. Upload train.jsonl (and eval.jsonl) to the Colab notebook.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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
        "--since-months",
        type=int,
        default=None,
        help="keep only turns from the last N months (train on current-you). "
        "Bohdan's code-switch climbs ~0.20 all-time -> ~0.30 over 12mo -> ~0.43 over 3mo, "
        "so all-time sounds dated. Omit for all-time.",
    )
    p.add_argument(
        "--no-system",
        action="store_true",
        help="omit the persona system turn (train on raw pairs only)",
    )
    args = p.parse_args()

    system = None if args.no_system else DEFAULT_SYSTEM
    if args.no_system:
        print(
            "WARNING: --no-system omits the persona anchor. Qwen's chat template then "
            "injects its OWN default 'You are Qwen...' system at train time, which will "
            "NOT match the THIN_SYSTEM the bot serves. Only use this to A/B deliberately."
        )
    out_dir = Path(args.out)

    train = list(
        iter_records(
            eval_split=False,
            system=system,
            min_reply_chars=args.min_reply_chars,
            max_ctx_chars=args.max_ctx_chars,
            since_months=args.since_months,
        )
    )
    held = list(
        iter_records(
            eval_split=True,
            system=system,
            min_reply_chars=args.min_reply_chars,
            max_ctx_chars=args.max_ctx_chars,
            since_months=args.since_months,
        )
    )
    n_train = write_jsonl(out_dir / "train.jsonl", train)
    n_eval = write_jsonl(out_dir / "eval.jsonl", held)
    window = "all-time" if args.since_months is None else f"last {args.since_months} months"
    print(f"train.jsonl: {n_train} pairs")
    print(f"eval.jsonl:  {n_eval} pairs")
    print(f"window:      {window}")
    print(f"system turn: {'(none)' if system is None else system!r}")
    print(f"-> {out_dir}/")
    _print_reference(train, held)


def _print_reference(train: list[dict[str, Any]], held: list[dict[str, Any]]) -> None:
    """Print the register-matched reference: the recipient-stratified split means
    train and eval share the same code-switch register, so these are the LoRA's
    HONEST, reachable targets (not the old temporal-split 0.468 artifact)."""
    from persona_rag.eval.distribution import (
        latin_script_rate,
        opener_top_share,
        paren_smiley_rate,
    )

    def replies(recs: list[dict[str, Any]]) -> list[str]:
        return [t["value"] for r in recs for t in r["conversations"] if t["from"] == "gpt"]

    tr, ev = replies(train), replies(held)
    print("\nregister-matched reference (the LoRA's honest target — train≈eval by design):")
    for name, fn in (
        ("latin_script_rate", latin_script_rate),
        ("paren_smiley_rate", paren_smiley_rate),
        ("opener_top_share", opener_top_share),
    ):
        print(f"  {name:18s} train={fn(tr):.3f}  eval={fn(ev):.3f}")


if __name__ == "__main__":
    main()
