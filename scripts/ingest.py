from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from persona_rag._logging import configure_logging
from persona_rag.ingest.pipeline import run_ingest


def main() -> None:
    configure_logging()
    p = argparse.ArgumentParser(description="Run the Persona-RAG ingest pipeline.")
    p.add_argument(
        "--tg",
        type=Path,
        default=Path("data/raw/telegram/result.json"),
        help="Path to Telegram export JSON",
    )
    p.add_argument(
        "--ig",
        type=Path,
        default=Path("data/raw/instagram"),
        help="Path to Instagram export root folder",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip embeddings (SQLite + BM25 still run, plus cost estimate)",
    )
    p.add_argument(
        "--estimate-only",
        action="store_true",
        help="Parse + extract turns + print token/cost estimate. No DB write, no embeddings.",
    )
    p.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Cap total raw messages parsed (for safe first runs on huge exports).",
    )
    args = p.parse_args()

    tg = args.tg if args.tg.exists() else None
    ig = args.ig if args.ig.exists() else None
    asyncio.run(
        run_ingest(
            telegram_path=tg,
            ig_root=ig,
            dry_run_embeddings=args.dry_run,
            estimate_only=args.estimate_only,
            max_messages=args.max_messages,
        )
    )


if __name__ == "__main__":
    main()
