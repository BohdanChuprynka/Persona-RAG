"""Ingest durable persona facts from the Obsidian drop-folder (spec 2026-06-03).

Full-rebuild each run: wipes prior source='vault' facts, then re-extracts the
current contents of VAULT_RAW_DIR. Raw note text never leaves this process —
only distilled identity facts are stored. The drop folder is gitignored.
"""

from __future__ import annotations

import asyncio

from persona_rag._logging import configure_logging
from persona_rag.config import get_settings
from persona_rag.index.qdrant_store import ensure_insights_collection, make_client
from persona_rag.insights.vault import rebuild_vault


async def _main() -> int:
    s = get_settings()
    client = make_client()
    ensure_insights_collection(client, s.QDRANT_INSIGHTS_COLLECTION)
    n = await rebuild_vault(
        directory=s.VAULT_RAW_DIR,
        qdrant_client=client,
        collection=s.QDRANT_INSIGHTS_COLLECTION,
        threshold=s.VAULT_CONFIDENCE_THRESHOLD,
    )
    print(f"vault facts ingested: {n}")
    return 0


def main() -> None:
    configure_logging()
    raise SystemExit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
