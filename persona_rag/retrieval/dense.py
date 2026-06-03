from __future__ import annotations

from qdrant_client import QdrantClient

from persona_rag.config import get_settings
from persona_rag.index.embedder import embed_batch
from persona_rag.index.qdrant_store import search_dense
from persona_rag.models import RetrievedTurn


async def retrieve_dense(
    client: QdrantClient,
    query: str,
    *,
    top_k: int,
    language: str | None = None,
    exclude_ids: set[str] | None = None,
) -> list[RetrievedTurn]:
    vec = (await embed_batch([query]))[0]
    return search_dense(
        client,
        get_settings().QDRANT_COLLECTION,
        vec,
        top_k=top_k,
        language=language,
        exclude_ids=exclude_ids,
    )
