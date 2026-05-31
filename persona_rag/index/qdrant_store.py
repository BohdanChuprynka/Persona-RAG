from __future__ import annotations

import uuid
from collections.abc import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    HasVectorCondition,
    IsEmptyCondition,
    IsNullCondition,
    MatchValue,
    NestedCondition,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from persona_rag.config import get_settings
from persona_rag.models import PersonaTurn, RetrievedTurn

VECTOR_SIZE = 1536  # text-embedding-3-small

# Qdrant accepts only unsigned int or UUID string as point IDs. Our insight rows
# use a 16-char sha1 hex as the stable SQLite primary key (deterministic across
# runs). This helper bridges the two: deterministic mapping sqlite_id → UUID5.
_QDRANT_ID_NS = uuid.UUID("6b8f1f4c-1c4a-4f3a-9b9e-1d3c2f5a7e09")


def to_qdrant_point_id(sqlite_id: str) -> str:
    """Map an InsightRow.id (16-hex) to a deterministic Qdrant UUID string."""
    return str(uuid.uuid5(_QDRANT_ID_NS, sqlite_id))


_Condition = (
    FieldCondition
    | IsEmptyCondition
    | IsNullCondition
    | HasIdCondition
    | HasVectorCondition
    | NestedCondition
    | Filter
)


def make_client() -> QdrantClient:
    """Build a Qdrant client.

    Only forwards ``QDRANT_API_KEY`` over HTTPS. Sending an API key over an
    insecure ``http://`` connection is meaningless (and triggers a warning),
    so local docker-compose runs ignore any key that's been set.
    """
    s = get_settings()
    api_key = (
        s.QDRANT_API_KEY if (s.QDRANT_API_KEY and s.QDRANT_URL.startswith("https://")) else None
    )
    return QdrantClient(url=s.QDRANT_URL, api_key=api_key)


def ensure_collection(client: QdrantClient, name: str, *, vector_size: int = VECTOR_SIZE) -> None:
    collections = {c.name for c in client.get_collections().collections}
    if name in collections:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    client.create_payload_index(name, field_name="language", field_schema=PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, field_name="eval_split", field_schema=PayloadSchemaType.BOOL)


def upsert_turns(
    client: QdrantClient,
    collection: str,
    items: Iterable[tuple[PersonaTurn, list[float]]],
) -> None:
    points = [
        PointStruct(id=turn.id, vector=vec, payload=turn.model_dump(mode="json"))
        for turn, vec in items
    ]
    if points:
        client.upsert(collection_name=collection, points=points)


def ensure_insights_collection(
    client: QdrantClient, name: str, *, vector_size: int = VECTOR_SIZE
) -> None:
    """Create the self_insights collection (idempotent). Mirrors ensure_collection."""
    collections = {c.name for c in client.get_collections().collections}
    if name in collections:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    client.create_payload_index(name, field_name="category", field_schema=PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, field_name="source", field_schema=PayloadSchemaType.KEYWORD)
    client.create_payload_index(
        name, field_name="review_status", field_schema=PayloadSchemaType.KEYWORD
    )


def search_dense(
    client: QdrantClient,
    collection: str,
    vector: list[float],
    *,
    top_k: int,
    language: str | None = None,
    exclude_eval: bool = True,
) -> list[RetrievedTurn]:
    conditions: list[_Condition] = []
    if exclude_eval:
        conditions.append(FieldCondition(key="eval_split", match=MatchValue(value=False)))
    if language:
        conditions.append(FieldCondition(key="language", match=MatchValue(value=language)))
    flt = Filter(must=conditions) if conditions else None
    response = client.query_points(
        collection_name=collection,
        query=vector,
        limit=top_k,
        query_filter=flt,
        with_payload=True,
        with_vectors=True,
    )
    out: list[RetrievedTurn] = []
    for h in response.points:
        turn = PersonaTurn.model_validate(h.payload)
        # Qdrant returns h.vector either as list[float] or dict (named vectors).
        # We use a single unnamed vector, so h.vector is list[float] or None.
        vec = getattr(h, "vector", None)
        embedding = vec if isinstance(vec, list) else None
        out.append(
            RetrievedTurn(
                turn=turn,
                score=float(h.score),
                score_dense=float(h.score),
                embedding=embedding,
            )
        )
    return out
