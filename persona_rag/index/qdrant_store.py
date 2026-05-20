from __future__ import annotations

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
    s = get_settings()
    return QdrantClient(url=s.QDRANT_URL, api_key=s.QDRANT_API_KEY)


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
    )
    out: list[RetrievedTurn] = []
    for h in response.points:
        turn = PersonaTurn.model_validate(h.payload)
        out.append(RetrievedTurn(turn=turn, score=float(h.score), score_dense=float(h.score)))
    return out
