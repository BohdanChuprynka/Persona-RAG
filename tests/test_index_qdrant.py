import uuid
from datetime import UTC, datetime

from qdrant_client import QdrantClient

from persona_rag.index.qdrant_store import (
    ensure_collection,
    search_dense,
    to_qdrant_point_id,
    upsert_turns,
)
from persona_rag.models import PersonaTurn


def test_to_qdrant_point_id_is_deterministic_uuid() -> None:
    sqlite_id = "f1dd6b24e325a29a"
    a = to_qdrant_point_id(sqlite_id)
    b = to_qdrant_point_id(sqlite_id)
    assert a == b
    # Must parse as a valid UUID (else Qdrant rejects with 400 Bad Request).
    uuid.UUID(a)
    # Different inputs must produce different IDs.
    assert to_qdrant_point_id("other") != a


def _client() -> QdrantClient:
    return QdrantClient(":memory:")


def _turn(reply: str = "hi") -> PersonaTurn:
    return PersonaTurn(
        id="00000000-0000-0000-0000-000000000001",
        your_reply=reply,
        incoming_context=["q"],
        channel="telegram",
        chat_id_hash="x",
        recipient_id_hash="y",
        timestamp=datetime.now(UTC),
        language="en",
        your_reply_len_chars=2,
        your_reply_emoji_count=0,
        eval_split=False,
    )


def test_ensure_collection_creates_once() -> None:
    c = _client()
    ensure_collection(c, "test_coll", vector_size=4)
    ensure_collection(c, "test_coll", vector_size=4)  # idempotent


def test_upsert_then_search_returns_top() -> None:
    c = _client()
    ensure_collection(c, "test_coll", vector_size=4)
    upsert_turns(c, "test_coll", [(_turn("hi"), [1.0, 0.0, 0.0, 0.0])])
    results = search_dense(c, "test_coll", [1.0, 0.0, 0.0, 0.0], top_k=5)
    assert len(results) == 1
    assert results[0].turn.your_reply == "hi"
