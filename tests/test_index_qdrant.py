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


def test_search_dense_populates_embedding_when_vectors_returned(monkeypatch):
    """search_dense must request and pass through dense vectors so MMR has them."""
    from unittest.mock import MagicMock

    from persona_rag.index.qdrant_store import search_dense

    fake_payload = {
        "id": "t1",
        "your_reply": "x",
        "incoming_context": ["y"],
        "channel": "telegram",
        "chat_id_hash": "c1",
        "recipient_id_hash": "r1",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "language": "uk",
        "your_reply_len_chars": 1,
        "your_reply_emoji_count": 0,
        "eval_split": False,
    }

    fake_point = MagicMock()
    fake_point.payload = fake_payload
    fake_point.score = 0.9
    fake_point.vector = [0.1, 0.2, 0.3]

    fake_response = MagicMock()
    fake_response.points = [fake_point]

    captured = {}

    def fake_query_points(**kwargs):
        captured.update(kwargs)
        return fake_response

    client = MagicMock()
    client.query_points = fake_query_points

    out = search_dense(client, "coll", [0.0, 0.0, 0.0], top_k=1)
    assert captured.get("with_vectors") is True
    assert len(out) == 1
    assert out[0].embedding == [0.1, 0.2, 0.3]


def test_search_dense_embedding_none_when_vector_missing():
    """Older Qdrant responses without .vector must not crash; embedding stays None."""
    from unittest.mock import MagicMock

    from persona_rag.index.qdrant_store import search_dense

    fake_payload = {
        "id": "t1",
        "your_reply": "x",
        "incoming_context": ["y"],
        "channel": "telegram",
        "chat_id_hash": "c1",
        "recipient_id_hash": "r1",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "language": "uk",
        "your_reply_len_chars": 1,
        "your_reply_emoji_count": 0,
        "eval_split": False,
    }
    fake_point = MagicMock(spec=["payload", "score"])  # spec restricts attributes — no .vector
    fake_point.payload = fake_payload
    fake_point.score = 0.5

    fake_response = MagicMock()
    fake_response.points = [fake_point]

    client = MagicMock()
    client.query_points = lambda **kw: fake_response

    out = search_dense(client, "coll", [0.0], top_k=1)
    assert out[0].embedding is None
