from datetime import UTC, datetime

from persona_rag.ingest.normalize import detect_language, hash_id, normalize_message
from persona_rag.models import RawMessage


def test_hash_deterministic() -> None:
    assert hash_id("user123") == hash_id("user123")
    assert hash_id("user123") != hash_id("user124")
    assert len(hash_id("user123")) == 16


def test_detect_language() -> None:
    assert detect_language("Hello, world. How are you?") == "en"
    # Ukrainian
    assert detect_language("Привіт, як справи?") == "uk"


def test_normalize_message() -> None:
    raw = RawMessage(
        channel="telegram",
        chat_id="c1",
        sender_id="u1",
        sender_name="Alice",
        text="Hello there",
        timestamp=datetime.now(UTC),
        is_group=False,
    )
    n = normalize_message(raw)
    assert n["chat_id_hash"] == hash_id("c1")
    assert n["sender_id_hash"] == hash_id("u1")
    assert n["language"] == "en"
    assert n["text"] == "Hello there"
