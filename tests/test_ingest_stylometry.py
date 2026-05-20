from datetime import UTC, datetime

from persona_rag.ingest.stylometry import compute_anchors
from persona_rag.models import PersonaTurn


def _t(reply: str, lang: str = "en") -> PersonaTurn:
    return PersonaTurn(
        id="x",
        your_reply=reply,
        incoming_context=[],
        channel="telegram",
        chat_id_hash="a",
        recipient_id_hash="b",
        timestamp=datetime.now(UTC),
        language=lang,
        your_reply_len_chars=len(reply),
        your_reply_emoji_count=0,
    )


def test_compute_anchors_basics():
    turns = [_t("hi"), _t("hello there"), _t("привіт", "uk")]
    a = compute_anchors(turns)
    assert a.n_turns == 3
    assert a.primary_language == "en"
    assert a.lang_distribution["en"] > a.lang_distribution["uk"]
    assert a.avg_len_chars > 0


def test_compute_anchors_empty():
    a = compute_anchors([])
    assert a.n_turns == 0
    assert a.primary_language == "en"
    assert a.top_bigrams == []


def test_compute_anchors_bigrams():
    turns = [_t("hello world foo"), _t("hello world bar")]
    a = compute_anchors(turns)
    assert "hello world" in a.top_bigrams


def test_compute_anchors_emoji_rate():
    turns = [_t("hi")]
    a = compute_anchors(turns)
    assert a.emoji_rate_per_char == 0.0
