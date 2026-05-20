from datetime import UTC, datetime

from persona_rag.generate.prompt import build_messages
from persona_rag.models import PersonaTurn, RetrievedTurn, StyleAnchors


def _r(reply: str, ctx: str) -> RetrievedTurn:
    return RetrievedTurn(
        turn=PersonaTurn(
            id=reply,
            your_reply=reply,
            incoming_context=[ctx],
            channel="telegram",
            chat_id_hash="x",
            recipient_id_hash="y",
            timestamp=datetime.now(UTC),
            language="en",
            your_reply_len_chars=len(reply),
            your_reply_emoji_count=0,
        ),
        score=1.0,
    )


def test_messages_have_cacheable_system_then_alternating_fewshot():
    anchors = StyleAnchors(
        avg_len_chars=20,
        median_len_chars=18,
        emoji_rate_per_char=0.01,
        lang_distribution={"en": 1.0},
        top_bigrams=["ok cool"],
        n_turns=10,
        primary_language="en",
    )
    msgs = build_messages(
        persona_name="Bob",
        persona_description="Test",
        style_anchors=anchors,
        user_memory="They like cats.",
        retrieved=[_r("yes", "do you like cats?")],
        session=[],
        incoming="how about dogs?",
    )
    assert msgs[0]["role"] == "system"
    assert "Bob" in msgs[0]["content"]
    assert "They like cats." in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "do you like cats?"}
    assert msgs[2] == {"role": "assistant", "content": "yes"}
    assert msgs[-1] == {"role": "user", "content": "how about dogs?"}
