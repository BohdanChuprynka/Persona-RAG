# ruff: noqa: RUF001
# Reason: Cyrillic test messages exercise register-aware prompt assembly.
"""build_messages must adapt the per-reply directive to the incoming register.

This is the wiring that fixes the 'emotionless' tone failure: a vulnerable
message must get an ENGAGEMENT directive and lose the brevity cap, while casual
pings keep the short-shape directive and heated messages get a fire-back nudge.
"""

from __future__ import annotations

from datetime import UTC, datetime

from persona_rag.config import get_settings
from persona_rag.generate.prompt import build_messages
from persona_rag.models import PersonaTurn, RetrievedTurn, StyleAnchors

_ANCHORS = StyleAnchors(
    avg_len_chars=20,
    median_len_chars=18,
    emoji_rate_per_char=0.01,
    lang_distribution={"uk": 1.0},
    top_bigrams=["ну да"],
    n_turns=10,
    primary_language="uk",
)

VULN = (
    "дивись останнім часом в мене є така проблема що я знаю що деякі речі "
    "***REMOVED***, що мені робити??"
)


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
            language="uk",
            your_reply_len_chars=len(reply),
            your_reply_emoji_count=0,
        ),
        score=1.0,
    )


def _sys_blob(incoming: str, retrieved: list[RetrievedTurn]) -> str:
    msgs = build_messages(
        persona_name="Bohdan",
        persona_description="t",
        style_anchors=_ANCHORS,
        user_memory="",
        retrieved=retrieved,
        session=[],
        incoming=incoming,
    )
    return "\n".join(m["content"] for m in msgs if m["role"] == "system")


def test_serious_injects_engagement_and_drops_shape_cap() -> None:
    blob = _sys_blob(VULN, [_r("ок", "привіт"), _r("ага", "як ти")])
    assert "opening up" in blob  # engagement directive present
    # the hard brevity cap must NOT be imposed on a vulnerable moment
    assert "send ONE short message" not in blob
    assert "send about" not in blob


def test_casual_keeps_shape_directive_and_no_engagement() -> None:
    blob = _sys_blob("шо там", [_r("норм", "як ти"), _r("та норм", "шо як")])
    assert ("send ONE short message" in blob) or ("send about" in blob)
    assert "opening up" not in blob


def test_heated_gets_fireback_nudge_not_engagement() -> None:
    blob = _sys_blob("сам ти даун", [_r("норм", "як"), _r("ок", "шо")])
    assert "they came at you" in blob  # heated directive
    assert "opening up" not in blob


def test_heated_nudge_survives_shape_hint_off(monkeypatch) -> None:
    # code-review #3: the fire-back nudge must not be coupled to SHAPE_HINT_ENABLED
    monkeypatch.setenv("REGISTER_AWARE_ENABLED", "true")
    monkeypatch.setenv("SHAPE_HINT_ENABLED", "false")
    get_settings.cache_clear()
    try:
        blob = _sys_blob("сам ти даун", [_r("норм", "як")])
        assert "they came at you" in blob
    finally:
        get_settings.cache_clear()


def test_last_message_is_the_incoming_even_when_serious() -> None:
    msgs = build_messages(
        persona_name="Bohdan",
        persona_description="t",
        style_anchors=_ANCHORS,
        user_memory="",
        retrieved=[_r("ок", "привіт")],
        session=[],
        incoming=VULN,
    )
    assert msgs[-1] == {"role": "user", "content": VULN}
