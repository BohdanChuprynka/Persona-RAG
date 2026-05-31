# ruff: noqa: RUF001
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from persona_rag.db.models import PersonaTurnRow
from persona_rag.insights.extractor import (
    EXTRACT_SYSTEM_PROMPT,
    RawInsight,
    extract_from_session,
    parse_extractor_response,
    render_session,
)
from persona_rag.insights.sessions import build_sessions


def _t(reply: str, ts: datetime, ctx: list[str] | None = None) -> PersonaTurnRow:
    import json as _json

    return PersonaTurnRow(
        id=f"t-{ts.isoformat()}",
        your_reply=reply,
        incoming_context_json=_json.dumps(ctx or []),
        channel="telegram",
        chat_id_hash="c1",
        recipient_id_hash="r1",
        timestamp=ts,
        language="uk",
        your_reply_len_chars=len(reply),
        your_reply_emoji_count=0,
    )


def test_system_prompt_mentions_persona_name():
    rendered = EXTRACT_SYSTEM_PROMPT.format(persona_name="TestPersona")
    assert "TestPersona" in rendered
    assert "bio|opinion|interest|behavior" in rendered or "bio" in rendered


def test_render_uses_me_and_contact_labels():
    """Spec §5.2 — speaker labels are Me: and Contact-<8>: not [friend]/[name]."""
    now = datetime(2025, 1, 1, tzinfo=UTC)
    rows = [
        _t("я кодю", now, ctx=["що робиш?"]),
        _t("так", now, ctx=["і норм?"]),
    ]
    sessions = build_sessions(rows, gap_hours=6)
    rendered = render_session(sessions[0], persona_name="Bohdan")
    assert "Me:" in rendered
    assert "Contact-" in rendered
    assert "[friend]" not in rendered
    assert "[Bohdan]" not in rendered
    assert "що робиш?" in rendered
    assert "я кодю" in rendered


def test_render_distinct_contacts_get_distinct_labels():
    """Two recipient_id_hash values render as two distinct Contact-XXXX labels."""
    import json as _json
    from datetime import timedelta

    now = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    def t(reply: str, ts: datetime, ctx: list[str], rcpt: str) -> PersonaTurnRow:
        return PersonaTurnRow(
            id=f"t-{ts.isoformat()}-{rcpt}",
            your_reply=reply,
            incoming_context_json=_json.dumps(ctx),
            channel="telegram",
            chat_id_hash="group1",
            recipient_id_hash=rcpt,
            timestamp=ts,
            language="uk",
            your_reply_len_chars=len(reply),
            your_reply_emoji_count=0,
        )

    rows = [
        t("hi", now, ["A"], "aaaaaaaa11111111"),
        t("ok", now + timedelta(minutes=1), ["B"], "bbbbbbbb22222222"),
    ]
    sessions = build_sessions(rows, gap_hours=6)
    rendered = render_session(sessions[0], persona_name="Bohdan")
    assert "Contact-aaaaaaaa" in rendered
    assert "Contact-bbbbbbbb" in rendered


def test_parse_extractor_response_happy():
    response = """{
        "insights": [
            {
                "category": "interest",
                "subject": "cyberpunk 2077",
                "text": "Plays Cyberpunk 2077",
                "confidence": 0.85,
                "source_quote": "грав вчора cyberpunk"
            }
        ]
    }"""
    out = parse_extractor_response(response, session_id="s1")
    assert len(out) == 1
    assert isinstance(out[0], RawInsight)
    assert out[0].subject == "cyberpunk 2077"
    assert out[0].session_id == "s1"


def test_parse_extractor_response_empty_list():
    response = '{"insights": []}'
    assert parse_extractor_response(response, session_id="s1") == []


def test_parse_extractor_response_handles_junk():
    with pytest.raises(ValueError):
        parse_extractor_response("not json at all", session_id="s1")


def test_parse_extractor_response_strips_markdown_fence():
    response = '```json\n{"insights": []}\n```'
    assert parse_extractor_response(response, session_id="s1") == []


def test_parse_extractor_response_rejects_unknown_category():
    item = (
        '{"category": "made_up", "subject": "x",'
        ' "text": "y", "confidence": 0.5, "source_quote": "z"}'
    )
    response = f'{{"insights": [{item}]}}'
    out = parse_extractor_response(response, session_id="s1")
    assert out == []


@pytest.mark.asyncio
async def test_extract_from_session_passes_entity_hints():
    now = datetime(2025, 1, 1, tzinfo=UTC)
    rows = [_t("я кодю на python", now, ctx=["що робиш?"])]
    sessions = build_sessions(rows, gap_hours=6)
    canned = '{"insights": []}'
    with patch(
        "persona_rag.insights.extractor.chat_complete",
        AsyncMock(return_value=canned),
    ) as mock_chat:
        out = await extract_from_session(
            sessions[0],
            persona_name="Bohdan",
            entity_hints=["python", "cyberpunk"],
        )
    assert out == []
    # Hints should be visible somewhere in the user message
    messages = mock_chat.call_args.args[0]
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "python" in user_msg["content"]
    assert "cyberpunk" in user_msg["content"]


@pytest.mark.asyncio
async def test_extract_from_session_returns_parsed_insights():
    now = datetime(2025, 1, 1, tzinfo=UTC)
    rows = [_t("я грав cyberpunk", now, ctx=["що грав?"])]
    sessions = build_sessions(rows, gap_hours=6)
    canned = (
        '{"insights": [{"category": "interest", "subject": "cyberpunk 2077",'
        ' "text": "Plays Cyberpunk", "confidence": 0.9, "source_quote": "я грав cyberpunk"}]}'
    )
    with patch(
        "persona_rag.insights.extractor.chat_complete",
        AsyncMock(return_value=canned),
    ):
        out = await extract_from_session(sessions[0], persona_name="Bohdan", entity_hints=[])
    assert len(out) == 1
    assert out[0].subject == "cyberpunk 2077"


@pytest.mark.asyncio
async def test_extract_from_session_swallows_parse_failure():
    now = datetime(2025, 1, 1, tzinfo=UTC)
    rows = [_t("hi", now, ctx=["yo"])]
    sessions = build_sessions(rows, gap_hours=6)
    with (
        patch(
            "persona_rag.insights.extractor.chat_complete",
            AsyncMock(return_value="garbage non-json"),
        ),
        pytest.raises(ValueError),
    ):
        await extract_from_session(sessions[0], persona_name="Bohdan", entity_hints=[])


@pytest.mark.asyncio
async def test_extract_from_session_uses_json_mode_and_low_temp():
    """Regression: prior to fix, extractor inherited chatbot defaults (temp=0.8, max=300,
    no response_format), causing truncated / non-JSON model output."""
    now = datetime(2025, 1, 1, tzinfo=UTC)
    rows = [_t("я кодю", now, ctx=["що робиш?"])]
    sessions = build_sessions(rows, gap_hours=6)
    with patch(
        "persona_rag.insights.extractor.chat_complete",
        AsyncMock(return_value='{"insights": []}'),
    ) as mock_chat:
        await extract_from_session(sessions[0], persona_name="Bohdan", entity_hints=[])
    kwargs = mock_chat.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] >= 1500


def test_parse_extractor_response_error_includes_preview():
    """Error msg must show what model actually returned, not just 'char 0'."""
    bad = "Here are the insights you requested:\n\nSorry, I cannot help."
    with pytest.raises(ValueError, match="preview="):
        parse_extractor_response(bad, session_id="s1")
    try:
        parse_extractor_response(bad, session_id="s1")
    except ValueError as e:
        assert "Here are the insights" in str(e)
