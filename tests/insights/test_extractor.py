# ruff: noqa: RUF001
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from persona_rag.db.models import PersonaTurnRow
from persona_rag.insights.extractor import (
    EXTRACT_SYSTEM_PROMPT,
    RawInsight,
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


def test_render_session_labels_speakers():
    now = datetime(2025, 1, 1, tzinfo=UTC)
    rows = [
        _t("я кодю", now, ctx=["що робиш?"]),
        _t("так", now, ctx=["і норм?"]),
    ]
    sessions = build_sessions(rows, gap_hours=6)
    rendered = render_session(sessions[0], persona_name="Bohdan")
    assert "[friend]" in rendered
    assert "[Bohdan]" in rendered
    assert "що робиш?" in rendered
    assert "я кодю" in rendered


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
