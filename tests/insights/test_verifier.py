# Reason: tests use Cyrillic to exercise the Ukrainian-aware verifier.
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from persona_rag.insights.extractor import RawInsight
from persona_rag.insights.sessions import SessionDoc


def _raw(
    subject: str = "basketball",
    text: str = "Bohdan plays basketball",
    quote: str = "soccer, basketball, track",
) -> RawInsight:
    return RawInsight(
        session_id="s1",
        category="interest",
        subject=subject,
        text=text,
        confidence=0.9,
        source_quote=quote,
        extracted_at=datetime.now(UTC),
    )


def _session() -> SessionDoc:
    now = datetime.now(UTC)
    return SessionDoc(
        session_id="s1",
        chat_id_hash="c1",
        start=now,
        end=now,
        n_persona_turns=0,
        persona_chars=0,
        primary_language="uk",
        turns=[],
    )


@pytest.mark.asyncio
async def test_verifier_returns_yes_for_clear_quote():
    from persona_rag.insights.verifier import VerificationVerdict, verify_raw

    canned = '{"verdict": "YES", "reason": "quote directly affirms"}'
    with patch(
        "persona_rag.insights.verifier.chat_complete",
        AsyncMock(return_value=canned),
    ):
        v = await verify_raw(_raw(), session=_session())
    assert isinstance(v, VerificationVerdict)
    assert v.verdict == "YES"
    assert "affirms" in v.reason.lower()


@pytest.mark.asyncio
async def test_verifier_returns_no_for_third_party_quote():
    from persona_rag.insights.verifier import verify_raw

    canned = '{"verdict": "NO", "reason": "quote is about another person"}'
    with patch(
        "persona_rag.insights.verifier.chat_complete",
        AsyncMock(return_value=canned),
    ):
        v = await verify_raw(_raw(), session=_session())
    assert v.verdict == "NO"


@pytest.mark.asyncio
async def test_verifier_returns_ambiguous_for_vague_quote():
    from persona_rag.insights.verifier import verify_raw

    canned = '{"verdict": "AMBIGUOUS", "reason": "quote could be sarcastic"}'
    with patch(
        "persona_rag.insights.verifier.chat_complete",
        AsyncMock(return_value=canned),
    ):
        v = await verify_raw(_raw(), session=_session())
    assert v.verdict == "AMBIGUOUS"


@pytest.mark.asyncio
async def test_verifier_fails_open_on_api_error():
    """Spec §5.5 — verifier API error must NOT silently drop a raw.
    Returns verdict=None so caller treats as keep + reason='verifier_error'."""
    from persona_rag.insights.verifier import verify_raw

    with patch(
        "persona_rag.insights.verifier.chat_complete",
        AsyncMock(side_effect=RuntimeError("network down")),
    ):
        v = await verify_raw(_raw(), session=_session())
    assert v.verdict is None
    assert "error" in v.reason.lower()


@pytest.mark.asyncio
async def test_verifier_fails_open_on_bad_json():
    from persona_rag.insights.verifier import verify_raw

    with patch(
        "persona_rag.insights.verifier.chat_complete",
        AsyncMock(return_value="not json at all"),
    ):
        v = await verify_raw(_raw(), session=_session())
    assert v.verdict is None
