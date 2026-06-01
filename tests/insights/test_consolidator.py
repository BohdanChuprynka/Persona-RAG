# ruff: noqa: RUF001
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from persona_rag.insights.consolidator import (
    ConsolidatedInsight,
    consolidate,
    load_synonyms,
    normalize_subject,
)
from persona_rag.insights.extractor import RawInsight


def _raw(category: str, subject: str, conf: float, when: datetime, sid: str = "s1") -> RawInsight:
    return RawInsight(
        session_id=sid,
        category=category,  # type: ignore[arg-type]
        subject=subject,
        text=f"about {subject}",
        confidence=conf,
        source_quote="quote",
        extracted_at=when,
    )


def test_normalize_subject_lowercases_and_strips():
    assert normalize_subject("  C++  ") == "c++"
    assert normalize_subject("School!") == "school"


def test_normalize_subject_applies_synonyms(tmp_path):
    syn_file = tmp_path / "synonyms.yaml"
    syn_file.write_text("python:\n  - пайтон\n  - pythonista\n")
    syns = load_synonyms(syn_file)
    assert normalize_subject("пайтон", syns) == "python"
    assert normalize_subject("Pythonista", syns) == "python"


@pytest.mark.asyncio
async def test_consolidate_merges_3_plus_via_llm():
    now = datetime(2025, 1, 1, tzinfo=UTC)
    raws = [_raw("interest", "cyberpunk", 0.8, now, sid=f"s{i}") for i in range(4)]
    canned = "Plays Cyberpunk 2077 — active 2024 → 2025\n\nTrajectory: active 2024-Q1 → 2025-Q1"
    with patch(
        "persona_rag.insights.consolidator.chat_complete",
        AsyncMock(return_value=canned),
    ):
        out = await consolidate(raws, synonyms={})
    assert len(out) == 1
    ci = out[0]
    assert isinstance(ci, ConsolidatedInsight)
    assert ci.evidence_count == 4
    assert ci.category == "interest"


@pytest.mark.asyncio
async def test_consolidate_keeps_singletons_no_llm():
    now = datetime(2025, 1, 1, tzinfo=UTC)
    raws = [_raw("interest", "rare_topic", 0.6, now)]
    with patch(
        "persona_rag.insights.consolidator.chat_complete",
        AsyncMock(),
    ) as mock_chat:
        out = await consolidate(raws, synonyms={})
    mock_chat.assert_not_called()
    assert len(out) == 1
    assert out[0].evidence_count == 1


@pytest.mark.asyncio
async def test_consolidate_groups_by_synonym(tmp_path):
    now = datetime(2025, 1, 1, tzinfo=UTC)
    raws = [
        _raw("interest", "C++", 0.8, now, sid="s1"),
        _raw("interest", "плюси", 0.7, now, sid="s2"),
        _raw("interest", "cpp", 0.75, now, sid="s3"),
    ]
    syns = {"c++": ["cpp", "плюси", "си плюс плюс"]}
    canned = "Learns C++"
    with patch(
        "persona_rag.insights.consolidator.chat_complete",
        AsyncMock(return_value=canned),
    ):
        out = await consolidate(raws, synonyms=syns)
    assert len(out) == 1
    assert out[0].canonical_subject == "c++"
    assert out[0].evidence_count == 3


def test_stable_insight_id_is_deterministic():
    from persona_rag.insights.consolidator import _stable_insight_id

    a = _stable_insight_id("interest", "cyberpunk 2077")
    b = _stable_insight_id("interest", "cyberpunk 2077")
    assert a == b
    assert len(a) == 16
    # Different inputs → different IDs
    assert a != _stable_insight_id("interest", "different")
    assert a != _stable_insight_id("bio", "cyberpunk 2077")


def test_consolidate_prompt_has_preserve_specifics_rule():
    """Spec §5.6 — consolidator prompt must instruct to keep concrete tokens."""
    from persona_rag.insights.consolidator import CONSOLIDATE_PROMPT

    rendered = CONSOLIDATE_PROMPT.format(n=3, observations="x")
    lower = rendered.lower()
    assert "preserve" in lower or "keep" in lower
    assert "concrete" in lower or "specific" in lower or "particulars" in lower
    assert "genre" in lower or "name" in lower or "number" in lower


@pytest.mark.asyncio
async def test_consolidate_computes_distinct_partners():
    """Two sessions with the same partner → distinct_partners=1; different → 2."""
    from datetime import UTC, datetime

    from persona_rag.insights.consolidator import consolidate
    from persona_rag.insights.extractor import RawInsight

    now = datetime(2026, 1, 1, tzinfo=UTC)
    raws = [
        RawInsight(
            session_id="s1",
            category="interest",
            subject="running",
            text="runs",
            confidence=0.9,
            source_quote="бігаю",
            extracted_at=now,
        ),
        RawInsight(
            session_id="s2",
            category="interest",
            subject="running",
            text="runs",
            confidence=0.9,
            source_quote="бігаю",
            extracted_at=now,
        ),
    ]
    out_same = await consolidate(
        raws,
        synonyms={},
        session_to_partners={"s1": {"rcpt-A"}, "s2": {"rcpt-A"}},
    )
    assert len(out_same) == 1
    assert out_same[0].distinct_partners == 1

    out_diff = await consolidate(
        raws,
        synonyms={},
        session_to_partners={"s1": {"rcpt-A"}, "s2": {"rcpt-B"}},
    )
    assert out_diff[0].distinct_partners == 2


@pytest.mark.asyncio
async def test_consolidate_defaults_partners_to_zero_when_no_mapping_given():
    """Backward-compat: callers that don't pass session_to_partners get 0."""
    from datetime import UTC, datetime

    from persona_rag.insights.consolidator import consolidate
    from persona_rag.insights.extractor import RawInsight

    now = datetime(2026, 1, 1, tzinfo=UTC)
    raws = [
        RawInsight(
            session_id="s1",
            category="interest",
            subject="running",
            text="runs",
            confidence=0.9,
            source_quote="бігаю",
            extracted_at=now,
        )
    ]
    out = await consolidate(raws, synonyms={})
    assert out[0].distinct_partners == 0
