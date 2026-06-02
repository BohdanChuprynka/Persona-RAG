# Reason: Cyrillic test data exercises the insights rendering.
from __future__ import annotations

from datetime import UTC, datetime

from persona_rag.generate.prompt import _render_insights_block
from persona_rag.insights.recency import RankedInsight


def _ins(category: str, subject: str, text: str, score: float = 0.5) -> RankedInsight:
    now = datetime.now(UTC)
    return RankedInsight(
        id=f"i-{subject}",
        category=category,
        subject=subject,
        text=text,
        confidence=1.0,
        evidence_count=5,
        earliest_date=now,
        latest_date=now,
        trajectory=None,
        source="chat",
        semantic_score=score,
        final_score=score,
    )


def test_prompt_does_not_render_recurring_topics():
    """Spec §5.1.c — the static entities line must be gone from runtime prompts."""
    insights = {
        "semantic": [_ins("interest", "running", "Bohdan runs")],
        "static": {
            "languages": [{"subject": "uk", "percentage": 0.6, "count": 100}],
            "entities": [
                {"subject": "мене", "count": 50},
                {"subject": "тебе", "count": 40},
            ],
        },
    }
    out = _render_insights_block(insights)
    assert "recurring topics" not in out
    assert "мене" not in out and "тебе" not in out


def test_prompt_keeps_language_mix():
    """Language line still grounds primary-language choice; only entities go."""
    insights = {
        "semantic": [],
        "static": {
            "languages": [{"subject": "uk", "percentage": 0.6, "count": 100}],
            "entities": [],
        },
    }
    out = _render_insights_block(insights)
    assert "uk" in out


def test_prompt_splits_bio_from_other_categories():
    """Spec §5.8 — bio insights render under their own header so the bio-anchor
    priority rule has something to point at."""
    insights = {
        "semantic": [
            _ins("bio", "school", "Bohdan attends Lincoln HS"),
            _ins("interest", "running", "Bohdan runs"),
            _ins("behavior", "planning", "Bohdan plans meticulously"),
        ],
        "static": {"languages": [], "entities": []},
    }
    out = _render_insights_block(insights)
    assert "What's true about you (bio facts):" in out
    assert "Lincoln" in out
    assert "Things you talk about / are into:" in out
    assert "Bohdan runs" in out
    assert out.index("What's true about you") < out.index("Things you talk about")


def test_prompt_omits_bio_header_when_no_bio_insights():
    insights = {
        "semantic": [_ins("interest", "running", "Bohdan runs")],
        "static": {"languages": [], "entities": []},
    }
    out = _render_insights_block(insights)
    assert "What's true about you" not in out
    assert "Things you talk about / are into:" in out


def test_system_template_has_bio_anchor_priority_rule():
    """Spec §5.8 — model is told to USE bio insights, not always deflect."""
    from persona_rag.generate.prompt import SYSTEM_TEMPLATE

    rendered = SYSTEM_TEMPLATE.format(
        persona_name="Bohdan",
        persona_description="test",
        avg_len_chars=42,
        median_len_chars=25,
        emoji_rate_per_char=0.001,
        primary_language="uk",
        top_bigrams_joined="x",
        user_memory="x",
        insights_block="",
    )
    assert "bio" in rendered.lower()
    assert "anchor" in rendered.lower() or "bio facts" in rendered.lower()
    assert "куди в школу" in rendered or "школу ходиш" in rendered


def test_system_template_has_bio_over_opinion_factual_rule():
    """Spec §5.9 — bio > opinion when answering yes/no factual questions."""
    from persona_rag.generate.prompt import SYSTEM_TEMPLATE

    rendered = SYSTEM_TEMPLATE.format(
        persona_name="Bohdan",
        persona_description="test",
        avg_len_chars=42,
        median_len_chars=25,
        emoji_rate_per_char=0.001,
        primary_language="uk",
        top_bigrams_joined="x",
        user_memory="x",
        insights_block="",
    )
    lower = rendered.lower()
    assert "bio" in lower and "opinion" in lower
    assert "yes/no" in lower or "factual" in lower


def test_prompt_no_longer_bans_paragraphs():
    """Spec §5.2 — the "NEVER paragraphs" rule must be gone."""
    from persona_rag.generate.prompt import SYSTEM_TEMPLATE

    rendered = SYSTEM_TEMPLATE.format(
        persona_name="Bohdan",
        persona_description="test",
        avg_len_chars=42,
        median_len_chars=25,
        emoji_rate_per_char=0.001,
        primary_language="uk",
        top_bigrams_joined="x",
        user_memory="x",
        insights_block="",
    )
    lower = rendered.lower()
    assert "never write polished multi-sentence paragraphs" not in lower
    assert "default is short bursts" not in lower
    assert "length is dynamic, not fixed" not in lower


def test_prompt_has_shape_matches_the_moment_rule():
    """Spec §5.2 — replace length prescriptions with one rule about matching
    the moment via the retrieved examples."""
    from persona_rag.generate.prompt import SYSTEM_TEMPLATE

    rendered = SYSTEM_TEMPLATE.format(
        persona_name="Bohdan",
        persona_description="test",
        avg_len_chars=42,
        median_len_chars=25,
        emoji_rate_per_char=0.001,
        primary_language="uk",
        top_bigrams_joined="x",
        user_memory="x",
        insights_block="",
    )
    lower = rendered.lower()
    assert "shape matches the moment" in lower
    # Must reference using the retrieved examples as the shape template.
    assert "retrieved past replies" in lower or "retrieved examples" in lower


def test_prompt_keeps_existing_voice_rules():
    """Regression guard: voice rules unrelated to length stay intact."""
    from persona_rag.generate.prompt import SYSTEM_TEMPLATE

    rendered = SYSTEM_TEMPLATE.format(
        persona_name="Bohdan",
        persona_description="test",
        avg_len_chars=42,
        median_len_chars=25,
        emoji_rate_per_char=0.001,
        primary_language="uk",
        top_bigrams_joined="x",
        user_memory="x",
        insights_block="",
    )
    lower = rendered.lower()
    # Casing rule
    assert "copy the casing" in lower
    # Bio anchor rule
    assert "bio anchor" in lower
    # Register-match-when-insulted rule
    assert "register match" in lower or "match the heat" in lower
    # Travel vs residence rule
    assert "travel vs residence" in lower or "trip is an event" in lower or "travel vs" in lower
    # Anti-fabrication for self-description
    assert "anti-fabrication" in lower or "fabrication" in lower
