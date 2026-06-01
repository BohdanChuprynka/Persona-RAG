from __future__ import annotations

from datetime import UTC, datetime, timedelta

from persona_rag.db.models import PersonaTurnRow
from persona_rag.insights.algo import (
    extract_counterparty_rhythms,
    extract_languages,
    extract_style,
)


def _t(
    reply: str, ts: datetime, recipient: str, chat: str = "c1", lang: str = "uk"
) -> PersonaTurnRow:
    return PersonaTurnRow(
        id=f"{chat}-{ts.isoformat()}",
        your_reply=reply,
        incoming_context_json="[]",
        channel="telegram",
        chat_id_hash=chat,
        recipient_id_hash=recipient,
        timestamp=ts,
        language=lang,
        your_reply_len_chars=len(reply),
        your_reply_emoji_count=0,
    )


def test_rhythms_groups_by_recipient():
    now = datetime(2025, 1, 1, tzinfo=UTC)
    rows = [_t("hi", now + timedelta(minutes=i), recipient="alice") for i in range(20)] + [
        _t("hey", now + timedelta(minutes=i), recipient="bob") for i in range(5)
    ]
    out = extract_counterparty_rhythms(rows)
    subjects = {s["subject"] for s in out}
    assert "alice" in subjects
    assert "bob" in subjects
    alice = next(s for s in out if s["subject"] == "alice")
    assert alice["count"] == 20


def test_languages_distribution():
    now = datetime(2025, 1, 1, tzinfo=UTC)
    rows = (
        [_t("hi", now, "a", lang="uk") for _ in range(75)]
        + [_t("hi", now, "a", lang="ru") for _ in range(20)]
        + [_t("hi", now, "a", lang="en") for _ in range(5)]
    )
    out = extract_languages(rows)
    by_subject = {s["subject"]: s for s in out}
    assert "uk" in by_subject
    assert by_subject["uk"]["count"] == 75
    assert by_subject["ru"]["count"] == 20


def test_style_counts_all_caps():
    now = datetime(2025, 1, 1, tzinfo=UTC)
    rows = [_t("AHAHAHA", now, "a"), _t("hi", now, "a"), _t("ОЛОЛО", now, "a")]
    out = extract_style(rows)
    by_subject = {s["subject"]: s for s in out}
    assert by_subject["all_caps"]["count"] == 2
