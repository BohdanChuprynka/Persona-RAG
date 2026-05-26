from __future__ import annotations

from datetime import UTC, datetime, timedelta

from persona_rag.db.models import PersonaTurnRow
from persona_rag.insights.algo import extract_phases


def _t(reply: str, ts: datetime, chat: str = "c1") -> PersonaTurnRow:
    return PersonaTurnRow(
        id=f"{chat}-{ts.isoformat()}",
        your_reply=reply,
        incoming_context_json="[]",
        channel="telegram",
        chat_id_hash=chat,
        recipient_id_hash="r1",
        timestamp=ts,
        language="uk",
        your_reply_len_chars=len(reply),
        your_reply_emoji_count=0,
    )


def test_extract_phases_buckets_by_quarter():
    rows = [
        _t("я кодю на python", datetime(2024, 2, 1, tzinfo=UTC) + timedelta(days=i))
        for i in range(15)
    ] + [
        _t("я кодю на cpp", datetime(2025, 5, 1, tzinfo=UTC) + timedelta(days=i)) for i in range(15)
    ]
    out = extract_phases(rows)
    subjects = {s["subject"] for s in out}
    assert "2024-Q1" in subjects
    assert "2025-Q2" in subjects
