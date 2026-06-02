# ruff: noqa: RUF003
from __future__ import annotations

from datetime import UTC, datetime

from persona_rag.db.models import PersonaTurnRow
from persona_rag.insights.algo import extract_entities


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


def test_extract_entities_drops_sentence_starts():
    rows = [
        _t("Давай завтра", datetime(2025, 1, i + 1, tzinfo=UTC), chat=f"c{i}") for i in range(15)
    ]
    out = extract_entities(rows)
    subjects = {e["subject"] for e in out}
    assert "Давай" not in subjects


def test_extract_entities_drops_urls():
    rows = [
        _t(
            "https://youtube.com/abc and more text here",
            datetime(2025, 1, i + 1, tzinfo=UTC),
            chat=f"c{i}",
        )
        for i in range(15)
    ]
    out = extract_entities(rows)
    subjects = {e["subject"] for e in out}
    assert "https" not in subjects
    assert "youtube" not in subjects
    assert "com" not in subjects


def test_extract_entities_keeps_real_entities():
    # token mentioned 12× across 4 sessions, mid-sentence
    rows = []
    for i in range(12):
        rows.append(
            _t("я грав cyberpunk учора", datetime(2025, 1, i + 1, tzinfo=UTC), chat=f"c{i % 4}")
        )
    out = extract_entities(rows)
    subjects = {e["subject"] for e in out}
    assert "cyberpunk" in subjects


def test_extract_entities_caps_at_top_50():
    # generate 100 distinct entities
    rows = []
    for i in range(100):
        for j in range(12):
            rows.append(
                _t(
                    f"мій topic{i:03d} там",
                    datetime(2025, 1, (j % 28) + 1, tzinfo=UTC),
                    chat=f"c{j % 4}",
                )
            )
    out = extract_entities(rows)
    assert len(out) <= 50


def test_extract_entities_passes_synonyms_as_whitelist():
    """Tokens listed in synonyms.yaml must bypass the function-word blocklist."""
    rows: list[PersonaTurnRow] = []
    for i in range(15):  # >=min_count and across 4 distinct chat_id_hashes
        rows.append(
            PersonaTurnRow(
                id=f"t{i}",
                your_reply="правда мене зовсім не цікавить",
                incoming_context_json="[]",
                channel="telegram",
                chat_id_hash=f"chat{i % 4}",
                recipient_id_hash=f"r{i % 4}",
                timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                language="uk",
                your_reply_len_chars=20,
                your_reply_emoji_count=0,
            )
        )

    out_no_wl = extract_entities(rows, whitelist=None)
    subjects_no_wl = {e["subject"].lower() for e in out_no_wl}
    assert "мене" not in subjects_no_wl

    out_wl = extract_entities(rows, whitelist={"мене"})
    subjects_wl = {e["subject"].lower() for e in out_wl}
    assert "мене" in subjects_wl
