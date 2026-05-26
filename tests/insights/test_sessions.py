from __future__ import annotations

from datetime import UTC, datetime, timedelta

from persona_rag.db.models import PersonaTurnRow
from persona_rag.insights.sessions import (
    build_sessions,
    filter_high_signal,
)


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


def test_build_sessions_splits_on_gap():
    base = datetime(2025, 1, 1, 10, 0, tzinfo=UTC)
    rows = [
        _t("hi", base),
        _t("hello", base + timedelta(minutes=1)),
        _t("how are you", base + timedelta(hours=7)),  # > 6h gap → new session
    ]
    out = build_sessions(rows, gap_hours=6)
    assert len(out) == 2
    assert out[0].n_persona_turns == 2
    assert out[1].n_persona_turns == 1


def test_filter_drops_short_sessions():
    base = datetime(2025, 1, 1, tzinfo=UTC)
    big = [_t(f"reply number {i} with content", base + timedelta(minutes=i)) for i in range(20)]
    small = [_t("ok", base + timedelta(days=10) + timedelta(minutes=i)) for i in range(3)]
    sessions = build_sessions(big + small, gap_hours=6)
    out = filter_high_signal(
        sessions,
        history_years=10,
        min_turns=10,
        min_chars=300,
        max_sessions=600,
    )
    assert len(out) == 1
    assert out[0].n_persona_turns == 20


def test_filter_applies_history_cutoff():
    long_ago = datetime(2020, 1, 1, tzinfo=UTC)
    recent = datetime(2025, 1, 1, tzinfo=UTC)
    rows_old = [_t(f"older content reply {i}", long_ago + timedelta(minutes=i)) for i in range(20)]
    rows_new = [_t(f"newer content reply {i}", recent + timedelta(minutes=i)) for i in range(20)]
    sessions = build_sessions(rows_old + rows_new, gap_hours=6)
    out = filter_high_signal(
        sessions,
        history_years=2.5,
        min_turns=10,
        min_chars=300,
        max_sessions=600,
        now=datetime(2026, 5, 22, tzinfo=UTC),
    )
    # 2020 session dropped; 2025 session kept
    assert all(s.start.year >= 2023 for s in out)


def test_filter_caps_at_max_sessions():
    base = datetime(2025, 1, 1, tzinfo=UTC)
    sessions_in = []
    for i in range(50):
        rows = [
            _t(f"longer content reply {j}", base + timedelta(days=i, minutes=j)) for j in range(15)
        ]
        sessions_in.extend(build_sessions(rows, gap_hours=6))
    out = filter_high_signal(
        sessions_in, history_years=10, min_turns=10, min_chars=300, max_sessions=5
    )
    assert len(out) == 5
