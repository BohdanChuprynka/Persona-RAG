from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from persona_rag.graph.nodes.load_session import (
    _SESSIONS,
    load_session,
)
from persona_rag.graph.nodes.update_session import update_session
from persona_rag.graph.state import GraphState
from persona_rag.models import ChatMessage


@pytest.fixture(autouse=True)
def _reset_sessions():
    _SESSIONS.clear()
    yield
    _SESSIONS.clear()


def test_load_session_empty_by_default():
    state: GraphState = {"user_id": 1, "chat_id": 1, "incoming": "x"}
    out = load_session(state)
    assert out["session"] == []


def test_update_session_persists_turn_for_next_load():
    state: GraphState = {
        "user_id": 42,
        "chat_id": 42,
        "incoming": "як справи",
        "reply": "норм",
        "session": [],
    }
    update_session(state)

    next_state: GraphState = {"user_id": 42, "chat_id": 42, "incoming": "?"}
    loaded = load_session(next_state)["session"]

    assert len(loaded) == 2
    assert loaded[0] == ChatMessage(role="user", content="як справи")
    assert loaded[1] == ChatMessage(role="assistant", content="норм")


def test_session_caps_to_window():
    uid = 9
    for i in range(15):
        update_session(
            {
                "user_id": uid,
                "chat_id": uid,
                "incoming": f"u{i}",
                "reply": f"a{i}",
                "session": [],
            }
        )

    loaded = load_session({"user_id": uid, "chat_id": uid, "incoming": "x"})["session"]
    # CURRENT_SESSION_WINDOW=10 → at most 10 messages (5 turns)
    assert len(loaded) <= 10
    assert loaded[-1].content == "a14"


def test_session_expires_after_timeout():
    uid = 100
    update_session(
        {
            "user_id": uid,
            "chat_id": uid,
            "incoming": "old",
            "reply": "ok",
            "session": [],
        }
    )
    # Force stale timestamp far past timeout.
    entry = _SESSIONS[uid]
    entry.last_seen = datetime.now(UTC) - timedelta(hours=10)

    loaded = load_session({"user_id": uid, "chat_id": uid, "incoming": "x"})["session"]
    assert loaded == []


def test_update_session_skips_when_no_reply():
    uid = 5
    update_session(
        {"user_id": uid, "chat_id": uid, "incoming": "hello", "reply": "", "session": []}
    )
    assert uid not in _SESSIONS


def test_update_session_mirrors_into_state():
    state: GraphState = {
        "user_id": 77,
        "chat_id": 77,
        "incoming": "hi",
        "reply": "hey",
        "session": [],
    }
    out = update_session(state)
    assert out["session"] == [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hey"),
    ]
