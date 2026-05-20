from persona_rag.bot.auth import (
    approve_user,
    block_user,
    ensure_user,
    get_user_state,
)
from persona_rag.db.engine import make_engine
from persona_rag.models import UserState


def test_ensure_user_creates_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "persona_rag.bot.auth.make_engine",
        lambda: make_engine(str(tmp_path / "p.db")),
    )
    state = ensure_user(99, "alice", "Alice")
    assert state == UserState.UNKNOWN


def test_approve_user_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "persona_rag.bot.auth.make_engine",
        lambda: make_engine(str(tmp_path / "p.db")),
    )
    ensure_user(101, "u", "U")
    approve_user(101, admin_id=42)
    assert get_user_state(101) == UserState.WHITELISTED


def test_block_user(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "persona_rag.bot.auth.make_engine",
        lambda: make_engine(str(tmp_path / "p.db")),
    )
    ensure_user(202, "u", "U")
    block_user(202, admin_id=42)
    assert get_user_state(202) == UserState.BLOCKED
