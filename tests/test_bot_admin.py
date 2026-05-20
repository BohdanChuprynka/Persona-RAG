from unittest.mock import AsyncMock, MagicMock

import pytest

from persona_rag.bot.auth import approve_user, ensure_user
from persona_rag.bot.handlers.admin import handle_users
from persona_rag.db.engine import make_engine


@pytest.mark.asyncio
async def test_users_command_lists_whitelisted(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "persona_rag.bot.auth.make_engine",
        lambda: make_engine(str(tmp_path / "p.db")),
    )
    monkeypatch.setattr(
        "persona_rag.bot.handlers.admin.make_engine",
        lambda: make_engine(str(tmp_path / "p.db")),
    )
    ensure_user(1, "alice", "Alice")
    ensure_user(2, "bob", "Bob")
    approve_user(1, admin_id=42)

    msg = MagicMock()
    msg.answer = AsyncMock()
    msg.from_user = MagicMock()
    msg.from_user.id = 42  # matches conftest ADMIN_TELEGRAM_ID=42

    await handle_users(msg)
    msg.answer.assert_called_once()
    text = msg.answer.call_args[0][0]
    assert "alice" in text
    assert "bob" not in text
