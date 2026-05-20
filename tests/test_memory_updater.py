from unittest.mock import AsyncMock, patch

import pytest

from persona_rag.db.engine import make_engine
from persona_rag.memory.updater import update_user_memory
from persona_rag.models import ChatMessage


@pytest.mark.asyncio
async def test_update_calls_llm_and_saves(tmp_path, monkeypatch):
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "persona_rag.memory.store.make_engine",
        lambda: make_engine(db_path),
    )

    session = [
        ChatMessage(role="user", content="i like cats"),
        ChatMessage(role="assistant", content="cool"),
    ]
    with patch(
        "persona_rag.memory.updater.chat_complete", AsyncMock(return_value="User likes cats.")
    ):
        await update_user_memory(user_id=42, session=session)

    from persona_rag.memory.store import load_memory

    assert "cats" in load_memory(42)
