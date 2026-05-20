import pytest

from persona_rag.bot.auth import block_user, ensure_user
from persona_rag.db.engine import make_engine
from persona_rag.graph.compile import build_graph


@pytest.mark.asyncio
async def test_blocked_user_short_circuits(tmp_path, monkeypatch):
    db_path = str(tmp_path / "p.db")
    monkeypatch.setattr(
        "persona_rag.bot.auth.make_engine",
        lambda: make_engine(db_path),
    )

    ensure_user(7, "u", "u")
    block_user(7, admin_id=999)

    graph = build_graph()
    final = await graph.ainvoke({"user_id": 7, "chat_id": 7, "incoming": "hi"})
    assert final.get("reply", "") == ""
