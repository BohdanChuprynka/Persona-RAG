from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from persona_rag.graph.nodes.retrieve_insights import retrieve_insights


@pytest.mark.asyncio
async def test_retrieve_insights_no_collection_returns_empty(monkeypatch):
    monkeypatch.setattr(
        "persona_rag.graph.nodes.retrieve_insights.make_client",
        lambda: (_ for _ in ()).throw(RuntimeError("no qdrant")),
    )
    state = {"user_id": 1, "chat_id": 1, "incoming": "hi"}
    out = await retrieve_insights(state)
    assert out["insights"]["semantic"] == []
    assert out["insights"]["static"] == {}


@pytest.mark.asyncio
async def test_retrieve_insights_pulls_pool_then_reranks(monkeypatch):
    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_resp.points = []
    fake_client.query_points = MagicMock(return_value=fake_resp)
    monkeypatch.setattr(
        "persona_rag.graph.nodes.retrieve_insights.make_client",
        lambda: fake_client,
    )
    with patch(
        "persona_rag.graph.nodes.retrieve_insights.embed_batch",
        AsyncMock(return_value=[[0.0] * 1536]),
    ):
        state = {"user_id": 1, "chat_id": 1, "incoming": "hi"}
        out = await retrieve_insights(state)
    assert out["insights"]["semantic"] == []
