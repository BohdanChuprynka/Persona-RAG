from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from persona_rag.graph.nodes.update_memory import update_memory_node
from persona_rag.graph.state import GraphState
from persona_rag.models import ChatMessage


def _session_of_turns(n_turns: int) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    for i in range(n_turns):
        out.append(ChatMessage(role="user", content=f"u{i}"))
        out.append(ChatMessage(role="assistant", content=f"a{i}"))
    return out


@pytest.mark.asyncio
async def test_skips_when_below_interval(monkeypatch):
    monkeypatch.setenv("MEMORY_UPDATE_INTERVAL_TURNS", "4")
    from persona_rag.config import get_settings

    get_settings.cache_clear()

    state: GraphState = {
        "user_id": 1,
        "chat_id": 1,
        "incoming": "x",
        "session": _session_of_turns(2),  # below 4
    }
    with patch(
        "persona_rag.graph.nodes.update_memory.update_user_memory", AsyncMock()
    ) as mock_update:
        await update_memory_node(state)
    mock_update.assert_not_called()


@pytest.mark.asyncio
async def test_fires_at_interval_boundary(monkeypatch):
    monkeypatch.setenv("MEMORY_UPDATE_INTERVAL_TURNS", "4")
    from persona_rag.config import get_settings

    get_settings.cache_clear()

    state: GraphState = {
        "user_id": 1,
        "chat_id": 1,
        "incoming": "x",
        "session": _session_of_turns(4),  # boundary
    }
    with patch(
        "persona_rag.graph.nodes.update_memory.update_user_memory", AsyncMock()
    ) as mock_update:
        await update_memory_node(state)
    mock_update.assert_called_once()


@pytest.mark.asyncio
async def test_disabled_when_interval_zero(monkeypatch):
    monkeypatch.setenv("MEMORY_UPDATE_INTERVAL_TURNS", "0")
    from persona_rag.config import get_settings

    get_settings.cache_clear()

    state: GraphState = {
        "user_id": 1,
        "chat_id": 1,
        "incoming": "x",
        "session": _session_of_turns(20),
    }
    with patch(
        "persona_rag.graph.nodes.update_memory.update_user_memory", AsyncMock()
    ) as mock_update:
        await update_memory_node(state)
    mock_update.assert_not_called()


@pytest.mark.asyncio
async def test_skips_when_session_empty(monkeypatch):
    monkeypatch.setenv("MEMORY_UPDATE_INTERVAL_TURNS", "4")
    from persona_rag.config import get_settings

    get_settings.cache_clear()

    state: GraphState = {"user_id": 1, "chat_id": 1, "incoming": "x", "session": []}
    with patch(
        "persona_rag.graph.nodes.update_memory.update_user_memory", AsyncMock()
    ) as mock_update:
        await update_memory_node(state)
    mock_update.assert_not_called()
