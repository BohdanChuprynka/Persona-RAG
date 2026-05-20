from __future__ import annotations

from persona_rag.graph.state import GraphState
from persona_rag.memory.updater import update_user_memory


async def update_memory_node(state: GraphState) -> GraphState:
    await update_user_memory(user_id=state["user_id"], session=state.get("session", []))
    return state
