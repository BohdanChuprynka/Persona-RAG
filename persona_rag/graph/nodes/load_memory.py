from __future__ import annotations

from persona_rag.graph.state import GraphState
from persona_rag.memory.store import load_memory


def load_memory_node(state: GraphState) -> GraphState:
    state["memory"] = load_memory(state["user_id"])
    return state
