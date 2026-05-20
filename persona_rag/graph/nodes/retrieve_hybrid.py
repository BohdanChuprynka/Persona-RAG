from __future__ import annotations

from persona_rag.config import get_settings
from persona_rag.graph.state import GraphState
from persona_rag.index.qdrant_store import make_client
from persona_rag.retrieval import retrieve


async def retrieve_hybrid(state: GraphState) -> GraphState:
    client = make_client()
    state["retrieved"] = await retrieve(
        state["incoming"],
        client=client,
        top_k=get_settings().TOP_K,
    )
    return state
