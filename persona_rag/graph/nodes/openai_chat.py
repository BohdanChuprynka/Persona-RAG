from __future__ import annotations

from persona_rag.generate.llm_client import chat_complete
from persona_rag.graph.state import GraphState


async def openai_chat(state: GraphState) -> GraphState:
    state["reply"] = await chat_complete(state["prompt"])
    return state
