from __future__ import annotations

from persona_rag.config import get_settings
from persona_rag.graph.state import GraphState
from persona_rag.ingest.normalize import hash_id
from persona_rag.shadow.logger import write_shadow_entry


def shadow_log(state: GraphState) -> GraphState:
    s = get_settings()
    write_shadow_entry(
        user_id_hash=hash_id(str(state["user_id"])),
        incoming=state["incoming"],
        context=[m.content for m in state.get("session", [])],
        retrieved_ids=[r.turn.id for r in state.get("retrieved", [])],
        memory=state.get("memory", ""),
        generated_reply=state.get("reply", ""),
        params={
            "top_k": s.TOP_K,
            "alpha": s.HYBRID_DENSE_ALPHA,
            "model": s.OPENAI_CHAT_MODEL,
            "temperature": s.TEMPERATURE,
        },
        session_id=state.get("session_id"),
    )
    return state
