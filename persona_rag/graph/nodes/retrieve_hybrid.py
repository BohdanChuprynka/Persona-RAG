from __future__ import annotations

from persona_rag._logging import get_logger
from persona_rag.config import get_settings
from persona_rag.graph.state import GraphState
from persona_rag.index.qdrant_store import make_client
from persona_rag.retrieval import retrieve

log = get_logger()


async def retrieve_hybrid(state: GraphState) -> GraphState:
    client = make_client()
    retrieved = await retrieve(
        state["incoming"],
        client=client,
        top_k=get_settings().TOP_K,
    )
    state["retrieved"] = retrieved
    log.info(
        "retrieved",
        query=state["incoming"][:80],
        count=len(retrieved),
        top=[
            {
                "id": r.turn.id,
                "score": round(r.score, 3),
                "score_dense": round(r.score_dense or 0.0, 3),
                "score_bm25": round(r.score_bm25 or 0.0, 3),
                "lang": r.turn.language,
                "reply_preview": r.turn.your_reply[:60],
            }
            for r in retrieved[:3]
        ],
    )
    return state
