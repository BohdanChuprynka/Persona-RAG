"""Runtime retrieval of self-insights. Inserted between load_memory and load_session."""

from __future__ import annotations

import json
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchAny
from sqlmodel import Session, select

from persona_rag._logging import get_logger
from persona_rag.config import get_settings
from persona_rag.db.engine import make_engine
from persona_rag.db.models import AlgoSignal
from persona_rag.graph.state import GraphState
from persona_rag.index.embedder import embed_batch
from persona_rag.index.qdrant_store import make_client
from persona_rag.insights.recency import from_qdrant_point, rerank_with_recency

log = get_logger()


async def retrieve_insights(state: GraphState) -> GraphState:
    s = get_settings()
    if not s.INSIGHTS_ENABLED:
        return state
    try:
        client = make_client()
        vec = (await embed_batch([state["incoming"]]))[0]
        pool_size = s.INSIGHTS_TOP_K_SEMANTIC * 3
        response = client.query_points(
            collection_name=s.QDRANT_INSIGHTS_COLLECTION,
            query=vec,
            limit=pool_size,
            with_payload=True,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="review_status",
                        match=MatchAny(any=["auto", "approved"]),
                    )
                ]
            ),
        )
        candidates = [from_qdrant_point(p) for p in response.points]
        reranked = rerank_with_recency(candidates, half_life_days=s.INSIGHTS_RECENCY_HALF_LIFE_DAYS)
        # Apply minimum score floor first, then take top-K from what survives.
        # Keeps strong matches even if K is bumped; weak matches don't leak in.
        floor = s.INSIGHTS_MIN_SCORE_FLOOR
        if floor > 0:
            reranked = [r for r in reranked if r.final_score >= floor]
        semantic = reranked[: s.INSIGHTS_TOP_K_SEMANTIC]

        static = (
            _load_static_signals(top_n=s.INSIGHTS_TOP_N_STATIC)
            if s.INSIGHTS_STATIC_PATTERNS_ENABLED
            else {}
        )
        state["insights"] = {"semantic": semantic, "static": static}
        log.info(
            "insights_retrieved",
            n_semantic=len(semantic),
            top_subjects=[r.subject for r in semantic],
        )
    except Exception as e:
        log.warning("insights_retrieval_failed", error=str(e))
        state["insights"] = {"semantic": [], "static": {}}
    return state


def _load_static_signals(top_n: int) -> dict[str, Any]:
    """Load top-N algo signals for the runtime prompt."""
    out: dict[str, Any] = {"languages": [], "entities": []}
    with Session(make_engine()) as s:
        langs = list(s.exec(select(AlgoSignal).where(AlgoSignal.kind == "language")).all())
        ents = list(s.exec(select(AlgoSignal).where(AlgoSignal.kind == "entity")).all())
    for row in langs:
        try:
            val = json.loads(row.value_json)
        except json.JSONDecodeError:
            val = {}
        out["languages"].append(
            {
                "subject": row.subject,
                "percentage": val.get("percentage", 0),
                "count": row.evidence_count,
            }
        )
    out["languages"].sort(key=lambda x: x["count"], reverse=True)
    out["entities"] = [{"subject": r.subject, "count": r.evidence_count} for r in ents]
    out["entities"].sort(key=lambda x: x["count"], reverse=True)
    out["entities"] = out["entities"][:top_n]
    return out
