from __future__ import annotations

import math
from datetime import UTC, datetime

from persona_rag.config import get_settings
from persona_rag.models import RetrievedTurn


def recency_decay(
    items: list[RetrievedTurn],
    *,
    half_life_days: int | None = None,
) -> list[RetrievedTurn]:
    half = half_life_days or get_settings().RECENCY_HALF_LIFE_DAYS
    now = datetime.now(UTC)
    reranked: list[RetrievedTurn] = []
    for item in items:
        age = (now - item.turn.timestamp).days
        factor = math.exp(-math.log(2) * age / half)
        reranked.append(item.model_copy(update={"score": item.score * factor}))
    reranked.sort(key=lambda x: x.score, reverse=True)
    return reranked
