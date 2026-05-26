from __future__ import annotations

import json
from pathlib import Path

from persona_rag.config import get_settings
from persona_rag.generate.prompt import build_messages
from persona_rag.graph.state import GraphState
from persona_rag.models import StyleAnchors


def _load_anchors() -> StyleAnchors:
    path = Path("data/style_anchors.json")
    if not path.exists():
        return StyleAnchors(
            avg_len_chars=0,
            median_len_chars=0,
            emoji_rate_per_char=0,
            lang_distribution={},
            top_bigrams=[],
            n_turns=0,
            primary_language=get_settings().PERSONA_LANGUAGE,
        )
    return StyleAnchors.model_validate(json.loads(path.read_text()))


def build_prompt_node(state: GraphState) -> GraphState:
    s = get_settings()
    anchors = _load_anchors()
    state["prompt"] = build_messages(
        persona_name=s.PERSONA_NAME,
        persona_description=s.PERSONA_DESCRIPTION,
        style_anchors=anchors,
        user_memory=state.get("memory", ""),
        retrieved=state.get("retrieved", []),
        session=state.get("session", []),
        incoming=state["incoming"],
        insights=state.get("insights"),
    )
    return state
