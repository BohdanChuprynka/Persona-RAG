from __future__ import annotations

from persona_rag.generate.guardrails import apply_guardrails
from persona_rag.graph.state import GraphState


def guardrails_node(state: GraphState) -> GraphState:
    cleaned, ok = apply_guardrails(state.get("reply", ""))
    state["reply"] = cleaned if ok else ""
    return state
