from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from persona_rag.config import get_settings
from persona_rag.graph.nodes.auth_check import auth_check
from persona_rag.graph.nodes.build_prompt import build_prompt_node
from persona_rag.graph.nodes.guardrails import guardrails_node
from persona_rag.graph.nodes.load_memory import load_memory_node
from persona_rag.graph.nodes.load_session import load_session
from persona_rag.graph.nodes.openai_chat import openai_chat
from persona_rag.graph.nodes.retrieve_hybrid import retrieve_hybrid
from persona_rag.graph.nodes.retrieve_insights import retrieve_insights
from persona_rag.graph.nodes.send_reply import send_reply
from persona_rag.graph.nodes.shadow_log import shadow_log
from persona_rag.graph.nodes.update_memory import update_memory_node
from persona_rag.graph.nodes.update_session import update_session
from persona_rag.graph.state import GraphState
from persona_rag.models import UserState


def _route_after_auth(state: GraphState) -> str:
    auth = state.get("auth_state")
    if auth == UserState.WHITELISTED.value:
        return "retrieve_hybrid"
    return END


def _route_after_guardrails(state: GraphState) -> str:
    if get_settings().SHADOW_MODE:
        return "shadow_log"
    return "send_reply"


def build_graph() -> Any:
    g = StateGraph(GraphState)
    g.add_node("auth_check", auth_check)
    g.add_node("retrieve_hybrid", retrieve_hybrid)
    g.add_node("load_memory", load_memory_node)
    g.add_node("retrieve_insights", retrieve_insights)
    g.add_node("load_session", load_session)
    g.add_node("build_prompt", build_prompt_node)
    g.add_node("openai_chat", openai_chat)
    g.add_node("guardrails", guardrails_node)
    g.add_node("send_reply", send_reply)
    g.add_node("shadow_log", shadow_log)
    g.add_node("update_session", update_session)
    g.add_node("update_memory", update_memory_node)

    g.set_entry_point("auth_check")
    g.add_conditional_edges("auth_check", _route_after_auth)
    g.add_edge("retrieve_hybrid", "load_memory")
    g.add_edge("load_memory", "retrieve_insights")
    g.add_edge("retrieve_insights", "load_session")
    g.add_edge("load_session", "build_prompt")
    g.add_edge("build_prompt", "openai_chat")
    g.add_edge("openai_chat", "guardrails")
    g.add_conditional_edges("guardrails", _route_after_guardrails)
    g.add_edge("send_reply", "update_session")
    g.add_edge("shadow_log", "update_session")
    g.add_edge("update_session", "update_memory")
    g.add_edge("update_memory", END)
    return g.compile()
