from __future__ import annotations

from persona_rag.config import get_settings
from persona_rag.generate.llm_client import chat_complete, chat_complete_n, voice_logit_bias
from persona_rag.generate.select import select_best_style
from persona_rag.graph.state import GraphState


async def openai_chat(state: GraphState) -> GraphState:
    s = get_settings()
    bias = voice_logit_bias()
    n = max(1, s.BEST_OF_N)
    if n == 1:
        state["reply"] = await chat_complete(state["prompt"], logit_bias=bias)
    else:
        # Sample N at a slightly higher temperature, keep the most on-voice one.
        candidates = await chat_complete_n(
            state["prompt"], n=n, temperature=s.BEST_OF_N_TEMPERATURE, logit_bias=bias
        )
        state["reply"] = select_best_style(candidates)
    return state
