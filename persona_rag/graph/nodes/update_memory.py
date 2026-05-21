from __future__ import annotations

from persona_rag._logging import get_logger
from persona_rag.config import get_settings
from persona_rag.graph.state import GraphState
from persona_rag.memory.updater import update_user_memory

log = get_logger()


async def update_memory_node(state: GraphState) -> GraphState:
    """Distill the running session into a long-form user memory summary.

    Throttled: only fires every ``MEMORY_UPDATE_INTERVAL_TURNS`` completed
    turns to avoid doubling OpenAI cost + latency on every message. A "turn"
    is one (user, assistant) pair, so the message count is twice the turn
    count.
    """
    session = state.get("session", [])
    if not session:
        return state

    interval = get_settings().MEMORY_UPDATE_INTERVAL_TURNS
    if interval <= 0:
        return state

    # session length is messages, not turns; pair = 2 messages.
    completed_turns = len(session) // 2
    if completed_turns == 0 or completed_turns % interval != 0:
        return state

    log.info("memory_update_triggered", user_id=state["user_id"], turns=completed_turns)
    await update_user_memory(user_id=state["user_id"], session=session)
    return state
