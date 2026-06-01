from __future__ import annotations

from persona_rag.config import get_settings
from persona_rag.generate.llm_client import chat_complete
from persona_rag.memory.store import load_contact_memory, save_contact_memory
from persona_rag.models import ChatMessage

MEMORY_PROMPT = """\
Below is a recent conversation between {persona_name} and a user.
Below that is the current memory summary {persona_name} has about this user.

Conversation:
{session_log}

Current memory:
{existing_summary}

Update the memory in ≤300 tokens. Keep:
- Their name / how they prefer to be addressed
- Topics they care about
- Any commitments or promises made
- Relationship context (friend, colleague, etc.)

Drop:
- Specific message content older than what's relevant
- Anything tagged <REDACTED>

Output ONLY the new summary text. No preamble.
"""


def _format_session(session: list[ChatMessage]) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in session)


async def update_contact_memory(*, user_id: int, session: list[ChatMessage]) -> None:
    existing = load_contact_memory(user_id) or "(none yet)"
    prompt = MEMORY_PROMPT.format(
        persona_name=get_settings().PERSONA_NAME,
        session_log=_format_session(session),
        existing_summary=existing,
    )
    new_summary = await chat_complete([{"role": "user", "content": prompt}])
    save_contact_memory(user_id, new_summary)
