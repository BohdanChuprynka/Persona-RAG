from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from persona_rag.config import get_settings


def write_shadow_entry(
    *,
    user_id_hash: str,
    incoming: str,
    context: list[str],
    retrieved_ids: list[str],
    memory: str,
    generated_reply: str,
    params: dict[str, Any],
    session_id: str | None = None,
) -> None:
    s = get_settings()
    s.SHADOW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "session_id": session_id or str(uuid.uuid4()),
        "user_id_hash": user_id_hash,
        "incoming": incoming,
        "context": context,
        "retrieved_ids": retrieved_ids,
        "memory_summary": memory,
        "generated_reply": generated_reply,
        "your_actual_reply": None,
        "model": s.OPENAI_CHAT_MODEL,
        "params": params,
    }
    with s.SHADOW_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
