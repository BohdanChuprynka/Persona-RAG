from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from persona_rag.config import get_settings
from persona_rag.models import RawMessage


def _extract_text(msg: dict[str, object]) -> str:
    raw = msg.get("text", "")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        # Telegram represents formatted runs as a list of dicts
        return "".join(p["text"] if isinstance(p, dict) else str(p) for p in raw)
    return ""


def _normalize_id(value: object) -> str:
    """Telegram exports user IDs as ``"user1037155651"`` (string with prefix) or
    plain int depending on version. Strip the ``user`` / ``channel`` prefix and
    return the numeric body so downstream matching against ``ADMIN_TELEGRAM_ID``
    works regardless of the source format.
    """
    s = str(value or "")
    for prefix in ("user", "channel"):
        if s.startswith(prefix):
            return s[len(prefix) :]
    return s


def parse_telegram_export(path: Path) -> Iterator[RawMessage]:
    settings = get_settings()
    data = json.loads(path.read_text(encoding="utf-8"))
    for chat in data.get("chats", {}).get("list", []):
        is_group = chat.get("type") not in ("personal_chat", "private_supergroup")
        if is_group and not settings.INCLUDE_GROUP_CHATS:
            continue
        chat_id = str(chat.get("id"))
        for msg in chat.get("messages", []):
            if msg.get("type") != "message":
                continue
            text = _extract_text(msg).strip()
            if not text:
                continue
            yield RawMessage(
                channel="telegram",
                chat_id=chat_id,
                sender_id=_normalize_id(msg.get("from_id")),
                sender_name=str(msg.get("from", "")),
                text=text,
                timestamp=datetime.fromisoformat(msg["date"]),
                is_group=is_group,
            )
