from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from persona_rag.config import get_settings
from persona_rag.models import RawMessage


def _decode_mojibake(text: str) -> str:
    """Instagram exports UTF-8 bytes encoded as Latin-1 codepoints.
    Re-encode as Latin-1 then decode as UTF-8 to recover the original chars."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def parse_instagram_export(path: Path) -> Iterator[RawMessage]:
    """Parse a single Instagram message_N.json file."""
    settings = get_settings()
    data = json.loads(path.read_text(encoding="utf-8"))
    is_group = len(data.get("participants", [])) > 2
    if is_group and not settings.INCLUDE_GROUP_CHATS:
        return
    thread = data.get("thread_path") or data.get("title") or path.stem
    chat_id = str(thread)
    for msg in data.get("messages", []):
        text = msg.get("content")
        if not text:
            continue
        yield RawMessage(
            channel="instagram",
            chat_id=chat_id,
            sender_id=_decode_mojibake(msg["sender_name"]),
            sender_name=_decode_mojibake(msg["sender_name"]),
            text=_decode_mojibake(text),
            timestamp=datetime.fromtimestamp(msg["timestamp_ms"] / 1000, tz=UTC),
            is_group=is_group,
        )


def walk_instagram_folder(root: Path) -> Iterator[RawMessage]:
    """Walk an Instagram messages/inbox/*/message_*.json tree."""
    for json_file in root.rglob("message_*.json"):
        yield from parse_instagram_export(json_file)
