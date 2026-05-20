from __future__ import annotations

import hashlib
from typing import Any

from langdetect import (  # type: ignore[import-not-found]
    DetectorFactory,
    LangDetectException,
    detect,
)

from persona_rag.config import get_settings
from persona_rag.models import RawMessage

DetectorFactory.seed = 0  # deterministic


def hash_id(value: str) -> str:
    """BLAKE2b keyed hash, 16-char hex. Key derived from PERSONA_NAME."""
    settings = get_settings()
    h = hashlib.blake2b(
        value.encode("utf-8"),
        key=settings.PERSONA_NAME.encode("utf-8")[:64],
        digest_size=8,
    )
    return h.hexdigest()


def detect_language(text: str) -> str:
    try:
        return detect(text)  # type: ignore[no-any-return]
    except LangDetectException:
        return get_settings().PERSONA_LANGUAGE


def normalize_message(raw: RawMessage) -> dict[str, Any]:
    return {
        "channel": raw.channel,
        "chat_id_hash": hash_id(raw.chat_id),
        "sender_id_hash": hash_id(raw.sender_id),
        "sender_name": raw.sender_name,
        "text": raw.text,
        "timestamp": raw.timestamp,
        "is_group": raw.is_group,
        "language": detect_language(raw.text),
    }
