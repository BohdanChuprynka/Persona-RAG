from __future__ import annotations

import re

_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_FALLBACK = "..."


def apply_guardrails(reply: str, *, max_chars: int = 1200) -> tuple[str, bool]:
    """Returns (cleaned_reply, ok). ok=False signals do-not-send."""
    if "<REDACTED>" in reply:
        return reply, False
    if not reply.strip():
        return _FALLBACK, True
    cleaned = _PHONE.sub("", reply)
    cleaned = _EMAIL.sub("", cleaned)
    if len(cleaned) > max_chars:
        truncated = cleaned[:max_chars]
        last_dot = truncated.rfind(".")
        cleaned = truncated[: last_dot + 1] if last_dot > max_chars * 0.5 else truncated
    return cleaned.strip(), True
