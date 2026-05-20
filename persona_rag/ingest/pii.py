from __future__ import annotations

import re

from persona_rag.config import get_settings

_PATTERNS: dict[str, re.Pattern[str]] = {
    "phone": re.compile(r"\+?\d[\d\s().-]{7,}\d"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "address": re.compile(
        r"\b\d{1,5}\s+\w+(?:\s+\w+){0,3}\s+"
        r"(St|Street|Ave|Avenue|Rd|Road|Dr|Drive|Ln|Lane|Blvd)\b",
        re.IGNORECASE,
    ),
}


def redact(
    text: str,
    *,
    patterns: list[str] | None = None,
    names: list[str] | None = None,
    token: str | None = None,
    strip_urls: bool | None = None,
) -> str:
    """Apply configured redaction rules. Returns new string."""
    s = get_settings()
    if patterns is None:
        patterns = [p.strip() for p in s.PII_PATTERNS.split(",") if p.strip()]
    if names is None:
        names = [n.strip() for n in s.PII_NAMES.split(",") if n.strip()]
    if token is None:
        token = s.PII_REPLACE_TOKEN
    if strip_urls is None:
        strip_urls = s.STRIP_URLS

    out = text
    for name in patterns:
        regex = _PATTERNS.get(name)
        if regex is not None:
            out = regex.sub(token, out)

    if strip_urls:
        out = re.sub(r"https?://\S+", token, out)

    for n in names:
        if n:
            out = re.sub(rf"\b{re.escape(n)}\b", token, out, flags=re.IGNORECASE)

    return out
