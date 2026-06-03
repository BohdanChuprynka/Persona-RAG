"""Detect the language/script of an incoming message: uk | ru | en.

Heuristic and dependency-free. Script first (Cyrillic vs Latin); within Cyrillic,
Ukrainian-distinctive letters win over Russian-distinctive ones. Defaults to uk
(the persona's primary register) when ambiguous. A char heuristic beats langdetect
here because chat queries are short and uk/ru are easily confused on a few words.
"""

from __future__ import annotations

from typing import Literal

Lang = Literal["uk", "ru", "en"]

_UK = set("іїєґ")
_RU = set("ыъэё")


def detect_language(text: str) -> Lang:
    t = (text or "").lower()
    cyr = sum(1 for c in t if "Ѐ" <= c <= "ӿ")
    lat = sum(1 for c in t if "a" <= c <= "z")
    if cyr == 0 and lat > 0:
        return "en"
    if cyr == 0 and lat == 0:
        return "uk"
    if any(c in _UK for c in t):
        return "uk"
    if any(c in _RU for c in t) and not any(c in _UK for c in t):
        return "ru"
    return "uk"
