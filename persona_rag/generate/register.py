# ruff: noqa: RUF001
# Reason: Ukrainian/Russian marker tokens contain Cyrillic that ruff flags as
# look-alike. They are intentional.
"""Incoming-message register detection.

The persona had a single register — terse banter — and no notion of emotional
tone. So when a friend opened up ("***REMOVED***") the bot
gave the same flippant 2-bubble brush-off it gives a one-word ping. That reads
as emotionless. Register detection lets generation adapt the *engagement*, not
just the voice:

    heated  -> they came at you; fire back, match the heat (existing behaviour)
    serious -> they're opening up / asking for real help; engage, take space
    casual  -> default; short bursts

The distinction that matters most is heated vs serious when profanity is
present: "хуйло, йди нахуй" (attack on me) is heated, but "мені хуєво" (their
own pain) is serious. We classify by WHO the charged words are about, not by
their mere presence.
"""

from __future__ import annotations

import re
from typing import Literal

Register = Literal["heated", "serious", "casual"]

# Directed insults / aggression aimed AT you. Substring match on lowercased text.
# Deliberately NOT bare "хуй" — that would swallow own-state distress ("хуєво")
# and casual dismissal ("похуй"). Only the directed forms.
_HEATED = (
    "даун",
    "хуйл",  # хуйло, хуйлуша
    "нахуй",
    "єблан",
    "єбан",
    "ебан",
    "єбл",
    "йобн",  # йобнутий
    "єбанат",
    "ебанат",
    "мудак",
    "мудил",
    "дебіл",
    "дебил",
    "ідіот",
    "идиот",
    "придур",
    "гандон",
    "гондон",
    "чмо",
    "підор",
    "пидор",
    "підар",
    "пидар",
    "гнида",
    "сам ти",
    "сам такий",
    "пішов ти",
    "пошел ти",
    "пошёл ти",
    "відвали",
    "отвали",
    "відчепись",
    "заткн",  # заткнись / заткни
    "тупиця",
)

# Distress, help-seeking, emotional disclosure, reflective self-doubt. The
# presence of any one of these flips an incoming to "serious".
_SERIOUS = (
    # help-seeking (specific phrasings; bare "що робити" / "не знаю як" are too
    # broad — they match technical or playful questions)
    "що мені робити",
    "шо мені робити",
    "не знаю що робити",
    "не знаю шо робити",
    "не знаю навіщо",
    "що зі мною",
    "як мені бути",
    # problem framing — first-person framed only, so "не проблема",
    # "вирішив проблеми" and "проблема з інтернетом" don't trip it
    "в мене проблема",
    "у мене проблема",
    "така проблема",
    "моя проблема",
    "проблема в тому",
    "проблема в тім",
    # compulsion / loss of control
    "не можу перестати",
    "не можу зупин",
    "не можу більше",
    # distress feelings
    "важко",
    "тяжко",
    "погано мені",
    "мені погано",
    "сумно",
    "самотн",
    "тривож",
    "депрес",
    "вигор",  # вигорів / вигоріла
    "втомився",
    "втомилася",
    "нема сил",
    "немає сил",
    "страшно",
    "боюс",
    "паніка",
    "паніку",
    "ненавиджу себе",
    "не хочу жити",
    # own-state profanity = pain, not an attack
    "хуєво",
    "хуево",
    "хуйово",
    "херово",
    "хєрово",
    "хріново",
    "паршиво",
    # self-harm framing
    "шкодять мені",
    "шкодить мені",
    "шкодит",
    # reflective self-doubt
    "не розумію",
    "навіщо я",
    "навіщо це",
    "який сенс",
    "сенс життя",
    "куди я йду",
    "правильно я живу",
    "роблю помилк",
    "одні й ті самі помилки",
    "розчарув",
    "сумніваюсь",
    "чи варто",
    "що зі мною не так",
)


# A serious marker immediately preceded by a standalone "не" is a negation
# ("не страшно", "не важко") and must NOT flip the message to serious.
_NEG_BEFORE = re.compile(r"(?:^|\s)не\s+$")


def _has_unnegated(text: str, markers: tuple[str, ...]) -> bool:
    """True if any marker occurs at least once NOT directly negated by 'не'."""
    for m in markers:
        start = 0
        while True:
            i = text.find(m, start)
            if i == -1:
                break
            if not _NEG_BEFORE.search(text[:i]):
                return True
            start = i + 1
    return False


def detect_register(incoming: str, context: list[str] | None = None) -> Register:
    """Classify the incoming message's register.

    heated is checked first: an insult fires back even if the message is also
    long and questiony (we don't counsel someone who's attacking us). Otherwise
    an un-negated distress/help-seeking/reflective marker makes it serious.
    Everything else is casual — the common case.
    """
    text = (incoming or "").lower()
    if not text.strip():
        return "casual"
    if any(tok in text for tok in _HEATED):
        return "heated"
    if _has_unnegated(text, _SERIOUS):
        return "serious"
    return "casual"
