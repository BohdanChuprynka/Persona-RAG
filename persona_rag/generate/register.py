# ruff: noqa: RUF001, RUF003
# Reason: Ukrainian/Russian marker tokens + examples contain Cyrillic that ruff
# flags as look-alike (in strings and comments). They are intentional.
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
    # help-seeking
    "що мені робити",
    "шо мені робити",
    "що робити",
    "шо робити",
    "не знаю що",
    "не знаю шо",
    "не знаю як",
    "не знаю навіщо",
    "що зі мною",
    "як мені бути",
    # problem framing (with the -а/-и ending, so casual "нема проблем" is excluded)
    "проблема",
    "проблеми",
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
    "набридло",
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


def detect_register(incoming: str, context: list[str] | None = None) -> Register:
    """Classify the incoming message's register.

    heated is checked first: an insult fires back even if the message is also
    long and questiony (we don't counsel someone who's attacking us). Otherwise
    a distress/help-seeking/reflective marker makes it serious. Everything else
    is casual — the common case.
    """
    text = (incoming or "").lower()
    if not text.strip():
        return "casual"
    if any(tok in text for tok in _HEATED):
        return "heated"
    if any(tok in text for tok in _SERIOUS):
        return "serious"
    return "casual"
