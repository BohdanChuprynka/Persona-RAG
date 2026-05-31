# ruff: noqa: RUF001
# Reason: intentional Cyrillic stopword corpus.
"""Noise filters for Stage A algorithmic signals.

The first probe over the live corpus showed raw token frequency mining is
dominated by sentence-start common words and URL fragments. These blocklists
+ helpers gate which tokens are eligible as entities.
"""

from __future__ import annotations

URL_PIECES: frozenset[str] = frozenset(
    {
        "https",
        "http",
        "www",
        "com",
        "org",
        "net",
        "ua",
        "ru",
        "youtube",
        "youtu",
        "redacted",
        "share",
        "feature",
        "watch",
        "user",
        "video",
        "html",
        "php",
    }
)

ENGLISH_FILLERS: frozenset[str] = frozenset(
    {
        "yeah",
        "good",
        "okay",
        "right",
        "going",
        "know",
        "think",
        "want",
        "need",
        "time",
        "make",
        "let",
        "see",
        "way",
        "yes",
        "well",
        "stuff",
        "thing",
        "things",
        "much",
        "just",
        "really",
        "back",
        "out",
        "all",
        "today",
        "tomorrow",
        "now",
        "alright",
        "sure",
        "feel",
        "feels",
        "got",
        "give",
        "gave",
        "look",
    }
)

# Tokens that look like proper nouns but are actually sentence-start common words.
CYRILLIC_SENT_STARTS: frozenset[str] = frozenset(
    {
        "Давай",
        "Зара",
        "Поняв",
        "Коли",
        "Дякую",
        "Може",
        "Треба",
        "Всьо",
        "Завтра",
        "Добраніч",
        "Виходи",
        "Добре",
        "Будь",
        "Надобраніч",
        "Піздєц",
        "Тільки",
        "Десь",
        "Капєц",
        "Доречі",
        "Скинь",
        "Можеш",
        "Сьогодні",
        "Чекай",
        "Дивись",
        "Солодких",
        "Пізда",
        "Можна",
        "Чому",
        "Кароче",
        "Потім",
        "Чесно",
        "Так",
        "Ні",
        "Ага",
        "Ок",
        "Окей",
    }
)


# Spec 2026-05-31 §5.1.a — narrow Slavic function-word blocklist.
# Pronouns + clearest non-content particles only. Deliberately excludes
# verbs (хочу, можу, буде), opinion adverbs (дуже, тільки, просто), and
# time/place adverbs (зараз, там, тут) because they carry signal.
SLAVIC_FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        # ua personal / possessive / reflexive pronouns
        "мене",
        "тебе",
        "себе",
        "нього",
        "неї",
        "них",
        "мені",
        "тобі",
        "собі",
        "йому",
        "їй",
        "мій",
        "моя",
        "моє",
        "мої",
        "твій",
        "твоя",
        "твоє",
        "твої",
        "його",
        "її",
        "їх",
        "їхній",
        "їхня",
        "їхнє",
        "наш",
        "наша",
        "наше",
        "наші",
        "ваш",
        "ваша",
        "ваше",
        "ваші",
        # ua clearest non-content particles / indefinite words
        "нічого",
        "щось",
        "хтось",
        "кудись",
        "якось",
        "якийсь",
        "така",
        "такий",
        "таке",
        "сюди",
        "звідти",
        # ru variants of the above
        "меня",
        "тебя",
        "себя",
        "его",
        "ее",
        "их",
        "мне",
        "ему",
        "ей",
        "им",
        "что-то",
        "кто-то",
        "куда-то",
        "как-то",
    }
)


def is_sentence_start_only(positions: list[int]) -> bool:
    """True if a token only appears at sentence-position 0 (i.e., capitalized
    only because it starts a sentence, not because it's an entity)."""
    return bool(positions) and all(p == 0 for p in positions)


def passes_entity_filter(
    token: str,
    *,
    count: int,
    n_sessions: int,
    all_zero_positions: bool,
    min_count: int = 10,
    min_sessions: int = 3,
    min_len: int = 4,
    whitelist: frozenset[str] | set[str] | None = None,
) -> bool:
    """Apply Stage A noise filters. Return True iff the token survives.

    `whitelist` (typically loaded from synonyms.yaml) bypasses content
    blocklists (URL pieces / English fillers / Cyrillic sentence-starts /
    Slavic function words) AND the length gate, but still enforces
    frequency / session-breadth / non-zero-positions checks.
    """
    lower = token.lower()
    if whitelist and lower in whitelist:
        if not token or " " in token:
            return False
        if count < min_count or n_sessions < min_sessions:
            return False
        return not all_zero_positions
    if len(token) < min_len:
        return False
    if lower in URL_PIECES:
        return False
    if lower in ENGLISH_FILLERS:
        return False
    if lower in SLAVIC_FUNCTION_WORDS:
        return False
    if token in CYRILLIC_SENT_STARTS:
        return False
    if count < min_count:
        return False
    if n_sessions < min_sessions:
        return False
    return not all_zero_positions
