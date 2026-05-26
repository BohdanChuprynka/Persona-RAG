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
) -> bool:
    """Apply Stage A noise filters. Return True iff the token survives."""
    lower = token.lower()
    if len(token) < min_len:
        return False
    if lower in URL_PIECES:
        return False
    if lower in ENGLISH_FILLERS:
        return False
    if token in CYRILLIC_SENT_STARTS:
        return False
    if count < min_count:
        return False
    if n_sessions < min_sessions:
        return False
    return not all_zero_positions
