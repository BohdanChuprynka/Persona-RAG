from __future__ import annotations

import itertools
import statistics
from collections import Counter
from collections.abc import Iterable

from persona_rag.models import PersonaTurn, StyleAnchors


def _bigrams(text: str) -> list[str]:
    words = text.lower().split()
    return [f"{a} {b}" for a, b in itertools.pairwise(words)]


def compute_anchors(turns: Iterable[PersonaTurn]) -> StyleAnchors:
    turns_list = list(turns)
    if not turns_list:
        return StyleAnchors(
            avg_len_chars=0,
            median_len_chars=0,
            emoji_rate_per_char=0,
            lang_distribution={},
            top_bigrams=[],
            n_turns=0,
            primary_language="en",
        )
    lens = [t.your_reply_len_chars for t in turns_list]
    emoji_total = sum(t.your_reply_emoji_count for t in turns_list)
    char_total = sum(lens) or 1
    lang_counts = Counter(t.language for t in turns_list)
    total = sum(lang_counts.values())
    lang_dist = {k: v / total for k, v in lang_counts.items()}
    primary = max(lang_dist.items(), key=lambda x: x[1])[0] if lang_dist else "en"
    bigram_counter: Counter[str] = Counter()
    for t in turns_list:
        bigram_counter.update(_bigrams(t.your_reply))
    return StyleAnchors(
        avg_len_chars=statistics.mean(lens),
        median_len_chars=statistics.median(lens),
        emoji_rate_per_char=emoji_total / char_total,
        lang_distribution=lang_dist,
        top_bigrams=[bg for bg, _ in bigram_counter.most_common(10)],
        n_turns=len(turns_list),
        primary_language=primary,
    )
