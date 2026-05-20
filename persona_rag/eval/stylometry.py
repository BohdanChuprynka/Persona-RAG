from __future__ import annotations

import re

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _is_emoji(c: str) -> bool:
    o = ord(c)
    return 0x1F300 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF


def compute_features(text: str) -> dict[str, float]:
    n_chars = len(text)
    if n_chars == 0:
        return {
            "len_chars": 0.0,
            "len_words": 0.0,
            "emoji_rate": 0.0,
            "caps_ratio": 0.0,
            "punct_density": 0.0,
            "avg_word_len": 0.0,
            "lexical_diversity": 0.0,
        }
    words = text.split()
    n_words = len(words)
    alpha = [c for c in text if c.isalpha()]
    caps = sum(1 for c in alpha if c.isupper())
    return {
        "len_chars": float(n_chars),
        "len_words": float(n_words),
        "emoji_rate": sum(1 for c in text if _is_emoji(c)) / n_chars,
        "caps_ratio": caps / len(alpha) if alpha else 0.0,
        "punct_density": len(_PUNCT.findall(text)) / max(n_words, 1),
        "avg_word_len": sum(len(w) for w in words) / max(n_words, 1),
        "lexical_diversity": len(set(words)) / max(n_words, 1),
    }


def mean_abs_deviation(generated: list[str], real: list[str]) -> dict[str, float]:
    gen_feats = [compute_features(t) for t in generated]
    real_feats = [compute_features(t) for t in real]
    out: dict[str, float] = {}
    if not gen_feats:
        return out
    for key in gen_feats[0]:
        g = sum(f[key] for f in gen_feats) / len(gen_feats)
        r = sum(f[key] for f in real_feats) / len(real_feats) if real_feats else 0.0
        out[key] = abs(g - r)
    return out
