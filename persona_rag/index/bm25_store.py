from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

_TOKENIZER = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKENIZER.findall(text.lower())


def build_bm25(corpus: list[str]) -> BM25Okapi:
    return BM25Okapi([tokenize(t) for t in corpus])


def score_bm25(bm25: BM25Okapi, query: str) -> list[float]:
    return list(bm25.get_scores(tokenize(query)))


def save(bm25: BM25Okapi, ids: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids}, f)


def load(path: Path) -> tuple[BM25Okapi, list[str]]:
    with path.open("rb") as f:
        data: dict[str, Any] = pickle.load(f)
    return data["bm25"], data["ids"]
