from persona_rag.index.bm25_store import build_bm25, score_bm25


def test_bm25_returns_higher_score_for_overlap():
    corpus = ["the quick brown fox", "lazy dog", "the brown bear"]
    bm = build_bm25(corpus)
    scores = score_bm25(bm, "brown fox")
    assert scores[0] > scores[1]
