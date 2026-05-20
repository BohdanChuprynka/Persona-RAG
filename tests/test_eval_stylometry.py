from persona_rag.eval.stylometry import compute_features


def test_emoji_rate_and_len():
    f = compute_features("hi 🎉 yes!")
    assert f["emoji_rate"] > 0
    assert f["len_chars"] == len("hi 🎉 yes!")
    assert f["len_words"] == 3
