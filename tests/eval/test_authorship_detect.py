# ruff: noqa: RUF001
"""Unit tests for the validated author-detector core (review fix #2).

Pure sklearn, no network/model download. The live data run is
`scripts/authorship_validation.py`; here we only exercise the scoring core on toy,
trivially-separable data.
"""

from __future__ import annotations

from persona_rag.eval.authorship_detect import (
    acceptance,
    p_owner,
    train_detector,
    validate_auc,
)

# Owner = Cyrillic, others = Latin -> trivially separable by char n-grams.
OWNER = ["привіт як справи", "ок норм", "та нема за що", "ще побачимось пізніше", "ага точно"]
OTHER = ["hello how are you", "yeah sure thing", "see you later", "okay sounds good", "no worries"]


def test_detector_separates_and_probs_in_range() -> None:
    model = train_detector(OWNER, OTHER)
    probs = p_owner(model, OWNER + OTHER)
    assert len(probs) == len(OWNER) + len(OTHER)
    assert all(0.0 <= p <= 1.0 for p in probs)
    # Owner texts should score higher than the others on average.
    owner_mean = sum(probs[: len(OWNER)]) / len(OWNER)
    other_mean = sum(probs[len(OWNER) :]) / len(OTHER)
    assert owner_mean > other_mean


def test_validate_auc_high_on_separable() -> None:
    model = train_detector(OWNER, OTHER)
    v = validate_auc(model, OWNER, OTHER, n_boot=200)
    assert v["auc"] > 0.9
    assert 0.0 <= v["auc_ci"][0] <= v["auc_ci"][1] <= 1.0
    assert v["n_owner"] == len(OWNER) and v["n_other"] == len(OTHER)


def test_acceptance_keys_and_empty() -> None:
    model = train_detector(OWNER, OTHER)
    a = acceptance(model, ["привіт", "ще трохи"])
    assert set(a) == {"n", "mean_p_owner", "accept_rate"}
    assert a["n"] == 2
    empty = acceptance(model, ["", "   "])
    assert empty["n"] == 0


def test_p_owner_empty_list() -> None:
    model = train_detector(OWNER, OTHER)
    assert p_owner(model, []) == []
