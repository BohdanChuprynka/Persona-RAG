"""Pure-math tests for the style-embedding 'is this me?' scorer.

The model-loading/encode shell needs network + torch, so we test only the
deterministic centroid + cosine helpers here. The full scorer is exercised by
scripts/eval_persona.py once the embedding model is available.
"""

from __future__ import annotations

import math

from persona_rag.eval.authorship import centroid, mean_cosine_to_ref


def test_centroid_of_identical_unit_vectors_is_that_unit_vector():
    assert centroid([[1.0, 0.0], [1.0, 0.0]]) == [1.0, 0.0]


def test_centroid_is_l2_normalized():
    c = centroid([[2.0, 0.0], [0.0, 2.0]])
    # mean is (1,1); normalized is (1/sqrt2, 1/sqrt2)
    assert math.isclose(c[0], 1 / math.sqrt(2), rel_tol=1e-9)
    assert math.isclose(c[1], 1 / math.sqrt(2), rel_tol=1e-9)
    assert math.isclose(math.hypot(*c), 1.0, rel_tol=1e-9)


def test_mean_cosine_identical_direction_is_one():
    assert math.isclose(mean_cosine_to_ref([[3.0, 0.0]], [1.0, 0.0]), 1.0, rel_tol=1e-9)


def test_mean_cosine_orthogonal_is_zero():
    assert math.isclose(mean_cosine_to_ref([[0.0, 5.0]], [1.0, 0.0]), 0.0, abs_tol=1e-9)


def test_mean_cosine_averages_over_vectors():
    # one identical (cos 1), one orthogonal (cos 0) -> mean 0.5
    got = mean_cosine_to_ref([[1.0, 0.0], [0.0, 1.0]], [1.0, 0.0])
    assert math.isclose(got, 0.5, rel_tol=1e-9)


def test_empty_inputs_are_safe():
    assert centroid([]) == []
    assert mean_cosine_to_ref([], [1.0, 0.0]) == 0.0
    assert mean_cosine_to_ref([[1.0, 0.0]], []) == 0.0
