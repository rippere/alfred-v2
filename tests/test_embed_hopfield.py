"""Coverage for alfred.embed.hopfield.HopfieldRefiner: deterministic vector
math, no LLM/network calls involved."""
from __future__ import annotations

import numpy as np

from alfred.embed.hopfield import HopfieldRefiner


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def test_refine_returns_unit_normalized_vector_of_same_shape():
    refiner = HopfieldRefiner(iterations=3, beta=2.0)
    query = _unit(np.array([1.0, 0.0, 0.0]))
    candidates = np.stack([
        _unit(np.array([1.0, 0.1, 0.0])),
        _unit(np.array([0.0, 1.0, 0.0])),
        _unit(np.array([0.0, 0.0, 1.0])),
    ])

    refined = refiner.refine(query, candidates)

    assert refined.shape == (3,)
    assert np.isclose(np.linalg.norm(refined), 1.0, atol=1e-6)


def test_refine_converges_toward_single_matching_candidate():
    refiner = HopfieldRefiner(iterations=5, beta=4.0)
    target = _unit(np.array([1.0, 0.0, 0.0]))
    query = _unit(np.array([0.9, 0.1, 0.0]))
    candidates = np.stack([target])

    refined = refiner.refine(query, candidates)

    # A single-pattern store is a fixed attractor: the refined vector must
    # end up (near-)identical to that one stored pattern.
    assert np.allclose(refined, target, atol=1e-3)


def test_refine_pulls_query_closer_to_nearest_of_several_candidates():
    refiner = HopfieldRefiner(iterations=5, beta=8.0)
    query = _unit(np.array([1.0, 0.05, 0.0]))
    nearest = _unit(np.array([1.0, 0.0, 0.0]))
    far1 = _unit(np.array([0.0, 1.0, 0.0]))
    far2 = _unit(np.array([0.0, 0.0, 1.0]))
    candidates = np.stack([nearest, far1, far2])

    refined = refiner.refine(query, candidates)

    sim_before = float(query @ nearest)
    sim_after = float(refined @ nearest)
    assert sim_after >= sim_before, "refinement (high beta) must sharpen toward the nearest attractor"


def test_refine_is_deterministic_for_same_inputs():
    refiner = HopfieldRefiner(iterations=3, beta=2.0)
    query = _unit(np.array([0.3, 0.7, 0.1]))
    candidates = np.stack([
        _unit(np.array([1.0, 0.0, 0.0])),
        _unit(np.array([0.0, 1.0, 0.0])),
    ])

    r1 = refiner.refine(query, candidates)
    r2 = refiner.refine(query, candidates)

    assert np.array_equal(r1, r2)


def test_refine_zero_iterations_returns_normalized_input_unchanged():
    refiner = HopfieldRefiner(iterations=0, beta=2.0)
    query = _unit(np.array([0.3, 0.7, 0.1]))
    candidates = np.stack([_unit(np.array([1.0, 0.0, 0.0]))])

    refined = refiner.refine(query, candidates)

    assert np.allclose(refined, query)
