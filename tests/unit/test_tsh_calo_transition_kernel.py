"""Change-A invariants for the canonical frozen-CALO transition kernel."""

from __future__ import annotations

import numpy as np
import pytest

from calo_rpd_studio.algorithms.calo import optimizer, training, transition_kernel


def test_runtime_and_training_import_one_transition_authority():
    assert optimizer.generate_offspring is transition_kernel.generate_offspring
    assert training.generate_offspring is transition_kernel.generate_offspring
    assert optimizer.complete_transition is transition_kernel.complete_transition
    assert training.complete_transition is transition_kernel.complete_transition
    assert optimizer.evaluate_candidates is transition_kernel.evaluate_candidates
    assert training.evaluate_candidates is transition_kernel.evaluate_candidates


def test_candidate_evaluation_reports_exact_and_partial_batches_without_padding():
    candidates = np.zeros((3, 2), dtype=float)
    complete = transition_kernel.evaluate_candidates(
        candidates, lambda rows: list(range(len(rows)))
    )
    partial = transition_kernel.evaluate_candidates(candidates, lambda _rows: ["only-one"])

    assert complete.requested == complete.completed == 3
    assert complete.complete is True
    assert partial.requested == 3
    assert partial.completed == 1
    assert partial.complete is False
    assert partial.evaluations == ["only-one"]


def test_candidate_evaluation_rejects_hidden_overcount():
    candidates = np.zeros((2, 1), dtype=float)
    with pytest.raises(RuntimeError, match="3 results for 2 requested"):
        transition_kernel.evaluate_candidates(candidates, lambda _rows: [1, 2, 3])


@pytest.mark.parametrize(
    ("global_regime", "context", "expected"),
    [(0, 0, 1), (0, 3, 3), (3, 2, 1), (2, 1, 2)],
)
def test_individual_regime_mapping_is_frozen(global_regime, context, expected):
    assert transition_kernel.individual_regime(global_regime, context) == expected


def test_invalid_probability_vector_has_deterministic_uniform_fallback():
    values = np.asarray([np.nan, np.inf, -1.0, 0.0])
    np.testing.assert_array_equal(
        transition_kernel.normalise_probabilities(values),
        np.full(4, 0.25),
    )
