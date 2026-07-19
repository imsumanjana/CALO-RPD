import warnings

import numpy as np

from calo_rpd_studio.algorithms.calo.operator_credit import OperatorCredit, blend_probabilities
from calo_rpd_studio.algorithms.calo.success_memory import SuccessMemory


def test_success_memory_sampling_survives_nonfinite_and_extreme_rewards():
    memory = SuccessMemory(capacity=8, decay=0.97)
    rng = np.random.default_rng(123)

    memory.add(np.ones(4), 0, objective_gain=np.nan, feasibility_gain=np.inf)
    memory.add(np.full(4, 2.0), 1, objective_gain=1e308, feasibility_gain=1e308)
    memory.add(np.full(4, 3.0), 2, objective_gain=0.0, feasibility_gain=0.0)
    memory.add(np.array([np.nan, 0.0, 0.0, 0.0]), 3, objective_gain=1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for _ in range(50):
            sampled = memory.sample_direction(4, rng)
            assert sampled.shape == (4,)
            assert np.all(np.isfinite(sampled))

    assert len(memory) == 3  # the non-finite direction is rejected
    assert np.all(np.isfinite(memory.success_rates()))


def test_success_memory_returns_zero_for_incompatible_dimension():
    memory = SuccessMemory(capacity=3)
    memory.add(np.ones(4), 0, objective_gain=1.0)
    sampled = memory.sample_direction(5, np.random.default_rng(1))
    assert np.array_equal(sampled, np.zeros(5))
    assert np.array_equal(memory.direction(5), np.zeros(5))


def test_operator_credit_cannot_be_poisoned_by_nonfinite_rewards():
    credit = OperatorCredit(n_operators=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        credit.update(0, np.nan, True)
        credit.update(1, np.inf, True)
        credit.update(2, 1e308, True)
        credit.update(3, -np.inf, False)

    probabilities = credit.probabilities()
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities > 0.0)
    assert np.isclose(probabilities.sum(), 1.0)


def test_probability_blend_sanitises_nonfinite_inputs():
    ai = np.array([np.nan, np.inf, -1.0, 0.0, 2.0, 3.0])
    credit = np.array([1.0, 2.0, np.nan, np.inf, 0.0, 1.0])
    probabilities = blend_probabilities(ai, credit)
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities > 0.0)
    assert np.isclose(probabilities.sum(), 1.0)
