from __future__ import annotations

import numpy as np

from calo_rpd_studio.algorithms.calo.archives import ConstraintBoundaryArchive, FeasibleEliteArchive
from calo_rpd_studio.algorithms.calo.environmental_selection import (
    epsilon_better,
    environmental_select,
)
from calo_rpd_studio.algorithms.calo.operator_credit import OperatorCredit, blend_probabilities
from calo_rpd_studio.algorithms.calo.learning_operators import mixed_variable_neighbourhood
from calo_rpd_studio.orpd.problem import Evaluation
from calo_rpd_studio.orpd.decision_variables import DecisionVariable, VariableKind


def _ev(value, violation, components=None):
    return Evaluation(
        value=float(value),
        feasible=violation <= 1e-12,
        violation=float(violation),
        metadata={"constraint_components": components or {}},
    )


def test_epsilon_rule_allows_near_feasible_objective_comparison():
    a = _ev(1.0, 0.03)
    b = _ev(2.0, 0.04)
    assert epsilon_better(a, b, epsilon=0.05)
    assert not epsilon_better(a, _ev(0.5, 0.0), epsilon=0.05)


def test_dual_archives_preserve_feasible_and_diverse_boundary_candidates():
    rng = np.random.default_rng(4)
    pop = rng.random((12, 5))
    evaluations = []
    for index in range(12):
        violation = 0.0 if index < 2 else 0.01 + 0.01 * index
        evaluations.append(
            _ev(
                5.0 - 0.1 * index,
                violation,
                {
                    "bus_voltage": violation if index % 2 else 0.0,
                    "generator_q": 0.0 if index % 2 else violation,
                },
            )
        )
    feasible = FeasibleEliteArchive(4)
    boundary = ConstraintBoundaryArchive(6)
    feasible.update(pop, evaluations)
    boundary.update(pop, evaluations)
    assert len(feasible) == 2
    assert 1 <= len(boundary) <= 6
    assert boundary.sample(rng, pop[0]).shape == (5,)


def test_environmental_selection_returns_requested_population_size():
    rng = np.random.default_rng(1)
    vectors = rng.random((20, 4))
    evaluations = [_ev(i, 0.0 if i < 3 else i / 100) for i in range(20)]
    selected, selected_ev = environmental_select(vectors, evaluations, 8, epsilon=0.05)
    assert selected.shape == (8, 4)
    assert len(selected_ev) == 8


def test_online_operator_credit_is_normalized_and_blend_is_valid():
    credit = OperatorCredit(6)
    for _ in range(5):
        credit.update(2, 1.0, True)
    probs = credit.probabilities()
    assert np.isclose(probs.sum(), 1.0)
    blended = blend_probabilities(np.full(6, 1 / 6), probs, 0.5)
    assert np.isclose(blended.sum(), 1.0)
    assert blended[2] == blended.max()


def test_mixed_variable_neighbourhood_moves_discrete_variable_on_lattice():
    rng = np.random.default_rng(11)
    variables = [
        DecisionVariable("continuous", 0, 1, VariableKind.CONTINUOUS),
        DecisionVariable("tap", 0.9, 1.1, VariableKind.DISCRETE, (0.9, 0.95, 1.0, 1.05, 1.1)),
    ]
    x = np.asarray([0.5, 0.5])
    candidates = [mixed_variable_neighbourhood(x, variables, rng, 0.02, 1) for _ in range(30)]
    assert all(np.all((candidate >= 0) & (candidate <= 1)) for candidate in candidates)
    assert any(not np.allclose(candidate, x) for candidate in candidates)


def test_training_rejects_final_benchmark_leakage_by_default(tmp_path):
    import pytest
    from calo_rpd_studio.algorithms.calo.training import TrainingConfig, train_policy

    with pytest.raises(ValueError, match="Final publication benchmark cases"):
        train_policy(
            TrainingConfig(
                epochs=1,
                episodes_per_epoch=1,
                horizon=2,
                population_size=4,
                development_cases=("case118",),
            ),
            tmp_path / "blocked.pt",
        )


def test_real_ppo_training_smoke_saves_core_v2_checkpoint(tmp_path):
    import torch
    from calo_rpd_studio.algorithms.calo.training import TrainingConfig, train_policy

    path, history = train_policy(
        TrainingConfig(
            epochs=1,
            episodes_per_epoch=1,
            horizon=2,
            population_size=4,
            ppo_epochs=1,
            minibatch_size=4,
            hidden_dim=16,
            seed=17,
        ),
        tmp_path / "policy.pt",
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["metadata"]["training_method"] == "PPO"
    assert payload["metadata"]["calo_core"] == "v5.0"
    assert payload["metadata"]["state_schema_version"] == "calo-state-v5.9-32"
    assert payload["metadata"]["action_schema_version"] == "calo-action-v5.9-raw-global-4r-6o-6p"
    assert payload["metadata"]["final_publication_benchmarks_used_for_training"] is False
    assert history


def test_resource_aware_parallel_training_records_execution_metadata(tmp_path):
    import torch
    from calo_rpd_studio.algorithms.calo.training import TrainingConfig, train_policy

    path, history = train_policy(
        TrainingConfig(
            epochs=1,
            episodes_per_epoch=2,
            horizon=2,
            population_size=4,
            ppo_epochs=1,
            minibatch_size=4,
            hidden_dim=16,
            rollout_workers=2,
            ppo_device="cpu",
            seed=23,
        ),
        tmp_path / "parallel_policy.pt",
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    execution = payload["metadata"]["execution"]
    assert execution["rollout_workers"] == 2
    assert execution["ppo_device"] == "cpu"
    assert history[0]["transitions"] == 4


def test_cuda_request_fails_cleanly_when_cuda_unavailable():
    import pytest
    import torch
    from calo_rpd_studio.algorithms.calo.training import _resolve_training_device

    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA training was requested"):
            _resolve_training_device("cuda")
