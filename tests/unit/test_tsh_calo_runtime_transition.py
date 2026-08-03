"""TSH-CALO per-learner group action and physics-proposal transition invariants."""

from __future__ import annotations

import numpy as np
import pytest

from calo_rpd_studio.algorithms.calo.training import SyntheticCALOEnvironment
from calo_rpd_studio.algorithms.calo.tsh_calo_physics_repair import (
    PhysicsRepairConfig,
    PhysicsRepairContext,
    PhysicsRepairOperator,
)
from calo_rpd_studio.algorithms.calo import tsh_calo_transition_kernel
from calo_rpd_studio.algorithms.calo.tsh_calo_transition_kernel import generate_tsh_offspring


def _arguments(seed: int = 41):
    rng = np.random.default_rng(seed)
    environment = SyntheticCALOEnvironment(rng, stage=1, population_size=8)
    variables = environment.problem.variables
    population_size = environment.population_size
    return dict(
        population=environment.population,
        evaluations=environment.evaluations,
        personal_best=environment.personal_best,
        rng=rng,
        dimension=environment.problem.dimension,
        variables=variables,
        quality_order=np.arange(population_size),
        contexts=np.arange(population_size) % 4,
        learner_groups=np.arange(population_size) % 3,
        learned_lanes=np.ones(population_size, dtype=np.int8),
        global_regime=2,
        learner_operators=np.arange(population_size) % 6,
        group_parameter_actions=np.full((3, 6), 0.5),
        memory=environment.memory,
        hpem=environment.hpem,
        feasible_archive=environment.feasible_archive,
        boundary_archive=environment.boundary_archive,
        credit=environment.credit,
        group_intelligence=environment.group_intelligence,
        precision=environment.precision,
        precision_active=False,
        precision_fraction=0.0,
        forced_recovery=set(),
        consensus=0.0,
        environment_deterministic=True,
    )


def test_group_conditioned_candidate_generation_is_seeded_and_per_learner():
    first = generate_tsh_offspring(**_arguments(41))
    second = generate_tsh_offspring(**_arguments(41))

    np.testing.assert_allclose(
        first.candidates.offspring, second.candidates.offspring, rtol=0, atol=0
    )
    np.testing.assert_array_equal(first.candidates.assigned_groups, np.arange(8) % 3)
    np.testing.assert_array_equal(first.candidates.assigned_operators, np.arange(8) % 6)
    assert first.group_parameter_values.shape == (3, 6)
    assert all(trace is None for trace in first.physics_repair_proposals)


def test_physics_operator_uses_supplied_counted_context_and_remains_group_focused():
    arguments = _arguments(53)
    dimension = arguments["dimension"]
    operators = np.zeros(8, dtype=int)
    operators[0] = 6
    arguments["learner_operators"] = operators
    contexts = [None] * 8
    contexts[0] = PhysicsRepairContext(
        converged=True,
        available_from_counted_evaluation=True,
        source_evaluation_id="fe-1",
        ac_jacobian=np.eye(dimension),
        control_sensitivity=np.eye(dimension),
        constraint_residual=np.linspace(0.1, 0.2, dimension),
        condition_number=1.0,
    )
    arguments["physics_contexts"] = tuple(contexts)
    arguments["physics_repair_operator"] = PhysicsRepairOperator(
        PhysicsRepairConfig(enabled=True, trust_radius=0.05)
    )

    batch = generate_tsh_offspring(**arguments)

    proposal = batch.physics_repair_proposals[0]
    assert proposal.source_evaluation_id == "fe-1"
    assert proposal.hidden_solver_calls == proposal.evaluator_calls == 0
    group_mask = arguments["group_intelligence"].mask(0, dimension)
    np.testing.assert_allclose(
        batch.candidates.offspring[0][~group_mask], arguments["population"][0][~group_mask]
    )


def test_shielded_physics_selection_fails_closed_if_context_disappears():
    arguments = _arguments(61)
    operators = np.zeros(8, dtype=int)
    operators[0] = 6
    arguments["learner_operators"] = operators
    arguments["physics_repair_operator"] = PhysicsRepairOperator(PhysicsRepairConfig(enabled=True))
    with pytest.raises(RuntimeError, match="became unavailable"):
        generate_tsh_offspring(**arguments)


def test_action_vectors_and_parameters_cannot_bypass_the_versioned_abi():
    arguments = _arguments()
    arguments["learner_groups"] = np.zeros(7, dtype=int)
    with pytest.raises(ValueError, match="align"):
        generate_tsh_offspring(**arguments)
    arguments = _arguments()
    arguments["group_parameter_actions"] = np.full((3, 6), 1.5)
    with pytest.raises(ValueError, match="within"):
        generate_tsh_offspring(**arguments)


def test_tsh_completion_reserves_a_distinct_precision_memory_channel(monkeypatch):
    observed = {}

    def capture(**kwargs):
        observed.update(kwargs)
        return "transition"

    monkeypatch.setattr(tsh_calo_transition_kernel, "complete_transition", capture)

    assert tsh_calo_transition_kernel.complete_tsh_transition(population="sentinel") == "transition"
    assert observed == {"population": "sentinel", "precision_memory_operator": 7}
