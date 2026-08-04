"""Change-E proposal, masking, lattice, accounting, and failure invariants."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from calo_rpd_studio.algorithms.calo.tsh_calo_physics_repair import (
    PhysicsRepairConfig,
    PhysicsRepairContext,
    PhysicsRepairOperator,
    PhysicsRepairStatus,
    evaluate_physics_repair_proposal,
    physics_repair_context_from_counted_evaluation,
    physics_repair_context_is_usable,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_policy import GroupActionMask
from calo_rpd_studio.algorithms.calo.tsh_calo_schema import (
    DEFAULT_TSH_CALO_FEATURES,
    N_OPERATORS,
)
from calo_rpd_studio.orpd.decision_variables import DecisionVariable, VariableKind
from calo_rpd_studio.orpd.mixed_variable_handler import decode_discrete
from calo_rpd_studio.orpd.problem import ORPDProblem
from calo_rpd_studio.power_system.case_loader import CaseLoader


def _variables():
    return [
        DecisionVariable("continuous", 0.0, 1.0),
        DecisionVariable("discrete", 0.0, 1.0, VariableKind.DISCRETE, (0.0, 0.5, 1.0)),
    ]


def _context(**changes) -> PhysicsRepairContext:
    values = dict(
        converged=True,
        available_from_counted_evaluation=True,
        source_evaluation_id="fe-17",
        ac_jacobian=np.eye(2),
        control_sensitivity=np.eye(2),
        constraint_residual=np.asarray([0.2, -0.1]),
        condition_number=1.0,
    )
    values.update(changes)
    return PhysicsRepairContext(**values)


def test_counted_orpd_linearization_adapts_without_an_evaluator_call(toy_case):
    problem = ORPDProblem(toy_case)
    _evaluation, counted = problem.evaluate_with_context(
        np.full(problem.dimension, 0.5), retain_control_linearization=True
    )

    context = physics_repair_context_from_counted_evaluation(counted)

    assert context is not None
    assert context.converged is True
    assert context.available_from_counted_evaluation is True
    assert len(context.source_evaluation_id) == 64
    assert context.control_sensitivity.shape[1] == problem.dimension
    assert "run_ac_power_flow" not in inspect.getsource(
        physics_repair_context_from_counted_evaluation
    )


def test_context_exposure_requires_residual_sensitivity_and_conditioning():
    assert physics_repair_context_is_usable(_context(), maximum_condition_number=1e10)
    assert not physics_repair_context_is_usable(
        _context(constraint_residual=np.zeros(2)), maximum_condition_number=1e10
    )
    assert not physics_repair_context_is_usable(
        _context(control_sensitivity=np.zeros((2, 2))), maximum_condition_number=1e10
    )
    assert not physics_repair_context_is_usable(
        _context(condition_number=1e12), maximum_condition_number=1e10
    )


def test_real_case30_counted_context_proposes_inside_trust_and_lattice():
    problem = ORPDProblem(CaseLoader.load("case30"))
    current = np.full(problem.dimension, 0.5)
    evaluation, counted = problem.evaluate_with_context(
        current,
        retain_control_linearization=True,
    )
    context = physics_repair_context_from_counted_evaluation(counted)
    operator = PhysicsRepairOperator(PhysicsRepairConfig(enabled=True, trust_radius=0.08))

    assert evaluation.violation > 0.0
    assert physics_repair_context_is_usable(context, maximum_condition_number=1e10)
    proposal = operator.propose(current, context, problem.decoder.variables)

    assert proposal.status is PhysicsRepairStatus.PROPOSED
    assert proposal.step_norm <= 0.08 + 1e-12
    assert proposal.hidden_solver_calls == proposal.evaluator_calls == 0
    assert proposal.declares_feasibility is False
    for index, variable in enumerate(problem.decoder.variables):
        if variable.values:
            assert decode_discrete(proposal.candidate[index], variable.values) in variable.values


def test_physics_repair_and_seventh_operator_are_disabled_by_default():
    assert DEFAULT_TSH_CALO_FEATURES.physics_repair is False
    proposal = PhysicsRepairOperator().propose(np.asarray([0.5, 0.5]), _context(), _variables())
    mask = GroupActionMask.from_control_groups([0, 1, 2])

    assert N_OPERATORS == 7
    assert proposal.status is PhysicsRepairStatus.MASKED
    assert proposal.candidate is None
    assert not bool(mask.allowed[:, 6].any())


def test_repair_uses_supplied_linearization_is_trust_bounded_and_snaps_lattice():
    operator = PhysicsRepairOperator(PhysicsRepairConfig(enabled=True, trust_radius=0.1))
    current = np.asarray([0.5, 0.5])
    first = operator.propose(current, _context(), _variables())
    second = operator.propose(current, _context(), _variables())

    assert first.status is PhysicsRepairStatus.PROPOSED
    assert first.declares_feasibility is False
    assert first.hidden_solver_calls == 0
    assert first.evaluator_calls == 0
    assert first.step_norm <= 0.1 + 1e-12
    assert first.candidate[1] in {0.0, 0.5, 1.0}
    np.testing.assert_allclose(first.candidate, second.candidate, rtol=0.0, atol=0.0)
    assert "run_ac_power_flow" not in inspect.getsource(PhysicsRepairOperator)


@pytest.mark.parametrize(
    ("context", "reason"),
    [
        (None, "unavailable"),
        (_context(converged=False), "did not converge"),
        (_context(available_from_counted_evaluation=False), "unavailable"),
        (_context(condition_number=1e15), "ill-conditioned"),
        (_context(ac_jacobian=None), "missing"),
    ],
)
def test_repair_masks_unavailable_unconverged_or_ill_conditioned_context(context, reason):
    proposal = PhysicsRepairOperator(PhysicsRepairConfig(enabled=True)).propose(
        np.asarray([0.5, 0.5]), context, _variables()
    )

    assert proposal.status is PhysicsRepairStatus.MASKED
    assert proposal.candidate is None
    assert reason in proposal.reason


def test_singular_supplied_jacobian_records_failure_without_candidate():
    proposal = PhysicsRepairOperator(PhysicsRepairConfig(enabled=True)).propose(
        np.asarray([0.5, 0.5]),
        _context(ac_jacobian=np.zeros((2, 2))),
        _variables(),
    )

    assert proposal.status is PhysicsRepairStatus.FAILED
    assert proposal.candidate is None
    assert "linear algebra failed" in proposal.reason


def test_repair_proposal_requires_one_explicit_trusted_evaluation():
    proposal = PhysicsRepairOperator(PhysicsRepairConfig(enabled=True)).propose(
        np.asarray([0.5, 0.5]), _context(), _variables()
    )
    calls = []

    def evaluator(population):
        calls.append(np.asarray(population).copy())
        return [{"trusted": True}]

    counted = evaluate_physics_repair_proposal(proposal, evaluator, remaining_evaluations=1)

    assert len(calls) == 1
    assert calls[0].shape == (1, 2)
    assert counted.requested_evaluations == counted.completed_evaluations == 1
    assert counted.evaluation == {"trusted": True}
    assert counted.source_evaluation_id == "fe-17"


def test_repair_evaluation_fails_closed_on_budget_or_incomplete_evaluator():
    proposal = PhysicsRepairOperator(PhysicsRepairConfig(enabled=True)).propose(
        np.asarray([0.5, 0.5]), _context(), _variables()
    )
    with pytest.raises(ValueError, match="remaining FE budget"):
        evaluate_physics_repair_proposal(
            proposal, lambda _population: [object()], remaining_evaluations=0
        )
    with pytest.raises(RuntimeError, match="did not complete"):
        evaluate_physics_repair_proposal(proposal, lambda _population: [], remaining_evaluations=1)
