"""Already-counted ORPD solver-context retention invariants for TSH-CALO."""

from __future__ import annotations

import numpy as np

import calo_rpd_studio.orpd.problem as problem_module
from calo_rpd_studio.orpd.problem import ORPDProblem
from calo_rpd_studio.robustness.scenario import Scenario


def test_context_retention_uses_exactly_the_existing_scenario_solver_calls(monkeypatch, toy_case):
    calls = 0
    original = problem_module.run_ac_power_flow

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(problem_module, "run_ac_power_flow", counted)
    problem = ORPDProblem(toy_case, scenarios=[Scenario("base"), Scenario("replay")])
    controls = np.full(problem.dimension, 0.5)

    evaluation, context = problem.evaluate_with_context(controls)

    assert calls == 2
    assert len(context.scenarios) == 2
    assert context.primary_converged_power_flow() is context.scenarios[0].power_flow
    np.testing.assert_allclose(context.normalized_controls, controls)
    assert "power_flow" not in evaluation.metadata


def test_ordinary_evaluate_remains_value_equivalent_and_does_not_retain_context(toy_case):
    problem = ORPDProblem(toy_case)
    controls = np.linspace(0.1, 0.9, problem.dimension)

    ordinary = problem.evaluate(controls)
    retained, context = problem.evaluate_with_context(controls)

    assert ordinary.value == retained.value
    assert ordinary.feasible == retained.feasible
    assert ordinary.violation == retained.violation
    assert ordinary.components == retained.components
    assert ordinary.physical_controls == retained.physical_controls
    assert ordinary.metadata == retained.metadata
    assert len(context.scenarios) == 1


def test_context_fails_closed_when_no_counted_scenario_converged(monkeypatch, toy_case):
    problem = ORPDProblem(toy_case)
    _evaluation, context = problem.evaluate_with_context(np.full(problem.dimension, 0.5))
    context.scenarios[0].power_flow.converged = False

    try:
        context.primary_converged_power_flow()
    except ValueError as exc:
        assert "No already-counted converged" in str(exc)
    else:
        raise AssertionError("Missing converged context must fail closed")
