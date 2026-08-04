"""Already-counted ORPD solver-context retention invariants for TSH-CALO."""

from __future__ import annotations

import numpy as np

import calo_rpd_studio.orpd.problem as problem_module
from calo_rpd_studio.orpd.problem import ORPDProblem, ORPDProblemConfig
from calo_rpd_studio.orpd.variable_decoder import ORPDVariableConfig
from calo_rpd_studio.power_system.case_loader import CaseLoader
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

    evaluation, context = problem.evaluate_with_context(controls, retain_control_linearization=True)

    assert calls == 2
    assert len(context.scenarios) == 2
    assert context.primary_converged_power_flow() is context.scenarios[0].power_flow
    np.testing.assert_allclose(context.normalized_controls, controls)
    assert "power_flow" not in evaluation.metadata
    linearization = context.primary_control_linearization()
    state_dimension = linearization.jacobian.shape[0]
    assert linearization.schema_version == "calo-rpd-counted-control-linearization-v1"
    assert len(linearization.source_evaluation_id) == 64
    assert linearization.control_sensitivity.shape == (state_dimension, problem.dimension)
    assert linearization.constraint_residual.shape == (state_dimension,)
    assert np.all(np.isfinite(linearization.control_sensitivity))
    assert np.all(np.isfinite(linearization.constraint_residual))
    assert np.isfinite(linearization.condition_number)
    assert "trusted FE" in linearization.sensitivity_semantics


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
    assert context.scenarios[0].control_linearization is None
    try:
        context.primary_control_linearization()
    except ValueError as exc:
        assert "No counted control linearization" in str(exc)
    else:
        raise AssertionError("Default counted topology context must not pay for Change E")


def test_counted_linearization_identity_and_values_are_deterministic(toy_case):
    problem = ORPDProblem(toy_case)
    controls = np.linspace(0.2, 0.8, problem.dimension)

    _first_evaluation, first = problem.evaluate_with_context(
        controls, retain_control_linearization=True
    )
    _second_evaluation, second = problem.evaluate_with_context(
        controls, retain_control_linearization=True
    )
    first_linearization = first.primary_control_linearization()
    second_linearization = second.primary_control_linearization()

    assert first_linearization.source_evaluation_id == second_linearization.source_evaluation_id
    np.testing.assert_allclose(
        first_linearization.control_sensitivity,
        second_linearization.control_sensitivity,
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        first_linearization.constraint_residual,
        second_linearization.constraint_residual,
        rtol=0,
        atol=0,
    )


def test_case30_analytic_control_sensitivity_matches_power_flow_finite_difference():
    problem = ORPDProblem(
        CaseLoader.load("case30"),
        ORPDProblemConfig(
            variables=ORPDVariableConfig(
                discrete_transformer_taps=False,
                discrete_shunts=False,
            )
        ),
    )
    controls = np.linspace(0.35, 0.65, problem.dimension)
    _evaluation, counted = problem.evaluate_with_context(
        controls,
        retain_control_linearization=True,
    )
    power_flow = counted.primary_converged_power_flow()
    newton = power_flow.linearization
    retained = counted.primary_control_linearization()
    step = 1e-5

    for column in range(problem.dimension):
        upper = controls.copy()
        lower = controls.copy()
        upper[column] += step
        lower[column] -= step
        _upper_evaluation, upper_context = problem.evaluate_with_context(upper)
        _lower_evaluation, lower_context = problem.evaluate_with_context(lower)
        upper_pf = upper_context.primary_converged_power_flow()
        lower_pf = lower_context.primary_converged_power_flow()
        upper_state = np.r_[
            np.angle(upper_pf.voltage)[newton.pvpq],
            upper_pf.vm_pu[newton.pq],
        ]
        lower_state = np.r_[
            np.angle(lower_pf.voltage)[newton.pvpq],
            lower_pf.vm_pu[newton.pq],
        ]
        finite_difference = (upper_state - lower_state) / (2.0 * step)
        np.testing.assert_allclose(
            retained.control_sensitivity[:, column],
            finite_difference,
            rtol=1e-4,
            atol=5e-7,
        )


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
