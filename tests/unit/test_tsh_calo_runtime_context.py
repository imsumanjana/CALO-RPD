"""Measured, already-counted runtime topology-context invariants."""

from __future__ import annotations

import inspect

import numpy as np

from calo_rpd_studio.algorithms.calo.tsh_calo_runtime_context import (
    build_runtime_topology_policy_context,
    measured_scenario_descriptors,
)
from calo_rpd_studio.orpd.problem import ORPDProblem
from calo_rpd_studio.power_system.case_model import BR_STATUS, PD, QD
from calo_rpd_studio.robustness.scenario import Scenario


def _scenarios():
    def load(case):
        case.bus[:, PD] *= 1.2
        case.bus[:, QD] *= 1.1
        return case

    def outage(case):
        case.branch[0, BR_STATUS] = 0
        return case

    return [
        Scenario("base", 0.5),
        Scenario("load_stress", 0.25, load),
        Scenario("branch_out_0", 0.25, outage),
    ]


def test_runtime_context_uses_measured_scenario_stress_and_counted_base_solve(toy_case):
    problem = ORPDProblem(toy_case, scenarios=_scenarios())
    controls = np.full(problem.dimension, 0.5)
    _evaluation, counted = problem.evaluate_with_context(controls)

    runtime = build_runtime_topology_policy_context(np.linspace(0.0, 1.0, 32), problem, counted)
    descriptors = measured_scenario_descriptors(problem, counted)

    assert runtime.reference_scenario == "base"
    assert runtime.scenario_names == ("base", "load_stress", "branch_out_0")
    assert descriptors[0].load_stress == 1.0
    assert descriptors[1].load_stress == 1.15
    assert descriptors[1].aggregation_role == 1.0 / 3.0
    assert descriptors[2].contingency_stress > 0.0
    assert descriptors[2].aggregation_role == 1.0
    runtime.policy_state.validate()


def test_runtime_context_builder_contains_no_solver_call(toy_case):
    problem = ORPDProblem(toy_case)
    _evaluation, counted = problem.evaluate_with_context(np.full(problem.dimension, 0.5))
    source = inspect.getsource(build_runtime_topology_policy_context)

    assert "run_ac_power_flow" not in source
    build_runtime_topology_policy_context(np.zeros(32), problem, counted).policy_state.validate()


def test_unknown_scenario_role_is_explicitly_neutral_not_ood_claim(toy_case):
    problem = ORPDProblem(toy_case, scenarios=[Scenario("custom_observed", 1.0)])
    _evaluation, counted = problem.evaluate_with_context(np.full(problem.dimension, 0.5))
    descriptor = measured_scenario_descriptors(problem, counted)[0]

    assert descriptor.aggregation_role == 0.5
    assert descriptor.is_ood == 0.0
    assert descriptor.observed == 1.0


def test_structural_zero_reactive_load_is_encoded_as_unchanged(toy_case):
    toy_case.bus[:, QD] = 0.0
    problem = ORPDProblem(toy_case)
    _evaluation, counted = problem.evaluate_with_context(np.full(problem.dimension, 0.5))

    descriptor = measured_scenario_descriptors(problem, counted)[0]

    assert descriptor.load_stress == 1.0
