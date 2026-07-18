from __future__ import annotations

import numpy as np
import pytest
import torch

from calo_rpd_studio.accelerated.torch_orpd import AcceleratedORPDProblem, parity_check
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig, RobustScenarioSettings
from calo_rpd_studio.experiments.experiment_runner import build_problem, build_scenarios
from calo_rpd_studio.orpd.problem import ORPDProblem
from calo_rpd_studio.orpd.variable_decoder import ORPDVariableConfig, ORPDVariableDecoder
from calo_rpd_studio.power_system.ac_power_flow import run_ac_power_flow
from calo_rpd_studio.power_system.case_loader import CaseLoader
from calo_rpd_studio.power_system.case_model import BR_R, BR_X, BS
from calo_rpd_studio.power_system.case_validation import validate_case
from calo_rpd_studio.robustness.cvar import weighted_cvar, weighted_cvar_torch


def test_weighted_cvar_fractional_tail_exact_and_cross_backend():
    values = np.array([0.0, 100.0])
    weights = np.array([0.96, 0.04])
    assert weighted_cvar(values, weights, 0.95) == pytest.approx(80.0)
    torch_value = weighted_cvar_torch(
        torch.tensor(values, dtype=torch.float64),
        torch.tensor(weights, dtype=torch.float64),
        0.95,
    )
    assert float(torch_value) == pytest.approx(80.0)


@pytest.mark.parametrize(
    ("values", "weights", "alpha"),
    [([], [], 0.95), ([1.0], [-1.0], 0.95), ([1.0], [0.0], 0.95), ([1.0], [1.0], 1.0)],
)
def test_weighted_cvar_rejects_invalid_inputs(values, weights, alpha):
    with pytest.raises((ValueError, RuntimeError)):
        weighted_cvar(values, weights, alpha)


@pytest.mark.parametrize("case_name", ["case30", "case57", "case118", "case300"])
def test_all_publication_cases_have_explicit_formulation_manifest_and_converged_base(case_name):
    case = CaseLoader.load(case_name)
    decoder = ORPDVariableDecoder(case, ORPDVariableConfig())
    manifest = decoder.formulation_manifest()
    assert manifest["profile_version"] == "ieee-orpd-controls-v3.4.0"
    assert manifest["case_name"] == case_name
    assert manifest["case_checksum"] == case.checksum()
    assert manifest["dimension"] == decoder.dimension > 0
    assert len(decoder.variables) == decoder.dimension
    assert run_ac_power_flow(case).converged


def test_case118_fixed_reactors_are_preserved_by_zero_and_mid_controls():
    case = CaseLoader.load("case118")
    decoder = ORPDVariableDecoder(case, ORPDVariableConfig())
    fixed = {int(row[0]): float(row[BS]) for row in case.bus if float(row[BS]) < 0.0}
    assert fixed[5] == pytest.approx(-40.0)
    assert fixed[37] == pytest.approx(-25.0)
    for vector in (np.zeros(decoder.dimension), np.full(decoder.dimension, 0.5)):
        decoded, _ = decoder.decode(vector)
        by_bus = {int(row[0]): float(row[BS]) for row in decoded.bus}
        assert by_bus[5] == pytest.approx(-40.0)
        assert by_bus[37] == pytest.approx(-25.0)


def test_scenario_validation_rejects_empty_and_out_of_range_requests():
    config = ExperimentConfig()
    config.scenarios = RobustScenarioSettings(mode="load_uncertainty", count=0)
    with pytest.raises(ValueError, match="count"):
        config.validate()

    config = ExperimentConfig()
    config.scenarios = RobustScenarioSettings(mode="branch_contingency", branch_outages=[10_000])
    with pytest.raises(ValueError, match="outside"):
        build_scenarios(config, 7, CaseLoader.load("case30"))


def test_scenario_manifest_is_nonempty_and_case_checksums_are_reproducible():
    config = ExperimentConfig()
    config.scenarios = RobustScenarioSettings(mode="load_uncertainty", count=3)
    problem_a = build_problem(config, 123)
    problem_b = build_problem(config, 123)
    checksums_a = [scenario.apply(problem_a.case).checksum() for scenario in problem_a.scenarios]
    checksums_b = [scenario.apply(problem_b.case).checksum() for scenario in problem_b.scenarios]
    assert len(checksums_a) == 3
    assert checksums_a == checksums_b
    assert pytest.approx(sum(item.weight for item in problem_a.scenarios)) == 1.0


def test_case_validator_rejects_active_zero_impedance_branch():
    case = CaseLoader.load("case30").clone()
    case.branch[0, [BR_R, BR_X]] = 0.0
    report = validate_case(case)
    assert not report.valid
    assert any("zero-impedance" in message for message in report.errors)


@pytest.mark.parametrize("case_name", ["case30", "case57"])
def test_mixed_vector_cpu_torch_parity_for_publication_cases(case_name):
    case = CaseLoader.load(case_name)
    reference = ORPDProblem(case)
    accelerated = AcceleratedORPDProblem(case, device="cpu", batch_size=1)
    candidate = np.random.default_rng(340).random((1, reference.dimension))
    report = parity_check(
        reference,
        accelerated,
        candidate,
        objective_tolerance=1e-5,
        violation_tolerance=1e-6,
        voltage_tolerance=1e-5,
    )
    assert report.passed, (case_name, report)


def test_case300_reference_bus_reports_actual_unclipped_reactive_requirement():
    from calo_rpd_studio.power_system.case_model import (
        BUS_I, BUS_TYPE, GEN_BUS, GEN_STATUS, QG, QMAX, REF,
    )

    case = CaseLoader.load("case300")
    result = run_ac_power_flow(case)
    assert result.converged
    ref_rows = np.where(result.case.bus[:, BUS_TYPE].astype(int) == REF)[0]
    assert ref_rows.size == 1
    ref_bus = int(result.case.bus[int(ref_rows[0]), BUS_I])
    gen_rows = np.where(
        (result.case.gen[:, GEN_STATUS] > 0)
        & (result.case.gen[:, GEN_BUS].astype(int) == ref_bus)
    )[0]
    actual_q = float(np.sum(result.case.gen[gen_rows, QG]))
    qmax = float(np.sum(result.case.gen[gen_rows, QMAX]))
    assert actual_q > qmax
    assert actual_q == pytest.approx(38.84697442, rel=0, abs=1e-5)
