from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from calo_rpd_studio.accelerated.torch_orpd import AcceleratedORPDProblem, parity_check
from calo_rpd_studio.compute.resource_scheduler import build_weighted_lane_plan
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.orpd.problem import ORPDProblem
from calo_rpd_studio.power_system.case_loader import CaseLoader


def test_v34_defaults_prioritize_device_resident_cuda():
    config = ExperimentConfig()
    assert config.execution_backend == "cuda_preferred"
    assert config.device_resident_execution is True
    assert not hasattr(config, "cuda_task_share")
    assert not hasattr(config, "cuda_priority_work_stealing")


def test_v33_all_primary_jobs_are_accelerator_eligible():
    plan = [SimpleNamespace(job_index=index, label=f"A{index}") for index in range(400)]
    assignments, summary = build_weighted_lane_plan(
        plan,
        "comparison",
        cuda_available=True,
    )
    assert len(assignments) == 400
    assert summary.accelerator_eligible_jobs == 400
    assert summary.cpu_only_jobs == 0
    assert (summary.cuda_jobs, summary.total_cpu_jobs) == (400, 0)


def test_v33_cuda_only_assigns_every_job_to_cuda():
    plan = [SimpleNamespace(job_index=index, label=f"A{index}") for index in range(400)]
    assignments, summary = build_weighted_lane_plan(
        plan,
        "comparison",
        cuda_available=True,
    )
    assert summary.cuda_jobs == 400
    assert summary.total_cpu_jobs == 0
    assert set(assignments.values()) == {"cuda"}


def test_v33_tensor_batch_stays_on_execution_device_until_materialized(toy_case):
    reference = ORPDProblem(toy_case)
    problem = AcceleratedORPDProblem(toy_case, device="cpu", batch_size=8, device_resident=True)
    candidates = torch.rand((6, reference.dimension), dtype=torch.float64)
    batch = problem.evaluate_population_tensor(candidates)
    assert batch.objective.device.type == "cpu"
    assert batch.decoded_values.device.type == "cpu"
    assert batch.metadata["device_resident_execution"] is True
    results = batch.to_evaluations()
    assert len(results) == 6
    assert all(result.metadata["host_materializations_per_population"] == 1 for result in results)
    assert all(
        len(result.metadata["normalized_decision_vector"]) == reference.dimension
        for result in results
    )
    report = parity_check(reference, problem, candidates.numpy())
    assert report.passed
    assert report.feasibility_mismatches == 0


def test_counted_context_batch_materializes_once_without_reference_power_flow_rerun(
    toy_case, monkeypatch
):
    reference = ORPDProblem(toy_case)
    problem = AcceleratedORPDProblem(
        toy_case,
        device="cpu",
        batch_size=8,
        device_resident=True,
        cuda_cpu_fallback_enabled=False,
    )
    candidates = np.vstack(
        (
            np.linspace(0.1, 0.9, reference.dimension),
            np.linspace(0.8, 0.2, reference.dimension),
        )
    )
    expected = [
        reference.evaluate_with_context(row, retain_control_linearization=True)
        for row in candidates
    ]

    def forbidden_reference_rerun(*_args, **_kwargs):
        raise AssertionError("accelerated counted context reran the CPU reference evaluator")

    monkeypatch.setattr(problem.reference, "evaluate_with_context", forbidden_reference_rerun)
    actual = problem.evaluate_population_with_context(candidates, retain_control_linearization=True)

    assert len(actual) == len(expected) == 2
    for (evaluation, context), (expected_evaluation, expected_context) in zip(
        actual, expected, strict=True
    ):
        assert evaluation.metadata["context_power_flow_reruns"] == 0
        assert evaluation.metadata["context_host_materializations_per_population"] == 1
        assert evaluation.feasible == expected_evaluation.feasible
        np.testing.assert_allclose(
            [evaluation.value, evaluation.violation],
            [expected_evaluation.value, expected_evaluation.violation],
            rtol=1e-6,
            atol=1e-8,
        )
        power_flow = context.primary_converged_power_flow()
        expected_power_flow = expected_context.primary_converged_power_flow()
        np.testing.assert_allclose(
            power_flow.voltage, expected_power_flow.voltage, rtol=1e-8, atol=1e-9
        )
        np.testing.assert_array_equal(
            power_flow.case.bus[:, 1].astype(int),
            expected_power_flow.case.bus[:, 1].astype(int),
        )
        np.testing.assert_allclose(
            power_flow.branch.s_from_mva,
            expected_power_flow.branch.s_from_mva,
            rtol=1e-8,
            atol=1e-8,
        )
        linearization = context.primary_control_linearization()
        expected_linearization = expected_context.primary_control_linearization()
        np.testing.assert_allclose(
            linearization.jacobian.toarray(),
            expected_linearization.jacobian.toarray(),
            rtol=1e-7,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            linearization.control_sensitivity,
            expected_linearization.control_sensitivity,
            rtol=1e-6,
            atol=1e-7,
        )


def test_joint_nonconvergence_uses_canonical_failure_record_not_discarded_iterates():
    case = CaseLoader.load("case57")
    reference = ORPDProblem(case)
    accelerated = AcceleratedORPDProblem(case, device="cpu", batch_size=1)
    candidate = np.arange(reference.dimension, dtype=float) % 2.0

    cpu = reference.evaluate(candidate)
    tensor = accelerated.evaluate(candidate)
    assert not cpu.feasible and not tensor.feasible
    assert np.isinf(cpu.value) and np.isinf(tensor.value)
    assert np.isinf(cpu.violation) and np.isinf(tensor.violation)
    assert tensor.components == cpu.components
    assert tensor.metadata["constraint_components"]["power_flow"] == float("inf")
    assert all(
        value == 0.0
        for name, value in tensor.metadata["constraint_components"].items()
        if name != "power_flow"
    )

    report = parity_check(reference, accelerated, candidate)
    assert report.passed
    assert report.jointly_nonconverged_scenarios == 1
    assert report.compared_converged_scenarios == 0
    assert report.max_voltage_error == 0.0
    assert report.max_angle_error_deg == 0.0
    assert report.max_constraint_component_error == 0.0
    assert report.max_objective_component_error == 0.0


def test_v33_legacy_cuda_only_config_migrates_to_cuda_preferred():
    config = ExperimentConfig.from_dict({"execution_backend": "cuda_only"})
    config.validate()
    assert not hasattr(config, "cuda_task_share")
    restored = ExperimentConfig.from_dict(config.to_dict())
    assert restored.execution_backend == "cuda_preferred"
    assert not hasattr(restored, "cpu_task_share")
