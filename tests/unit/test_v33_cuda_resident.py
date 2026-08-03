from __future__ import annotations

from types import SimpleNamespace

import torch

from calo_rpd_studio.accelerated.torch_orpd import AcceleratedORPDProblem, parity_check
from calo_rpd_studio.compute.resource_scheduler import build_weighted_lane_plan
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.orpd.problem import ORPDProblem


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


def test_v33_legacy_cuda_only_config_migrates_to_cuda_preferred():
    config = ExperimentConfig.from_dict({"execution_backend": "cuda_only"})
    config.validate()
    assert not hasattr(config, "cuda_task_share")
    restored = ExperimentConfig.from_dict(config.to_dict())
    assert restored.execution_backend == "cuda_preferred"
    assert not hasattr(restored, "cpu_task_share")
