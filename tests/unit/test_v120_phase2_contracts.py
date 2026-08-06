"""Phase 2 runtime binding, exact accounting, ownership, and provenance contracts.

These tests run only through the user's Git-ignored Phase 2 validator; Codex does not execute them.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from calo_rpd_studio.accelerated.torch_orpd import AcceleratedORPDProblem
from calo_rpd_studio.accelerated.vram_residency import (
    CudaCapacityExhausted,
    VramResidencyStats,
)
from calo_rpd_studio.algorithms.base_optimizer import (
    BaseOptimizer,
    EvaluationBatchInvariantError,
    OptimizerConfig,
)
from calo_rpd_studio.compute.device_lease import (
    DeviceLeaseCancelled,
    DeviceLeaseUnavailable,
    ExclusiveDeviceLease,
)
from calo_rpd_studio.compute.device_binding import (
    bind_config_to_device,
    resolve_config_for_entrypoint,
)
from calo_rpd_studio.compute.execution_contract import (
    ExecutionResolutionError,
    FormalCudaRequired,
    execution_claim_eligibility,
    resolve_execution,
)
from calo_rpd_studio.compute.resource_scheduler import DeviceSnapshot, ResourceSnapshot
from calo_rpd_studio.compute.topology import ComputeTopologyService
from calo_rpd_studio.benchmarking.campaign import BenchmarkCampaignConfig
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.experiments.experiment_runner import (
    _failure_kind,
    _optimizer_failure_state,
)
from calo_rpd_studio.orpd.problem import Evaluation
from calo_rpd_studio.scripts.run_benchmark import parser as benchmark_parser
from calo_rpd_studio.scripts.run_final_benchmark import parser as final_parser


def _config(**overrides):
    values = {
        "execution_backend": "cuda_preferred",
        "execution_purpose": "exploratory",
        "requested_compute_device": "auto",
        "cuda_cpu_fallback_enabled": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _cuda(device_id="cuda:0", *, uuid="GPU-ABC", pci="0000:65:00.0"):
    return DeviceSnapshot(
        device_id=device_id,
        backend="cuda",
        index=int(device_id.split(":", 1)[1]),
        name="NVIDIA test GPU",
        available=True,
        memory_total_bytes=16 * 1024**3,
        memory_available_bytes=12 * 1024**3,
        hardware_uuid=uuid,
        pci_bus_id=pci,
    )


def _snapshot(*devices):
    return ResourceSnapshot(
        cpu_percent=0.0,
        devices=tuple(devices),
        system_memory_total_bytes=32 * 1024**3,
        system_memory_available_bytes=24 * 1024**3,
    )


def test_cpu_only_resolves_to_concrete_cpu_without_fallback():
    resolved = resolve_execution(
        _config(execution_backend="cpu_only", requested_compute_device="cpu"),
        snapshot=_snapshot(),
    )
    assert resolved.runtime_compute_device == "cpu"
    assert resolved.actual_computation_device == "cpu"
    assert resolved.fallback_policy == "not_applicable"
    assert resolved.fallback_permitted is False


def test_exploratory_cuda_unavailable_is_explicit_cpu_restart():
    resolved = resolve_execution(_config(), snapshot=_snapshot())
    assert resolved.runtime_compute_device == "cpu"
    assert resolved.fallback_policy == "full_request_cpu_restart"
    assert resolved.fallback_reason == "cuda_unavailable_at_resolution"
    assert resolved.fallback_stratification_required is True
    assert not any(execution_claim_eligibility(resolved).values())


def test_formal_cuda_unavailable_fails_closed():
    with pytest.raises(FormalCudaRequired):
        resolve_execution(
            _config(execution_purpose="formal", cuda_cpu_fallback_enabled=False),
            snapshot=_snapshot(),
        )


def test_formal_cuda_uses_uuid_and_is_cuda_claim_eligible():
    resolved = resolve_execution(
        _config(execution_purpose="formal", cuda_cpu_fallback_enabled=False),
        snapshot=_snapshot(_cuda()),
    )
    assert resolved.assigned_physical_device == "gpu-uuid:gpu-abc"
    assert resolved.physical_identity_authority == "hardware_uuid"
    assert resolved.assigned_logical_device == "cuda:0"
    assert resolved.fallback_policy == "forbidden"
    assert all(execution_claim_eligibility(resolved).values())


def test_pci_identity_is_controlled_fallback_when_uuid_absent():
    resolved = resolve_execution(_config(), snapshot=_snapshot(_cuda(uuid="")))
    assert resolved.assigned_physical_device == "pci:65:00.0"
    assert resolved.physical_identity_authority == "normalized_pci_bus_id"


def test_xpu_is_never_executable():
    with pytest.raises(ExecutionResolutionError, match="view-only"):
        resolve_execution(_config(requested_compute_device="xpu:0"), snapshot=_snapshot())


def test_direct_and_gui_scheduler_bindings_share_cpu_only_semantics():
    config = _config(
        execution_backend="cpu_only",
        requested_compute_device="cpu",
        algorithms=["TLBO"],
        algorithm_parameters={},
        scientific_backend="torch_fp64",
    )
    direct = resolve_config_for_entrypoint(config)
    scheduled = bind_config_to_device(config, "cpu")
    for bound in (direct, scheduled):
        assert bound.runtime_compute_device == "cpu"
        assert bound.runtime_fallback_policy == "not_applicable"
        assert bound.runtime_assigned_logical_device == "cpu"
        assert bound.algorithm_parameters["TLBO"]["execution_device"] == "cpu"


def test_process_local_resolution_marker_is_never_serialized_or_trusted():
    config = ExperimentConfig()
    config.runtime_resolution_process_id = 12345
    assert "runtime_resolution_process_id" not in config.to_dict()
    with pytest.raises(ValueError, match="Unknown experiment configuration field"):
        ExperimentConfig.from_dict({"runtime_resolution_process_id": 12345})


def test_cli_and_final_campaign_expose_the_same_purpose_and_device_vocabulary():
    ordinary = benchmark_parser().parse_args(
        [
            "--compute-mode",
            "cuda_preferred",
            "--execution-purpose",
            "formal",
            "--device",
            "cuda:0",
            "--no-allow-cpu-fallback",
        ]
    )
    final = final_parser().parse_args(["--device", "cuda:0"])
    campaign = BenchmarkCampaignConfig()
    assert ordinary.execution_purpose == campaign.execution_purpose == "formal"
    assert ordinary.device == final.device == "cuda:0"
    assert ordinary.allow_cpu_fallback is False


def _capacity_problem(*, fallback_enabled):
    class Evaluator:
        vram_governor = SimpleNamespace(note_cpu_fallback=lambda _reason: None)

        @staticmethod
        def evaluate_tensor(_population):
            raise CudaCapacityExhausted("injected capacity", {"minimum_microbatch": 1})

    class Reference:
        @staticmethod
        def evaluate(candidate):
            return _evaluation(candidate)

        @staticmethod
        def evaluate_with_context(candidate, *, retain_control_linearization=False):
            return _evaluation(candidate), SimpleNamespace(
                retained=bool(retain_control_linearization)
            )

    problem = object.__new__(AcceleratedORPDProblem)
    problem._broker = None
    problem._large_case_reference_fallback = False
    problem._device_resident_evaluator = Evaluator()
    problem.cuda_cpu_fallback_enabled = bool(fallback_enabled)
    problem.runtime_fallback_policy = (
        "full_request_cpu_restart" if fallback_enabled else "forbidden"
    )
    problem.device = "cuda:0"
    problem.reference = Reference()
    problem._execution_provenance = {
        "actual_computation_device": "cuda:0",
        "fallback_used": False,
        "fallback_reason": "",
        "fallback_count": 0,
        "cuda_only_claims_eligible": not fallback_enabled,
    }
    return problem


def test_capacity_exhaustion_formal_fails_exploratory_restarts_with_provenance():
    population = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(CudaCapacityExhausted):
        _capacity_problem(fallback_enabled=False).evaluate_population(population)

    exploratory = _capacity_problem(fallback_enabled=True)
    results = exploratory.evaluate_population(population)
    assert len(results) == 2
    assert all(item.metadata["actual_computation_device"] == "cpu" for item in results)
    assert all(item.metadata["cuda_only_claims_eligible"] is False for item in results)
    provenance = exploratory.execution_provenance()
    assert provenance["fallback_used"] is True
    assert provenance["fallback_reason"] == "cuda_capacity_exhausted"


def test_dense_tensor_api_obeys_the_same_fallback_flag():
    population = np.asarray([[0.1, 0.2]])
    formal = _capacity_problem(fallback_enabled=False)
    formal._large_case_reference_fallback = True
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        formal.evaluate_population_tensor(population)

    exploratory = _capacity_problem(fallback_enabled=True)
    exploratory._large_case_reference_fallback = True
    result = exploratory.evaluate_population_tensor(population)
    assert result[0].metadata["actual_computation_device"] == "cpu"


def test_dense_counted_context_api_obeys_the_same_fallback_flag():
    population = np.asarray([[0.1, 0.2]])
    formal = _capacity_problem(fallback_enabled=False)
    formal._large_case_reference_fallback = True
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        formal.evaluate_population_with_context(population)

    exploratory = _capacity_problem(fallback_enabled=True)
    exploratory._large_case_reference_fallback = True
    result = exploratory.evaluate_population_with_context(population)
    assert result[0][0].metadata["actual_computation_device"] == "cpu"


def test_one_physical_uuid_has_one_lease_despite_logical_reordering(tmp_path):
    first = ExclusiveDeviceLease(
        "cuda:0", physical_device_id="gpu-uuid:gpu-abc", host_scope="host-a", root=tmp_path
    )
    second = ExclusiveDeviceLease(
        "cuda:3", physical_device_id="gpu-uuid:gpu-abc", host_scope="host-a", root=tmp_path
    )
    other = ExclusiveDeviceLease(
        "cuda:0", physical_device_id="gpu-uuid:gpu-def", host_scope="host-a", root=tmp_path
    )
    try:
        assert first.key == second.key
        assert first.key != other.key
    finally:
        other.close()
        second.close()
        first.close()


def test_ordinary_lease_contention_waits_for_queue_admission(monkeypatch, tmp_path):
    attempts = []

    def flaky_lock(_stream):
        attempts.append(1)
        if len(attempts) == 1:
            raise DeviceLeaseUnavailable("injected contention")

    monkeypatch.setattr(ExclusiveDeviceLease, "_lock_stream", staticmethod(flaky_lock))
    monkeypatch.setattr(ExclusiveDeviceLease, "_unlock_stream", staticmethod(lambda _stream: None))
    monkeypatch.setattr("calo_rpd_studio.compute.device_lease.time.sleep", lambda _seconds: None)
    lease = ExclusiveDeviceLease(
        "cuda:0",
        physical_device_id="gpu-uuid:queued",
        host_scope="host-a",
        root=tmp_path,
        wait=True,
        timeout_seconds=1.0,
    )
    try:
        assert len(attempts) == 2
    finally:
        lease.close()


def test_lease_cancellation_and_contention_have_distinct_failure_classes():
    assert _failure_kind(DeviceLeaseCancelled("cancelled"), 0) == "cancellation"
    assert _failure_kind(DeviceLeaseUnavailable("busy"), 0) == "lease_contention"


class _BatchProblem:
    dimension = 2

    def __init__(self, output):
        self.output = output

    def evaluate_population(self, population):
        return self.output(np.asarray(population, dtype=float))


def _evaluation(vector):
    return Evaluation(
        1.0,
        True,
        0.0,
        metadata={"normalized_decision_vector": np.asarray(vector, dtype=float).tolist()},
    )


@pytest.mark.parametrize("returned", [0, 1, 3])
def test_batch_cardinality_fails_before_any_fe_is_registered(returned):
    def output(population):
        rows = list(population[:returned])
        if returned > len(population):
            rows.append(population[-1])
        return [_evaluation(row) for row in rows]

    problem = _BatchProblem(output)
    optimizer = BaseOptimizer(problem, OptimizerConfig(max_evaluations=10))
    with pytest.raises(EvaluationBatchInvariantError):
        optimizer.evaluate_population(np.asarray([[0.1, 0.2], [0.3, 0.4]]))
    assert optimizer.evaluations == 0


def test_reordered_batch_identity_fails_before_any_fe_is_registered():
    problem = _BatchProblem(
        lambda population: [_evaluation(population[1]), _evaluation(population[0])]
    )
    optimizer = BaseOptimizer(problem, OptimizerConfig(max_evaluations=10))
    with pytest.raises(EvaluationBatchInvariantError, match="candidate_identity_mismatch"):
        optimizer.evaluate_population(np.asarray([[0.1, 0.2], [0.3, 0.4]]))
    assert optimizer.evaluations == 0


def test_partial_failure_envelope_preserves_exact_count_and_incumbent():
    incumbent = Evaluation(3.0, False, 0.25, metadata={"constraint_components": {"v": 0.25}})
    optimizer = SimpleNamespace(
        evaluations=17,
        iteration=4,
        best_evaluation=incumbent,
        best_vector=np.asarray([0.2, 0.8]),
        evaluation_history=[5, 10, 17],
        best_feasible_objective_history=[float("nan")] * 3,
        best_constraint_violation_history=[0.8, 0.4, 0.25],
        config=SimpleNamespace(parameters={"run_checkpoint_path": "checkpoint.json"}),
    )
    config = SimpleNamespace(
        runtime_compute_device="cuda:0",
        runtime_device_resolution={"schema_version": "calo-runtime-execution-contract-v2"},
        runtime_fallback_policy="forbidden",
        runtime_fallback_reason="",
    )
    state = _optimizer_failure_state(optimizer, config, RuntimeError("injected"))
    assert state["failure_kind"] == "partial_failure"
    assert state["evaluation_count"] == 17
    assert state["last_incumbent"]["normalized_vector"] == [0.2, 0.8]
    assert state["last_incumbent"]["violation"] == 0.25
    assert state["checkpoint_reference"] == "checkpoint.json"


def test_request_and_lifetime_vram_statistics_are_separate():
    stats = VramResidencyStats(device="cuda:0", enabled=True, budget_fraction=0.80)
    stats.request_count = 2
    stats.last_request_microbatches = 1
    stats.microbatches = 7
    stats.last_request_oom_retries = 0
    stats.oom_retries = 3
    payload = stats.to_dict()
    assert payload["request_statistics"]["microbatches"] == 1
    assert payload["governor_lifetime_statistics"]["microbatches"] == 7
    assert payload["request_statistics"]["oom_retries"] == 0
    assert payload["governor_lifetime_statistics"]["oom_retries"] == 3


def test_cpu_only_synthetic_topology_scan_is_fail_closed(monkeypatch):
    monitor = SimpleNamespace(sample=lambda: _snapshot())
    service = ComputeTopologyService(monitor=monitor)
    monkeypatch.setattr(service, "_windows_adapters", lambda: [])
    topology = service.scan()
    assert topology.devices == ()
    assert topology.fingerprint


def test_active_status_is_phase2_cuda_cpu_only_and_xpu_nonexecutable():
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "ACTIVE_DEVELOPMENT_STATUS.json").read_text(encoding="utf-8"))
    assert payload["phase"] == 2
    assert payload["supported_execution_modes"] == ["cuda-preferred", "cpu-only"]
    assert payload["intel_xpu_executable"] is False
    assert (
        payload["phase_2_validation"]
        == "failed_phase2-20260807-003024_formatting_corrected_rerun_pending"
    )
