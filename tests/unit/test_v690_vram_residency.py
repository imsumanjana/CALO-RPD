from __future__ import annotations

from pathlib import Path
import multiprocessing

import numpy as np
import pytest

from calo_rpd_studio.accelerated.device_resident_orpd import DeviceResidentBatch
from calo_rpd_studio.accelerated.vram_residency import (
    CudaCapacityExhausted,
    VramResidencyGovernor,
    VramResidencyPolicy,
    VramResidencyStats,
)
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.compute.memory_budget import calculate_available_memory_admission
from calo_rpd_studio.compute.device_lease import DeviceLeaseUnavailable, ExclusiveDeviceLease


def _lease_attempt(root: str, queue) -> None:
    try:
        lease = ExclusiveDeviceLease("cuda:0", root=root)
    except DeviceLeaseUnavailable:
        queue.put("busy")
    else:
        queue.put("acquired")
        lease.close()


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_v690_defaults_use_fixed_80_percent_of_currently_free_vram():
    config = ExperimentConfig()
    assert config.device_resident_execution is True
    assert config.cuda_vram_budget_fraction == pytest.approx(0.80)
    assert config.cuda_oom_retry_count == 4
    assert config.cuda_minimum_microbatch == 1
    assert config.cuda_resident_hot_loop is True
    config.validate()


def test_v690_config_round_trip_normalizes_memory_fraction_and_preserves_retry_settings():
    config = ExperimentConfig()
    config.cuda_vram_budget_fraction = 0.82
    config.cuda_oom_retry_count = 5
    config.cuda_minimum_microbatch = 2
    config.cuda_resident_hot_loop = False
    restored = ExperimentConfig.from_dict(config.to_dict())
    assert restored.cuda_vram_budget_fraction == pytest.approx(0.80)
    assert restored.cuda_oom_retry_count == 5
    assert restored.cuda_minimum_microbatch == 2
    assert restored.cuda_resident_hot_loop is False
    restored.validate()


def test_v690_direct_non_80_percent_configuration_is_rejected():
    config = ExperimentConfig(cuda_vram_budget_fraction=0.82)
    with pytest.raises(ValueError, match="fixed at 0.80"):
        config.validate()


def test_v690_vram_policy_rejects_unsafe_hard_100_percent():
    with pytest.raises(ValueError, match="no greater than 0.80"):
        VramResidencyPolicy(budget_fraction=1.0).validate()


def test_available_vram_admission_uses_free_not_total_capacity():
    gib = 1024**3
    admission = calculate_available_memory_admission(
        total_bytes=8 * gib,
        available_bytes=5 * gib,
        requested_fraction=0.80,
    )
    assert admission.additional_allowance_bytes == 4 * gib
    assert admission.process_ceiling_bytes == 4 * gib
    assert admission.allocator_fraction_of_total == pytest.approx(0.50)


def test_available_vram_admission_includes_but_does_not_rebudget_existing_reservation():
    gib = 1024**3
    admission = calculate_available_memory_admission(
        total_bytes=8 * gib,
        available_bytes=5 * gib,
        baseline_reserved_bytes=gib // 2,
        requested_fraction=0.80,
    )
    assert admission.additional_allowance_bytes == 4 * gib
    assert admission.process_ceiling_bytes == 4 * gib + gib // 2


def test_device_lease_is_shared_in_process_and_exclusive_across_processes(tmp_path):
    first = ExclusiveDeviceLease("cuda:0", root=tmp_path)
    second = ExclusiveDeviceLease("cuda:0", root=tmp_path)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    contender = context.Process(target=_lease_attempt, args=(str(tmp_path), queue))
    contender.start()
    contender.join(timeout=10)
    assert contender.exitcode == 0
    assert queue.get(timeout=2) == "busy"

    first.close()
    # The same-process second reference still owns the OS lease.
    contender = context.Process(target=_lease_attempt, args=(str(tmp_path), queue))
    contender.start()
    contender.join(timeout=10)
    assert contender.exitcode == 0
    assert queue.get(timeout=2) == "busy"

    second.close()
    contender = context.Process(target=_lease_attempt, args=(str(tmp_path), queue))
    contender.start()
    contender.join(timeout=10)
    assert contender.exitcode == 0
    assert queue.get(timeout=2) == "acquired"


def test_v690_cpu_governor_is_transparent():
    governor = VramResidencyGovernor("cpu", VramResidencyPolicy())
    values = np.arange(12).reshape(4, 3)
    calls = []

    def evaluate_once(chunk):
        calls.append(len(chunk))
        return chunk * 2

    result = governor.run_microbatched(values, evaluate_once, lambda chunks, _meta: chunks[0])
    assert not governor.enabled
    assert calls == [4]
    assert np.array_equal(result, values * 2)


def test_v690_oom_backoff_retries_on_accelerator_without_cpu_fallback():
    class FakeOutOfMemoryError(RuntimeError):
        pass

    class FakeCuda:
        empty_cache_calls = 0

        @classmethod
        def empty_cache(cls):
            cls.empty_cache_calls += 1

        @staticmethod
        def reset_peak_memory_stats(_device):
            return None

        @staticmethod
        def max_memory_allocated(_device):
            return 123

        @staticmethod
        def max_memory_reserved(_device):
            return 456

    class FakeTorch:
        OutOfMemoryError = FakeOutOfMemoryError
        cuda = FakeCuda

    governor = object.__new__(VramResidencyGovernor)
    governor.policy = VramResidencyPolicy(
        budget_fraction=0.80,
        oom_retry_count=4,
        minimum_microbatch=1,
    )
    governor.device = "cuda:0"
    governor.device_text = "cuda:0"
    governor._torch = FakeTorch
    governor._enabled = True
    governor.stats = VramResidencyStats(device="cuda:0", enabled=True, budget_fraction=0.80)

    values = np.arange(15).reshape(5, 3)
    observed = []

    def evaluate_once(chunk):
        observed.append(len(chunk))
        if len(chunk) > 2:
            raise FakeOutOfMemoryError("CUDA out of memory")
        return chunk.copy()

    def concatenate(chunks, metadata):
        return np.concatenate(chunks, axis=0), metadata

    result, metadata = governor.run_microbatched(
        values,
        evaluate_once,
        concatenate,
        preferred_microbatch=5,
    )
    assert np.array_equal(result, values)
    assert observed[:2] == [5, 2]
    assert metadata["request_oom_retries"] == 1
    assert metadata["cpu_fallbacks"] == 0
    assert metadata["cpu_inner_loop_participation"] is False
    assert FakeCuda.empty_cache_calls == 1


def test_minimum_cuda_microbatch_raises_typed_capacity_exhaustion_with_provenance():
    class FakeOutOfMemoryError(RuntimeError):
        pass

    class FakeCuda:
        @staticmethod
        def empty_cache():
            return None

        @staticmethod
        def reset_peak_memory_stats(_device):
            return None

        @staticmethod
        def max_memory_allocated(_device):
            return 321

        @staticmethod
        def max_memory_reserved(_device):
            return 654

    class FakeTorch:
        OutOfMemoryError = FakeOutOfMemoryError
        cuda = FakeCuda

    governor = object.__new__(VramResidencyGovernor)
    governor.policy = VramResidencyPolicy(oom_retry_count=1, minimum_microbatch=1)
    governor.device = "cuda:0"
    governor.device_text = "cuda:0"
    governor._torch = FakeTorch
    governor._enabled = True
    governor._device_lease = None
    governor.stats = VramResidencyStats(device="cuda:0", enabled=True, budget_fraction=0.80)

    values = np.arange(6).reshape(2, 3)
    with pytest.raises(CudaCapacityExhausted) as captured:
        governor.run_microbatched(
            values,
            lambda _chunk: (_ for _ in ()).throw(FakeOutOfMemoryError("out of memory")),
            lambda _chunks, _metadata: None,
            preferred_microbatch=2,
        )
    metadata = captured.value.metadata
    assert metadata["execution_state"] == "cuda_capacity_exhausted"
    assert metadata["last_fallback_reason"] == "minimum_cuda_microbatch_exhausted"
    assert metadata["input_staged_from_host"] is True


def test_v690_device_batch_concatenation_stays_on_torch_device():
    torch = pytest.importorskip("torch")

    def make(offset: float) -> DeviceResidentBatch:
        objective = torch.tensor([offset, offset + 1.0], dtype=torch.float64)
        violation = torch.tensor([0.0, 0.1], dtype=torch.float64)
        feasible = torch.tensor([True, False])
        normalized = torch.tensor([[0.1], [0.2]], dtype=torch.float64)
        decoded = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
        scenarios = torch.tensor([[offset], [offset + 1.0]], dtype=torch.float64)
        objectives = {
            name: objective.clone()
            for name in (
                "active_power_loss_mw",
                "voltage_deviation_pu",
                "l_index_max",
                "scenario_objective_mean",
                "scenario_objective_std",
            )
        }
        constraints = {
            name: violation.clone()
            for name in (
                "bus_voltage",
                "generator_q",
                "generator_p",
                "branch_thermal",
                "branch_angle",
                "power_flow",
            )
        }
        scenario_constraints = torch.zeros((2, 1, 6), dtype=torch.float64)
        return DeviceResidentBatch(
            objective,
            violation,
            feasible,
            normalized,
            decoded,
            scenarios,
            objectives,
            constraints,
            scenario_constraints,
            ("x",),
            {"device_resident_execution": True},
        )

    joined = DeviceResidentBatch.concatenate([make(1.0), make(3.0)], {"budget_fraction": 0.80})
    assert joined.count == 4
    assert joined.objective.device.type == "cpu"  # test runtime device; no NumPy materialisation
    assert joined.metadata["device_microbatch_count"] == 2
    assert joined.metadata["vram_residency"]["budget_fraction"] == pytest.approx(0.80)


def test_v690_hot_loop_and_training_sources_avoid_per_step_host_scalar_reads():
    root = _root()
    nr = (root / "calo_rpd_studio/accelerated/torch_power_flow.py").read_text(encoding="utf-8")
    evaluator = (root / "calo_rpd_studio/accelerated/device_resident_orpd.py").read_text(
        encoding="utf-8"
    )
    training = (root / "calo_rpd_studio/algorithms/calo/heterogeneous_training.py").read_text(
        encoding="utf-8"
    )
    assert "host_early_exit=False" not in evaluator  # controlled by v6.9 config, not hard-coded
    assert (
        'host_early_exit=not bool(getattr(self.problem, "cuda_resident_hot_loop", True))'
        in evaluator
    )
    assert "fixed-shape mask" in nr
    assert "epoch_loss_tensors.append(loss.detach())" in training
    assert 'torch.stack(epoch_loss_tensors).detach().to("cpu").tolist()' in training
    assert "epoch_losses.append(float(loss.detach().cpu().item()))" not in training
