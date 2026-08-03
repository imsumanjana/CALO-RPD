"""Safe-80 memory and exclusive-device admission for independent TSH-CALO training."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import calo_rpd_studio.algorithms.calo.tsh_calo_training_resources as resources
from calo_rpd_studio.algorithms.calo.tsh_calo_policy import TSHCALOPolicyNetwork
from calo_rpd_studio.algorithms.calo.tsh_calo_training_resources import (
    TSHCALOTrainingDeviceGuard,
    TSHCALOTrainingResourceEnvelope,
    estimate_tsh_calo_training_working_set,
    validate_tsh_calo_training_device_provenance,
)
from calo_rpd_studio.compute.device_lease import DeviceLeaseUnavailable


def _envelope(**changes) -> TSHCALOTrainingResourceEnvelope:
    values = dict(
        rollout_capacity=8,
        maximum_population_size=32,
        maximum_topology_nodes=64,
        maximum_topology_edges=192,
        maximum_topology_controls=64,
        maximum_scenarios=8,
    )
    values.update(changes)
    return TSHCALOTrainingResourceEnvelope(**values)


def _estimate(envelope=None):
    return estimate_tsh_calo_training_working_set(
        TSHCALOPolicyNetwork(hidden_dim=16, graph_steps=1), envelope or _envelope()
    )


def test_declared_shape_estimator_is_deterministic_and_monotone():
    small = _estimate()
    repeated = _estimate()
    larger = _estimate(
        _envelope(
            rollout_capacity=16,
            maximum_topology_nodes=128,
            maximum_topology_edges=384,
        )
    )

    assert small.to_dict() == repeated.to_dict()
    assert small.estimated_working_set_bytes >= 64 << 20
    assert larger.retained_rollout_bytes > small.retained_rollout_bytes
    assert larger.autograd_activation_bytes > small.autograd_activation_bytes
    assert larger.estimated_working_set_bytes >= small.estimated_working_set_bytes
    assert larger.envelope["rollout_capacity"] == 16


def test_cpu_admission_uses_eighty_percent_of_current_available_ram(monkeypatch):
    estimate = _estimate()
    working = estimate.estimated_working_set_bytes
    monkeypatch.setattr(
        resources.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=4 * working, available=2 * working),
    )

    guard = TSHCALOTrainingDeviceGuard.admit(
        estimate, requested_device="cpu", allow_cpu_fallback=False
    )

    assert guard.admission.selected_device == "cpu"
    assert guard.admission.computation_device == "cpu"
    assert guard.admission.allowance_bytes == int(0.8 * 2 * working)
    assert guard.admission.available_bytes_at_admission == 2 * working
    guard.close()

    monkeypatch.setattr(
        resources.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=4 * working, available=working),
    )
    with pytest.raises(MemoryError, match="currently available CPU RAM"):
        TSHCALOTrainingDeviceGuard.admit(estimate, requested_device="cpu", allow_cpu_fallback=False)


def test_cuda_admission_uses_current_free_vram_and_holds_exclusive_lease(monkeypatch):
    estimate = _estimate()
    working = estimate.estimated_working_set_bytes
    events: list[tuple] = []

    class Lease:
        def __init__(self, device):
            events.append(("lease", device))

        def close(self):
            events.append(("close",))

    monkeypatch.setattr(resources, "ExclusiveDeviceLease", Lease)
    monkeypatch.setattr(resources.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        resources.torch.cuda,
        "mem_get_info",
        lambda _device: (2 * working, 8 * working),
    )
    monkeypatch.setattr(resources.torch.cuda, "memory_reserved", lambda _device: 0)
    monkeypatch.setattr(
        resources.torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, index: events.append(("fraction", fraction, index)),
    )

    guard = TSHCALOTrainingDeviceGuard.admit(
        estimate, requested_device="auto", allow_cpu_fallback=True
    )

    assert guard.admission.selected_device == "cuda:0"
    assert guard.admission.computation_device == "nvidia_gpu"
    assert guard.admission.available_bytes_at_admission == 2 * working
    assert guard.admission.allowance_bytes == int(0.8 * 2 * working)
    assert events[0] == ("lease", "cuda:0")
    assert events[1][0] == "fraction"
    with pytest.raises(DeviceLeaseUnavailable, match="in this process"):
        TSHCALOTrainingDeviceGuard.admit(estimate, requested_device="auto", allow_cpu_fallback=True)
    guard.close()
    assert events[-1] == ("close",)


def test_cuda_vram_shortfall_falls_back_only_under_explicit_permission(monkeypatch):
    estimate = _estimate()
    working = estimate.estimated_working_set_bytes

    class Lease:
        def __init__(self, _device):
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(resources, "ExclusiveDeviceLease", Lease)
    monkeypatch.setattr(resources.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        resources.torch.cuda,
        "mem_get_info",
        lambda _device: (working, 8 * working),
    )
    monkeypatch.setattr(resources.torch.cuda, "memory_reserved", lambda _device: 0)
    monkeypatch.setattr(
        resources.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=8 * working, available=4 * working),
    )

    fallback = TSHCALOTrainingDeviceGuard.admit(
        estimate, requested_device="auto", allow_cpu_fallback=True
    )
    assert fallback.admission.selected_device == "cpu"
    assert "currently free VRAM" in fallback.admission.fallback_reason

    with pytest.raises(MemoryError, match="currently free VRAM"):
        TSHCALOTrainingDeviceGuard.admit(
            estimate, requested_device="cuda", allow_cpu_fallback=False
        )


def test_cuda_lease_contention_blocks_instead_of_spilling_to_cpu(monkeypatch):
    estimate = _estimate()

    class BusyLease:
        def __init__(self, _device):
            raise DeviceLeaseUnavailable("busy")

    monkeypatch.setattr(resources, "ExclusiveDeviceLease", BusyLease)
    monkeypatch.setattr(resources.torch.cuda, "is_available", lambda: True)

    with pytest.raises(DeviceLeaseUnavailable, match="busy"):
        TSHCALOTrainingDeviceGuard.admit(estimate, requested_device="auto", allow_cpu_fallback=True)


def test_no_xpu_training_resource_path_exists():
    source = resources.__file__
    assert source is not None
    text = Path(source).read_text(encoding="utf-8")
    assert "xpu" not in text.lower()
    with pytest.raises(ValueError, match="cpu, cuda"):
        TSHCALOTrainingDeviceGuard.admit(
            _estimate(), requested_device="xpu", allow_cpu_fallback=True
        )


def test_candidate_device_provenance_rejects_safe80_or_compute_mutation(monkeypatch):
    estimate = _estimate()
    working = estimate.estimated_working_set_bytes
    monkeypatch.setattr(
        resources.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=4 * working, available=2 * working),
    )
    guard = TSHCALOTrainingDeviceGuard.admit(
        estimate, requested_device="cpu", allow_cpu_fallback=False
    )
    provenance = {
        "memory_estimate": estimate.to_dict(),
        "memory_admission": guard.admission.to_dict(),
        "computation_semantics": "CPU computes; system RAM is admitted storage",
    }
    validate_tsh_calo_training_device_provenance(provenance)

    changed = {
        **provenance,
        "memory_admission": {
            **provenance["memory_admission"],
            "allowance_bytes": 2 * working,
            "process_ceiling_bytes": 2 * working,
            "allocator_fraction_of_total": 0.5,
        },
    }
    with pytest.raises(ValueError, match="Safe-80"):
        validate_tsh_calo_training_device_provenance(changed)
    with pytest.raises(ValueError, match="computation semantics"):
        validate_tsh_calo_training_device_provenance(
            {**provenance, "computation_semantics": "memory computes"}
        )
