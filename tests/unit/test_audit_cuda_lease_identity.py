"""CUDA identity tests mock runtime discovery; no GPU allocation or work is executed."""

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID
import os
import subprocess
import sys
import pytest
import torch
from calo_rpd_studio.compute.device_lease import ExclusiveDeviceLease as Lease

UUID_A = "12345678-1234-5678-9234-567812345678"
UUID_B = "22345678-1234-5678-9234-567812345678"


@pytest.fixture(autouse=True)
def fake_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("CALO_DEVICE_LEASE_DIR", str(tmp_path / "leases"))
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="GPU-" + UUID_A)
    )


def test_logical_ordinals_and_display_scopes_share_one_physical_lock():
    with Lease.for_cuda(
        "cuda:0", physical_device_id="PNP:display-adapter", host_scope="host-a"
    ) as first:
        with Lease.for_cuda(
            "cuda:3",
            physical_device_id="gpu-uuid:GPU-" + UUID_A,
            host_scope="host-b",
            container_scope="container-b",
        ) as second:
            assert first.key == second.key
            assert Lease._process_leases[first.key].references == 2


def test_different_physical_gpus_do_not_share_a_lock(monkeypatch):
    with Lease.for_cuda("cuda:0") as first:
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid=UUID_B)
        )
        with Lease.for_cuda("cuda:0") as second:
            assert first.key != second.key


@pytest.mark.parametrize("invalid", [None, "", "not-a-uuid", str(UUID(int=0))])
def test_missing_or_invalid_runtime_uuid_never_falls_back_to_ordinal(
    monkeypatch, tmp_path, invalid
):
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid=invalid)
    )
    with pytest.raises(RuntimeError, match="physical UUID"):
        Lease.for_cuda("cuda:0")
    assert not (tmp_path / "leases").exists()


@pytest.mark.parametrize("claim", ["gpu-uuid:gpu-" + UUID_B, "gpu-invalid"])
def test_mismatching_or_malformed_uuid_claim_is_rejected(claim):
    with pytest.raises(ValueError, match="physical UUID"):
        Lease.for_cuda("cuda:0", physical_device_id=claim)


def test_runtime_uuid_byte_object_and_current_device_are_supported(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(uuid=SimpleNamespace(bytes=list(UUID(UUID_A).bytes))),
    )
    with Lease.for_cuda("cuda") as lease:
        assert lease.device_id == "cuda:3"
        assert lease.physical_device_id == "gpu-uuid:gpu-" + UUID_A


def test_cpu_request_is_rejected_before_cuda_discovery(monkeypatch):
    def forbidden(*args):
        pytest.fail("CPU request must not query CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_properties", forbidden)
    with pytest.raises(ValueError, match="NVIDIA CUDA"):
        Lease.for_cuda("cpu")


CHILD_CODE = """
from types import SimpleNamespace
import sys
import torch
from calo_rpd_studio.algorithms.calo.tsh_calo_training_resources import TSHCALOTrainingDeviceGuard
from calo_rpd_studio.compute.device_lease import DeviceLeaseUnavailable
torch.version.hip = None
torch.cuda.is_available = lambda: True
torch.cuda.get_device_properties = lambda _: SimpleNamespace(uuid=sys.argv[1])
torch.cuda.mem_get_info = lambda _: (2**30, 2**31)
torch.cuda.memory_reserved = lambda _: 0
torch.cuda.set_per_process_memory_fraction = lambda *a: None
estimate = SimpleNamespace(estimated_working_set_bytes=2**26, estimator_version='synthetic')
try:
    guard = TSHCALOTrainingDeviceGuard.admit(estimate, requested_device=sys.argv[2], allow_cpu_fallback=False)
except DeviceLeaseUnavailable:
    print('blocked')
else:
    guard.close()
    print('acquired')
"""


@pytest.mark.parametrize("child_device", ["cuda:0", "cuda:7"])
def test_experiment_governor_excludes_training_in_another_process(monkeypatch, child_device):
    from calo_rpd_studio.accelerated.vram_residency import VramResidencyGovernor

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (2**30, 2**31))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 0)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda *a: None)
    monkeypatch.setattr(VramResidencyGovernor, "_configured_fractions", {})

    def child(uuid):
        result = subprocess.run(
            [sys.executable, "-B", "-c", CHILD_CODE, uuid, child_device],
            capture_output=True,
            text=True,
            timeout=60,
            env=os.environ.copy(),
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    governor = VramResidencyGovernor(
        "cuda:3", physical_device_id="PNP:experiment-adapter", lease_host_scope="display-host"
    )
    try:
        assert governor.enabled
        assert child(UUID_A) == "blocked"
        assert child(UUID_B) == "acquired"
    finally:
        governor.close()
    assert child(UUID_A) == "acquired"


def test_all_production_cuda_lease_callers_use_the_canonical_factory():
    root = Path(__file__).resolve().parents[2]
    for relative in (
        "algorithms/calo/tsh_calo_training_resources.py",
        "accelerated/vram_residency.py",
        "compute/soak.py",
        "scripts/validate_resource_recovery.py",
    ):
        text = (root / "calo_rpd_studio" / relative).read_text(encoding="utf-8")
        assert "ExclusiveDeviceLease.for_cuda(" in text
        assert "ExclusiveDeviceLease(" not in text
