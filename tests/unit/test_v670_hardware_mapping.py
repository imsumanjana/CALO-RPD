from __future__ import annotations

from types import SimpleNamespace
import sys

from calo_rpd_studio.compute.device_binding import bind_config_to_device, runtime_device_attestation
from calo_rpd_studio.compute.resource_scheduler import DeviceSnapshot, ResourceMonitor, ResourceSnapshot
from calo_rpd_studio.compute.topology import ComputeTopologyService
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig


def _fake_cuda_torch(*, uuids=("GPU-A",), names=("NVIDIA A",), utilization_error=None):
    class Props:
        total_memory = 8 * 1024**3
        pci_bus_id = ""

        def __init__(self, uuid):
            self.uuid = uuid

    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return len(names)

        @staticmethod
        def get_device_name(index):
            return names[index]

        @staticmethod
        def get_device_properties(index):
            return Props(uuids[index])

        @staticmethod
        def utilization(index):
            if utilization_error is not None:
                raise utilization_error
            return 12 + index

        @staticmethod
        def mem_get_info(index):
            return 6 * 1024**3, 8 * 1024**3

    return SimpleNamespace(cuda=Cuda(), version=SimpleNamespace(cuda="13.0"))


def test_missing_nvml_telemetry_never_removes_cuda_device(monkeypatch):
    fake = _fake_cuda_torch(utilization_error=ModuleNotFoundError("pynvml missing"))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monitor = ResourceMonitor(xpu_interpreter="")
    monitor._nvidia_smi = None

    snapshots = monitor._sample_cuda()

    assert len(snapshots) == 1
    assert snapshots[0].device_id == "cuda:0"
    assert snapshots[0].available is True
    assert snapshots[0].utilization_percent is None
    assert snapshots[0].memory_total_bytes == 8 * 1024**3


def test_nvidia_smi_rows_are_mapped_by_uuid_not_row_index(monkeypatch):
    fake = _fake_cuda_torch(uuids=("GPU-A", "GPU-B"), names=("NVIDIA A", "NVIDIA B"))
    monkeypatch.setitem(sys.modules, "torch", fake)

    def fake_run(*args, **kwargs):
        # Deliberately reverse physical rows. Runtime cuda:0 must still map to GPU-A / 11%.
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "GPU-B, 00000000:02:00.0, 0x28E010DE, 1, NVIDIA B, 77, 2048, 8192, 50, 20, 80, 581.86\n"
                "GPU-A, 00000000:01:00.0, 0x28E010DE, 0, NVIDIA A, 11, 1024, 8192, 45, 15, 80, 581.86\n"
            ),
        )

    import calo_rpd_studio.compute.resource_scheduler as rs

    monkeypatch.setattr(rs.subprocess, "run", fake_run)
    monitor = ResourceMonitor(xpu_interpreter="")
    monitor._nvidia_smi = "nvidia-smi"

    snapshots = monitor._sample_cuda()

    assert [s.utilization_percent for s in snapshots] == [11.0, 77.0]
    assert [s.hardware_uuid for s in snapshots] == ["GPU-A", "GPU-B"]
    assert snapshots[0].pci_bus_id.endswith("01:00.0")
    assert snapshots[1].pci_bus_id.endswith("02:00.0")


def test_canonical_device_binding_sets_evaluator_optimizer_and_policy_devices():
    config = ExperimentConfig(algorithms=["CALO", "PSO"])
    config.scientific_backend = "torch_fp64"
    local = bind_config_to_device(config, "xpu:0")

    assert local.runtime_compute_device == "xpu:0"
    assert local.algorithm_parameters["CALO"]["execution_device"] == "xpu:0"
    assert local.algorithm_parameters["CALO"]["inference_device"] == "xpu:0"
    assert local.algorithm_parameters["CALO"]["optimizer_backend"] == "torch"
    assert local.algorithm_parameters["PSO"]["execution_device"] == "xpu:0"
    assert local.algorithm_parameters["PSO"]["optimizer_backend"] == "torch"


def test_sidecar_memory_and_fp64_capability_are_taken_from_sidecar_snapshot(monkeypatch):
    class Monitor:
        def sample(self):
            return ResourceSnapshot(
                cpu_percent=10.0,
                devices=(
                    DeviceSnapshot(
                        "xpu:0",
                        "xpu",
                        0,
                        "Intel Arc Test",
                        True,
                        5.0,
                        10.0,
                        runtime="sidecar",
                        memory_total_bytes=16 * 1024**3,
                        vendor_id="8086",
                        product_id="1234",
                        fp64_test_passed=True,
                    ),
                ),
            )

    monkeypatch.setattr(ComputeTopologyService, "_windows_adapters", staticmethod(lambda: []))
    topology = ComputeTopologyService(Monitor()).scan()
    device = topology.devices[0]

    assert device.runtime_id == "xpu:0"
    assert device.memory_total_bytes == 16 * 1024**3
    assert device.orpd_evaluator is True
    assert device.full_training_branch is False
    assert "FP64" in device.capability_status
    assert device.vendor_id == "8086"


def test_cpu_runtime_attestation_is_explicit_and_truthful():
    result = runtime_device_attestation("cpu")
    assert result["available"] is True
    assert result["resolved_device"] == "cpu"
    assert result["tensor_probe_passed"] is True


def test_windows_adapter_matching_prefers_vendor_product_identity_over_order():
    adapters = [
        {"Name": "NVIDIA Generic", "PNPDeviceID": r"PCI\\VEN_10DE&DEV_9999"},
        {"Name": "NVIDIA Generic", "PNPDeviceID": r"PCI\\VEN_10DE&DEV_28E0"},
    ]
    used: set[int] = set()
    name, index = ComputeTopologyService._match_adapter(
        "NVIDIA GeForce RTX 4060 Laptop GPU",
        adapters,
        used,
        vendor_id="10DE",
        product_id="28E0",
    )
    assert index == 1
    assert name == "NVIDIA Generic"
