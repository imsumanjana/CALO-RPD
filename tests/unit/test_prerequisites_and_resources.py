from __future__ import annotations

from calo_bootstrap.prerequisites import NvidiaInfo, candidate_torch_channels
from calo_rpd_studio.compute.resource_scheduler import (
    DeviceSnapshot,
    ResourceSnapshot,
    accelerator_admission_allowed,
    cpu_admission_allowed,
    item_uses_calo_ai,
    prioritized_accelerators,
)
from calo_rpd_studio.experiments.calo_ablation import AblationSpec
from calo_rpd_studio.experiments.execution_plan import PlannedItem


def test_cuda_channel_selection_respects_driver_capability():
    channels = candidate_torch_channels(NvidiaInfo(True, "GPU", "999", "12.6", ""))
    assert channels[0] == "cu126"
    assert "cu130" not in channels
    assert channels[-1] == "cpu"


def test_no_nvidia_selects_cpu_pytorch_channel():
    assert candidate_torch_channels(NvidiaInfo()) == ["cpu"]


def test_gpu_capability_classification_covers_comparison_and_ablation():
    comparison = PlannedItem(0, 0, "CALO", None)
    tlbo = PlannedItem(1, 0, "TLBO", None)
    complete = PlannedItem(2, 0, "Complete CALO", AblationSpec("Complete CALO", "CALO", {}))
    no_ai = PlannedItem(
        3,
        0,
        "CALO Core v2 without AI",
        AblationSpec("CALO Core v2 without AI", "CALO", {"use_ai": False}),
    )
    assert item_uses_calo_ai("comparison", comparison)
    assert item_uses_calo_ai("comparison", tlbo)
    assert item_uses_calo_ai("ablation", complete)
    assert item_uses_calo_ai("ablation", no_ai)


def _snapshot(cuda_utilization=40.0, ram=30.0):
    cuda = DeviceSnapshot(
        device_id="cuda:0",
        backend="cuda",
        index=0,
        name="NVIDIA GPU",
        available=True,
        utilization_percent=cuda_utilization,
        memory_percent=30.0,
    )
    return ResourceSnapshot(cpu_percent=20.0, devices=(cuda,), system_memory_percent=ram)


def test_resource_admission_thresholds_are_soft_and_bounded():
    low = _snapshot(40.0)
    high_gpu = _snapshot(75.0)
    cuda_device = low.by_backend("cuda")[0]
    cuda_device_high = high_gpu.by_backend("cuda")[0]
    assert accelerator_admission_allowed(cuda_device, 70, 85, 0, 1)
    assert not accelerator_admission_allowed(cuda_device_high, 70, 85, 0, 1)
    assert not accelerator_admission_allowed(cuda_device, 70, 85, 1, 1)
    assert cpu_admission_allowed(low, 50, 0)


def test_accelerator_priority_is_cuda_then_xpu():
    xpu = DeviceSnapshot("xpu:0", "xpu", 0, "Intel GPU", True, None, 20.0)
    cuda = DeviceSnapshot("cuda:0", "cuda", 0, "NVIDIA GPU", True, 30.0, 25.0)
    snapshot = ResourceSnapshot(10.0, devices=(xpu, cuda), system_memory_percent=20.0)
    assert [device.device_id for device in prioritized_accelerators(snapshot)] == [
        "cuda:0",
        "xpu:0",
    ]


def test_xpu_without_utilization_uses_memory_and_job_cap_for_admission():
    xpu = DeviceSnapshot("xpu:0", "xpu", 0, "Intel GPU", True, None, 30.0)
    assert accelerator_admission_allowed(xpu, 70, 85, active_jobs=0, max_jobs=2)
    assert not accelerator_admission_allowed(xpu, 70, 85, active_jobs=2, max_jobs=2)
    full_memory = DeviceSnapshot("xpu:0", "xpu", 0, "Intel GPU", True, None, 90.0)
    assert not accelerator_admission_allowed(full_memory, 70, 85, active_jobs=0, max_jobs=2)


def test_cpu_admission_respects_utilization_and_system_memory_safety_limits():
    low_memory = ResourceSnapshot(20.0, system_memory_percent=40.0)
    high_memory = ResourceSnapshot(20.0, system_memory_percent=90.0)
    high_cpu = ResourceSnapshot(75.0, system_memory_percent=40.0)
    assert cpu_admission_allowed(low_memory, 50, 0, memory_limit_percent=85)
    assert not cpu_admission_allowed(high_memory, 50, 0, memory_limit_percent=85)
    assert not cpu_admission_allowed(high_cpu, 50, 0, memory_limit_percent=85)


def test_pip_raw_progress_parser_and_download_item_extraction():
    from calo_bootstrap.prerequisites import _human_download_item, _parse_pip_raw_progress

    assert _parse_pip_raw_progress("Progress 1048576 of 8388608") == (1048576, 8388608)
    assert _parse_pip_raw_progress("Collecting torch") is None
    assert (
        _human_download_item(
            "Downloading https://download.pytorch.org/whl/cu126/torch-2.7.1%2Bcu126-cp311-cp311-win_amd64.whl (2.5 GB)"
        )
        == "torch-2.7.1%2Bcu126-cp311-cp311-win_amd64.whl"
    )


def test_weighted_lane_plan_splits_accelerator_eligible_jobs_50_30_20():
    from calo_rpd_studio.compute.resource_scheduler import build_weighted_lane_plan

    plan = [PlannedItem(i, i, "CALO", None) for i in range(10)]
    lanes, summary = build_weighted_lane_plan(
        plan,
        "comparison",
        cuda_available=True,
        xpu_available=True,
        cuda_share=50,
        xpu_share=30,
        cpu_share=20,
    )
    assert sum(lane == "cuda" for lane in lanes.values()) == 5
    assert sum(lane == "xpu" for lane in lanes.values()) == 3
    assert sum(lane == "cpu" for lane in lanes.values()) == 2
    assert summary.accelerator_eligible_jobs == 10
    assert summary.cpu_only_jobs == 0


def test_weighted_lane_plan_routes_every_v3_algorithm_to_accelerator_lanes():
    from calo_rpd_studio.compute.resource_scheduler import build_weighted_lane_plan

    plan = [
        PlannedItem(0, 0, "CALO", None),
        PlannedItem(1, 0, "TLBO", None),
        PlannedItem(2, 0, "PSO", None),
        PlannedItem(3, 1, "QODE", None),
    ]
    lanes, summary = build_weighted_lane_plan(
        plan,
        "comparison",
        cuda_available=True,
        xpu_available=True,
        cuda_share=50,
        xpu_share=30,
        cpu_share=20,
    )
    assert summary.accelerator_eligible_jobs == 4
    assert summary.cpu_only_jobs == 0
    assert all(lane in {"cuda", "xpu", "cpu"} for lane in lanes.values())


def test_weighted_lane_plan_assigns_all_jobs_to_cuda_when_requested():
    from calo_rpd_studio.compute.resource_scheduler import build_weighted_lane_plan

    plan = [
        PlannedItem(i, i // 8, name, None)
        for i, name in enumerate(
            (["CALO", "TLBO", "PSO", "CLPSO", "MTLA-DE", "QODE", "GWO", "WOA"] * 50)
        )
    ]
    lanes, summary = build_weighted_lane_plan(
        plan,
        "comparison",
        cuda_available=True,
        xpu_available=True,
        cuda_share=100,
        xpu_share=0,
        cpu_share=0,
    )
    assert len(plan) == 400
    assert summary.cuda_jobs == 400
    assert summary.xpu_jobs == 0
    assert summary.total_cpu_jobs == 0
    assert set(lanes.values()) == {"cuda"}


def test_weighted_lane_plan_redistributes_when_xpu_is_unavailable():
    from calo_rpd_studio.compute.resource_scheduler import build_weighted_lane_plan

    plan = [PlannedItem(i, i, "CALO", None) for i in range(10)]
    lanes, summary = build_weighted_lane_plan(
        plan,
        "comparison",
        cuda_available=True,
        xpu_available=False,
        cuda_share=50,
        xpu_share=30,
        cpu_share=20,
    )
    assert summary.xpu_jobs == 0
    assert sum(lane == "cuda" for lane in lanes.values()) == 7
    assert sum(lane == "cpu" for lane in lanes.values()) == 3
