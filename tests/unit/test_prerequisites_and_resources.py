from __future__ import annotations

import pytest

import calo_bootstrap.prerequisites as prerequisites
from calo_bootstrap.prerequisites import (
    NvidiaInfo,
    TorchInfo,
    candidate_torch_channels,
    project_torch_requirement,
    torch_version_satisfies_requirement,
)
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


def test_project_torch_requirement_is_read_from_project_metadata(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "example"\ndependencies = ["torch>=2.10,<2.11", "numpy>=1.26"]\n',
        encoding="utf-8",
    )
    assert project_torch_requirement(tmp_path) == "torch>=2.10,<2.11"


def test_torch_version_contract_accepts_cuda_local_version_and_rejects_drift():
    requirement = "torch>=2.10,<2.11"
    assert torch_version_satisfies_requirement("2.10.0+cu128", requirement)
    assert torch_version_satisfies_requirement("2.10.1+cpu", requirement)
    assert not torch_version_satisfies_requirement("2.11.0+cu128", requirement)
    assert not torch_version_satisfies_requirement("2.9.1+cu128", requirement)


def test_scan_environment_rejects_pytorch_outside_project_contract(monkeypatch):
    monkeypatch.setattr(prerequisites, "_distribution_version", lambda unused: "1.0")
    monkeypatch.setattr(prerequisites, "detect_core_import_errors", lambda: {})
    monkeypatch.setattr(prerequisites, "detect_nvidia", lambda: NvidiaInfo())
    monkeypatch.setattr(
        prerequisites,
        "detect_torch",
        lambda: TorchInfo(installed=True, version="2.11.0+cu128"),
    )
    monkeypatch.setattr(
        prerequisites, "project_torch_requirement", lambda unused=None: "torch>=2.10,<2.11"
    )

    report = prerequisites.scan_environment()

    assert not report.torch_version_compatible
    assert not report.mandatory_ready
    assert "does not satisfy" in report.message


def test_installer_repairs_broken_scientific_import_before_pytorch(
    monkeypatch, tmp_path
):
    calls: list[list[str]] = []
    reports = iter(
        [
            {"NumPy": "ImportError: missing _multiarray_umath"},
            {},
        ]
    )
    monkeypatch.setattr(prerequisites, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        prerequisites,
        "_pip",
        lambda args, *unused_args, **unused_kwargs: calls.append(list(args)) or 0,
    )
    monkeypatch.setattr(prerequisites, "detect_core_import_errors", lambda: next(reports))
    monkeypatch.setattr(
        prerequisites,
        "detect_torch",
        lambda: (_ for _ in ()).throw(AssertionError("stop after scientific repair test")),
    )

    with pytest.raises(AssertionError, match="scientific repair test"):
        prerequisites.install_or_repair()

    repair_calls = [call for call in calls if "--force-reinstall" in call]
    assert len(repair_calls) == 1
    assert any("numpy" in item.lower() for item in repair_calls[0])
    assert not any("torch" in " ".join(call).lower() for call in calls)


def test_installer_stops_before_pytorch_when_targeted_scientific_repair_fails(
    monkeypatch, tmp_path
):
    calls: list[list[str]] = []
    monkeypatch.setattr(prerequisites, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        prerequisites,
        "_pip",
        lambda args, *unused_args, **unused_kwargs: calls.append(list(args)) or 0,
    )
    monkeypatch.setattr(
        prerequisites,
        "detect_core_import_errors",
        lambda: {"NumPy": "ImportError: missing _multiarray_umath"},
    )
    monkeypatch.setattr(
        prerequisites,
        "detect_torch",
        lambda: (_ for _ in ()).throw(AssertionError("PyTorch inspection must not run")),
    )

    with pytest.raises(RuntimeError, match="PyTorch was not changed"):
        prerequisites.install_or_repair()

    assert not any("torch" in " ".join(call).lower() for call in calls)


def test_installer_keeps_existing_torch_until_replacement_install_succeeds(
    monkeypatch, tmp_path
):
    calls: list[list[str]] = []
    reports = iter(
        [
            TorchInfo(installed=True, version="2.11.0+cu128", cuda_available=True, gpu_test_passed=True),
            TorchInfo(installed=True, version="2.10.0+cu128", cuda_available=True, gpu_test_passed=True),
        ]
    )
    monkeypatch.setattr(prerequisites, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        prerequisites,
        "_pip",
        lambda args, *unused_args, **unused_kwargs: calls.append(list(args)) or 0,
    )
    monkeypatch.setattr(prerequisites, "detect_core_import_errors", lambda: {})
    monkeypatch.setattr(
        prerequisites,
        "detect_nvidia",
        lambda: NvidiaInfo(True, "GPU", "999", "12.8", ""),
    )
    monkeypatch.setattr(prerequisites, "detect_torch", lambda: next(reports))
    monkeypatch.setattr(prerequisites, "candidate_torch_channels", lambda unused: ["cu128"])
    monkeypatch.setattr(
        prerequisites, "project_torch_requirement", lambda unused=None: "torch>=2.10,<2.11"
    )
    monkeypatch.setattr(
        prerequisites,
        "scan_environment",
        lambda: prerequisites.EnvironmentReport(
            python_ok=True,
            python_version="3.11",
            interpreter="python",
            virtual_environment=True,
            core_packages={},
            missing_core_packages=[],
            core_import_errors={},
            nvidia=NvidiaInfo(True, "GPU", "999", "12.8", ""),
            torch=TorchInfo(
                installed=True,
                version="2.10.0+cu128",
                cuda_available=True,
                gpu_test_passed=True,
            ),
            torch_requirement="torch>=2.10,<2.11",
            torch_version_compatible=True,
            mandatory_ready=True,
            gpu_ready=True,
            recommended_backend="cuda:0",
            message="ready",
        ),
    )
    monkeypatch.setattr(prerequisites, "save_environment_state", lambda unused: None)

    prerequisites.install_or_repair()

    first_torch_action = next(
        call for call in calls if any("torch" in item.lower() for item in call)
    )
    assert first_torch_action[0] == "install"
    assert ["uninstall", "-y", "torch"] not in calls


def test_installer_does_not_cycle_wheels_after_pytorch_verification_exception(
    monkeypatch, tmp_path
):
    calls: list[list[str]] = []
    reports = iter(
        [
            TorchInfo(),
            TorchInfo(
                installed=True,
                version="2.10.0+cu128",
                error_stage="import",
                error="ImportError: dependent DLL could not be loaded",
            ),
        ]
    )
    monkeypatch.setattr(prerequisites, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        prerequisites,
        "_pip",
        lambda args, *unused_args, **unused_kwargs: calls.append(list(args)) or 0,
    )
    monkeypatch.setattr(prerequisites, "detect_core_import_errors", lambda: {})
    monkeypatch.setattr(
        prerequisites,
        "detect_nvidia",
        lambda: NvidiaInfo(True, "GPU", "999", "12.8", ""),
    )
    monkeypatch.setattr(prerequisites, "detect_torch", lambda: next(reports))
    monkeypatch.setattr(prerequisites, "candidate_torch_channels", lambda unused: ["cu128", "cu126"])
    monkeypatch.setattr(
        prerequisites, "project_torch_requirement", lambda unused=None: "torch>=2.10,<2.11"
    )

    with pytest.raises(RuntimeError, match="No additional multi-gigabyte wheel channels"):
        prerequisites.install_or_repair()

    torch_installs = [
        call for call in calls if call and call[0] == "install" and any("torch" in item for item in call)
    ]
    torch_uninstalls = [call for call in calls if call[:3] == ["uninstall", "-y", "torch"]]
    assert len(torch_installs) == 1
    assert torch_uninstalls == []

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
    gib = 1024**3
    cuda = DeviceSnapshot(
        device_id="cuda:0",
        backend="cuda",
        index=0,
        name="NVIDIA GPU",
        available=True,
        utilization_percent=cuda_utilization,
        memory_percent=30.0,
        memory_total_bytes=8 * gib,
        memory_available_bytes=5 * gib,
    )
    return ResourceSnapshot(
        cpu_percent=20.0,
        devices=(cuda,),
        system_memory_percent=ram,
        system_memory_total_bytes=32 * gib,
        system_memory_available_bytes=int(32 * gib * (100.0 - ram) / 100.0),
    )


def test_resource_admission_ignores_utilization_and_respects_job_caps():
    low = _snapshot(40.0)
    high_gpu = _snapshot(75.0)
    cuda_device = low.by_backend("cuda")[0]
    cuda_device_high = high_gpu.by_backend("cuda")[0]
    assert accelerator_admission_allowed(cuda_device, 0, 1)
    assert accelerator_admission_allowed(cuda_device_high, 0, 1)
    assert not accelerator_admission_allowed(cuda_device, 1, 1)
    assert cpu_admission_allowed(low, 0, 1)


def test_accelerator_priority_contains_only_cuda_devices():
    unsupported = DeviceSnapshot("other:0", "other", 0, "Unsupported GPU", True, None, 20.0)
    cuda = DeviceSnapshot("cuda:0", "cuda", 0, "NVIDIA GPU", True, 30.0, 25.0)
    snapshot = ResourceSnapshot(10.0, devices=(unsupported, cuda), system_memory_percent=20.0)
    assert [device.device_id for device in prioritized_accelerators(snapshot)] == ["cuda:0"]


def test_accelerator_without_utilization_uses_memory_and_job_cap_for_admission():
    gib = 1024**3
    cuda = DeviceSnapshot(
        "cuda:0",
        "cuda",
        0,
        "NVIDIA GPU",
        True,
        None,
        30.0,
        memory_total_bytes=8 * gib,
        memory_available_bytes=5 * gib,
    )
    assert accelerator_admission_allowed(cuda, active_jobs=0, max_jobs=2)
    assert not accelerator_admission_allowed(cuda, active_jobs=2, max_jobs=2)
    full_memory = DeviceSnapshot(
        "cuda:0",
        "cuda",
        0,
        "NVIDIA GPU",
        True,
        None,
        100.0,
        memory_total_bytes=8 * gib,
        memory_available_bytes=0,
    )
    assert not accelerator_admission_allowed(full_memory, active_jobs=0, max_jobs=2)


def test_cpu_admission_uses_available_ram_not_cpu_utilization():
    gib = 1024**3
    available = ResourceSnapshot(
        99.0,
        system_memory_percent=90.0,
        system_memory_total_bytes=32 * gib,
        system_memory_available_bytes=2 * gib,
    )
    exhausted = ResourceSnapshot(
        0.0,
        system_memory_percent=0.0,
        system_memory_total_bytes=32 * gib,
        system_memory_available_bytes=0,
    )
    assert cpu_admission_allowed(available, active_cpu_jobs=0, max_jobs=2)
    assert not cpu_admission_allowed(available, active_cpu_jobs=2, max_jobs=2)
    assert not cpu_admission_allowed(exhausted, active_cpu_jobs=0, max_jobs=2)


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


def test_automatic_lane_plan_routes_all_compatible_jobs_to_cuda():
    from calo_rpd_studio.compute.resource_scheduler import build_weighted_lane_plan

    plan = [PlannedItem(i, i, "CALO", None) for i in range(10)]
    lanes, summary = build_weighted_lane_plan(
        plan,
        "comparison",
        cuda_available=True,
    )
    assert sum(lane == "cuda" for lane in lanes.values()) == 10
    assert sum(lane == "cpu" for lane in lanes.values()) == 0
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
    )
    assert summary.accelerator_eligible_jobs == 4
    assert summary.cpu_only_jobs == 0
    assert set(lanes.values()) == {"cuda"}


def test_automatic_lane_plan_assigns_large_campaign_to_cuda():
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
    )
    assert len(plan) == 400
    assert summary.cuda_jobs == 400
    assert summary.total_cpu_jobs == 0
    assert set(lanes.values()) == {"cuda"}


def test_weighted_lane_plan_redistributes_when_cuda_is_unavailable():
    from calo_rpd_studio.compute.resource_scheduler import build_weighted_lane_plan

    plan = [PlannedItem(i, i, "CALO", None) for i in range(10)]
    lanes, summary = build_weighted_lane_plan(
        plan,
        "comparison",
        cuda_available=False,
    )
    assert sum(lane == "cuda" for lane in lanes.values()) == 0
    assert sum(lane == "cpu" for lane in lanes.values()) == 10
