from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tomllib

from calo_rpd_studio.benchmarking.freeze import verify_freeze_manifest
from calo_rpd_studio.version import FREEZE_ID, FREEZE_MANIFEST, RELEASE_NAME, VERSION


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_v670_release_identity_is_consistent():
    root = _root()
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text(encoding="utf-8"))
    assert project["version"] == VERSION == metadata["version"] == "6.7.0"
    assert RELEASE_NAME == metadata["release_name"] == "Hardware Runtime Binding & Telemetry Integrity"
    assert FREEZE_ID == "calo_v670_software_release"
    assert FREEZE_MANIFEST == "calo_v670_freeze.json"


def test_v670_freeze_verifies_hardware_runtime_scope():
    root = _root()
    freeze_path = root / "calo_rpd_studio" / "data" / "frozen" / FREEZE_MANIFEST
    result = verify_freeze_manifest(freeze_path, project_root=root)
    assert result.passed
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert "calo_rpd_studio/compute/device_binding.py" in payload["files"]
    scope = payload["frozen_scope"]
    for key in (
        "v670_nvml_dependency_declared",
        "v670_cuda_compute_independent_from_telemetry",
        "v670_stable_nvidia_uuid_pci_mapping",
        "v670_canonical_cross_runtime_device_binding",
        "v670_xpu_sidecar_fp64_identity_memory_telemetry",
        "v670_planned_vs_actual_device_attestation",
        "v670_nonmisleading_os_adapter_labels",
    ):
        assert scope[key] is True, key
    assert scope["v650_torch_newton_backtracking_cpu_parity"] is True
    assert scope["v660_unified_feasibility_tolerance_and_ordering"] is True


def test_v670_dependency_and_binding_contract_is_declared():
    root = _root()
    requirements = (root / "requirements-core.txt").read_text(encoding="utf-8")
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    prereqs = (root / "calo_bootstrap" / "prerequisites.py").read_text(encoding="utf-8")
    assert "nvidia-ml-py>=13,<14" in requirements
    assert "nvidia-ml-py>=13,<14" in project["dependencies"]
    assert "nvidia-ml-py>=13,<14" in prereqs
    for relative in (
        "calo_rpd_studio/app/experiment_manager.py",
        "calo_rpd_studio/compute/persistent_accelerator_worker.py",
        "calo_rpd_studio/compute/persistent_accelerator_sidecar.py",
        "calo_rpd_studio/compute/xpu_worker.py",
    ):
        text = (root / relative).read_text(encoding="utf-8")
        assert "bind_config_to_device" in text, relative


def test_v670_release_evidence_records_all_runtime_findings():
    root = _root()
    for name in (
        "CALO-RPD-v6.7.0_IMPLEMENTATION_REPORT.md",
        "CALO-RPD-v6.7.0_DEEP_POST_GENERATION_AUDIT.txt",
        "FINDINGS_CLOSURE_v6.7.0.csv",
        "HARDWARE_QUALIFICATION_STATUS.json",
        "SCIENTIFIC_EQUIVALENCE_STATUS.json",
        "STAGE_B_ACCELERATOR_QUALIFICATION_STATUS.json",
        "RELEASE_VALIDATION.md",
        "RELEASE_METADATA.json",
    ):
        assert (root / name).is_file(), name
    with (root / "FINDINGS_CLOSURE_v6.7.0.csv").open(encoding="utf-8", newline="") as stream:
        rows = {row["id"]: row for row in csv.DictReader(stream)}
    assert set(rows) == {f"V67-R{i:02d}" for i in range(1, 8)}
    assert all(row["status"] == "RESOLVED" for row in rows.values())


def test_v670_root_manifest_matches_every_packaged_file():
    root = _root()
    manifest = root / "MANIFEST.sha256"
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative = line.split("  ", 1)
            rows[relative] = digest
    assert "MANIFEST.sha256" not in rows
    for relative, expected in rows.items():
        path = root / relative
        assert path.is_file(), relative
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == expected, relative
    excluded_parts = {"__pycache__", ".pytest_cache", ".git"}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "MANIFEST.sha256"
        and not any(part in excluded_parts for part in path.parts)
        and path.suffix not in {".pyc", ".pyo"}
    }
    assert set(rows) == actual


def test_v670_metadata_is_truthful_about_validation_boundaries():
    root = _root()
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text(encoding="utf-8"))
    assert metadata["freeze"]["status"] == "VERIFIED"
    assert metadata["freeze"]["checked_files"] == 148
    assert metadata["validation"]["compileall"] == "PASS"
    assert metadata["validation"]["focused_v670_hardware_runtime"] == "6 passed"
    assert metadata["validation"]["physical_cuda_xpu"] == "NOT AVAILABLE IN BUILD RUNTIME"
    assert metadata["validation"]["guaranteed_high_gpu_utilization"] is False
    assert metadata["v670_hardware_runtime_closure"]["resolved"] == 7
