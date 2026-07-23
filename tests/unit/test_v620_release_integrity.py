from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tomllib

from calo_rpd_studio.benchmarking.freeze import verify_freeze_manifest
from calo_rpd_studio.version import FREEZE_ID, FREEZE_MANIFEST, RELEASE_NAME, VERSION


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_v620_release_identity_is_consistent():
    root = _root()
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text(encoding="utf-8"))
    assert project["version"] == VERSION == metadata["version"] == "6.2.0"
    assert RELEASE_NAME == metadata["release_name"] == "Adaptive Compute Protection, Recovery and Scientific Qualification"
    assert FREEZE_ID == "calo_v620_software_release"
    assert FREEZE_MANIFEST == "calo_v620_freeze.json"


def test_v620_freeze_verifies_and_covers_rc_final_architecture():
    root = _root()
    freeze_path = root / "calo_rpd_studio" / "data" / "frozen" / FREEZE_MANIFEST
    result = verify_freeze_manifest(freeze_path, project_root=root)
    assert result.passed
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    required = {
        "calo_rpd_studio/compute/governor.py",
        "calo_rpd_studio/compute/provenance.py",
        "calo_rpd_studio/compute/soak.py",
        "calo_rpd_studio/compute/scientific_equivalence.py",
        "calo_rpd_studio/app/session_recovery.py",
        "calo_rpd_studio/app/workspaces.py",
        "calo_rpd_studio/app/main_window.py",
        "calo_rpd_studio/algorithms/calo/competitive_training.py",
        "calo_rpd_studio/app/experiment_manager.py",
        "calo_rpd_studio/validation/gui_contract.py",
    }
    assert required <= set(payload["files"])
    scope = payload["frozen_scope"]
    assert scope["dynamic_thermal_power_governor"] is True
    assert scope["staged_compute_startup"] is True
    assert scope["hash_chained_compute_provenance"] is True
    assert scope["workspace_schema_v3_migration"] is True
    assert scope["unclean_application_session_recovery"] is True
    assert scope["hardware_soak_qualification_protocol"] is True
    assert scope["scheduling_scientific_equivalence_protocol"] is True
    assert scope["physical_multi_hour_hardware_soak_certified_in_build_runtime"] is False


def test_v620_release_evidence_files_exist_and_are_truthful():
    root = _root()
    for name in (
        "CALO-RPD-v6.2.0_IMPLEMENTATION_REPORT.md",
        "CALO-RPD-v6.2.0_DEEP_POST_GENERATION_AUDIT.txt",
        "FINDINGS_CLOSURE_v6.2.0.csv",
        "HARDWARE_QUALIFICATION_STATUS.json",
        "SCIENTIFIC_EQUIVALENCE_STATUS.json",
        "RELEASE_VALIDATION.md",
        "RELEASE_METADATA.json",
    ):
        assert (root / name).is_file(), name
    hardware = json.loads((root / "HARDWARE_QUALIFICATION_STATUS.json").read_text(encoding="utf-8"))
    assert "PENDING" in hardware["status"]
    assert "NOT_EXECUTED" in hardware["physical_multi_hour_cuda_soak"]


def test_v620_root_manifest_matches_every_packaged_file():
    root = _root()
    manifest = root / "MANIFEST.sha256"
    rows = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative = line.split("  ", 1)
            rows[relative] = digest
    assert "MANIFEST.sha256" not in rows
    for relative, expected in rows.items():
        path = root / relative
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
    # Complete package accounting: every non-cache regular file except the manifest itself is listed.
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


def test_v620_metadata_records_environment_boundaries():
    root = _root()
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text(encoding="utf-8"))
    assert metadata["freeze"]["status"] == "VERIFIED"
    assert metadata["validation"]["compileall"] == "PASS"
    assert metadata["validation"]["physical_cuda_xpu"] == "NOT AVAILABLE IN BUILD RUNTIME"
    assert metadata["validation"]["pyqt6"] == "NOT AVAILABLE IN BUILD RUNTIME"
    assert metadata["validation"]["pypower"] == "NOT AVAILABLE IN BUILD RUNTIME"
