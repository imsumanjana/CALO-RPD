from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tomllib

import pytest

from calo_rpd_studio.benchmarking.freeze import verify_freeze_manifest
from calo_rpd_studio.version import FREEZE_ID, FREEZE_MANIFEST, RELEASE_NAME, VERSION

pytestmark = pytest.mark.skipif(VERSION != "6.3.0", reason="historical v6.3.0 release gate")


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_v630_release_identity_is_consistent():
    root = _root()
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text(encoding="utf-8"))
    assert project["version"] == VERSION == metadata["version"] == "6.3.0"
    assert RELEASE_NAME == metadata["release_name"] == "Training Status and Device Reporting Correctness"
    assert FREEZE_ID == "calo_v630_software_release"
    assert FREEZE_MANIFEST == "calo_v630_freeze.json"


def test_v630_freeze_verifies_and_covers_stage_a_reporting_contract():
    root = _root()
    freeze_path = root / "calo_rpd_studio" / "data" / "frozen" / FREEZE_MANIFEST
    result = verify_freeze_manifest(freeze_path, project_root=root)
    assert result.passed
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    required = {
        "calo_rpd_studio/algorithms/calo/competitive_training.py",
        "calo_rpd_studio/compute/training_resources.py",
        "calo_rpd_studio/gui/panels/calo_intelligence_panel.py",
        "calo_rpd_studio/gui/panels/dashboard_panel.py",
        "calo_rpd_studio/version.py",
    }
    assert required <= set(payload["files"])
    scope = payload["frozen_scope"]
    assert scope["truthful_training_progress_reporting"] is True
    assert scope["selected_vs_recommended_rollout_reporting_separated"] is True
    assert scope["protected_runtime_device_mapping_reporting"] is True
    assert scope["stage_a_gpu_resident_environment_claim"] is False


def test_v630_release_evidence_files_exist_and_stage_b_boundary_is_explicit():
    root = _root()
    for name in (
        "CALO-RPD-v6.3.0_IMPLEMENTATION_REPORT.md",
        "CALO-RPD-v6.3.0_DEEP_POST_GENERATION_AUDIT.txt",
        "FINDINGS_CLOSURE_v6.3.0.csv",
        "HARDWARE_QUALIFICATION_STATUS.json",
        "SCIENTIFIC_EQUIVALENCE_STATUS.json",
        "RELEASE_VALIDATION.md",
        "RELEASE_METADATA.json",
    ):
        assert (root / name).is_file(), name
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text(encoding="utf-8"))
    assert metadata["stage_a"]["synthetic_calo_environment_gpu_resident"] is False
    assert metadata["stage_a"]["normal_gui_development_cases_configured"] is False


def test_v630_root_manifest_matches_every_packaged_file():
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


def test_v630_metadata_records_environment_boundaries():
    root = _root()
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text(encoding="utf-8"))
    assert metadata["freeze"]["status"] == "VERIFIED"
    assert metadata["validation"]["compileall"] == "PASS"
    assert metadata["validation"]["physical_cuda_xpu"] == "NOT AVAILABLE IN BUILD RUNTIME"
    assert metadata["validation"]["pyqt6"] == "NOT AVAILABLE IN BUILD RUNTIME"
    assert metadata["validation"]["pypower"] == "NOT AVAILABLE IN BUILD RUNTIME"
