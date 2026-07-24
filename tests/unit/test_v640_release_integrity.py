from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tomllib

import pytest

from calo_rpd_studio.benchmarking.freeze import verify_freeze_manifest
from calo_rpd_studio.version import FREEZE_ID, FREEZE_MANIFEST, RELEASE_NAME, VERSION

pytestmark = pytest.mark.skipif(VERSION != "6.4.0", reason="historical v6.4.0 release gate")


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_v640_release_identity_is_consistent():
    root = _root()
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text(encoding="utf-8"))
    assert project["version"] == VERSION == metadata["version"] == "6.4.0"
    assert RELEASE_NAME == metadata["release_name"] == "Stage-B Device-Resident Policy Training"
    assert FREEZE_ID == "calo_v640_software_release"
    assert FREEZE_MANIFEST == "calo_v640_freeze.json"


def test_v640_freeze_verifies_and_covers_stage_b_science():
    root = _root()
    freeze_path = root / "calo_rpd_studio" / "data" / "frozen" / FREEZE_MANIFEST
    result = verify_freeze_manifest(freeze_path, project_root=root)
    assert result.passed
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    required = {
        "calo_rpd_studio/algorithms/calo/device_resident_synthetic.py",
        "calo_rpd_studio/algorithms/calo/heterogeneous_training.py",
        "calo_rpd_studio/algorithms/calo/training.py",
        "calo_rpd_studio/gui/panels/calo_intelligence_panel.py",
        "calo_rpd_studio/scripts/validate_stage_b_synthetic.py",
        "calo_rpd_studio/data/examples/policy_development_active_loss.yaml",
    }
    assert required <= set(payload["files"])
    scope = payload["frozen_scope"]
    assert scope["stage_b_device_resident_synthetic_evaluation"] is True
    assert scope["stage_b_cross_episode_synthetic_microbatching"] is True
    assert scope["stage_b_synthetic_startup_parity_fail_closed"] is True
    assert scope["stage_b_real_orpd_development_suite_configurable"] is True
    assert scope["stage_b_full_stochastic_calo_controller_gpu_resident"] is False


def test_v640_release_evidence_and_stage_b_boundaries_exist():
    root = _root()
    for name in (
        "CALO-RPD-v6.4.0_IMPLEMENTATION_REPORT.md",
        "CALO-RPD-v6.4.0_DEEP_POST_GENERATION_AUDIT.txt",
        "FINDINGS_CLOSURE_v6.4.0.csv",
        "HARDWARE_QUALIFICATION_STATUS.json",
        "SCIENTIFIC_EQUIVALENCE_STATUS.json",
        "RELEASE_VALIDATION.md",
        "RELEASE_METADATA.json",
    ):
        assert (root / name).is_file(), name
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text(encoding="utf-8"))
    assert metadata["stage_b"]["device_resident_synthetic_evaluation"] is True
    assert metadata["stage_b"]["real_orpd_development_suite_configured"] is True
    assert metadata["stage_b"]["full_stochastic_calo_controller_gpu_resident"] is False


def test_v640_root_manifest_matches_every_packaged_file():
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


def test_v640_metadata_records_target_hardware_boundary():
    root = _root()
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text(encoding="utf-8"))
    assert metadata["freeze"]["status"] == "VERIFIED"
    assert metadata["validation"]["compileall"] == "PASS"
    assert metadata["validation"]["physical_cuda_xpu"] == "NOT AVAILABLE IN BUILD RUNTIME"
    assert metadata["validation"]["pyqt6"] == "NOT AVAILABLE IN BUILD RUNTIME"
    assert metadata["validation"]["pypower"] == "NOT AVAILABLE IN BUILD RUNTIME"
    assert metadata["validation"]["guaranteed_high_gpu_utilization"] is False
