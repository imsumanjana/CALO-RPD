from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tomllib

import pytest

from calo_rpd_studio.benchmarking.freeze import verify_freeze_manifest
from calo_rpd_studio.version import FREEZE_ID, FREEZE_MANIFEST, RELEASE_NAME, VERSION

pytestmark = pytest.mark.skipif(VERSION != "6.1.0", reason="historical v6.1 release gate")


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_v610_release_identity_is_consistent():
    root = _root()
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text(encoding="utf-8"))
    assert project["version"] == VERSION == metadata["version"] == "6.1.0"
    assert RELEASE_NAME == metadata["release_name"] == "Protected Training Queue and Capability-Aware Scheduling"
    assert FREEZE_ID == "calo_v610_software_release"
    assert FREEZE_MANIFEST == "calo_v610_freeze.json"


def test_v610_freeze_verifies_and_covers_beta_architecture():
    root = _root()
    freeze_path = root / "calo_rpd_studio" / "data" / "frozen" / FREEZE_MANIFEST
    result = verify_freeze_manifest(freeze_path, project_root=root)
    assert result.passed is True
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert payload["software_version"] == "6.1.0"
    required = {
        "calo_rpd_studio/app/state_manager.py",
        "calo_rpd_studio/app/workflow_manager.py",
        "calo_rpd_studio/app/main_window.py",
        "calo_rpd_studio/compute/topology.py",
        "calo_rpd_studio/compute/training_resources.py",
        "calo_rpd_studio/algorithms/calo/training.py",
        "calo_rpd_studio/algorithms/calo/competitive_training.py",
        "calo_rpd_studio/algorithms/calo/heterogeneous_training.py",
        "calo_rpd_studio/gui/panels/dashboard_panel.py",
        "calo_rpd_studio/gui/panels/calo_intelligence_panel.py",
    }
    assert required <= set(payload["files"])
    scope = payload["frozen_scope"]
    assert scope["global_training_exclusive_ui_lock"] is True
    assert scope["queued_total_vs_concurrent_branch_scheduler"] is True
    assert scope["exact_resume_queue_time_slicing"] is True
    assert scope["global_cpu_worker_budget_enforced"] is True
    assert scope["automatic_accelerator_to_cpu_branch_spillover"] is False
    assert scope["xpu_capability_aware_scheduling"] is True
    assert scope["xpu_sidecar_full_branch_certified"] is False
    assert scope["dynamic_thermal_power_governor"] is False


def test_v610_release_evidence_files_exist():
    root = _root()
    for name in (
        "CALO-RPD-v6.1.0_IMPLEMENTATION_REPORT.md",
        "CALO-RPD-v6.1.0_DEEP_POST_GENERATION_AUDIT.txt",
        "FINDINGS_CLOSURE_v6.1.0.csv",
        "RELEASE_VALIDATION.md",
        "RELEASE_METADATA.json",
    ):
        assert (root / name).is_file(), name
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "CALO-RPD Studio v6.1.0" in readme
    assert "Global Training Exclusive Lock" in readme
    assert "Maximum simultaneous branches" in readme
    assert "XPU capability-aware scheduling" in readme


def test_v610_root_manifest_matches_every_listed_file_and_current_freeze():
    root = _root()
    manifest = root / "MANIFEST.sha256"
    assert manifest.is_file()
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        rows[relative] = digest
    assert "MANIFEST.sha256" not in rows
    for relative, expected in rows.items():
        path = root / relative
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
    assert "calo_rpd_studio/data/frozen/calo_v610_freeze.json" in rows
    assert "tests/unit/test_v610_beta_architecture.py" in rows
    assert "tests/unit/test_v610_release_integrity.py" in rows


def test_v610_metadata_records_verified_release_artifacts():
    root = _root()
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text(encoding="utf-8"))
    freeze = metadata["freeze"]
    assert freeze["status"] == "VERIFIED"
    assert int(freeze["checked_files"]) > 0
    assert freeze["missing_files"] == 0
    assert freeze["changed_files"] == 0
    validation = metadata["validation"]
    assert validation["compileall"] == "PASS"
    assert str(validation["root_manifest"]).startswith("VERIFIED")
